#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <ctime>
#include <cstring>
#include "cuda.h"
#include <cuda_runtime.h>
#include "common_functions.h"
#include "sm_20_intrinsics.h"
#include "host_defines.h"
#include <thrust/sort.h>
#include <thrust/scan.h>
#include "cudamatrix_types.cuh"



#  define CUDA_SAFE_KERNEL(call) {                                         \
	call;																					\
	cudaDeviceSynchronize();														\
	cudaError err = cudaGetLastError();										\
    if ( cudaSuccess != err) {                                               \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
                exit(EXIT_FAILURE);                                                  \
    } }

#define MAX_SMEM_PER_C2MESH 6144

__constant__ float pi_const = 3.1415927;
__constant__ int cells_per_bin = 8;


extern int3 ncells_per_bin_g;


extern "C" __host__
void xplist_transpose_(long int* xplist_d,
									   float* xplist_h,float* dt_prec,float* vzinit,int* ipf,
									   int* npartmax,int* ndims,int* direction,int* xpdata_only);


class Particlebin
{
public:
	uint ifirstp;
	uint ilastp;
	uint binid;

	__device__
	uint3 get_bin_position(int3 ncells);
};

class Mesh_data
{
public:
	// Mesh parameters
	int nr,nth,npsi,nrused,nthused,npsiused,nrfull,nthfull,npsifull;
	int nrpre,ntpre,nppre;
	int nvel,nQth;
	float rfac,tfac,pfac;
	float debyelen;

	float bdyfc,Ti,vd,cd,cB,Bz;

	float dp,dth,dpinv,dthinv,dpsi;
	float colnwt;

	int lat0,lap0;

	// Mesh arrays
	cudaMatrixf phi;
	cudaMatrixf phiaxis;
	cudaMatrixf rho;
	cudaMatrixf rhoDiag;
	cudaMatrixf rmesh; // r(nrsize)
	cudaMatrixf rccmesh; // rcc(nrsize)
	cudaMatrixf thmesh; // th(nthsize)
	cudaMatrixf tccmesh; // tcc(nthsize)
	cudaMatrixf thang; // thang(nthsize)
	cudaMatrixf pcc; // pcc(npsisize)
	cudaMatrixf volinv; // volinv(nrsize)
	cudaMatrixf zeta; // zeta(nrsize+1)
	cudaMatrixf zetahalf; // zetahalf(nrsize+1)
	cudaMatrixi irpre; // irpre(nrpre)
	cudaMatrixi itpre; // itpre(ntpre)
	cudaMatrixi ippre; // ippre(nppre)
	cudaMatrixf Qcom; // Qcom(nQth)
	cudaMatrixf Gcom; // Gcom(nvel,nQth)
	cudaMatrixf Vcom; // Vcom(nvel))

	cudaMatrixf nincell;
	cudaMatrixf vrincell;
	cudaMatrixf vr2incell;

	cudaMatrixf fluxofangle,fincellave;

	int nbins;
	Particlebin* bins;

	cudaMatrixf psum; // psum(1:nrsize-1,1:nthsize-1,1:npsisize-1)


	template<int doih>
	__device__
	float4 ptomesh(float x, float y, float z,int4* icell,float4* cellf,float &zetap);

	__device__
	int interppsi(float sp,float cp,float* pf);
	__device__
	int interpth(float ct,float* thf);
	__device__
	int interpr(float rp,float* rf);
	__device__
	float3 getaccel(float px,float py,float pz);
	__device__
	int boundary_intersection(float px1,float py1,float pz1,float& px2,float& py2,float& pz2);

	 __device__
	void probe_diags(float3 pin, float3 vin);
};


class XPdiags
{
public:
	float3* momout;
	float4* momprobe;
	int* ninner;
	bool device;

	__host__
	XPdiags(int location)
	{
		device = location;
		if(device == 0)
		{
			// Allocate Host Memory
			CUDA_SAFE_CALL(cudaMallocHost((void**)&momout,sizeof(float3)));
			CUDA_SAFE_CALL(cudaMallocHost((void**)&momprobe,sizeof(float4)));
			CUDA_SAFE_CALL(cudaMallocHost((void**)&ninner,sizeof(int)));

			momout->x = 0;
			momout->y = 0;
			momout->z = 0;

			momprobe->x = 0;
			momprobe->y = 0;
			momprobe->z = 0;
			momprobe->w = 0;

			*ninner= 0;
		}
		else
		{
			// Allocate device Memory
			CUDA_SAFE_CALL(cudaMalloc((void**)&momout,sizeof(float3)));
			CUDA_SAFE_CALL(cudaMalloc((void**)&momprobe,sizeof(float4)));
			CUDA_SAFE_CALL(cudaMalloc((void**)&ninner,sizeof(int)));


			CUDA_SAFE_CALL(cudaMemset(momout,0,sizeof(float3)));
			CUDA_SAFE_CALL(cudaMemset(momprobe,0,sizeof(float4)));
			CUDA_SAFE_CALL(cudaMemset(ninner,0,sizeof(int)));
		}
	}

	__host__
	void free(void)
	{
		if(device == 0)
		{
			// Free Host Memory
			CUDA_SAFE_CALL(cudaFreeHost(momout));
			CUDA_SAFE_CALL(cudaFreeHost(momprobe));
			CUDA_SAFE_CALL(cudaFreeHost(ninner));
		}
		else
		{
			// Free device Memory
			CUDA_SAFE_CALL(cudaFree(momout));
			CUDA_SAFE_CALL(cudaFree(momprobe));
			CUDA_SAFE_CALL(cudaFree(ninner));
		}
	}
};

#define nfloats_XPlist 9
#define nints_XPlist 3
class XPlist
{
public:
	float* px;
	float* py;
	float* pz;
	float* vx;
	float* vy;
	float* vz;
	float* dt_prec;
	float* vzinit;


	int* ipf;
	int* didileave;

	float* buffer;

	ushort* binid;
	int* particle_id;

	int nptcls;

	// Allocate particle list in specified memory
	__host__
	void allocate(int nptcls);

	__device__ __host__
	float** get_float_ptr(int i);

	__device__ __host__
	int** get_int_ptr(int i);

	__host__
	void sort(Particlebin* bins);

	__device__
	void calc_binid(Mesh_data* mesh,int3 ncells,int idx);

	__host__
	void advance(Mesh_data mesh,XPdiags diags,float* reinjlist_h,float dt,int &reinject_counter);

	 __device__
	void move(Mesh_data* mesh,float3 &pout,float3 &vout,float dtin,int gidx);

	__host__
	void free(void)
	{
		for(int i=0;i<9;i++)
		{
			CUDA_SAFE_CALL(cudaFree(*get_float_ptr(i)));
		}

		for(int i=0;i<3;i++)
		{
			CUDA_SAFE_CALL(cudaFree(*get_int_ptr(i)));
		}

		CUDA_SAFE_CALL(cudaFree(binid));
	}
};




__inline__ __device__
int zorder(int ix,int iy,int iz)
{
	ix = (ix | (ix << 16)) & 0x030000FF;
	ix = (ix | (ix <<  8)) & 0x0300F00F;
	ix = (ix | (ix <<  4)) & 0x030C30C3;
	ix = (ix | (ix <<  2)) & 0x09249249;

	iy = (iy | (iy << 16)) & 0x030000FF;
	iy = (iy | (iy <<  8)) & 0x0300F00F;
	iy = (iy | (iy <<  4)) & 0x030C30C3;
	iy = (iy | (iy <<  2)) & 0x09249249;

	iz = (iz | (iz << 16)) & 0x030000FF;
	iz = (iz | (iz <<  8)) & 0x0300F00F;
	iz = (iz | (iz <<  4)) & 0x030C30C3;
	iz = (iz | (iz <<  2)) & 0x09249249;

	return ix | (iy << 1) | (iz << 2);

}

__inline__ __device__ __host__
float** XPlist::get_float_ptr(int i)
{
	float** result;
	switch(i)
	{
	case 0: result = &px; break;
	case 1: result = &py; break;
	case 2: result = &pz; break;
	case 3: result = &vx; break;
	case 4: result = &vy; break;
	case 5: result = &vz; break;
	case 6: result = &dt_prec; break;
	case 7: result = &vzinit;  break;
	case 8: result = &buffer; break;
	default:
		break;
	}

	return result;
}

__inline__ __device__ __host__
int** XPlist::get_int_ptr(int i)
{
	int** result;
	switch(i)
	{
	case 0: result = &ipf; break;
	case 1: result = &particle_id; break;
	case 2: result = &didileave; break;
	default:
		break;
	}

	return result;
}

__inline__ __device__
void XPlist::calc_binid(Mesh_data* mesh,int3 ncells,int idx)
{
	int4 icell;
	float4 cellf;
	float zetap;

	mesh -> ptomesh<0>(px[idx],py[idx],pz[idx],&icell,&cellf,zetap);

	icell.x = (icell.x-1)/ncells.x;
	icell.y = (icell.y-1)/ncells.y;
	icell.z = (icell.z-1)/ncells.z;


	binid[idx] = zorder(icell.x,icell.y,icell.z);

#ifdef debug
	if(binid[idx] >= mesh->nbins)
	{
		mesh -> ptomesh(px[idx],py[idx],pz[idx],&icell,&cellf,zetap);
		printf("particle %i is in bin %i (%i,%i,%i) %f, %f, %f \n",idx,binid[idx],icell.x,icell.y,icell.z,px[idx],py[idx],pz[idx]);
	}
#endif
	//printf("particle %i is in bin %i, %i, %i = %i\n",idx,icell.x,icell.y,icell.z,binid[idx]);


}

__inline__ __device__
void XPlist::move(Mesh_data* mesh,float3 &pout,float3 &vout,float dtin,int gidx)
{
	float dtnow,dt;
	float sd,sB;
	float3 accel;
	float Eneutral = 0;

	pout.x = px[gidx];
	pout.y = py[gidx];
	pout.z = pz[gidx];
	vout.x = vx[gidx];
	vout.y = vy[gidx];
	vout.z = vz[gidx];

	sd = sqrt(1.0f-mesh->cd*mesh->cd);
	sB = sqrt(1.0f-mesh->cB*mesh->cB);

	dt = dtin;

	dtnow = 0.5*(dt+dt_prec[gidx]);

	accel = mesh->getaccel(pout.x,pout.y,pout.z);

	// Not using the verlet integrator

	// If using collisions
	accel.z = accel.z+Eneutral*mesh->cd;
	accel.y = accel.y+Eneutral*sd;

	vout.x += accel.x*dtnow;
	vout.y += accel.y*dtnow;
	vout.z += accel.z*dtnow;

	if(mesh->Bz != 0.0f)
	{
		vout.y -= mesh->vd*sd;
		vout.z -= mesh->vd*mesh->cd;

		if(mesh->cB < 0.999f)
		{
			float temp = pout.y;
			pout.y = temp*mesh->cB-pout.z*sB;
			pout.z = pout.z*mesh->cB + temp*sB;
			temp = vout.y;
			vout.y = temp*mesh->cB-vout.z*sB;
			vout.z = vout.z*mesh->cB+temp*sB;
		}

		float cosomdt = cos(mesh->Bz*dt);
		float sinomdt = sin(mesh->Bz*dt);

		pout.x += (vout.y*(1.0f-cosomdt)+vout.x*sinomdt)/mesh->Bz;
		pout.y += (vout.x*(cosomdt-1.0f)+vout.y*sinomdt)/mesh->Bz;

		float temp = vout.x;

		vout.x = temp*cosomdt+vout.y*sinomdt;
		vout.y = vout.y*cosomdt-temp*sinomdt;

		pout.z += vout.z*dt;

		if(mesh->cB < 0.999f)
		{
			float temp = pout.y;
			pout.y = temp*mesh->cB+pout.z*sB;
			pout.z = pout.z*mesh->cB-temp*sB;
			temp = vout.y;
			vout.y = temp*mesh->cB + pout.z*sB;
			vout.z = vout.z*mesh->cB - temp*sB;
		}

		pout.y += mesh->vd*sd*dt;
		pout.z += mesh->vd*mesh->cd*dt;
		vout.y += mesh->vd*sd;
		vout.z += mesh->vd*mesh->cd;

	}
	else
	{
		pout.x += vout.x*dt;
		pout.y += vout.y*dt;
		pout.z += vout.z*dt;
	}
}

__inline__ __device__
uint condense_bits(uint in)
{
	in = ((in & 0x08208208) >> 2) ^ (in & ~(0x08208208));
	in = ((in & 0x400C00C0) >> 4) ^ (in &  ~(0x400C00C0));
	in = ((in & 0x0000F000) >> 8) ^ (in &  ~(0x0000F000));
	in = ((in & 0x07000000) >> 16) ^ (in &  ~(0x07000000));

	return in;

}

__inline__ __device__
uint3 Particlebin::get_bin_position(int3 ncells)
{
	uint3 result;
	uint temp;

	temp = binid & 0x49249249;

	result.x = condense_bits(temp);
	temp = binid & (0x49249249 << 1);

	result.y = condense_bits((temp >> 1));
	temp = binid & (0x49249249 << 2);
	result.z = condense_bits((temp >> 2));

	result.x *= ncells.x;
	result.y *= ncells.y;
	result.z *= ncells.z;

	return result;

}



template<int doih>
__inline__ __device__
float4 Mesh_data::ptomesh(float x, float y, float z,int4* icell,float4* cellf,float &zetap)
{
	float4 ang;
	float rsp,rp,hf,hp;
	int ih;

	// ang.x = cp
	// ang.y = sp
	// ang.z = ct
	// ang.w = st

	rsp = (x*x)+(y*y);

	rp = sqrt(rsp+(z*z));

	// psi sin/cos
	rsp = sqrt(rsp);
	if(rsp > 1.0e-9f)
	{
		ang.x = x/rsp;
		ang.y = y/rsp;
	}
	else
	{
		ang.x = 1.0f;
		ang.y = 0.0f;
	}

	icell->z = interppsi(ang.y,ang.x,&(cellf->z));

	// theta sin/cos
	ang.z = z/rp;
	ang.w = rsp/rp;

	ang.z = max(-1.0,min(1.0,ang.z));

	icell->y = interpth(ang.z,&(cellf->y));

	icell->x = interpr(rp,&(cellf->x));

	if(doih)
	{
		ih = icell->x + 1;
		hp = abs(rp - rmesh(1));
		zetap = sqrt(2.0f*hp);
		hf = zetap - zetahalf(ih);
		if(hf < 0.0f) ih -= 1;

		hf = (zetap - zetahalf(ih))/(zetahalf(ih+1)-zetahalf(ih));

		icell -> w = ih;
		cellf -> w = hf;
	}

	return ang;


}

__inline__ __device__
int Mesh_data::interppsi(float sp,float cp,float* pf)
{
	int ipl;
	float psi;

	psi = atan2(sp,cp);
	if(psi < 0.0f) psi+=2.0f*pi_const;

	ipl = abs(floor((psi-pcc(1))*pfac));

	ipl = ippre(ipl);

	pf[0] = (psi - pcc(ipl))/(pcc(ipl+1)-pcc(ipl));

	if(pf[0] > 1.0f)
	{
		if(ipl+2 <= npsifull)
		{
			ipl+=1;
			pf[0] = (psi - pcc(ipl))/(pcc(ipl+1)-pcc(ipl));
		}
		else
		{
			pf[0] = 1.0f;
		}
	}

	return ipl;


}

__inline__ __device__
int Mesh_data::interpth(float ct,float* thf)
{
	int ithl;

	ithl = abs(floor((ct-thmesh(1))*tfac));

	ithl = itpre(ithl);
	thf[0] = (ct-thmesh(ithl))/(thmesh(ithl+1)-thmesh(ithl));
/*
	 if(thf[0] < 0.0f)
	{
			if(ithl > 1)
			{
				ithl -= 1;
			}
			else
			{
				ithl += nthfull-1;
			}

			thf[0] = (ct-thmesh(ithl))/(thmesh(ithl+1)-thmesh(ithl));
	}
*/
	if(thf[0] > 1.0f)
	{
		if(ithl+2 <= nthfull)
		{
			ithl += 1;
			thf[0] = (ct-thmesh(ithl))/(thmesh(ithl+1)-thmesh(ithl));
		}
	}

	//thf[0] = abs(thf[0]);


	return ithl;


}

__inline__ __device__
int Mesh_data::interpr(float rp,float* rf)
{
	int irl;

	irl = irpre(max((int)(floor((rp-rmesh(1))*rfac)),0));

	rf[0] = (rp-rmesh(irl))/(rmesh(irl+1)-rmesh(irl));

	while(rf[0] > 1.0f)
	{
		if(irl < nr)
		{
			irl += 1;
			rf[0] = (rp-rmesh(irl))/(rmesh(irl+1)-rmesh(irl));
		}
		else
		{
			break;
		}
	}

	return irl;


}

__inline__ __device__
int Mesh_data::boundary_intersection(float px1,float py1,float pz1,float& px2,float& py2,float& pz2)
{
	// Return the position where the track intersects either the probe or the outer boundary

	float dx,dy,dz,dr;
	float r1,r2;
	float dl;
	float dp,dm,d;
	float radical;
	float ldotc;
	int result = 0;

	/*
	if(isnan(px2+py2+pz2))
	{
		result = 2;
		py2 = rmesh(nr)/1.73205f;
		pz2 = rmesh(nr)/1.73205f;
		px2 = rmesh(nr)/1.73205f+1.0e-5f;

		return result;
	}
	*/

	dx = px2-px1;
	dy = py2-py1;
	dz = pz2-pz1;
	dl = sqrt(dx*dx+dy*dy+dz*dz);

	dx /= dl;
	dy /= dl;
	dz /= dl;

	ldotc = -dx*px1-dy*py1-dz*pz1;

	radical = ldotc*ldotc - (px1*px1+py1*py1+pz1*pz1);

	// First we check the probe
	if(((radical + rmesh(1)*rmesh(1)) >= 0.0f)&&(ldotc > 0.0f))
	{
		// We have an intersection with the probe
		dp = ldotc + sqrt(radical + rmesh(1)*rmesh(1));
		dm = ldotc - sqrt(radical + rmesh(1)*rmesh(1));

		// Make sure that we don't take intersections resulting from projecting the track in the wrong direction

		// Take the closest intersection
		d = min(dp,dm);

		if(d <= dl)
		{
			//printf("particle left the grid with position %f, %f, %f\n",px2,py2,pz2);
			// Particle intersects the probe before it finishes its step;
			px2 = px1 + d*dx;
			py2 = py1 + d*dy;
			pz2 = pz1 + d*dz;
			result = 1;

			return result;
		}

	}

	// Now we check the outer boundary
	if((px2*px2+py2*py2+pz2*pz2) >= rmesh(nr)*rmesh(nr))
	{
		// We already know that it intersects the sphere at least once, all we need know is where it intersects
		result = 1;
		r1 = sqrt(px1*px1+py1*py1+pz1*pz1);
		r2 = sqrt(px2*px2+py2*py2+pz2*pz2);
		dr = r2 - r1;

		d = (rmesh(nr)+1.0e-5f - r1)/dr;

		//printf("particle left the grid with position %f, %f, %f\n",px2,py2,pz2);

		px2 = px1 + d*dx;
		py2 = py1 + d*dy;
		pz2 = pz1 + d*dz;
		return result;

	}

	return result;



}

__inline__ __device__
float3 Mesh_data::getaccel(float px,float py,float pz)
{
	int4 icell;
	float4 cellf;
	float4 ang;
	float3 result;
	float zetap;
	float rl,rr,pf,rlm1,rf,hf;
	int ir,irl,ith,ipl,ih,ilm1,ithp1,ithp2,ithm1,iplp1,iplp2,iplm1;

    float philm1tt,philm1pt,philm1tp,philm1pp;
    float philm1t,philm1p;

    float phihp1tt,phihp1pt,phihp1mt,phihp12t;
    float phihp1tp,phihp1pp,phihp1mp,phihp12p;
    float phihp1tm,phihp1pm;
    float phihp1t2,phihp1p2;

    float phihp1t,phihp1p,phihp1m,phihp12;
    float phihp1tX,phihp1pX,phihp1mX,phihp12X;

    float phih1tt,phih1tp,phih1tm, phih1t2;
    float phih1pt,phih1pp,phih1pm,phih1p2;
    float phih1mt,phih1mp;
    float phih12t,phih12p;

    float phih1t,phih1p,phih1m,phih12;
    float phih1tX,phih1pX,phih1mX,phih12X;

    //float rp,rsp,ct,st,cp,sp;
    float &ct = ang.z;
    float &st = ang.w;
    float &cp = ang.x;
    float &sp = ang.y;

    float ar,at,ap;

	ang = ptomesh<1>(px,py,pz,&icell,&cellf,zetap);
/*
	rsp = (px*px+py*py);
	rp = sqrt(rsp+pz*pz);

	rsp = sqrt(rsp);

	if(rsp > 1.0e-9f)
	{
		cp = px/rsp;
		sp = py/rsp;
	}
	else
	{
		cp = 1.0f;
		sp = 0.0f;
	}



	ct = pz/rp;
	st = rsp/rp;

	ct = max(-1.0,min(1.0,ct));
*/

	cp = ang.x;
	sp = ang.y;
	ct = ang.z;
	st = ang.w;


	pf = cellf.z;
	hf = cellf.w;
	rf = cellf.x;
	irl = icell.x;


	ih = icell.w;
	ith = icell.y;
	ipl = icell.z;
	ir = ih+1;

	rl = rmesh(ih);
	ilm1 = ih-1;

	// Theta indexes
	ithp1 = ith+1;
	ithp2 = ith+2;
	ithm1 = ith-1;

	// Psi indexes
	iplp1 = (ipl%npsiused)+1;
	iplp2 = ((ipl+1)%npsiused)+1;
	iplm1 = ((ipl+npsiused-2)%npsiused)+1;

	// Potential at i=ih
    phih1tt=phi(ih,ith,ipl);
    phih1tp=phi(ih,ith,iplp1);
    phih1tm=phi(ih,ith,iplm1);
    phih1t2=phi(ih,ith,iplp2);
    phih1pt=phi(ih,ithp1,ipl);
    phih1pp=phi(ih,ithp1,iplp1);
    phih1pm=phi(ih,ithp1,iplm1);
    phih1p2=phi(ih,ithp1,iplp2);
    phih1mt=phi(ih,ithm1,ipl);
    phih1mp=phi(ih,ithm1,iplp1);
    phih12t=phi(ih,ithp2,ipl);
    phih12p=phi(ih,ithp2,iplp1);

    phih1t=phih1tp*cellf.z+phih1tt*(1-cellf.z);
    phih1p=phih1pp*cellf.z+phih1pt*(1-cellf.z);
    phih1m=phih1mp*cellf.z+phih1mt*(1-cellf.z);
    phih12=phih12p*cellf.z+phih12t*(1-cellf.z);


    float tflin=(acos(ct)-thang(ith))/(thang(ith+1)-thang(ith));

    phih1tX=phih1pt*tflin +phih1tt*(1-tflin);
    phih1pX=phih1pp*tflin +phih1tp*(1-tflin);
    phih1mX=phih1pm*tflin +phih1tm*(1-tflin);
    phih12X=phih1p2*tflin +phih1t2*(1-tflin);

    if(ih == nr)
    {
        phihp1tt=2.0f*phih1tt-phi(ih-1,ith,ipl);
        phihp1pt=2.0f*phih1pt-phi(ih-1,ithp1,ipl);
        phihp1mt=2.0f*phih1mt-phi(ih-1,ithm1,ipl);
        phihp12t=2.0f*phih12t-phi(ih-1,ithp2,ipl);

        phihp1tp=2.0f*phih1tp-phi(ih-1,ith,iplp1);
        phihp1pp=2.0f*phih1pp-phi(ih-1,ithp1,iplp1);
        phihp1mp=2.0f*phih1mp-phi(ih-1,ithm1,iplp1);
        phihp12p=2.0f*phih12p-phi(ih-1,ithp2,iplp1);

        phihp1tm=2.0f*phih1tm-phi(ih-1,ith,iplm1);
        phihp1pm=2.0f*phih1pm-phi(ih-1,ithp1,iplm1);

        phihp1t2=2.0f*phih1t2-phi(ih-1,ith,iplp2);
        phihp1p2=2.0f*phih1p2-phi(ih-1,ithp1,iplp2);

         rr=2*rl-rmesh(ih-1);
    }
    else
    {
        phihp1tt=phi(ir,ith,ipl);
        phihp1tp=phi(ir,ith,iplp1);
        phihp1tm=phi(ir,ith,iplm1);
        phihp1t2=phi(ir,ith,iplp2);
        phihp1pt=phi(ir,ithp1,ipl);
        phihp1pp=phi(ir,ithp1,iplp1);
        phihp1pm=phi(ir,ithp1,iplm1);
        phihp1p2=phi(ir,ithp1,iplp2);
        phihp1mt=phi(ir,ithm1,ipl);
        phihp1mp=phi(ir,ithm1,iplp1);
        phihp12t=phi(ir,ithp2,ipl);
        phihp12p=phi(ir,ithp2,iplp1);

        rr=rmesh(ir);
    }


    // For the radial and theta acceleration, use values of the potential
   // already weighted in the psi direction
	  phihp1t=phihp1tp*pf+phihp1tt*(1-pf);
	  phihp1p=phihp1pp*pf+phihp1pt*(1-pf);
	  phihp1m=phihp1mp*pf+phihp1mt*(1-pf);
	  phihp12=phihp12p*pf+phihp12t*(1-pf);

//     Theta weighting
	  phihp1tX=phihp1pt*tflin +phihp1tt*(1-tflin);
	  phihp1pX=phihp1pp*tflin +phihp1tp*(1-tflin);
	  phihp1mX=phihp1pm*tflin +phihp1tm*(1-tflin);
	  phihp12X=phihp1p2*tflin +phihp1t2*(1-tflin);

	  if(debyelen < 1.0e-2f)
	  {
		  if(ih == 1)
		  {
	            philm1tt=2.0f*phi(irl,ith,ipl)-phi(ir,ith,ipl);
	            philm1pt=2.0f*phi(irl,ithp1,ipl)-phi(ir,ithp1,ipl);
	            philm1tp=2.0f*phi(irl,ith,iplp1)-phi(ir,ith,iplp1);
	            philm1pp=2.0f*phi(irl,ithp1,iplp1)-phi(ir,ithp1,iplp1);

	            rlm1=2.0f*rl - rr;
		  }
		  else
		  {
	            philm1tt=phi(ilm1,ith,ipl);
	            philm1pt=phi(ilm1,ithp1,ipl);
	            philm1tp=phi(ilm1,ith,iplp1);
	            philm1pp=phi(ilm1,ithp1,iplp1);
	            rlm1=rmesh(ilm1);
		  }

		 philm1t=philm1tp*pf+philm1tt*(1.0f-pf);
		 philm1p=philm1pp*pf+philm1pt*(1.0f-pf);

		  if(zetap <= 1.0e-2f) zetap = 1.0e-2f;

	      ar = (  ( (phihp1t-phih1t)/(zeta(ir)-zeta(ih))*hf +
	             (phih1t-philm1t)/(zeta(ih)-zeta(ilm1))*(1.0f-hf))*(1.0f-tflin)
	             + ( (phihp1p-phih1p)/(zeta(ir)-zeta(ih))*hf +
	             (phih1p-philm1p)/(zeta(ih)-zeta(ilm1))*(1.0f-hf))*tflin )/zetap;



	  }
	  else
	  {
	         philm1tt=phi(ilm1,ith,ipl);
	         philm1pt=phi(ilm1,ithp1,ipl);
	         philm1tp=phi(ilm1,ith,iplp1);
	         philm1pp=phi(ilm1,ithp1,iplp1);
	         rlm1=rmesh(ilm1);

	         philm1t=philm1tp*pf+philm1tt*(1-pf);
	         philm1p=philm1pp*pf+philm1pt*(1-pf);

	         ar=(  ( (phihp1t-phih1t)/(rr-rl)*hf +
	            (phih1t-philm1t)/(rl-rlm1)*(1.0f-hf))*(1.0f-tflin)
	            +( (phihp1p-phih1p)/(rr-rl)*hf +
	            (phih1p-philm1p)/(rl-rlm1)*(1.0f-hf))*tflin );
	  }

	  ar = -ar;

	  if(tflin <= 0.5f)
	  {
	       at= ( (phih1p-phih1t)*(tflin)*2.0f
	               /(rl*(thang(ithp1)-thang(ith)))
	             +(phih1p-phih1m)*(0.5f-tflin)*2.0f
	               /(rl*(thang(ithp1)-thang(ithm1))) ) * (1.0f-rf)
	             + ( (phihp1p-phihp1t)*(tflin)*2.0f
	               /(rr*(thang(ithp1)-thang(ith)))
	             +(phihp1p-phihp1m)*(0.5f-tflin)*2.0f
	               /(rr*(thang(ithp1)-thang(ithm1))) ) * rf;
	  }
	  else
	  {
	         at= ( (phih12-phih1t)*(tflin-0.5f)*2.0f
	               /(rl*(thang(ithp2)-thang(ith)))
	             +(phih1p-phih1t)*(1.0f-tflin)*2.0f
	               /(rl*(thang(ithp1)-thang(ith))) ) * (1.0f-rf)
	             + ( (phihp12-phihp1t)*(tflin-0.5f)*2.0f
	               /(rr*(thang(ithp2)-thang(ith)))
	             +(phihp1p-phihp1t)*(1.0f-tflin)*2.0f
	               /(rr*(thang(ithp1)-thang(ith))) ) * rf;
	  }

	  at = -at;

	  if(lat0) at = 0;

	  if(pf<=0.5f)
	  {
	         ap= ( (phih1pX-phih1tX)*(pf)*2./rl +0.5f*(phih1pX-phih1mX)*(0.5f
	             -pf)*2.0f/rl ) * (1.0f-rf) + ( (phihp1pX-phihp1tX)*(pf)*2.0f/rr
	             +0.5f*(phihp1pX-phihp1mX)*(0.5f-pf)*2.0f/rr ) * rf;
	  }
	  else
	  {
	         ap= ( 0.5f*(phih12X-phih1tX)*(pf-0.5f)*2.0f/rl +(phih1pX-phih1tX)
	             *(1.0f-pf)*2.0f/rl ) * (1.0f-rf) + ( 0.5f*(phihp12X-phihp1tX)*(pf
	             -0.5f)*2.0f/rr +(phihp1pX-phihp1tX)*(1.0f-pf)*2.0f/rr ) * rf;
	  }

	  ap=-ap/(st*dpsi+1.0e-7f);

	  if(lap0) ap=0.0;

	  // 3D acceleration
	  result.z = ar*ct - at*st;
	  result.y = (ar*st+at*ct)*sp+ap*cp;
	  result.x = (ar*st+at*ct)*cp-ap*sp;
/*
	  bool Excessive_Accel;

	  Excessive_Accel = !(abs(result.x)<1.0e5f);
	  Excessive_Accel = Excessive_Accel||(!(abs(result.y)<1.0e5f));
	  Excessive_Accel = Excessive_Accel||(!(abs(result.z)<1.0e5f));

	  if(Excessive_Accel)
	  {
		  printf("Accel Excessive at %g, %g, %g\n",px,py,pz);
		  printf("ar, at, ap, ct, st, sp, cp\n%g, %g, %g, %g, %g, %g, %g\n",ar,at,ap,acos(ct),st,sp,cp);
		  printf("phi at ith and ithp1:\n%g, %g, %g\n%g, %g, %g\n",philm1t,phih1t,phihp1t,philm1p,phih1p,phihp1p);
		  printf("zetap= %g\n",zetap);
		  printf("zeta \n%g, %g, %g\n",zeta(ilm1),zeta(ih),zeta(ir));
		  printf("ih, ith, ipl, iplp1, hf, rf, tflin, pf\n%i, %i, %i, %i, %g, %g, %g, %g\n",ih,ith,ipl,iplp1,hf,rf,tflin,pf);
		  result.x = 1.0e5f*(((int)result.x) & 0x8000);
		  result.y = 1.0e5f*(((int)result.y) & 0x8000);
		  result.z = 1.0e5f*(((int)result.z) & 0x8000);
	  }

	  if(!(abs(result.x)<1.0e5f))
	  {
		  result.x = 1.0e5f*(result.x & 0x8000);
	  }

	  if(!(abs(result.y)<1.0e5f))
	  {
		  result.y = 1.0e5f*(result.y & 0x8000);
	  }

	  if(!(abs(result.z)<1.0e5f))
	  {
		  result.z = 1.0e5f*(result.z & 0x8000);
	  }
*/



	  return result;
}

__inline__ __device__
void Mesh_data::probe_diags(float3 pin, float3 vin)
{
	float4 cellfractions;
	int4 my_cell;
	float zetap;
	float vr,vr2;
	float frac;

	vr2 = vin.x*vin.x+vin.y*vin.y+vin.z*vin.z;
	vr = sqrt(vr2);

	//my_cell.w = 0;
	ptomesh<0>(pin.x,pin.y,pin.z,&my_cell,&cellfractions,zetap);



	my_cell.y -= 1;
	my_cell.z -= 1;

	frac = (1.0f-cellfractions.y)*(1.0f-cellfractions.z);
	atomicAdd(&nincell(my_cell.y,my_cell.z),frac);
	atomicAdd(&vrincell(my_cell.y,my_cell.z),frac*vr);
	atomicAdd(&vr2incell(my_cell.y,my_cell.z),frac*vr2);

	frac = (cellfractions.y)*(1.0f-cellfractions.z);
	atomicAdd(&nincell(my_cell.y+1,my_cell.z),frac);
	atomicAdd(&vrincell(my_cell.y+1,my_cell.z),frac*vr);
	atomicAdd(&vr2incell(my_cell.y+1,my_cell.z),frac*vr2);

	if(my_cell.z == npsi-1) my_cell.z = -1;

	frac = (1.0f-cellfractions.y)*(cellfractions.z);
	atomicAdd(&nincell(my_cell.y,my_cell.z+1),frac);
	atomicAdd(&vrincell(my_cell.y,my_cell.z+1),frac*vr);
	atomicAdd(&vr2incell(my_cell.y,my_cell.z+1),frac*vr2);

	frac = (cellfractions.y)*(cellfractions.z);
	atomicAdd(&nincell(my_cell.y+1,my_cell.z+1),frac);
	atomicAdd(&vrincell(my_cell.y+1,my_cell.z+1),frac*vr);
	atomicAdd(&vr2incell(my_cell.y+1,my_cell.z+1),frac*vr2);
}













































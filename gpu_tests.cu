
#include "XPlist.cuh"


extern "C" { void ptomesh_interface_(int &i,int &irl,float &rf,int &ithl,float &thf,
				int &ipl,float &pf,float &st,float &ct,float &sp,float &cp,float &rp,
				float &zetap,int &ih,float &hf);}

extern "C" { void chargetomesh_interface_(void);}

extern "C" void gpu_chargeassign_(long int* XP_ptr,long int* mesh_ptr,float* psum);

extern "C" void gpu_padvnc_(long int* XP_ptr,long int* mesh_ptr,
												float* phi,float* xpreinject,float* dt,int* reinject_counter);

extern "C" void xplist_transpose_(long int* xplist_d,
									   float* xplist_h,float* dt_prec,float* vzinit,int* ipf,
									   int* npartmax,int* ndims,int* direction,int* xpdata_only);

extern "C" {void getaccel_interface_(int &i,float* accel);}

__global__
void populate_ptomesh_stuff(XPlist particles, Mesh_data mesh,int4* indexes,float4* fractions,int nptcls)
{
	int idx = threadIdx.x;
	int gidx = blockIdx.x*blockDim.x+idx;

	int4 local_index;
	float4 local_fraction;
	float zetap;

	if(gidx < nptcls)
	{
		local_index.w = 1;
		mesh.ptomesh(particles.px[gidx],particles.py[gidx],particles.pz[gidx],&local_index,&local_fraction,zetap);

		indexes[gidx] = local_index;
		fractions[gidx] = local_fraction;
	}
}

extern "C" __host__
void test_gpu_ptomesh_(long int* XP_ptr,long int* mesh_ptr)
{
	int cudaBlockSize;
	int cudaGridSize;
	int irl,ithl,ipl,ih;
	float rf,thf,pf,st,ct,sp,cp,rp,zetap,hf;
	double error = 1.0e-4;
	double pf_error;
	double rf_error;
	double thf_error;

	double rf_error_total = 0.0;
	double thf_error_total = 0.0;
	double pf_error_total = 0.0;

	XPlist* particles = (XPlist*)(*XP_ptr);
	Mesh_data mesh_d = *(Mesh_data*)(*mesh_ptr);
	int nptcls = particles->nptcls;

	float4* fractions_h = (float4*)malloc(nptcls*sizeof(float4));
	int4* index_h = (int4*)malloc(nptcls*sizeof(int4));

	float4* fractions_d;
	int4* index_d;

	CUDA_SAFE_CALL(cudaMalloc((void**)&fractions_d,nptcls*sizeof(float4)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&index_d,nptcls*sizeof(int4)));

	cudaBlockSize = 512;
	cudaGridSize = (nptcls+cudaBlockSize-1)/cudaBlockSize;

	CUDA_SAFE_KERNEL((populate_ptomesh_stuff<<<cudaGridSize,cudaBlockSize>>>
								 (*particles,mesh_d,index_d,fractions_d,nptcls)));

	CUDA_SAFE_CALL(cudaMemcpy(fractions_h,fractions_d,nptcls*sizeof(float4),cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(index_h,index_d,nptcls*sizeof(int4),cudaMemcpyDeviceToHost));

	for(int i=1;i<=nptcls;i++)
	{
		ptomesh_interface_(i,irl,rf,ithl,thf,ipl,pf,st,ct,sp,cp,rp,zetap,ih,hf);

		if(irl != index_h[i-1].x)
		{
			printf("Wrong r index for particle %i, %i != %i\n",i,irl,index_h[i-1].x);
		}
		if(ithl != index_h[i-1].y)
		{
			printf("Wrong theta index for particle %i, %i != %i\n",i,ithl,index_h[i-1].y);
		}
		if(ipl != index_h[i-1].z)
		{
			printf("Wrong psi index for particle %i, %i != %i\n",i,ipl,index_h[i-1].z);
		}
		if(ih != index_h[i-1].w)
		{
			printf("Wrong half index for particle %i, %i != %i\n",i,ih,index_h[i-1].w);
		}

		rf_error = abs(rf - fractions_h[i-1].x);
		thf_error = abs(thf - fractions_h[i-1].y);
		pf_error = abs(pf - fractions_h[i-1].z);

		rf_error_total += rf_error;
		thf_error_total += thf_error;
		pf_error_total += pf_error;
		if(rf_error >error)
		{
			//printf("Wrong r fraction for particle %i, %f != %f error = %g\n",i,rf,fractions_h[i-1].x,rf_error);
		}
		if(thf_error > error)
		{
			//printf("Wrong theta fraction for particle %i, %f != %f error = %g\n",i,thf,fractions_h[i-1].y,thf_error);
		}
		if(pf_error > error)
		{
			//printf("Wrong psi fraction for particle %i, %f != %f error = %g\n",i,pf,fractions_h[i-1].z,pf_error);
		}
	}

	printf("Average fractional errors = %g, %g, %g\n",rf_error_total/((double)nptcls),thf_error_total/((double)nptcls),pf_error_total/((double)nptcls));


	free(fractions_h);
	free(index_h);
	cudaFree(fractions_d);
	cudaFree(index_d);

}



extern "C" void test_chargetomesh_(long int* XP_ptr,long int* mesh_ptr,float* psum)
{
	size_t freemem = 0;
	size_t total = 0;
	// See how much memory is allocated / free
	cudaMemGetInfo(&freemem,&total);
	printf("Free Memory = %i mb\nUsed mememory = %i mb\n",(int)(freemem)/(1<<20),(int)(total-freemem)/(1<<20));

	XPlist* particles = (XPlist*)(*XP_ptr);
	Mesh_data mesh_d = *(Mesh_data*)(*mesh_ptr);
	int nptcls = particles->nptcls;

	int psum_size = (mesh_d.nrfull-1)*(mesh_d.nthfull-1)*(mesh_d.npsifull-1);

	float* psum_temp = (float*)malloc(psum_size*sizeof(int));

	double difference;
	double total_difference = 0;
	double tolerance = 1.0e-6;
	int index;

	unsigned int timer = 0;
	unsigned int timer2 = 0;
	cutCreateTimer(&timer);
	cutCreateTimer(&timer2);

	// Call the fortran version of chargeassign
	cutStartTimer(timer);
	chargetomesh_interface_();
	cutStopTimer(timer);


	// Call the gpu version of the chargeassign
	cutStartTimer(timer2);
	gpu_chargeassign_(XP_ptr,mesh_ptr,psum_temp);
	cutStopTimer(timer2);

	printf( "\nFortran Charge assign took: %f (ms)\n\n", cutGetTimerValue( timer));
	printf( "\GPU Charge assign took: %f (ms)\n\n", cutGetTimerValue( timer2));

//	printf("dims = %i, %i, %i\n",mesh_d.nrfull,mesh_d.nthfull,mesh_d.npsifull);



	// Compare the results
	for(int i=0;i<mesh_d.nr;i++)
	{
		for(int j=0;j<mesh_d.nth;j++)
		{
			for(int k=0;k<mesh_d.npsi;k++)
			{
				index = i+(mesh_d.nrfull-1)*(j+(mesh_d.nthfull-1)*k);
				difference = abs(psum[index] - psum_temp[index])/max(psum[index],tolerance);
				total_difference += difference;

				if(difference > tolerance)
				{
					//printf("Sums are not equal %f != %f at %i, %i, %i diff = %g\n",psum[index],psum_temp[index],i,j,k,difference);
				}
			}
		}
	}

	printf(" average difference = %g \n",total_difference/(mesh_d.nr*mesh_d.nth*mesh_d.npsi));

	free(psum_temp);

	cutDeleteTimer( timer);
	cutDeleteTimer( timer2);


}



__global__
void populate_getaccel_stuff(XPlist particles, Mesh_data mesh,float3* accels,int nptcls)
{
	int idx = threadIdx.x;
	int gidx = blockIdx.x*blockDim.x+idx;

	float3 my_accel;
	float zetap;

	if(gidx < nptcls)
	{
		my_accel = mesh.getaccel(particles.px[gidx],particles.py[gidx],particles.pz[gidx]);

		accels[gidx] = my_accel;
	}
}

extern "C" __host__
void test_gpu_getaccel_(long int* XP_ptr,long int* mesh_ptr,float* phi)
{
	int cudaBlockSize;
	int cudaGridSize;
	double error = 1.0e-4;
	double xerror;
	double yerror;
	double zerror;

	double xerror_total = 0.0;
	double yerror_total = 0.0;
	double zerror_total = 0.0;


	int irl,ithl,ipl,ih;
	float rf,thf,pf,st,ct,sp,cp,rp,zetap,hf;
	double pf_error;
	double rf_error;
	double thf_error;
	double hf_error;

	double rf_error_total = 0.0;
	double thf_error_total = 0.0;
	double pf_error_total = 0.0;


	float accel[3];

	XPlist* particles = (XPlist*)(*XP_ptr);
	Mesh_data mesh_d = *(Mesh_data*)(*mesh_ptr);
	int nptcls = particles->nptcls;

	mesh_d.phi.cudaMatrixcpy(phi,cudaMemcpyHostToDevice);

	float3* accel_h = (float3*)malloc(nptcls*sizeof(float3));
	float3* accel_d;


	CUDA_SAFE_CALL(cudaMalloc((void**)&accel_d,nptcls*sizeof(float3)));

	float4* fractions_h = (float4*)malloc(nptcls*sizeof(float4));
	int4* index_h = (int4*)malloc(nptcls*sizeof(int4));

	float4* fractions_d;
	int4* index_d;

	CUDA_SAFE_CALL(cudaMalloc((void**)&fractions_d,nptcls*sizeof(float4)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&index_d,nptcls*sizeof(int4)));


	cudaBlockSize = 512;
	cudaGridSize = (nptcls+cudaBlockSize-1)/cudaBlockSize;

	CUDA_SAFE_KERNEL((populate_getaccel_stuff<<<cudaGridSize,cudaBlockSize>>>
								 (*particles,mesh_d,accel_d,nptcls)));

	CUDA_SAFE_CALL(cudaMemcpy(accel_h,accel_d,nptcls*sizeof(float3),cudaMemcpyDeviceToHost));

	CUDA_SAFE_KERNEL((populate_ptomesh_stuff<<<cudaGridSize,cudaBlockSize>>>
								 (*particles,mesh_d,index_d,fractions_d,nptcls)));

	CUDA_SAFE_CALL(cudaMemcpy(fractions_h,fractions_d,nptcls*sizeof(float4),cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(index_h,index_d,nptcls*sizeof(int4),cudaMemcpyDeviceToHost));

	for(int i=1;i<=nptcls;i++)
	{
		getaccel_interface_(i,accel);

		xerror = abs(accel[0] - accel_h[i-1].x)/abs(accel[0] + accel_h[i-1].x);
		yerror = abs(accel[1] - accel_h[i-1].y)/abs(accel[1] + accel_h[i-1].y);
		zerror = abs(accel[2] - accel_h[i-1].z)/abs(accel[2] + accel_h[i-1].z);

		xerror = abs(accel[0] - accel_h[i-1].x);
		yerror = abs(accel[1] - accel_h[i-1].y);
		zerror = abs(accel[2] - accel_h[i-1].z);

		xerror_total += xerror;
		yerror_total += yerror;
		zerror_total += zerror;


		if(xerror > error)
		{
			printf("Wrong x accel for particle %i, %g != %g error = %g\n",i,accel[0],accel_h[i-1].x,xerror);
		}
		if(yerror > error)
		{
			printf("Wrong y accel  for particle %i, %g != %g error = %g\n",i,accel[1],accel_h[i-1].y,yerror);
		}
		if(zerror > error)
		{
			printf("Wrong z accel  for particle %i, %g != %g error = %g\n",i,accel[2],accel_h[i-1].z,zerror);
		}

		ih = 1;

		ptomesh_interface_(i,irl,rf,ithl,thf,ipl,pf,st,ct,sp,cp,rp,zetap,ih,hf);

		if(irl != index_h[i-1].x)
		{
			printf("Wrong r index for particle %i, %i != %i\n",i,irl,index_h[i-1].x);
		}
		if(ithl != index_h[i-1].y)
		{
			printf("Wrong theta index for particle %i, %i != %i\n",i,ithl,index_h[i-1].y);
		}
		if(ipl != index_h[i-1].z)
		{
			printf("Wrong psi index for particle %i, %i != %i\n",i,ipl,index_h[i-1].z);
		}
		if(ih != index_h[i-1].w)
		{
			printf("Wrong half index for particle %i, %i != %i\n",i,ih,index_h[i-1].w);
		}

		rf_error = abs(rf - fractions_h[i-1].x);
		thf_error = abs(thf - fractions_h[i-1].y);
		pf_error = abs(pf - fractions_h[i-1].z);
		hf_error = abs(hf - fractions_h[i-1].w);

		rf_error_total += rf_error;
		thf_error_total += thf_error;
		pf_error_total += pf_error;
		if(rf_error > error)
		{
			printf("Wrong r fraction for particle %i, %f != %f error = %g\n",i,rf,fractions_h[i-1].x,rf_error);
		}
		if(thf_error > error)
		{
			printf("Wrong theta fraction for particle %i, %f != %f error = %g\n",i,thf,fractions_h[i-1].y,thf_error);
		}
		if(pf_error > error)
		{
			printf("Wrong psi fraction for particle %i, %f != %f error = %g\n",i,pf,fractions_h[i-1].z,pf_error);
		}
		if(hf_error > error)
		{
			printf("Wrong half fraction for particle %i, %f != %f error = %g\n",i,hf,fractions_h[i-1].w,hf_error);
		}



	}

	printf("Average fractional errors = %g, %g, %g\n",xerror_total/((double)nptcls),yerror_total/((double)nptcls),zerror_total/((double)nptcls));


	free(accel_h);
	cudaFree(accel_d);

	free(fractions_h);
	free(index_h);
	cudaFree(fractions_d);
	cudaFree(index_d);

}


extern "C" void test_gpu_padvnc_(long int* XP_ptr,long int* mesh_ptr,
												float* xp,float* phi,float* xpreinject,float* dt)
{
	// Only call this after a call to padvnc2

	XPlist* particles = (XPlist*)(*XP_ptr);
	Mesh_data mesh_d = *(Mesh_data*)(*mesh_ptr);
	int nptcls = particles->nptcls;
	int reinject_counter = 0;
	int ndims = 6;

	int xpdata_only = 1;
	int transpose_dir = 1;

	float* dt_prec_dum;
	float* vzinit_dum;
	int* ipf_dum;

	double error = 1.0e-2;

	double perror;

	double perror_total = 0;

	unsigned int timer = 0;
	cutCreateTimer(&timer);

	// A temporary cpu array to store particles moved by the gpu
	float* xp_gpu_temp = (float*)malloc(ndims*nptcls*sizeof(float));

	// Particles on the host should have already been moved. If not then you fail
	// Time to move the particles on the gpu
	cutStartTimer(timer);
	gpu_padvnc_(XP_ptr,mesh_ptr,phi,xpreinject,dt,&reinject_counter);
	cutStopTimer(timer);
	printf( "\GPU Particle Move took: %f (ms)\n\n", cutGetTimerValue( timer));

	cutDeleteTimer(timer);

	// Copy the gpu results to the array on the host
	xplist_transpose_(XP_ptr,xp_gpu_temp,dt_prec_dum,vzinit_dum,ipf_dum,&nptcls,&ndims,&transpose_dir,&xpdata_only);

	// This should work under the assumption that none of the reinjected particles immediately leave the grid

	// Compare the results
	for(int i=0;i<nptcls;i++)
	{
		for(int j=0;j<6;j++)
		{
			perror = abs(xp[6*i+j] - xp_gpu_temp[6*i+j])/abs(xp[6*i+j] + xp_gpu_temp[6*i+j]);

			perror_total += perror;

			if(perror > error)
			{
				//printf("Dim %i for particle %i, %f != %f error = %g\n",j,i,xp[6*i+j],xp_gpu_temp[6*i+j],perror);
			}
		}
	}

	perror_total /= (6*nptcls);

	printf("Average error = %g\n",perror_total);

	free(xp_gpu_temp);

}































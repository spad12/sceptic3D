#include "gpu_solver.cuh"

#define ATIMES_DOMAIN_DIM 8

template<typename T>
class sMatrixT
{
public:
	int3 dims;
	T* shared_ptr;

	__device__
	sMatrixT(){;}

	__device__
	sMatrixT(T* sdata,int x,int y, int z){init(sdata,x,y,z);}

	__device__
	void init(T* sdata,int x,int y=1, int z=1)
	{
		dims.x = x;
		dims.y = y;
		dims.z = z;
		shared_ptr = sdata;
	}

	__device__
	T & operator() (int ix,int iy, int iz)
	{
		return shared_ptr[ix+dims.x*(iy+dims.y*iz)];
	}

	__device__
	const T & operator() (int ix,int iy, int iz)
	const
	{
		return shared_ptr[ix+dims.x*(iy+dims.y*iz)];
	}

	__device__
	T & operator() (int ix,int iy)
	{
		return shared_ptr[ix+dims.x*(iy)];
	}

	__device__
	const T & operator() (int ix,int iy)
	const
	{
		return shared_ptr[ix+dims.x*(iy)];
	}

	__device__
	T & operator() (int ix)
	{
		return shared_ptr[ix];
	}

	__device__
	const T & operator() (int ix)
	const
	{
		return shared_ptr[ix];
	}


};


typedef sMatrixT<float> sMatrixf;

extern "C" {void atimes_(int &n1,int &n2, int &n3,float* x, float* res,uint* itrnsp);}
extern "C" {void asolve_(int &n1,int &n2, int &n3,float* b, float* z,float* zerror);}
extern "C" {void cg3d_(int &n1,int &n2,int &n3,float* b,float* x,float &tol,int &iter,int&itmax);}

extern "C" void gpu_psolver_init_(long int* PsolvPtr,float* apc,float* bpc,float* cpc, float* dpc,float* epc,
													   float* fpc,float* gpc,int* nrsize,int* nthsize,int* npsisize)
{
	PoissonSolver* solver = (PoissonSolver*)malloc(sizeof(PoissonSolver));

	solver->allocate(*nrsize,*nthsize,*npsisize);

	solver->apc.cudaMatrixcpy(apc,cudaMemcpyHostToDevice);
	solver->bpc.cudaMatrixcpy(bpc,cudaMemcpyHostToDevice);
	solver->cpc.cudaMatrixcpy(cpc,cudaMemcpyHostToDevice);
	solver->dpc.cudaMatrixcpy(dpc,cudaMemcpyHostToDevice);
	solver->epc.cudaMatrixcpy(epc,cudaMemcpyHostToDevice);
	solver->fpc.cudaMatrixcpy(fpc,cudaMemcpyHostToDevice);
	solver->gpc.cudaMatrixcpy(gpc,cudaMemcpyHostToDevice);

	printf("Setting up Psolve Arrays\n");

	*PsolvPtr = (long int)solver;

}


__host__
void PoissonSolver::allocate(int nrsize_in,int nthsize_in,int npsisize_in)
{
	nrsize = nrsize_in;
	nthsize = nthsize_in;
	npsisize = npsisize_in;

	apc.cudaMatrix_allocate(nrsize+1,1,1);
	bpc.cudaMatrix_allocate(nrsize+1,1,1);
	cpc.cudaMatrix_allocate(nrsize+1,nthsize+1,1);
	dpc.cudaMatrix_allocate(nrsize+1,nthsize+1,1);
	epc.cudaMatrix_allocate(nrsize+1,nthsize+1,1);
	fpc.cudaMatrix_allocate(nrsize+1,nthsize+1,1);
	gpc.cudaMatrix_allocate(nthsize+1,npsisize+1,5);

	x.cudaMatrix_allocate(nrsize-1,nthsize+1,npsisize+1);
	phi.cudaMatrix_allocate(nrsize+1,nthsize+1,npsisize+1);
	z.cudaMatrix_allocate(nrsize-1,nthsize+1,npsisize+1);
	zz.cudaMatrix_allocate(nrsize-1,nthsize+1,npsisize+1);
	res.cudaMatrix_allocate(nrsize-1,nthsize+1,npsisize+1);
	resr.cudaMatrix_allocate(nrsize-1,nthsize+1,npsisize+1);
	p.cudaMatrix_allocate(nrsize-1,nthsize+1,npsisize+1);
	pp.cudaMatrix_allocate(nrsize-1,nthsize+1,npsisize+1);
	b.cudaMatrix_allocate(nrsize-1,nthsize+1,npsisize+1);

	CUDA_SAFE_CALL(cudaMalloc((void**)&sum_array,512*(npsisize+1)*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemset(sum_array,0,512*(npsisize+1)*sizeof(float)));

}

__host__
void PoissonSolver::psfree(void)
{
	apc.cudaMatrixFree();
	bpc.cudaMatrixFree();
	cpc.cudaMatrixFree();
	dpc.cudaMatrixFree();
	epc.cudaMatrixFree();
	fpc.cudaMatrixFree();
	gpc.cudaMatrixFree();

	x.cudaMatrixFree();
	phi.cudaMatrixFree();
	z.cudaMatrixFree();
	zz.cudaMatrixFree();
	res.cudaMatrixFree();
	resr.cudaMatrixFree();
	p.cudaMatrixFree();
	pp.cudaMatrixFree();
	b.cudaMatrixFree();

	CUDA_SAFE_CALL(cudaFree(sum_array));
}

static __inline__ __device__
void setup_shared(sMatrixf &apcin,sMatrixf &bpcin,sMatrixf &cpcin,
							    sMatrixf &dpcin,sMatrixf &epcin,sMatrixf &fpcin,
							    sMatrixf &gpcin,sMatrixf &expphiin,sMatrixf &xin,
							    sMatrixf &resin)
{
	const int d = ATIMES_DOMAIN_DIM+4;
	__shared__ float apc[d];
	__shared__ float bpc[d];
	__shared__ float cpc[d*d];
	__shared__ float dpc[d*d];
	__shared__ float epc[d*d];
	__shared__ float fpc[d*d];
	__shared__ float gpc[d*d*5];
	__shared__ float expphi[d*d*d];
	__shared__ float x[d*d*d];
	__shared__ float res[d*d*d];

	if(threadIdx.x+threadIdx.y+threadIdx.z == 0)
	{
		apcin.init(apc,d,1,1);
		bpcin.init(bpc,d,1,1);
		cpcin.init(cpc,d,d,1);
		dpcin.init(dpc,d,d,1);
		epcin.init(epc,d,d,1);
		fpcin.init(fpc,d,d,1);
		gpcin.init(gpc,d,d,5);

		expphiin.init(expphi,d,d,d);
		xin.init(x,d,d,d);
		resin.init(res,d,d,d);
	}
	__syncthreads();





}


__global__
void atimes_transp_kernels(PoissonSolver solver, cudaMatrixf xin,cudaMatrixf resin)
{
	int idx = threadIdx.x;
	int idy = threadIdx.y;
	int idz = threadIdx.z;
	int bidx = blockIdx.x*blockDim.x;
	int bidy = blockIdx.y*blockDim.y;
	int bidz = blockIdx.z*blockDim.z;
	int gidx = bidx+idx;
	int gidy = bidy+idy;
	int gidz = bidz+idz;

	float result;

	__shared__ sMatrixf apcs,bpcs,cpcs,dpcs,epcs,fpcs,gpcs;
	__shared__ sMatrixf x,res,expphi;

	setup_shared(apcs,bpcs,cpcs,dpcs,epcs,fpcs,gpcs,expphi,x,res);

	// Load all the data into shared memory
	while((idx < (blockDim.x+4))&&(gidx <= (solver.n1+3)))
	{
		idy = threadIdx.y;
		gidy = bidy+idy;
		while((idy < (blockDim.y+4))&&(gidy <= (solver.n2+3)))
		{
			idz = threadIdx.z;
			gidz = bidz+idz;
			if(idz == 0)
			{
				if(idy == 0)
				{
					apcs(idx) = solver.apc(gidx+1);
					bpcs(idx) = solver.bpc(gidx+1);
				}

				cpcs(idx,idy) = solver.cpc(gidx+1,gidy);
				dpcs(idx,idy) = solver.dpc(gidx+1,gidy);
				epcs(idx,idy) = solver.epc(gidx+1,gidy);
				fpcs(idx,idy) = solver.fpc(gidx+1,gidy);
			}



			while((idz < (blockDim.z+4))&&(gidz <= (solver.n3+3)))
			{
				expphi(idx,idy,idz) = exp(solver.phi(gidx+1,gidy,gidz));

				if(idx < 5) gpcs(idy,idz,idx) = solver.gpc(gidy,gidz,idx);

				if(gidx <= (solver.n1+3))
				{
					if(gidz == 0)
					{
						x(idx,idy,idz) = xin(gidx,gidy,solver.n3);
					}
					else if(gidz == solver.n3+1)
					{
						x(idx,idy,idz) = xin(gidx,gidy,1);
					}
					else
					{
						x(idx,idy,idz) = xin(gidx,gidy,gidz);
					}

				}
				else
				{
					x(idx,idy,idz) = 0.0;
				}

				// Do the edge cases, periodic boundary

				idz += blockDim.z;
				gidz = bidz+idz;
			}
			idy += blockDim.y;
			gidy = bidy+idy;
		}
		idx += blockDim.x;
		gidx = bidx+idx;
	}

	__syncthreads();

	idx = threadIdx.x+1;
	idy = threadIdx.y+1;
	idz = threadIdx.z+1;
	gidx = bidx+idx;
	gidy = bidy+idy;
	gidz = bidz+idz;

	if(blockIdx.x+blockIdx.y+blockIdx.z == 0)
	{
		float sdata = cpcs(idx,idy);
		float gdata = solver.cpc(gidx,gidy);
		//if(sdata!=gdata)
		//printf("shared x = %f, global x = %f, at %i, %i, %i\n",sdata,gdata,gidx,gidy,gidz);
	}

	// End of shared memory load

	// Reset thread ID's
	idx = threadIdx.x+1;
	idy = threadIdx.y+1;
	idz = threadIdx.z+1;
	gidx = bidx+idx;
	gidy = bidy+idy;
	gidz = bidz+idz;

	// Bulk loop
	if((gidz > 0)&&(gidz <= solver.n3))
	{
		if((gidy > 0)&&(gidy <= solver.n2))
		{
			if((gidx > 0)&&(gidx <= (solver.n1-3)))
			{
				res(idx,idy,idz) = bpcs(idx+1)*x(idx+1,idy,idz)
						+ apcs(idx-1)*x(idx-1,idy,idz)
						+ dpcs(idx,idy+1)*x(idx,idy+1,idz)
						+ cpcs(idx,idy-1)*x(idx,idy-1,idz)
						+ epcs(idx,idy)*(x(idx,idy,idz+1)+x(idx,idy,idz-1))
						- (fpcs(idx,idy)+expphi(idx,idy,idz))*x(idx,idy,idz);
			}
			else if(gidx == (solver.n1-2))
			{
				res(idx,idy,idz) = ((bpcs(idx+1)+gpcs(idy,idz,0)*apcs(idx+1))*x(idx+1,idy,idz)
						+ apcs(idx-1)*x(idx-1,idy,idz)
						+ dpcs(idx,idy+1)*x(idx,idy+1,idz)
						+ cpcs(idx,idy-1)*x(idx,idy-1,idz)
						+ epcs(idx,idy)*(x(idx,idy,idz+1)+x(idx,idy,idz-1))
						- (fpcs(idx,idy)+expphi(idx,idy,idz))*x(idx,idy,idz));
				//printf("Solving res(%i, %i, %i) = %f\n",gidx,gidy,gidz,res(idx,idy,idz));
			}
			else if(gidx == (solver.n1-1))
			{
				res(idx,idy,idz) = bpcs(idx+1)*x(idx+1,idy,idz)
						+ apcs(idx-1)*x(idx-1,idy,idz)
						+ (dpcs(idx,idy+1)+gpcs(idy+1,idz,1)*apcs(idx))*x(idx,idy+1,idz)
						+ (cpcs(idx,idy-1)+gpcs(idy-1,idz,2)*apcs(idx))*x(idx,idy-1,idz)
						+ epcs(idx,idy)*(x(idx,idy,idz+1)+x(idx,idy,idz-1))
						- (fpcs(idx,idy)+expphi(idx,idy,idz)-gpcs(idy,idz,4)*apcs(idx))*x(idx,idy,idz);
			}
			else
			{
				res(idx,idy,idz) = 0;
			}
		}
	}
	__syncthreads();
	if(gidx <= solver.n1)
	{
		if(gidy <= solver.n2)
		{
			if(gidz <= solver.n3)
			{
				 result = res(idx,idy,idz);
				if(isnan(result))
				{
					result = 0.0;
				}
				 resin(gidx,gidy,gidz) = result;
			}
		}
	}

}

__global__
void atimes_transp_kernel(PoissonSolver solver, cudaMatrixf x,cudaMatrixf resin)
{
	int idx = threadIdx.x+1;
	int idy = threadIdx.y;
	int idz = threadIdx.z;
	int bidx = blockIdx.x*blockDim.x;
	int bidy = blockIdx.y*blockDim.y;
	int bidz = blockIdx.z*blockDim.z;
	int gidx = bidx+idx;
	int gidy = bidy+idy;
	int gidz = bidz+idz;

	int gidzp = gidz+1;
	int gidzm = gidz-1;

	if(gidz == 1) gidzm = solver.n3;
	if(gidz == solver.n3) gidzp = 1;

	// Bulk loop
	if((gidz > 0)&&(gidz <= solver.n3))
	{
		if((gidy > 0)&&(gidy <= solver.n2))
		{
			if((gidx > 0)&&(gidx <= (solver.n1-2)))
			{
				resin(gidx,gidy,gidz) = solver.bpc(gidx+1+1)*x(gidx+1,gidy,gidz)
						+ solver.apc(gidx-1+1)*x(gidx-1,gidy,gidz)
						+ solver.dpc(gidx+1,gidy+1)*x(gidx,gidy+1,gidz)
						+ solver.cpc(gidx+1,gidy-1)*x(gidx,gidy-1,gidz)
						+ solver.epc(gidx+1,gidy)*(x(gidx,gidy,gidzp)+x(gidx,gidy,gidzm))
						- (solver.fpc(gidx+1,gidy)+exp(solver.phi(gidx+1,gidy,gidz)))*x(gidx,gidy,gidz);
			}
			else if(gidx == (solver.n1-1))
			{
				resin(gidx,gidy,gidz) = ((solver.bpc(gidx+1+1)+solver.gpc(gidy,gidz,0)*solver.apc(gidx+1+1))*x(gidx+1,gidy,gidz)
						+ solver.apc(gidx-1+1)*x(gidx-1,gidy,gidz)
						+ solver.dpc(gidx+1,gidy+1)*x(gidx,gidy+1,gidz)
						+ solver.cpc(gidx+1,gidy-1)*x(gidx,gidy-1,gidz)
						+ solver.epc(gidx+1,gidy)*(x(gidx,gidy,gidzp)+x(gidx,gidy,gidzm))
						- (solver.fpc(gidx+1,gidy)+exp(solver.phi(gidx+1,gidy,gidz)))*x(gidx,gidy,gidz));
			}
			else if(gidx == (solver.n1))
			{
				resin(gidx,gidy,gidz) = solver.bpc(gidx+1+1)*x(gidx+1,gidy,gidz)
						+ solver.apc(gidx-1+1)*x(gidx-1,gidy,gidz)
						+ (solver.dpc(gidx+1,gidy+1)+solver.gpc(gidy+1,gidz,1)*solver.apc(gidx+1))*x(gidx,gidy+1,gidz)
						+ (solver.cpc(gidx+1,gidy-1)+solver.gpc(gidy-1,gidz,2)*solver.apc(gidx+1))*x(gidx,gidy-1,gidz)
						+ solver.epc(gidx+1,gidy)*(x(gidx,gidy,gidzp)+x(gidx,gidy,gidzm))
						- (solver.fpc(gidx+1,gidy)+exp(solver.phi(gidx+1,gidy,gidz))-solver.gpc(gidy,gidz,4)*solver.apc(gidx+1))*x(gidx,gidy,gidz);
			}
		}
	}

}

__global__
void atimes_kernels(PoissonSolver solver, cudaMatrixf xin,cudaMatrixf resin)
{
	int idx = threadIdx.x;
	int idy = threadIdx.y;
	int idz = threadIdx.z;
	int bidx = blockIdx.x*blockDim.x;
	int bidy = blockIdx.y*blockDim.y;
	int bidz = blockIdx.z*blockDim.z;
	int gidx = bidx+idx;
	int gidy = bidy+idy;
	int gidz = bidz+idz;

	__shared__ sMatrixf apcs,bpcs,cpcs,dpcs,epcs,fpcs,gpcs;
	__shared__ sMatrixf x,res,expphi;

	setup_shared(apcs,bpcs,cpcs,dpcs,epcs,fpcs,gpcs,expphi,x,res);

	// Load all the data into shared memory
	while((idx < (blockDim.x+2))&&(gidx <= (solver.n1+3)))
	{
		idy = threadIdx.y;
		gidy = bidy+idy;
		while((idy < (blockDim.y+2))&&(gidy <= (solver.n2+3)))
		{
			idz = threadIdx.z;
			gidz = bidz+idz;
			if(idz == 0)
			{
				if(idy == 0)
				{
					apcs(idx) = solver.apc(gidx+1);
					bpcs(idx) = solver.bpc(gidx+1);
				}

				cpcs(idx,idy) = solver.cpc(gidx+1,gidy);
				dpcs(idx,idy) = solver.dpc(gidx+1,gidy);
				epcs(idx,idy) = solver.epc(gidx+1,gidy);
				fpcs(idx,idy) = solver.fpc(gidx+1,gidy);
			}



			while((idz < (blockDim.z+2))&&(gidz <= (solver.n3+3)))
			{
				expphi(idx,idy,idz) = exp(solver.phi(gidx+1,gidy,gidz));

				if(idx < 5) gpcs(idy,idz,idx) = solver.gpc(gidy,gidz,idx);

				if(gidx <= (solver.n1+2))
				{
					if(gidz == 0)
					{
						x(idx,idy,idz) = xin(gidx,gidy,solver.n3);
					}
					else if(gidz == solver.n3+1)
					{
						x(idx,idy,idz) = xin(gidx,gidy,1);
					}
					else
					{
						x(idx,idy,idz) = xin(gidx,gidy,gidz);
					}

				}

				// Do the edge cases, periodic boundary

				idz += blockDim.z;
				gidz = bidz+idz;
			}
			idy += blockDim.y;
			gidy = bidy+idy;
		}
		idx += blockDim.x;
		gidx = bidx+idx;
	}

	__syncthreads();
	// End of shared memory load

	// Reset thread ID's
	idx = threadIdx.x+1;
	idy = threadIdx.y+1;
	idz = threadIdx.z+1;
	gidx = bidx+idx;
	gidy = bidy+idy;
	gidz = bidz+idz;

	// Bulk loop
	if((gidz > 0)&&(gidz <= solver.n3+1))
	{
		if((gidy > 0)&&(gidy <= solver.n2+1))
		{
			if((gidx > 0)&&(gidx <= (solver.n1)))
			{
				res(idx,idy,idz) = apcs(idx)*x(idx+1,idy,idz)
						+ bpcs(idx)*x(idx-1,idy,idz)
						+ cpcs(idx,idy)*x(idx,idy+1,idz)
						+ dpcs(idx,idy)*x(idx,idy-1,idz)
						+ epcs(idx,idy)*(x(idx,idy,idz+1)+x(idx,idy,idz-1))
						- (fpcs(idx,idy)+expphi(idx,idy,idz))*x(idx,idy,idz);
			}
			else
			{
				res(idx,idy,idz) = 0;
			}
		}
	}
	__syncthreads();

	if((gidz > 0)&&(gidz <= solver.n3+1))
	{
		if((gidy > 0)&&(gidy <= solver.n2+1))
		{
			if((gidx == (solver.n1+1)))
			{
				x(idx+1,idy,idz) = gpcs(idy,idz,0)*x(idx-1,idy,idz)
						+ gpcs(idy,idz,1)*x(idx,idy-1,idz)
						+ gpcs(idy,idz,2)*x(idx,idy+1,idz)
						+ 0.0f*gpcs(idy,idz,3)
						+ gpcs(idy,idz,4)*x(idx,idy,idz);

				res(idx,idy,idz) = apcs(idx)*x(idx+1,idy,idz)
						+ bpcs(idx)*x(idx-1,idy,idz)
						+ cpcs(idx,idy)*x(idx,idy+1,idz)
						+ dpcs(idx,idy)*x(idx,idy-1,idz)
						+ epcs(idx,idy)*(x(idx,idy,idz+1)+x(idx,idy,idz-1))
						- (fpcs(idx,idy)+expphi(idx,idy,idz))*x(idx,idy,idz);
			}
		}
	}
	__syncthreads();

	idx = threadIdx.x+1;
	idy = threadIdx.y+1;
	idz = threadIdx.z+1;
	gidx = bidx+idx;
	gidy = bidy+idy;
	gidz = bidz+idz;
	if(gidx <= solver.n1+1)
	{
		if(gidy <= solver.n2+1)
		{
			if(gidz <= solver.n3+1)
			{
				resin(gidx,gidy,gidz) = res(idx,idy,idz);

				if(gidx == solver.n1-1) xin(gidx+2,gidy,gidz) = x(idx+2,idy,idz);

			}
		}
	}

}


__global__
void asolve_kernel(PoissonSolver solver,cudaMatrixf bin,cudaMatrixf zin)
{
	int idx = threadIdx.x+1;
	int idy = threadIdx.y+1;
	int idz = threadIdx.z+1;
	int gidx = blockDim.x*blockIdx.x+idx;
	int gidy = blockDim.y*blockIdx.y+idy;
	int gidz = blockDim.z*blockIdx.z+idz;

	float result;

	if(gidz <= solver.n3)
	{
		if(gidy <= solver.n2)
		{
			if(gidx <= solver.n1)
			{
				result = bin(gidx,gidy,gidz)/(-solver.fpc(gidx+1,gidy)-exp(solver.phi(gidx+1,gidy,gidz)));
				if(isnan(result))
				{
					result = 0.0;
				}
				zin(gidx,gidy,gidz) = result;

			}
			else if(gidx == solver.n1)
			{
				result = bin(gidx,gidy,gidz)/(-solver.fpc(gidx+1,gidy)-exp(solver.phi(gidx+1,gidy,gidz))
						+ solver.apc(gidx+1)*solver.gpc(gidy,gidz,4));
				if(isnan(result))
				{
					result = 0.0;
				}
				zin(gidx,gidy,gidz) = result;
			}
		}
	}
}

__global__
void setup_res_kernel(PoissonSolver solver)
{
	int idx = threadIdx.x+1;
	int idy = threadIdx.y+1;
	int idz = threadIdx.z+1;
	int gidx = blockDim.x*blockIdx.x+idx;
	int gidy = blockDim.y*blockIdx.y+idy;
	int gidz = blockDim.z*blockIdx.z+idz;

	float res;

	while(gidy <= solver.n2)
	{
		gidx = blockDim.x*blockIdx.x+idx;
		while(gidx <= solver.n1)
		{

			res = solver.b(gidx,gidy,gidz)-solver.res(gidx,gidy,gidz);

			solver.res(gidx,gidy,gidz) = res;
			solver.resr(gidx,gidy,gidz) = res;

			if(gidx == 1)
			{
				solver.res(gidx-1,gidy,gidz) = 0;
				solver.resr(gidx-1,gidy,gidz) = 0;
			}

			gidx += blockDim.x;
		}
		gidy += blockDim.y;
	}

}

__global__
void pppp_kernel(PoissonSolver solver)
{
	int idx = threadIdx.x+1;
	int idy = threadIdx.y+1;
	int idz = threadIdx.z+1;
	int gidx = blockDim.x*blockIdx.x+idx;
	int gidy = blockDim.y*blockIdx.y+idy;
	int gidz = blockDim.z*blockIdx.z+idz;

	float bk;

	if(solver.bkden == 0.0f)
	{
		bk = 0.0;
	}
	else
	{
		bk = solver.bknum/solver.bkden;
	}

	float result;


	while(gidy <= solver.n2)
	{
		gidx = blockDim.x*blockIdx.x+idx;
		while(gidx <= solver.n1)
		{

			result=bk*solver.p(gidx,gidy,gidz)+solver.z(gidx,gidy,gidz);
			if(isnan(result))
			{
				result = 0.0;
			}
			solver.p(gidx,gidy,gidz) = result;
			result = bk*solver.pp(gidx,gidy,gidz)+solver.zz(gidx,gidy,gidz);
			if(isnan(result))
			{
				result = 0.0;
			}
			solver.pp(gidx,gidy,gidz) = result;

			if(gidx == 1)
			{
				solver.p(gidx-1,gidy,gidz) = 0;
				solver.pp(gidx-1,gidy,gidz) = 0;
			}

			gidx += blockDim.x;
		}
		gidy += blockDim.y;
	}

}

__global__
void Psolve_reduce_kernel(PoissonSolver solver,const int operation)
{
	int idx = threadIdx.x+1;
	int idy = threadIdx.y+1;
	int idz = threadIdx.z+1;
	int gidx = blockDim.x*blockIdx.x+idx;
	int gidy = blockDim.y*blockIdx.y+idy;
	int gidz = blockDim.z*blockIdx.z+idz;

	int thid = gidx-1+blockDim.x*(gidy-1+blockDim.y*(gidz-1));

	float my_val = 0.0;


	while(gidy <= solver.n2)
	{
		gidx = blockDim.x*blockIdx.x+idx;
		while(gidx <= solver.n1)
		{
			 switch(operation)
			 {
			 case 0:
				 my_val = solver.bknum_eval(my_val,gidx,gidy,gidz);
				// printf("my_val = %f\n",my_val);
				 break;
			 case 1:
				 my_val = solver.aknum_eval(my_val,gidx,gidy,gidz);
				 break;
			 case 2:
				 my_val = solver.delta_eval(my_val,gidx,gidy,gidz);
				 break;
			 default:
				 break;
			 }


			gidx += blockDim.x;
		}
		gidy += blockDim.y;
	}

	solver.sum_array[thid] = my_val;

}

__host__
void PoissonSolver::eval_sum(const int operation)
{
	dim3 cudaBlockSize(32,16,1);
	dim3 cudaGridSize(1,1,n3);

	CUDA_SAFE_CALL(cudaMemset(sum_array,0,512*(npsisize+1)*sizeof(float)));

	float result;

	CUDA_SAFE_KERNEL((Psolve_reduce_kernel<<<cudaGridSize,cudaBlockSize>>>(*this,operation)));

	thrust::device_ptr<float> reduce_ptr(sum_array);

	if(operation < 2)
	{
		// bknum and
		result = thrust::reduce(reduce_ptr,reduce_ptr+(n3)*512);
	}
	else
	{
		// delta is a maximum
		result = thrust::reduce(reduce_ptr,reduce_ptr+(n3)*512,(float) 0.0,thrust::maximum<float>());
	}

	 switch(operation)
	 {
	 case 0:
		 bknum = result;
		 break;
	 case 1:
		akden = result;
		 break;
	 case 2:
		 deltamax = max(result,deltamax);
		 break;
	 default:
		 break;
	 }

}

__host__
void PoissonSolver::setup_res(void)
{
	dim3 cudaBlockSize(32,16,1);
	dim3 cudaGridSize(1,1,n3);

	CUDA_SAFE_KERNEL((setup_res_kernel<<<cudaGridSize,cudaBlockSize>>>(*this)));
}

__host__
void PoissonSolver::pppp(void)
{
	dim3 cudaBlockSize(32,16,1);
	dim3 cudaGridSize(1,1,n3);

	CUDA_SAFE_KERNEL((pppp_kernel<<<cudaGridSize,cudaBlockSize>>>(*this)));
}

__host__
void PoissonSolver::asolve(int n1_in,int n2_in,int n3_in,cudaMatrixf bin, cudaMatrixf zin)
{
	dim3 cudaBlockSize(8,8,8);
	dim3 cudaGridSize(1,1,1);
	n1 = n1_in;
	n2 = n2_in;
	n3 = n3_in;

	cudaGridSize.x = (n1+cudaBlockSize.x+4)/cudaBlockSize.x;
	cudaGridSize.y = (n2+cudaBlockSize.y+4)/cudaBlockSize.y;
	cudaGridSize.z = (n3+cudaBlockSize.z+4)/cudaBlockSize.z;

	CUDA_SAFE_KERNEL((asolve_kernel<<<cudaGridSize,cudaBlockSize>>>(*this,bin,zin)));

}

__host__
void PoissonSolver::atimes(int n1_in,int n2_in,int n3_in,cudaMatrixf xin, cudaMatrixf resin,int itransp)
{
	dim3 cudaBlockSize(8,8,8);
	dim3 cudaGridSize(1,1,1);
	n1 = n1_in;
	n2 = n2_in;
	n3 = n3_in;

	cudaGridSize.x = (n1+cudaBlockSize.x+4)/cudaBlockSize.x;
	cudaGridSize.y = (n2+cudaBlockSize.y+4)/cudaBlockSize.y;
	cudaGridSize.z = (n3+cudaBlockSize.z+4)/cudaBlockSize.z;



	if(itransp == 0)
	{
		// No transpose case
		CUDA_SAFE_KERNEL((atimes_kernels<<<cudaGridSize,cudaBlockSize>>>(*this,xin,resin)));
	}
	else if(itransp == 1) // Transpose case
	{
		CUDA_SAFE_KERNEL((atimes_transp_kernels<<<cudaGridSize,cudaBlockSize>>>(*this,xin,resin)));
	}

}

__host__
void PoissonSolver::cg3D(int n1_in,int n2_in,int n3_in,float* bin,float* xin,float &tol,int &iter,int &itmax,int &lbcg)
{
	n1 = n1_in;
	n2 = n2_in;
	n3 = n3_in;

	iter = 0;
	// Initialize the denominators
	bknum = 0;
	bkden = 0.0;
	akden = 0;
	deltamax = tol*1.5f;

	x.cudaMatrixcpy(xin,cudaMemcpyHostToDevice);
	b.cudaMatrixcpy(bin,cudaMemcpyHostToDevice);

	atimes(n1,n2,n3,x,res,0);

	setup_res();

	if(lbcg == 0)
	{
		atimes(n1,n2,n3,res,resr,0);
	}

	asolve(n1,n2,n3,res,z);

	// Main Loop
	while(deltamax >= tol)
	{
		iter++;

		printf("deltamax = %f\n",deltamax);
		asolve(n1,n2,n3,resr,zz);
		bknum = 0;

		// evaluate bknum;
		eval_sum(0);

		//printf("bknum = %f\n",bknum);

		// do the p's
		pppp();

		bkden = bknum;

		atimes(n1,n2,n3,p,z,0);

		akden = 0;

		// evaluate akden
		eval_sum(1);

		atimes(n1,n2,n3,pp,zz,lbcg);
		// set deltamax = 0 so that we can reevaluate it.
		deltamax = 0;

		// evaluate deltamax
		eval_sum(2);

		if(iter >= itmax)
			break;

		asolve(n1,n2,n3,res,z);
	}



	// copy results back to the cpu
	x.cudaMatrixcpy(xin,cudaMemcpyDeviceToHost);
	b.cudaMatrixcpy(bin,cudaMemcpyDeviceToHost);
}

extern "C" void cg3d_gpu_(long int* solverPtr,float* phi,int* lbcg,int* n1,int* n2,int* n3,
										    float* bin,float* xin,float* tol,int* iter,int* itmax)
{
	PoissonSolver* solver;

	solver = ((PoissonSolver*)(*solverPtr));

	solver -> phi.cudaMatrixcpy(phi,cudaMemcpyHostToDevice);

	solver -> cg3D(*n1,*n2,*n3,bin,xin,*tol,*iter,*itmax,*lbcg);

}




extern "C" void asolve_test_(long int* solverPtr,float* phi,float* bin,float* zin,
														int* n1,int* n2,int* n3)
{
	printf("Executing asolve test\n");
	PoissonSolver* solver;

	solver = ((PoissonSolver*)(*solverPtr));

	int nrsize = solver->nrsize;
	int nthsize = solver->nthsize;
	int npsisize = solver->npsisize;

	float tolerance = 1.0e-9;
	float zerror;

	int gidx;

	// Allocate temporary storage for the results.
	float* b_gpu = (float*)malloc((nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));
	float* b_cpu = (float*)malloc((nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));

	float* z_gpu = (float*)malloc((nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));
	float* z_cpu = (float*)malloc((nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));


	cudaMatrixf b_d = solver->b;
	cudaMatrixf z_d = solver->z;

	b_d.cudaMatrixcpy(bin,cudaMemcpyHostToDevice);
	b_d.cudaMatrixcpy(b_cpu,cudaMemcpyDeviceToHost);
	z_d.cudaMatrixcpy(zin,cudaMemcpyHostToDevice);
	z_d.cudaMatrixcpy(z_cpu,cudaMemcpyDeviceToHost);
	solver->phi.cudaMatrixcpy(phi,cudaMemcpyHostToDevice);

	// Do cpu asolve
	asolve_(*n1,*n2,*n3,b_cpu,z_cpu,&zerror);

	// Do gpu asolve
	solver->asolve(*n1,*n2,*n3,b_d,z_d);



	// Copy results back to the host
	z_d.cudaMatrixcpy(z_gpu,cudaMemcpyDeviceToHost);
	b_d.cudaMatrixcpy(b_gpu,cudaMemcpyDeviceToHost);

	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	// Check the results with the results of the host routine
	for(int i=0;i<(*n1+1);i++)
	{
		for(int j=1;j<(*n2+1);j++)
		{
			for(int k=1;k<(*n3+1);k++)
			{
				gidx = i+(nrsize-1)*(j+(nthsize+1)*k);

				float gpu_data = z_gpu[gidx];
				float cpu_data = z_cpu[gidx];

				float terror = 2.0*abs(gpu_data-cpu_data)/max(1.0*tolerance,abs(cpu_data+gpu_data));

				if(terror > tolerance)
				{
					printf("Error res %f != %f with error %f at %i, %i, %i\n",gpu_data,cpu_data,terror,i,j,k);
				}
			}
		}
	}



	free(b_gpu);
	free(b_cpu);
	free(z_gpu);
	free(z_cpu);



}

extern "C" void atimes_test_(long int* solverPtr,float* phi,float* xin,float* res,
														int* n1,int* n2,int* n3,uint* itrnsp)
{
	PoissonSolver* solver;

	solver = ((PoissonSolver*)(*solverPtr));

	int nrsize = solver->nrsize;
	int nthsize = solver->nthsize;
	int npsisize = solver->npsisize;

	float tolerance = 1.0e-4;

	int gidx;

	// Allocate temporary storage for the results.
	float* res_gpu = (float*)malloc((nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));
	float* res_cpu = (float*)malloc((nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));

	float* x_gpu = (float*)malloc((nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));
	float* x_cpu = (float*)malloc((nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));


	cudaMatrixf x_d = solver->x;
	cudaMatrixf res_d(nrsize-1,nthsize+1,npsisize+1);

	res_d.cudaMatrixcpy(res,cudaMemcpyHostToDevice);
	res_d.cudaMatrixcpy(res_cpu,cudaMemcpyDeviceToHost);
	solver->phi.cudaMatrixcpy(phi,cudaMemcpyHostToDevice);
	solver->x.cudaMatrixcpy(xin,cudaMemcpyHostToDevice);
	solver->x.cudaMatrixcpy(x_cpu,cudaMemcpyDeviceToHost);


	// Do cpu atimes
	atimes_(*n1,*n2,*n3,x_cpu,res_cpu,itrnsp);

	// Do gpu atimes
	solver->atimes(*n1,*n2,*n3,x_d,res_d,1);



	// Copy results back to the host
	res_d.cudaMatrixcpy(res_gpu,cudaMemcpyDeviceToHost);
	solver->x.cudaMatrixcpy(x_gpu,cudaMemcpyDeviceToHost);

	// Check the results with the results of the host routine
	for(int i=1;i<(*n1);i++)
	{
		for(int j=1;j<(*n2+1);j++)
		{
			for(int k=1;k<(*n3+1);k++)
			{
				gidx = i+(nrsize-1)*(j+(nthsize+1)*k);

				float gpu_data = res_gpu[gidx];
				float cpu_data = res_cpu[gidx];

				float terror = 1.0*abs(gpu_data-cpu_data)/max(10.0*tolerance,abs(cpu_data+gpu_data));

				if(terror > tolerance)
				{
					printf("Error res %f != %f with error %f at %i, %i, %i\n",gpu_data,cpu_data,terror,i,j,k);
				}
			}
		}
	}

	free(res_gpu);
	free(res_cpu);
	free(x_gpu);
	free(x_cpu);
	res_d.cudaMatrixFree();



}


extern "C" void cg3d_test_(long int* solverPtr,float* phi,int* lbcg,int* n1,int* n2,int* n3,
	    									 float* bin,float* xin,float* tol,int* iter,int* itmax)
{
	PoissonSolver* solver;

	solver = ((PoissonSolver*)(*solverPtr));

	int nrsize = solver->nrsize;
	int nthsize = solver->nthsize;
	int npsisize = solver->npsisize;

	int iter_gpu;
	int iter_cpu;
	float gpu_tol = *tol;

	float tolerance = 1.0e-4;

	int gidx;

	// Allocate temporary storage for the results.
	float* b_gpu = (float*)malloc((nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));
	float* b_cpu = (float*)malloc((nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));

	float* x_gpu = (float*)malloc((nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));
	float* x_cpu = (float*)malloc((nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));

	memcpy(b_gpu,bin,(nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));
	memcpy(b_cpu,bin,(nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));

	memcpy(x_gpu,xin,(nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));
	memcpy(x_cpu,xin,(nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));

	// Do gpu atimes
	cg3d_gpu_(solverPtr,phi,lbcg,n1,n2,n3,b_gpu,x_gpu,&gpu_tol,&iter_gpu,itmax);
	// Do cpu cg3d
	cg3d_(*n1,*n2,*n3,b_cpu,x_cpu,*tol,iter_cpu,*itmax);

	printf("CPU took %i iterations\n",iter_cpu);



	printf("GPU took %i iterations\n",iter_gpu);

	// Check the results with the results of the host routine
	for(int i=1;i<(*n1+1);i++)
	{
		for(int j=1;j<(*n2+1);j++)
		{
			for(int k=1;k<(*n3+1);k++)
			{
				gidx = i+(nrsize-1)*(j+(nthsize+1)*k);

				float gpu_data = x_gpu[gidx];
				float cpu_data = x_cpu[gidx];

				float terror = 1.0*abs(gpu_data-cpu_data)/max(100.0*tolerance,abs(cpu_data+gpu_data));

				if(terror > tolerance)
				{
					printf("Error res %f != %f with error %f at %i, %i, %i\n",gpu_data,cpu_data,terror,i,j,k);
				}
			}
		}
	}

	free(b_gpu);
	free(b_cpu);
	free(x_gpu);
	free(x_cpu);
}

























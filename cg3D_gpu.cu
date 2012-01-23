#include "gpu_solver.cuh"
#include "XPlist.cuh"

#define ATIMES_DOMAIN_DIMx 12
#define ATIMES_DOMAIN_DIMy 12
#define ATIMES_DOMAIN_DIMz 12

#define ATIMES_DOMAIN_DIM 13

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

__device__
void calc_dims(int &idx,int &idy,int &idz,int thidin,int3 dimin)
{
	idz = thidin/(dimin.x*dimin.y);
	idy = thidin/dimin.x - idz*dimin.y;
	idx = thidin - dimin.x*(idy + dimin.y*idz);

}

__host__
void print_cudaMatrixf(cudaMatrixf M_in,int n1 = 0,int n2 = 0, int n3 = 0)
{
	int nx,ny,nz;
	cudaExtent h_extent;

	h_extent = M_in.getdims();

	nx = h_extent.width/sizeof(float);
	ny = h_extent.height;
	nz = h_extent.depth;

	if(n1 != 0)
		nx = n1;

	if(n2 != 0)
		ny = n2;

	if(n3 != 0)
		nz = n3;

	// Allocate host memory
	float* h_data = (float*)malloc(nx*ny*nz*sizeof(float));

	// Copy gpu data to the cpu
	M_in.cudaMatrixcpy(h_data,cudaMemcpyDeviceToHost);

	for(int i=0;i<nx;i++)
	{
		for(int j=0;j<ny;j++)
		{
			for(int k=0;k<nz;k++)
			{
				int thid = i+nx*(j+ny*k);
				printf("M(%i,%i,%i) = %f\n",i,j,k,h_data[thid]);
			}
		}
	}

	free(h_data);

}


extern "C" void gpu_psolver_init_(long int* PsolvPtr,long int* mesh_ptr,float* apc,float* bpc,float* cpc, float* dpc,float* epc,
													   float* fpc,float* gpc,int* nrsize,int* nthsize,int* npsisize)
{
	Mesh_data mesh_d = *(Mesh_data*)(*mesh_ptr);
	PoissonSolver* solver = (PoissonSolver*)malloc(sizeof(PoissonSolver));

	solver->phi = mesh_d.phi;
	solver->rho = mesh_d.rho;
	solver->phiaxis = mesh_d.phiaxis;

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
//	phi.cudaMatrix_allocate(nrsize+1,nthsize+1,npsisize+1);
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
//	phi.cudaMatrixFree();
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
	const int dx = ATIMES_DOMAIN_DIMx+2;
	const int dy = ATIMES_DOMAIN_DIMy+2;
	const int dz = ATIMES_DOMAIN_DIMz+2;
	__shared__ float apc[dx];
	__shared__ float bpc[dx];
	__shared__ float cpc[dx*dy];
	__shared__ float dpc[dx*dy];
	__shared__ float epc[dx*dy];
	__shared__ float fpc[dx*dy];
	__shared__ float gpc[dy*dz*5];
	__shared__ float expphi[dx*dy*dz];
	__shared__ float x[dx*dy*dz];
	__shared__ float res[dx*dy*dz];

	if(threadIdx.x+threadIdx.y+threadIdx.z == 0)
	{
		apcin.init(apc,dx,1,1);
		bpcin.init(bpc,dx,1,1);
		cpcin.init(cpc,dx,dy,1);
		dpcin.init(dpc,dx,dy,1);
		epcin.init(epc,dx,dy,1);
		fpcin.init(fpc,dx,dy,1);
		gpcin.init(gpc,dy,dz,5);

		expphiin.init(expphi,dx,dy,dz);
		xin.init(x,dx,dy,dz);
		resin.init(res,dx,dy,dz);
	}
	__syncthreads();





}


__global__
void atimes_transp_kernels(PoissonSolver solver, cudaMatrixf xin,cudaMatrixf resin,int3 ndo)
{
	int idx = threadIdx.x;
	int idy = threadIdx.y;
	int idz = threadIdx.z;
	int bidx = ndo.x*blockIdx.x*blockDim.x;
	int bidy = ndo.y*blockIdx.y*blockDim.y;
	int bidz = ndo.z*blockIdx.z*blockDim.z;
	int gidx = bidx+idx;
	int gidy = bidy+idy;
	int gidz = bidz+idz;

	float result;

	__shared__ sMatrixf apcs,bpcs,cpcs,dpcs,epcs,fpcs,gpcs;
	__shared__ sMatrixf x,res,expphi;

	setup_shared(apcs,bpcs,cpcs,dpcs,epcs,fpcs,gpcs,expphi,x,res);
/*
	// Load all the data into shared memory
	while((idx < (ndo.x*blockDim.x+2))&&(gidx <= (solver.n1+1)))
	{
		idy = threadIdx.y;
		gidy = bidy+idy;
		while((idy < (ndo.y*blockDim.y+2))&&(gidy <= (solver.n2+1)))
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



			while((idz < (blockDim.z+2))&&(gidz <= (solver.n3+1)))
			{
				expphi(idx,idy,idz) = exp(solver.phi(gidx+1,gidy,gidz));

				if(idx < 5) gpcs(idy,idz,idx) = solver.gpc(gidy,gidz,idx);

				if(gidx <= (solver.n1))
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
	*/

	// Load all the data into shared memory
	while((idz < (ndo.z*blockDim.z+2))&&(gidz <= (solver.n3+1)))
	{
		idy = threadIdx.y;
		gidy = bidy+idy;
		while((idy < (ndo.y*blockDim.y+2))&&(gidy <= (solver.n2+1)))
		{
			idx = threadIdx.x;
			gidx = bidx+idx;
			while((idx < (ndo.x*blockDim.x+2))&&(gidx <= (solver.n1+1)))
			{
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

				expphi(idx,idy,idz) = exp(solver.phi(gidx+1,gidy,gidz));

				if(idx < 5) gpcs(idy,idz,idx) = solver.gpc(gidy,gidz,idx);

				if(gidx <= (solver.n1))
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

				idx += blockDim.x;
				gidx = bidx+idx;
			}
			idy += blockDim.y;
			gidy = bidy+idy;
		}
		idz += blockDim.z;
		gidz = bidz+idz;
	}

	__syncthreads();

	// End of shared memory load

	// Reset thread ID's
	idz = threadIdx.z+1;
	gidz = bidz+idz;

	// Bulk loop
	while((idz < (ndo.z*blockDim.z+1))&&(gidz <= solver.n3))
	{
		idy = threadIdx.y+1;
		gidy = bidy+idy;
		while((idy < (ndo.y*blockDim.y+1))&&(gidy <= solver.n2))
		{
			idx = threadIdx.x+1;
			gidx = bidx+idx;
			while((idx < (ndo.x*blockDim.x+1))&&(gidx < solver.n1))
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

				idx += blockDim.x;
				gidx = bidx+idx;
			}
			idy += blockDim.y;
			gidy = bidy+idy;
		}
		idz += blockDim.z;
		gidz = bidz+idz;
	}

	// Reset thread ID's
	idz = threadIdx.z+1;
	gidz = bidz+idz;
	while((idz < (ndo.z*blockDim.z+1))&&(gidz <= solver.n3))
	{
		idy = threadIdx.y+1;
		gidy = bidy+idy;
		while((idy < (ndo.y*blockDim.y+1))&&(gidy <= solver.n2))
		{
			idx = threadIdx.x+1;
			gidx = bidx+idx;
			while((idx < (ndo.x*blockDim.x+1))&&(gidx < solver.n1))
			{
				result = res(idx,idy,idz);
				resin(gidx,gidy,gidz) = result;

				idx += blockDim.x;
				gidx = bidx+idx;
			}
			idy += blockDim.y;
			gidy = bidy+idy;
		}
		idz += blockDim.z;
		gidz = bidz+idz;
	}

}


template<int itransp>
__global__
void atimes_kernel(PoissonSolver solver, cudaMatrixf x,cudaMatrixf resin)
{
	int idx = threadIdx.x+1;
	int idy = threadIdx.y+1;
	int idz = threadIdx.z+1;
	int bidx = blockIdx.x*blockDim.x;
	int bidy = blockIdx.y*blockDim.y;
	int bidz = blockIdx.z*blockDim.z;
	int gidx = bidx+idx;
	int gidy = bidy+idy;
	int gidz = bidz+idz;




	// Bulk loop
	while(gidz <= solver.n3)
	{
		int gidzp = gidz+1;
		int gidzm = gidz-1;
		if(gidz == 1) gidzm = solver.n3;
		if(gidz == solver.n3) gidzp = 1;

		gidy = idy+bidy;
		while(gidy <= solver.n2)
		{
			gidx = idx+bidx;
			while(gidx < solver.n1)
			{
				switch(itransp)
				{
				case 1:
					if((gidx > 0)&&(gidx <= (solver.n1-3)))
					{
						resin(gidx,gidy,gidz) = solver.bpc(gidx+1+1)*x(gidx+1,gidy,gidz)
								+ solver.apc(gidx-1+1)*x(gidx-1,gidy,gidz)
								+ solver.dpc(gidx+1,gidy+1)*x(gidx,gidy+1,gidz)
								+ solver.cpc(gidx+1,gidy-1)*x(gidx,gidy-1,gidz)
								+ solver.epc(gidx+1,gidy)*(x(gidx,gidy,gidzp)+x(gidx,gidy,gidzm))
								- (solver.fpc(gidx+1,gidy)+exp(solver.phi(gidx+1,gidy,gidz)))*x(gidx,gidy,gidz);
					}
					else if(gidx == (solver.n1-2))
					{
						resin(gidx,gidy,gidz) = ((solver.bpc(gidx+1+1)+solver.gpc(gidy,gidz,0)*solver.apc(gidx+1+1))*x(gidx+1,gidy,gidz)
								+ solver.apc(gidx-1+1)*x(gidx-1,gidy,gidz)
								+ solver.dpc(gidx+1,gidy+1)*x(gidx,gidy+1,gidz)
								+ solver.cpc(gidx+1,gidy-1)*x(gidx,gidy-1,gidz)
								+ solver.epc(gidx+1,gidy)*(x(gidx,gidy,gidzp)+x(gidx,gidy,gidzm))
								- (solver.fpc(gidx+1,gidy)+exp(solver.phi(gidx+1,gidy,gidz)))*x(gidx,gidy,gidz));
					}
					else if(gidx == (solver.n1-1))
					{
						resin(gidx,gidy,gidz) = solver.bpc(gidx+1+1)*x(gidx+1,gidy,gidz)
								+ solver.apc(gidx-1+1)*x(gidx-1,gidy,gidz)
								+ (solver.dpc(gidx+1,gidy+1)+solver.gpc(gidy+1,gidz,1)*solver.apc(gidx+1))*x(gidx,gidy+1,gidz)
								+ (solver.cpc(gidx+1,gidy-1)+solver.gpc(gidy-1,gidz,2)*solver.apc(gidx+1))*x(gidx,gidy-1,gidz)
								+ solver.epc(gidx+1,gidy)*(x(gidx,gidy,gidzp)+x(gidx,gidy,gidzm))
								- (solver.fpc(gidx+1,gidy)+exp(solver.phi(gidx+1,gidy,gidz))-solver.gpc(gidy,gidz,4)*solver.apc(gidx+1))*x(gidx,gidy,gidz);
					}
					break;
				case 0:
					if((gidx > 0)&&(gidx <= (solver.n1-2)))
					{
						resin(gidx,gidy,gidz) = solver.apc(gidx+1)*x(gidx+1,gidy,gidz)
								+ solver.bpc(gidx+1)*x(gidx-1,gidy,gidz)
								+ solver.cpc(gidx+1,gidy)*x(gidx,gidy+1,gidz)
								+ solver.dpc(gidx+1,gidy)*x(gidx,gidy-1,gidz)
								+ solver.epc(gidx+1,gidy)*(x(gidx,gidy,gidzp)+x(gidx,gidy,gidzm))
								- (solver.fpc(gidx+1,gidy)+exp(solver.phi(gidx+1,gidy,gidz)))*x(gidx,gidy,gidz);
					}
					else 	if((gidx == (solver.n1-1)))
					{
						x(gidx+1,gidy,gidz) = solver.gpc(gidy,gidz,0)*x(gidx-1,gidy,gidz)
								+ solver.gpc(gidy,gidz,1)*x(gidx,gidy-1,gidz)
								+ solver.gpc(gidy,gidz,2)*x(gidx,gidy+1,gidz)
								//+ 0.0f*gpcs(gidy,gidz,3)
								+ solver.gpc(gidy,gidz,4)*x(gidx,gidy,gidz);

						resin(gidx,gidy,gidz) = solver.apc(gidx+1)*x(gidx+1,gidy,gidz)
								+ solver.bpc(gidx+1)*x(gidx-1,gidy,gidz)
								+ solver.cpc(gidx+1,gidy)*x(gidx,gidy+1,gidz)
								+ solver.dpc(gidx+1,gidy)*x(gidx,gidy-1,gidz)
								+ solver.epc(gidx+1,gidy)*(x(gidx,gidy,gidzp)+x(gidx,gidy,gidzm))
								- (solver.fpc(gidx+1,gidy)+exp(solver.phi(gidx+1,gidy,gidz)))*x(gidx,gidy,gidz);
					}
					else
					{
						resin(gidx,gidy,gidz) = 0;
					}

					break;

				default:
					break;
				}

				gidx += blockDim.x*gridDim.x;
			}
			gidy += blockDim.y*gridDim.y;
		}
		gidz += blockDim.z*gridDim.z;
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
	while((idx < (blockDim.x+2))&&(gidx <= (solver.n1+1)))
	{
		idy = threadIdx.y;
		gidy = bidy+idy;
		while((idy < (blockDim.y+2))&&(gidy <= (solver.n2+1)))
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



			while((idz < (blockDim.z+2))&&(gidz <= (solver.n3+1)))
			{
				expphi(idx,idy,idz) = exp(solver.phi(gidx+1,gidy,gidz));

				if(idx < 5) gpcs(idy,idz,idx) = solver.gpc(gidy,gidz,idx);

				if(gidx <= (solver.n1+1))
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
			if((gidx > 0)&&(gidx <= (solver.n1-2)))
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

	if((gidz > 0)&&(gidz <= solver.n3))
	{
		if((gidy > 0)&&(gidy <= solver.n2))
		{
			if((gidx == (solver.n1-1)))
			{
				x(idx+1,idy,idz) = gpcs(idy,idz,0)*x(idx-1,idy,idz)
						+ gpcs(idy,idz,1)*x(idx,idy-1,idz)
						+ gpcs(idy,idz,2)*x(idx,idy+1,idz)
						//+ 0.0f*gpcs(idy,idz,3)
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
	if(gidx < solver.n1)
	{
		if(gidy <= solver.n2)
		{
			if(gidz <= solver.n3)
			{
				resin(gidx,gidy,gidz) = res(idx,idy,idz);

				if(gidx == solver.n1-1) xin(gidx+1,gidy,gidz) = x(idx+1,idy,idz);

			}
		}
	}

}

template<int itransp>
__global__
void atimes_kernelc(PoissonSolver solver, cudaMatrixf xin,cudaMatrixf resin)
{
	int idx = threadIdx.x;
	int idy = threadIdx.y;
	int idz = threadIdx.z;
	int bidx = blockIdx.x*ATIMES_DOMAIN_DIM;
	int bidy = blockIdx.y*ATIMES_DOMAIN_DIM;
	int bidz = blockIdx.z*ATIMES_DOMAIN_DIM;
	int gidx = bidx+idx;
	int gidy = bidy+idy;
	int gidz = bidz+idz;

	__shared__ sMatrixf apcs,bpcs,cpcs,dpcs,epcs,fpcs,gpcs;
	__shared__ sMatrixf x,res,expphi;

	setup_shared(apcs,bpcs,cpcs,dpcs,epcs,fpcs,gpcs,expphi,x,res);
/*
	// Load all the data into shared memory
	while((idx < (ndo.x*blockDim.x+2))&&(gidx <= (solver.n1+1)))
	{
		idy = threadIdx.y;
		gidy = bidy+idy;
		while((idy < (ndo.y*blockDim.y+2))&&(gidy <= (solver.n2+1)))
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



			while((idz < (blockDim.z+2))&&(gidz <= (solver.n3+1)))
			{
				expphi(idx,idy,idz) = exp(solver.phi(gidx+1,gidy,gidz));

				if(idx < 5) gpcs(idy,idz,idx) = solver.gpc(gidy,gidz,idx);

				if(gidx <= (solver.n1))
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
	*/

	// Load all the data into shared memory
	while((idz < (ATIMES_DOMAIN_DIM+2))&&(gidz <= (solver.n3+1)))
	{
		idy = threadIdx.y;
		gidy = bidy+idy;
		while((idy < (ATIMES_DOMAIN_DIM+2))&&(gidy <= (solver.n2+1)))
		{
			idx = threadIdx.x;
			gidx = bidx+idx;
			while((idx < (ATIMES_DOMAIN_DIM+2))&&(gidx <= (solver.n1+1)))
			{
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

				expphi(idx,idy,idz) = exp(solver.phi(gidx+1,gidy,gidz));

				if(idx < 5) gpcs(idy,idz,idx) = solver.gpc(gidy,gidz,idx);

				if(gidx <= (solver.n1))
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

				idx += blockDim.x;
				gidx = bidx+idx;
			}
			idy += blockDim.y;
			gidy = bidy+idy;
		}
		idz += blockDim.z;
		gidz = bidz+idz;
	}

	__syncthreads();

	// End of shared memory load

	// Reset thread ID's
	idz = threadIdx.z+1;
	gidz = bidz+idz;

	// Bulk loop
	while((idz < (ATIMES_DOMAIN_DIM+1))&&(gidz <= solver.n3))
	{
		idy = threadIdx.y+1;
		gidy = bidy+idy;
		while((idy < (ATIMES_DOMAIN_DIM+1))&&(gidy <= solver.n2))
		{
			idx = threadIdx.x+1;
			gidx = bidx+idx;
			while((idx < (ATIMES_DOMAIN_DIM+1))&&(gidx < solver.n1))
			{
				switch(itransp)
				{
				case 0:
					if((gidx > 0)&&(gidx <= (solver.n1-2)))
					{
						res(idx,idy,idz) = apcs(idx)*x(idx+1,idy,idz)
								+ bpcs(idx)*x(idx-1,idy,idz)
								+ cpcs(idx,idy)*x(idx,idy+1,idz)
								+ dpcs(idx,idy)*x(idx,idy-1,idz)
								+ epcs(idx,idy)*(x(idx,idy,idz+1)+x(idx,idy,idz-1))
								- (fpcs(idx,idy)+expphi(idx,idy,idz))*x(idx,idy,idz);
					}
					else 	if((gidx == (solver.n1-1)))
					{
						x(idx+1,idy,idz) = gpcs(idy,idz,0)*x(idx-1,idy,idz)
								+ gpcs(idy,idz,1)*x(idx,idy-1,idz)
								+ gpcs(idy,idz,2)*x(idx,idy+1,idz)
								//+ 0.0f*gpcs(idy,idz,3)
								+ gpcs(idy,idz,4)*x(idx,idy,idz);

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
					break;

				case 1:
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
					break;
				default:
					break;
				}

				idx += blockDim.x;
				gidx = bidx+idx;
			}
			idy += blockDim.y;
			gidy = bidy+idy;
		}
		idz += blockDim.z;
		gidz = bidz+idz;
	}

	// Reset thread ID's
	idz = threadIdx.z+1;
	gidz = bidz+idz;
	while((idz < (ATIMES_DOMAIN_DIM+1))&&(gidz <= solver.n3))
	{
		idy = threadIdx.y+1;
		gidy = bidy+idy;
		while((idy < (ATIMES_DOMAIN_DIM+1))&&(gidy <= solver.n2))
		{
			idx = threadIdx.x+1;
			gidx = bidx+idx;
			while((idx < (ATIMES_DOMAIN_DIM+1))&&(gidx < solver.n1))
			{

				resin(gidx,gidy,gidz) = res(idx,idy,idz);

				if(itransp == 0)
					if(gidx == solver.n1-1) xin(gidx+1,gidy,gidz) = x(idx+1,idy,idz);

				idx += blockDim.x;
				gidx = bidx+idx;
			}
			idy += blockDim.y;
			gidy = bidy+idy;
		}
		idz += blockDim.z;
		gidz = bidz+idz;
	}

}

template<int itransp>
__global__
void atimes_kernelc2(PoissonSolver solver, cudaMatrixf xin,cudaMatrixf resin)
{
	int thid = threadIdx.x;
	int idx;
	int idy;
	int idz;
	int bidx = blockIdx.x*ATIMES_DOMAIN_DIMx;
	int bidy = blockIdx.y*ATIMES_DOMAIN_DIMy;
	int bidz = blockIdx.z*ATIMES_DOMAIN_DIMz;
	int gidx;
	int gidy;
	int gidz;
	int3 bdim;


	bdim.x = min(ATIMES_DOMAIN_DIMx+2,(solver.n1+2-bidx));
	bdim.y = min(ATIMES_DOMAIN_DIMy+2,(solver.n2+2-bidy));
	bdim.z = min(ATIMES_DOMAIN_DIMz+2,(solver.n3+2-bidz));

	int nelements = bdim.x*bdim.y*bdim.z;

	bool dimcheck;



	__shared__ sMatrixf apcs,bpcs,cpcs,dpcs,epcs,fpcs,gpcs;
	__shared__ sMatrixf x,res,expphi;

	setup_shared(apcs,bpcs,cpcs,dpcs,epcs,fpcs,gpcs,expphi,x,res);


	// Load all the data into shared memory
	while(thid < (nelements))
	{
		// Calculate idx, idy, & idz from thid

		calc_dims(idx,idy,idz,thid,bdim);

		gidx = idx+bidx;
		gidy = idy+bidy;
		gidz = idz+bidz;

		// Check dimensions
		dimcheck = ((idx < (ATIMES_DOMAIN_DIMx+2))&&(gidx <= (solver.n1+1)));
		dimcheck = dimcheck&&((idy < (ATIMES_DOMAIN_DIMy+2))&&(gidy <= (solver.n2+1)));
		dimcheck = dimcheck&&((idz < (ATIMES_DOMAIN_DIMz+2))&&(gidz <= (solver.n3+1)));

		if(dimcheck)
		{
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

			expphi(idx,idy,idz) = exp(solver.phi(gidx+1,gidy,gidz));

			if(idx < 5) gpcs(idy,idz,idx) = solver.gpc(gidy,gidz,idx);

			if(gidx <= (solver.n1))
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
		}
//		else
//		{
//			if((blockIdx.x==7)&&(blockIdx.y==7)&&(blockIdx.z==7))
//				printf("thid = %i, %i = %i, %i, %i, bdim = %i, %i, %i\n",thid,idx,gidx,gidy,gidz,bdim.x,bdim.y,bdim.z);


//			break;
//		}

		thid += blockDim.x;
	}

	__syncthreads();

	// End of shared memory load

	// Reset thread ID's
	thid = threadIdx.x;
	bdim.x = min(ATIMES_DOMAIN_DIMx,(solver.n1-bidx));
	bdim.y = min(ATIMES_DOMAIN_DIMy,(solver.n2-bidy));
	bdim.z = min(ATIMES_DOMAIN_DIMz,(solver.n3-bidz));
	nelements = bdim.x*bdim.y*bdim.z;

	// Bulk loop
	while(thid < nelements)
	{
		// Calculate idx, idy, & idz from thid
		calc_dims(idx,idy,idz,thid,bdim);
		idx += 1;
		idy += 1;
		idz += 1;

		gidx = idx+bidx;
		gidy = idy+bidy;
		gidz = idz+bidz;


		// Check dimensions
		dimcheck = ((idx < (ATIMES_DOMAIN_DIMx+1))&&(gidx < (solver.n1)));
		dimcheck = dimcheck&&((idy < (ATIMES_DOMAIN_DIMy+1))&&(gidy <= (solver.n2)));
		dimcheck = dimcheck&&((idz < (ATIMES_DOMAIN_DIMz+1))&&(gidz <= (solver.n3)));

		if(dimcheck)
		{
			switch(itransp)
			{
			case 0:
				if((gidx > 0)&&(gidx <= (solver.n1-2)))
				{
					res(idx,idy,idz) = apcs(idx)*x(idx+1,idy,idz)
							+ bpcs(idx)*x(idx-1,idy,idz)
							+ cpcs(idx,idy)*x(idx,idy+1,idz)
							+ dpcs(idx,idy)*x(idx,idy-1,idz)
							+ epcs(idx,idy)*(x(idx,idy,idz+1)+x(idx,idy,idz-1))
							- (fpcs(idx,idy)+expphi(idx,idy,idz))*x(idx,idy,idz);
				}
				else 	if((gidx == (solver.n1-1)))
				{
					x(idx+1,idy,idz) = gpcs(idy,idz,0)*x(idx-1,idy,idz)
							+ gpcs(idy,idz,1)*x(idx,idy-1,idz)
							+ gpcs(idy,idz,2)*x(idx,idy+1,idz)
							//+ 0.0f*gpcs(idy,idz,3)
							+ gpcs(idy,idz,4)*x(idx,idy,idz);

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
				break;

			case 1:
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
				break;
			default:
				break;
			}
		}

		thid += blockDim.x;
	}

	__syncthreads();

	// Reset thread ID's
	thid = threadIdx.x;
#pragma unroll 3
	while(thid < nelements)
	{
		// Calculate idx, idy, & idz from thid
		calc_dims(idx,idy,idz,thid,bdim);
		idx += 1;
		idy += 1;
		idz += 1;

		gidx = idx+bidx;
		gidy = idy+bidy;
		gidz = idz+bidz;

		// Check dimensions
		dimcheck = ((idx < (ATIMES_DOMAIN_DIMx+1))&&(gidx < (solver.n1)));
		dimcheck = dimcheck&&((idy < (ATIMES_DOMAIN_DIMy+1))&&(gidy <= (solver.n2)));
		dimcheck = dimcheck&&((idz < (ATIMES_DOMAIN_DIMz+1))&&(gidz <= (solver.n3)));

		if(dimcheck)
		{

			resin(gidx,gidy,gidz) = res(idx,idy,idz);

			if(itransp == 0)
				if(gidx == solver.n1-1) xin(gidx+1,gidy,gidz) = x(idx+1,idy,idz);

		}

		thid += blockDim.x;
	}

}

__global__
void asolve_kernel(PoissonSolver solver,cudaMatrixf bin,cudaMatrixf zin)
{
	int idx = threadIdx.x+1;
	int idy = threadIdx.y+1;
	int idz = threadIdx.z+1;
	int bidx = blockIdx.x*blockDim.x;
	int bidy = blockIdx.y*blockDim.y;
	int bidz = blockIdx.z*blockDim.z;
	int gidx = bidx+idx;
	int gidy = bidy+idy;
	int gidz = bidz+idz;

	float result;

	while((gidz <= solver.n3))
	{
		gidy = bidy+idy;
		while((gidy <= solver.n2))
		{
			gidx = bidx+idx;
			while((gidx < solver.n1))
			{
				if(gidx < solver.n1-1)
				{
					result = bin(gidx,gidy,gidz)/(-1*solver.fpc(gidx+1,gidy)-exp(solver.phi(gidx+1,gidy,gidz)));

					zin(gidx,gidy,gidz) = result;

				}
				else if(gidx == solver.n1-1)
				{
					result = bin(gidx,gidy,gidz)/((-1*solver.fpc(gidx+1,gidy)-exp(solver.phi(gidx+1,gidy,gidz)))
							+ solver.apc(gidx+1)*solver.gpc(gidy,gidz,4));

					zin(gidx,gidy,gidz) = result;
				}
				gidx += blockDim.x*gridDim.x;
			}
			gidy += blockDim.y*gridDim.y;
		}
		gidz += blockDim.z*gridDim.z;
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
		while(gidx < solver.n1)
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

	if(solver.t_iter == 1)
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
		while(gidx < solver.n1)
		{

			result=bk*solver.p(gidx,gidy,gidz)+solver.z(gidx,gidy,gidz);

			solver.p(gidx,gidy,gidz) = result;
			result = bk*solver.pp(gidx,gidy,gidz)+solver.zz(gidx,gidy,gidz);

			solver.pp(gidx,gidy,gidz) = result;

			if(gidx == 1)
			{
				solver.p(gidx-1,gidy,gidz) = 0;
				solver.pp(gidx-1,gidy,gidz) = 0;
			}

			gidx += blockDim.x*gridDim.x;
		}
		gidy += blockDim.y*gridDim.y;
	}

}

template<int operation>
__global__
void Psolve_reduce_kernel(PoissonSolver solver)
{
	int idx = threadIdx.x+1;
	int idy = threadIdx.y+1;
	int idz = threadIdx.z+1;
	int gidx = blockDim.x*blockIdx.x+idx;
	int gidy = blockDim.y*blockIdx.y+idy;
	int gidz = blockDim.z*blockIdx.z+idz;

	int thid = gidx-1+blockDim.x*gridDim.x*(gidy-1+blockDim.y*gridDim.y*(gidz-1));

	float my_val = 0;


	while(gidy <= solver.n2)
	{
		gidx = blockDim.x*blockIdx.x+idx;
		while(gidx < solver.n1)
		{
			 switch(operation)
			 {
			 case 0:
				 my_val += solver.bknum_eval(gidx,gidy,gidz);
				// printf("my_val = %f\n",my_val);
				 break;
			 case 1:
				 my_val += solver.aknum_eval(gidx,gidy,gidz);
				 break;
			 case 2:
				 my_val = max(my_val,solver.delta_eval(gidx,gidy,gidz));
				 break;
			 default:
				 break;
			 }


			gidx += blockDim.x*gridDim.x;
		}
		gidy += blockDim.y*gridDim.y;
	}

	//if(operation == 0)
	//	printf("my_va(%i,%i,%i) = %f\n",gidx,gidy,gidz,my_val);

	solver.sum_array[thid] = my_val;

}

template<int operation>
__host__
void PoissonSolver::eval_sum(void)
{
	dim3 cudaBlockSize(32,4,1);
	dim3 cudaGridSize(1,1,n3);

	CUDA_SAFE_CALL(cudaMemset(sum_array,0,cudaBlockSize.x*cudaBlockSize.y*(n3+1)*sizeof(float)));

	float result = 0.0;


	CUDA_SAFE_KERNEL((Psolve_reduce_kernel<operation><<<cudaGridSize,cudaBlockSize>>>(*this)));



	thrust::device_ptr<float> reduce_ptr(sum_array);

	if(operation < 2)
	{
		// bknum and akden
		result = thrust::reduce(reduce_ptr,reduce_ptr+(n3+1)*cudaBlockSize.x*cudaBlockSize.y);
	}
	else
	{
		// delta is a maximum
		result = thrust::reduce(reduce_ptr,reduce_ptr+(n3+1)*cudaBlockSize.x*cudaBlockSize.y,(float) 0.0,thrust::maximum<float>());
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
		 deltamax = result;
		 break;
	 default:
		 break;
	 }

}

__host__
void PoissonSolver::setup_res(void)
{
	dim3 cudaBlockSize(32,8,1);
	dim3 cudaGridSize(1,1,n3);

	CUDA_SAFE_KERNEL((setup_res_kernel<<<cudaGridSize,cudaBlockSize>>>(*this)));
}

__host__
void PoissonSolver::pppp(void)
{
	dim3 cudaBlockSize(32,8,1);
	dim3 cudaGridSize(1,1,n3);

	CUDA_SAFE_KERNEL((pppp_kernel<<<cudaGridSize,cudaBlockSize>>>(*this)));
}

__host__
void PoissonSolver::asolve(int n1_in,int n2_in,int n3_in,cudaMatrixf bin, cudaMatrixf zin)
{
	dim3 cudaBlockSize(32,8,1);
	dim3 cudaGridSize(1,1,1);
	n1 = n1_in;
	n2 = n2_in;
	n3 = n3_in;

	//cudaGridSize.x = (n1+cudaBlockSize.x-1)/cudaBlockSize.x;
	//cudaGridSize.y = (n2+2*cudaBlockSize.y-1)/(2*cudaBlockSize.y);
	cudaGridSize.z = (n3+cudaBlockSize.z-1)/cudaBlockSize.z;

	CUDA_SAFE_KERNEL((asolve_kernel<<<cudaGridSize,cudaBlockSize>>>(*this,bin,zin)));

}

int atimes_shared = 0;

__host__
void PoissonSolver::atimes(int n1_in,int n2_in,int n3_in,cudaMatrixf xin, cudaMatrixf resin,const int itransp)
{
	dim3 cudaBlockSize(1,1,1);
	dim3 cudaGridSize(1,1,1);
	n1 = n1_in;
	n2 = n2_in;
	n3 = n3_in;

	if(atimes_shared)
	{
		cudaBlockSize.x = 512;
		cudaBlockSize.y = 1;
		cudaBlockSize.z = 1;

		cudaGridSize.x = (n1+ATIMES_DOMAIN_DIMx-2)/(ATIMES_DOMAIN_DIMx);
		cudaGridSize.y = (n2+ATIMES_DOMAIN_DIMy-1)/(ATIMES_DOMAIN_DIMy);
		cudaGridSize.z = (n3+ATIMES_DOMAIN_DIMz-1)/(ATIMES_DOMAIN_DIMz);

		switch(itransp)
		{
		case 0:
			CUDA_SAFE_KERNEL((atimes_kernelc2<0><<<cudaGridSize,cudaBlockSize>>>(*this,xin,resin)));
			break;
		case 1:
			CUDA_SAFE_KERNEL((atimes_kernelc2<1><<<cudaGridSize,cudaBlockSize>>>(*this,xin,resin)));
			break;
		default:
			break;
		}

		//atimes_shared = 0;
	}
	else
	{
		cudaBlockSize.x = 32;
		cudaBlockSize.y = 8;
		cudaBlockSize.z = 1;

		cudaGridSize.x = (n1+cudaBlockSize.x-1)/(cudaBlockSize.x);
		cudaGridSize.y = (n2+cudaBlockSize.y-1)/(cudaBlockSize.y);
		cudaGridSize.z = (n3+cudaBlockSize.z-1)/(cudaBlockSize.z);

		cudaGridSize.x = 1;
		cudaGridSize.y = 1;
		cudaGridSize.z = (n3+cudaBlockSize.z-1)/(cudaBlockSize.z);

		CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
		switch(itransp)
		{
		case 0:
			CUDA_SAFE_KERNEL((atimes_kernel<0><<<cudaGridSize,cudaBlockSize>>>(*this,xin,resin)));
			break;
		case 1:
			CUDA_SAFE_KERNEL((atimes_kernel<1><<<cudaGridSize,cudaBlockSize>>>(*this,xin,resin)));
			break;
		default:
			break;
		}
		CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
		//atimes_shared = 1;
	}


/*
	if(itransp == 0)
	{
		// No transpose case
		CUDA_SAFE_KERNEL((atimes_kernels<<<cudaGridSize,cudaBlockSize>>>(*this,xin,resin)));
	}
	else if(itransp == 1) // Transpose case
	{
		cudaBlockSize.x = 8;
		cudaBlockSize.y = 8;
		cudaBlockSize.z = 2;

		ndo.x = (ATIMES_DOMAIN_DIM+cudaBlockSize.x-1)/cudaBlockSize.x;
		ndo.y = (ATIMES_DOMAIN_DIM+cudaBlockSize.y-1)/cudaBlockSize.y;
		ndo.z = (ATIMES_DOMAIN_DIM+cudaBlockSize.z-1)/cudaBlockSize.z;

		cudaGridSize.x = (n1+ndo.x*cudaBlockSize.x-1)/(ndo.x*cudaBlockSize.x);
		cudaGridSize.y = (n2+ndo.y*cudaBlockSize.y-1)/(ndo.y*cudaBlockSize.y);
		cudaGridSize.z = (n3+ndo.z*cudaBlockSize.z-1)/(ndo.z*cudaBlockSize.z);

		CUDA_SAFE_KERNEL((atimes_transp_kernels<<<cudaGridSize,cudaBlockSize>>>(*this,xin,resin,ndo)));
	}
	*/

}

__host__
void PoissonSolver::cg3D(int n1_in,int n2_in,int n3_in,float tol,int &iter,int itmax,const int lbcg)
{
	n1 = n1_in;
	n2 = n2_in;
	n3 = n3_in;

	iter = 0;
	// Initialize the denominators
	bknum = 0;
	bkden = 0;
	akden = 0;
	deltamax = tol*1.5f;

	atimes(n1,n2,n3,x,res,0);

	setup_res();
	//printf("lbcg = %i\n",lbcg);
	if(lbcg == 0)
	{
		atimes(n1,n2,n3,res,resr,0);
	}

	asolve(n1,n2,n3,res,z);

	// Main Loop
	while(deltamax >= tol)
	{
		iter++;
		t_iter = iter;


		asolve(n1,n2,n3,resr,zz);
		bknum = 0.0;

		// evaluate bknum;
		eval_sum<0>();



		// do the p's
		pppp();

		bkden = bknum;

		atimes(n1,n2,n3,p,z,0);

		akden = 0.0;

		// evaluate akden
		eval_sum<1>();

		atimes(n1,n2,n3,pp,zz,lbcg);
		// set deltamax = 0 so that we can reevaluate it.
		deltamax = 0.0;

		// evaluate deltamax
		eval_sum<2>();

		if(iter >= itmax)
			break;

//		printf("bknum, akden, deltamax = %f, %f, %f\n",bknum,akden,deltamax);

		if(deltamax >= tol)
			asolve(n1,n2,n3,res,z);
		else
			break;


	}




}


extern "C" void cg3d_gpu_(long int* solverPtr,float* phi,int* lbcg,int* n1,int* n2,int* n3,
										    float* bin,float* xin,float* tol,float* gpc,int* iter,int* itmax)
{
	PoissonSolver* solver;

	solver = ((PoissonSolver*)(*solverPtr));

	solver -> phi.cudaMatrixcpy(phi,cudaMemcpyHostToDevice);
	solver -> gpc.cudaMatrixcpy(gpc,cudaMemcpyHostToDevice);

	solver -> x.cudaMatrixcpy(xin,cudaMemcpyHostToDevice);
	solver -> b.cudaMatrixcpy(bin,cudaMemcpyHostToDevice);

	bool tlbcg = 0;

	if((*lbcg) == 1)
	{
		tlbcg = 1;
	}

	cudaDeviceSynchronize();
	if(tlbcg)
	{
		solver -> cg3D(*n1,*n2,*n3,*tol,*iter,*itmax,1);
	}
	else
	{
		solver -> cg3D(*n1,*n2,*n3,*tol,*iter,*itmax,0);
	}

	cudaDeviceSynchronize();
	// copy results back to the cpu
	solver -> x.cudaMatrixcpy(xin,cudaMemcpyDeviceToHost);

}

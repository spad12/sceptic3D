/*
 * This software contains source code provided by NVIDIA Corporation.
 */
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

#include "cuda_texture_types.h"
#include "texture_fetch_functions.h"

#  define CUDA_SAFE_KERNEL(call) {                                         \
	call;																					\
	cudaDeviceSynchronize();														\
	cudaError err = cudaGetLastError();										\
    if ( cudaSuccess != err) {                                               \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
                exit(EXIT_FAILURE);                                                  \
    } }


//#define Texture_Solver

#define ATIMES_BLOCK_SIZE 256

extern "C" void cg3d_gpu_(long int* solverPtr,float* phi,int* lbcg,int* n1,int* n2,int* n3,
	    									float* bin,float* xin,float* tol,float* gpc,int* iter,int* itmax);

extern "C" void shielding3d_gpu_(long int* solverPtr,float* phi,float* rho,float* phiaxis,
														float* gpc,float* dt, int* lbcg,
														int* n1,int* n2,int* n3,int* nrused,int* iter);

extern "C" void start_timer_(uint* timer);

extern "C" void stop_timer_(float* time,uint* timer);

typedef float (*texFunctionPtr)(float,float);

class texMatrix
{
public:

	__host__ __device__
	texMatrix(){;}

	__host__
	void allocate(int nx_in, int ny_in, const char* texture_name_in);

	__host__
	void setup(float* src);


	__device__
	float operator()(int ix,int iy)
	{
		return fetchFunction(ix,iy);
	}

	__device__
	float operator()(int ix)
	{
		return fetchFunction(ix,0);
	}

	__host__
	void texMatrixFree(void)
	{
		const textureReference* texRefPtr;
		char* texrefstring = (char*)malloc(sizeof(char)*30);

		sprintf(texrefstring,"%s_t",texture_name);

		// Get the texture reference
		CUDA_SAFE_CALL(cudaGetTextureReference(&texRefPtr, texrefstring));
		CUDA_SAFE_CALL(cudaUnbindTexture(texRefPtr));

		cudaFree(cuArray);
	}

private:
	int nx,ny;
	texFunctionPtr fetchFunction;
	cudaArray* cuArray;
	const char* texture_name;


};


class PoissonSolver
{
public:
	int nrsize,nthsize,npsisize;
	int n1,n2,n3;
	int t_iter;

#ifdef Texture_Solver
	texMatrix apc,bpc,cpc,dpc,epc,fpc;
#else
	cudaMatrixf apc,bpc,cpc,dpc,epc,fpc;
#endif
	cudaMatrixf gpc;
	cudaMatrixf x;
	cudaMatrixf b;
	cudaMatrixf p,z,pp,zz,res,resr;
	cudaMatrixf phi,phiaxis;
	cudaMatrixf rho;

	float* sum_array;

	float bknum,bkden;
	float akden;
	float deltamax;

	__host__
	PoissonSolver(){;}

	__host__
	PoissonSolver(int n1_in,int n2_in,int n3_in){allocate(n1_in,n2_in,n3_in);}

	__host__
	void allocate(int n1_in, int n2_in, int n3_in);
	__host__
	void atimes(int n1,int n2,int n3,cudaMatrixf x_in,cudaMatrixf res_in,const int itrnsp);
	__host__
	void asolve(int n1,int n2,int n3,cudaMatrixf b_in, cudaMatrixf z_in);
	__host__
	void setup_res(void);
	__host__
	void pppp(void);

	template<int operation>
	__host__
	void eval_sum(void);
	__host__
	void cg3D(int n1_in,int n2_in,int n3_in,float tol,int &iter,int itmax,const int lbcg);
	__host__
	void shielding3D(float dt, int n1, int n2, int n3,int &iter,int nrused,int lbcg);


	__device__
	float bknum_eval(int gidx,int gidy,int gidz)
	{
		float tval = z(gidx,gidy,gidz)*resr(gidx,gidy,gidz);

		return tval;
	}
	__device__
	float aknum_eval(int gidx,int gidy,int gidz)
	{
		float tval = z(gidx,gidy,gidz)*pp(gidx,gidy,gidz);

		return tval;
	}
	__device__
	float delta_eval(int gidx,int gidy,int gidz)
	{
		float ak = bknum/akden;

		float delta = ak*p(gidx,gidy,gidz);

	//	printf("bknum, akden, ak = %f, %f, %f\n",bknum,akden,ak);

		x(gidx,gidy,gidz) += delta;
		res(gidx,gidy,gidz) = res(gidx,gidy,gidz) - ak*z(gidx,gidy,gidz);
		resr(gidx,gidy,gidz) = resr(gidx,gidy,gidz) - ak*zz(gidx,gidy,gidz);

		return abs(delta);
	}

	__host__
	void psfree(void);
};

















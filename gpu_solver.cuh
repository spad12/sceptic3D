#include <mathimf.h>
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




#define ATIMES_BLOCK_SIZE 256


class PoissonSolver
{
public:
	int nrsize,nthsize,npsisize;
	int n1,n2,n3;

	cudaMatrixf apc,bpc,cpc,dpc,epc,fpc,gpc;
	cudaMatrixf x;
	cudaMatrixf b;
	cudaMatrixf p,z,pp,zz,res,resr;
	cudaMatrixf phi;
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
	void atimes(int n1,int n2,int n3,cudaMatrixf x_in,cudaMatrixf res_in,int itrnsp);
	__host__
	void asolve(int n1,int n2,int n3,cudaMatrixf b_in, cudaMatrixf z_in);
	__host__
	void setup_res(void);
	__host__
	void pppp(void);
	__host__
	void eval_sum(const int operation);
	__host__
	void cg3D(int n1_in,int n2_in,int n3_in,float* bin,float* xin, float &tol,int &iter,int &itmax,int &lbcg);


	__device__
	float bknum_eval(float val_in,int gidx,int gidy,int gidz)
	{
		float tval = z(gidx,gidy,gidz)*resr(gidx,gidy,gidz);
		if(isnan(tval))
		{
			tval = 0.0;
		}


		return val_in+tval;
	}
	__device__
	float aknum_eval(float val_in,int gidx,int gidy,int gidz)
	{
		float tval = z(gidx,gidy,gidz)*pp(gidx,gidy,gidz);

		if(isnan(tval))
		{
			tval = 0.0;
		}


		return val_in+tval;
	}
	__device__
	float delta_eval(float val_in,int gidx,int gidy,int gidz)
	{
		float ak = bknum/akden;
		float delta = ak*p(gidx,gidy,gidz);
		if(isnan(delta))
		{
			delta = 0.0;
		}

		x(gidx,gidy,gidz) += delta;
		res(gidx,gidy,gidz) -= ak*z(gidx,gidy,gidz);
		resr(gidx,gidy,gidz) -= ak*zz(gidx,gidy,gidz);

		return max(abs(val_in),max(abs(delta),0.0));
	}

	__host__
	void psfree(void);
};

















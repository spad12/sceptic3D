/*
 * "This software contains source code provided by NVIDIA Corporation."
 */

#include "XPlist.cuh"

#ifdef INTEL_MKL
#include "mkl_trans.h"
#endif



int transposeblockSize = 512;
float* xplist_h_t ;

int icall_transpose = 0;

__global__
void transpose_gpu_kernel(XPlist device_data, cudaMatrixf host_data, int nptcls,int ndims,int direction)
{
	int idx = threadIdx.x;
	int gidx = blockIdx.x*blockDim.x+idx;

	int ogidx;
	if(gidx < nptcls)
	{
		for(int i=0;i<6;i++)
		{
			if(direction == 0)
			{
				(*device_data.get_float_ptr(i))[gidx] = host_data(i,gidx);

			}
			else if(direction == 1)
			{
				ogidx = device_data.particle_id[gidx];
				//ogidx = gidx;
				if(host_data(i,ogidx) != (*device_data.get_float_ptr(i))[gidx]){
					//printf("host_data(%i,%i = %i) = %f = %f\n",i,ogidx,gidx,host_data(i,ogidx),(*device_data.get_float_ptr(i))[gidx]);
				}
				host_data(i,gidx) = (*device_data.get_float_ptr(i))[gidx];
			}

		}
	}

}


extern "C" __host__
void xplist_transpose_(long int* xplist_d,
									   float* xplist_h,float* dt_prec,float* vzinit,int* ipf,
									   int* npartmax,int* ndims,int* direction,int* xpdata_only)
{

#ifdef time_run
	unsigned int timer = 0;
	cutCreateTimer(&timer);
	cutStartTimer(timer);
#endif

	XPlist particles = *(XPlist*)(*xplist_d);

	int nptcls = particles.nptcls;
	//printf("nptcls = %i \n",nptcls);
	int gridSize = (nptcls+transposeblockSize-1)/transposeblockSize;

	//cudaMatrixf host_data(6,nptcls);

	if(icall_transpose == 0)
	{
		xplist_h_t = (float*)malloc(nptcls*6*sizeof(float));
	//	CUDA_SAFE_CALL(cudaMallocHost((void**)&xplist_h_t,nptcls*6*sizeof(float)));
	}

	// Direction == 0 means a transfer to the GPU
	// Direction == 1 means a transfer to the CPU

	if((*direction == 0)&&(*xpdata_only == 0))
	{
		//host_data.cudaMatrixcpy(xplist_h,cudaMemcpyHostToDevice);
		CUDA_SAFE_CALL(cudaMemcpy(particles.dt_prec,dt_prec,nptcls*sizeof(float),cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(particles.vzinit,vzinit,nptcls*sizeof(float),cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(particles.ipf,ipf,nptcls*sizeof(int),cudaMemcpyHostToDevice));
	}

	if(*direction == 0)
	{
#ifdef INTEL_MKL
		MKL_Somatcopy('C','T',6,nptcls,1.0f,xplist_h,6,xplist_h_t,nptcls);
#else
		for(int i=0;i<nptcls;i++)
		{
			for(int j=0;j<6;j++)
			{
				xplist_h_t[i+nptcls*j] = xplist_h[j+i*6];
			}
		}

#endif


		for(int i=0;i<6;i++)
		{
			CUDA_SAFE_CALL(cudaMemcpy(*(particles.get_float_ptr(i)),xplist_h_t+nptcls*i,nptcls*sizeof(float),cudaMemcpyHostToDevice));
		}
	}
	else
	{
		for(int i=0;i<6;i++)
		{
			CUDA_SAFE_CALL(cudaMemcpy(xplist_h_t+nptcls*i,*(particles.get_float_ptr(i)),nptcls*sizeof(float),cudaMemcpyDeviceToHost));
		}
#ifdef INTEL_MKL
		MKL_Somatcopy('C','T',nptcls,6,1.0,xplist_h_t,nptcls,xplist_h,6);
#else

		for(int i=0;i<nptcls;i++)
		{
			for(int j=0;j<6;j++)
			{
				xplist_h[i*6+j] = xplist_h_t[i+nptcls*j];;
			}
		}

#endif

	}




	//CUDA_SAFE_KERNEL((transpose_gpu_kernel<<<gridSize,transposeblockSize>>>
					//			 (particles,host_data,nptcls,ndims[0],direction[0])));

	if((*direction == 1)&&(*xpdata_only == 0))
	{
		//host_data.cudaMatrixcpy(xplist_h,cudaMemcpyDeviceToHost);
		CUDA_SAFE_CALL(cudaMemcpy(dt_prec,particles.dt_prec,nptcls*sizeof(float),cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(vzinit,particles.vzinit,nptcls*sizeof(float),cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(ipf,particles.ipf,nptcls*sizeof(int),cudaMemcpyDeviceToHost));
	}

	icall_transpose++;

//	host_data.cudaMatrixFree();

	//CUDA_SAFE_CALL(cudaFreeHost(xplist_h_t));
	//free(xplist_h_t);

#ifdef time_run
	printf( "Particle List transpose took: %f (ms)\n\n", cutGetTimerValue( timer));
	cutStopTimer(timer);
	cutDeleteTimer( timer);
#endif

}

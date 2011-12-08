#include "XPlist.cuh"

int transposeblockSize = 512;


__global__
void transpose_gpu_kernel(XPlist device_data, float* host_data, int nptcls,int ndims,int direction)
{
	int idx = threadIdx.x;
	int gidx = blockIdx.x*blockDim.x+idx;

	if(gidx < nptcls)
	{
		for(int i=0;i<ndims;i++)
		{
			if(direction == 0)
			{
				(*device_data.get_float_ptr(i))[gidx] = host_data[ndims*gidx+i];
			}
			else if(direction == 1)
			{
				host_data[ndims*gidx+i] = (*device_data.get_float_ptr(i))[gidx];
			}
		}
	}

}


extern "C" __host__
void XPlist_transpose_(int* xplist_d,
									   float* xplist_h,float* dt_prec,float* vzinit,int* ipf,
									   int* ndims,int* direction)
{
	XPlist particles = *(XPlist*)(*xplist_d);

	int nptcls = particles.nptcls;
	int gridSize = (nptcls+transposeblockSize-1)/transposeblockSize;

	float* host_data;
	CUDA_SAFE_CALL(cudaMalloc((void**)&host_data,nptcls*ndims[0]*sizeof(float)));

	if(*direction == 0)
	{
		CUDA_SAFE_CALL(cudaMemcpy(host_data,xplist_h,nptcls*ndims[0]*sizeof(float),cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(particles.dt_prec,dt_prec,nptcls*sizeof(float),cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(particles.vzinit,vzinit,nptcls*sizeof(float),cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(particles.ipf,ipf,nptcls*sizeof(int),cudaMemcpyHostToDevice));
	}

	// Direction == 0 means a transfer to the GPU
	// Direction == 1 means a transfer to the CPU

	CUDA_SAFE_KERNEL((transpose_gpu_kernel<<<gridSize,transposeblockSize>>>
								 (particles,xplist_h,nptcls,ndims[0],direction[0])));

	if(*direction == 1)
	{
		CUDA_SAFE_CALL(cudaMemcpy(xplist_h,host_data,nptcls*ndims[0]*sizeof(float),cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(dt_prec,particles.dt_prec,nptcls*sizeof(float),cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(vzinit,particles.vzinit,nptcls*sizeof(float),cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(ipf,particles.ipf,nptcls*sizeof(int),cudaMemcpyDeviceToHost));
	}

	cudaFree(host_data);


}































#include <stdio.h>
#include <stdlib.h>
#include "cutil.h"
#include "cuda.h"
#include "stupid_sort.cuh"
// This is a very slow sort that is simple, so that I can actually profile the code without breaking the profiler

int compare(const void* a, const void* b)
{
	return (((int2*)a)->x < ((int2*)b)->x);
}

void stupid_sort(int* keys_d, int* values_d, int nelements)
{

	int* keys_h = (int*)malloc(nelements*sizeof(int));
	int* values_h = (int*)malloc(nelements*sizeof(int));

	int2* dict = (int2*)malloc(nelements*sizeof(int2));

	CUDA_SAFE_CALL(cudaMemcpy(keys_h,keys_d,nelements*sizeof(int),cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(values_h,values_d,nelements*sizeof(int),cudaMemcpyDeviceToHost));

	for(int i=0;i<nelements;i++)
	{
		dict[i].x = keys_h[i];
		dict[i].y = values_h[i];
	}

	qsort(dict,nelements,sizeof(int2),compare);

	for(int i=0;i<nelements;i++)
	{
		keys_h[i] = dict[i].x;
		values_h[i] = dict[i].y;
	}

	CUDA_SAFE_CALL(cudaMemcpy(keys_d,keys_h,nelements*sizeof(int),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(values_d,values_h,nelements*sizeof(int),cudaMemcpyHostToDevice));

	free(keys_h);
	free(values_h);
	free(dict);
}

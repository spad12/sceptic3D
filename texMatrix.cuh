#define __cplusplus
#define __CUDACC__

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
#include "cuda_texture_types.h"
#include "texture_types.h"
#include "texture_fetch_functions.h"
#include "cutil.h"

typedef float (*texFunctionPtr)(float,float,float);

class texMatrix
{
public:

	__host__ __device__
	texMatrix(){;}

	__host__
	void allocate(int nx,int ny,int nz,int textype);

	__host__
	void get_tex_string(char* texrefstring,char* texfetchstring);

	__host__
	void copy(float* src,enum cudaMemcpyKind kind);

	__device__
	float operator()(int ix,int iy=0,int iz=0);
private:
	texFunctionPtr fetchFunction;
	cudaArray* cuArray;
	int3 dims;
	int texture_ref_index;
	int textureType;
	bool is_bound;

};

__inline__ __device__
float texMatrix::operator()(int ix,int iy,int iz)
{
	return fetchFunction(ix,iy,iz);
}









































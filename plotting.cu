#include "XPlist.cuh"
#include <stdlib.h>

float* xgrid;
float* ygrid;

float* particles_x;
float* particles_y;

float* nptcls_block;

#define resolution 2048

__global__
void generate_mesh_image(Mesh_data mesh,float* xout, float* yout,
									char* rout, float* gout, float* bout,float rmax)
{
	int gidx = threadIdx.x+blockDim.x*blockIdx.x;
	int gidy = threadIdx.y+blockDim.y*blockIdx.y;


	float x = (2.0*rmax*gidx)/resolution - rmax;
	float y = (2.0*rmax*gidy)/resolution - rmax;

	float4 cellf;
	int4 icell;



}


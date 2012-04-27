#define __cplusplus
#define __CUDACC__
#include "cuda_texture_types.h"
#include "texture_types.h"
#include "texture_fetch_functions.h"

typedef float (*texFunctionPtr2)(float,float);
// Textures for Poisson Solve
texture<float,cudaTextureType1D,cudaReadModeElementType> apc_t;
texture<float,cudaTextureType1D,cudaReadModeElementType> bpc_t;
texture<float,cudaTextureType2D,cudaReadModeElementType> cpc_t;
texture<float,cudaTextureType2D,cudaReadModeElementType> dpc_t;
texture<float,cudaTextureType2D,cudaReadModeElementType> epc_t;
texture<float,cudaTextureType2D,cudaReadModeElementType> fpc_t;
texture<float,cudaTextureType2DLayered,cudaReadModeElementType> gpc_t;



static __inline__ __device__
float fetch_apc_t(float x,float y)
{
	return tex1D(apc_t,x);
}

static __inline__ __device__
float fetch_bpc_t(float x,float y)
{
	return tex1D(bpc_t,x);
}

static __inline__ __device__
float fetch_cpc_t(float x,float y)
{
	return tex2D(cpc_t,x,y);
}

static __inline__ __device__
float fetch_dpc_t(float x,float y)
{
	return tex2D(dpc_t,x,y);
}

static __inline__ __device__
float fetch_epc_t(float x,float y)
{
	return tex2D(epc_t,x,y);
}

static __inline__ __device__
float fetch_fpc_t(float x,float y)
{
	return tex2D(fpc_t,x,y);
}


__constant__ texFunctionPtr2 fetch_apc_t_ptr = 	&fetch_apc_t;
__constant__ texFunctionPtr2 fetch_bpc_t_ptr = 	&fetch_bpc_t;
__constant__ texFunctionPtr2 fetch_cpc_t_ptr = 	&fetch_cpc_t;
__constant__ texFunctionPtr2 fetch_dpc_t_ptr = 	&fetch_dpc_t;
__constant__ texFunctionPtr2 fetch_epc_t_ptr = 	&fetch_epc_t;
__constant__ texFunctionPtr2 fetch_fpc_t_ptr = 	&fetch_fpc_t;



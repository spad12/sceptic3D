#include "texMatrix.cuh"


// We need a dummy kernal to force ptxas to actually compile this
__global__
void dummy_kernel(void){;}


int next_tex1D = 0;
int next_tex2D = 0;
int next_tex1DLayered = 0;
int next_tex2DLayered = 0;
int next_tex3D = 0;

__inline__ __host__
int get_next_tex(int textureType)
{
	int result;
	switch(textureType)
	{
	case cudaTextureType1D:
		result = next_tex1D;
		next_tex1D++;
		break;
	case cudaTextureType1DLayered:
		result = next_tex1DLayered;
		next_tex1DLayered++;
		break;
	case cudaTextureType2D:
		result = next_tex2D;
		next_tex2D++;
		break;
	case cudaTextureType2DLayered:
		result = next_tex2DLayered;
		next_tex2DLayered++;
		break;
	case cudaTextureType3D:
		result = next_tex3D;
		next_tex3D++;
		break;
	default:
		break;
	}

	return result;
}
__host__
void texMatrix::get_tex_string(char* texrefstring,char* texfetchstring)
{
	switch(textureType)
	{
	case cudaTextureType1D:
		sprintf(texrefstring,"texref1D%i",texture_ref_index);
		sprintf(texfetchstring,"fetchtexref1DPtr%i",texture_ref_index);
		break;
	case cudaTextureType1DLayered:
		sprintf(texrefstring,"texref1DLayered%i",texture_ref_index);
		sprintf(texfetchstring,"fetchtexref1DLayeredPtr%i",texture_ref_index);
		break;
	case cudaTextureType2D:
		sprintf(texrefstring,"texref2D%i",texture_ref_index);
		sprintf(texfetchstring,"fetchtexref2DPtr%i",texture_ref_index);
		break;
	case cudaTextureType2DLayered:
		sprintf(texrefstring,"texref2DLayered%i",texture_ref_index);
		sprintf(texfetchstring,"fetchtexref2DLayeredPtr%i",texture_ref_index);
		break;
	case cudaTextureType3D:
		sprintf(texrefstring,"texref3D%i",texture_ref_index);
		sprintf(texfetchstring,"fetchtexref3DPtr%i",texture_ref_index);
		break;
	default:
		break;
	}

}


__host__
void texMatrix::allocate(int nx,int ny,int nz,int textype)
{
	cudaExtent extent;

	textureType = textype;

	dims.x = nx;
	dims.y = ny;
	dims.z = nz;

	char* texrefstring = (char*)malloc(sizeof(char)*30);
	char* texfetchstring = (char*)malloc(sizeof(char)*30);

	texture_ref_index = get_next_tex(textureType);

	get_tex_string(texrefstring,texfetchstring);

	char* symbol = texrefstring;

	//CUDA_SAFE_CALL(cudaGetSymbolAddress((void**)&fetchFunction,texfetchstring));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol((void*)&fetchFunction,texfetchstring,sizeof(texFunctionPtr)));
	//printf(" fill2D nx = %i, ny = %i \n", nx,ny);

	switch(textureType)
	{
	case cudaTextureType1D:
		extent = make_cudaExtent(nx,0,0);
		break;
	case cudaTextureType1DLayered:
		extent = make_cudaExtent(nx,0,ny);
		break;
	case cudaTextureType2D:
		extent = make_cudaExtent(nx,ny,0);
		break;
	case cudaTextureType2DLayered:
		extent = make_cudaExtent(nx,ny,nz);
		break;
	case cudaTextureType3D:
		extent = make_cudaExtent(nx,ny,nz);
		break;
	default:
		break;
	}

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();


	CUDA_SAFE_CALL(cudaMalloc3DArray(&cuArray,&desc,extent));

	is_bound = 0;

	free(texrefstring);
	free(texfetchstring);
}


__host__
void texMatrix::copy(float* src,enum cudaMemcpyKind kind)
{
	int nx = dims.x;
	int ny = dims.y;
	int nz = dims.z;

	cudaMemcpy3DParms params = {0};
	const textureReference* texRefPtr;
	cudaChannelFormatDesc channelDesc;
	CUDA_SAFE_CALL(cudaGetChannelDesc(&channelDesc, cuArray));

	char* texrefstring = (char*)malloc(sizeof(char)*30);
	char* texfetchstring = (char*)malloc(sizeof(char)*30);

	// Get the symbol for the texture reference
	get_tex_string(texrefstring,texfetchstring);

	char* symbol = texrefstring;

	// Get the texture reference
	CUDA_SAFE_CALL(cudaGetTextureReference(&texRefPtr, symbol));

	// Make sure that the texture isn't already bound
	if(is_bound)
		CUDA_SAFE_CALL(cudaUnbindTexture(texRefPtr));

	params.dstArray = cuArray;
	params.srcPtr.ptr = (void**)src;
	params.srcPtr.pitch = nx*sizeof(float);
	params.srcPtr.xsize = nx;
	params.kind = kind;

	switch(textureType)
	{
	case cudaTextureType1D:
		params.srcPtr.ysize = 1;
		params.extent = make_cudaExtent(nx,1,1);
		break;
	case cudaTextureType1DLayered:
		params.srcPtr.ysize = 1;
		params.extent = make_cudaExtent(nx,1,ny);
		break;
	case cudaTextureType2D:
		params.srcPtr.ysize = ny;
		params.extent = make_cudaExtent(nx,ny,1);
		break;
	case cudaTextureType2DLayered:
		params.srcPtr.ysize = ny;
		//params.extent = make_cudaExtent(nx,ny,nz);
		break;
	case cudaTextureType3D:
		params.srcPtr.ysize = ny;
		params.extent = make_cudaExtent(nx,ny,nz);
		break;
	default:
		break;
	}

	printf("ref # %i\n",texture_ref_index);

	// Do the copy
	CUDA_SAFE_CALL(cudaMemcpy3D(&params));

	cudaDeviceSynchronize();

	// Get the cudaArray's channel descriptor
	CUDA_SAFE_CALL(cudaGetChannelDesc(&channelDesc, cuArray));

	// Bind the cudaArray to the texture reference
	CUDA_SAFE_CALL(cudaBindTextureToArray(texRefPtr, cuArray, &channelDesc));
	cudaDeviceSynchronize();

	is_bound = 1;

	free(texrefstring);
	free(texfetchstring);

}







/*
 * "This software contains source code provided by NVIDIA Corporation."
 */
#include "XPlist.cuh"
#include "gpu_timing.cuh"
//__constant__ int ncells_per_binr_d;
//__constant__ int ncells_per_binth_d;
//__constant__ int ncells_per_binpsi_d;


#ifdef TEXTURE_PHI
texture<float,cudaTextureType3D,cudaReadModeElementType> phi_t;
#endif

#define SORT_BLOCK_SIZE 512
#define SORT_NPTCLS_PER_THREAD 4
#define CHARGE_ASSIGN_BLOCK_SIZE 512
#define PADVANCE_BLOCK_SIZE 128
#define PADVNC_NPTCLS_PER_THREAD 16

//#define time_run

// Use the stupid sort so that we can profile the code
//#define STUPID_SORT

#ifdef STUPID_SORT
#include "stupid_sort.cuh"
#endif




double sort_timer = 0;
double chargetomesh_timer = 0;
double padvnc_timer = 0;

int nptclstotal;
int nsorts = 0;
int nchargeassign = 0;
int npadvnc = 0;

int current_step = 0;


extern "C" void start_timer_(uint* timer)
{
	cutCreateTimer(timer);
	cutStartTimer(*timer);
}

extern "C" void stop_timer_(float* time,uint* timer)
{
	cutStopTimer(*timer);
	*time += cutGetTimerValue(*timer);
	cutDeleteTimer(*timer);
}


extern "C" void init_times_(uint* timer)
{
	cutCreateTimer(timer);
	cutStartTimer(*timer);

}

extern "C" void get_times_(uint* timer)
{
	cutStopTimer(*timer);

//	printf( "\nAverage Sort Time is: %e (ms)\n\n", (sort_timer/nsorts)/nptclstotal);
//	printf( "\nAverage Charge Assign Time is: %e (ms)\n\n", (chargetomesh_timer/nchargeassign)/nptclstotal);
//	printf( "\nAverage Padvnc Time is: %e (ms)\n\n", (padvnc_timer/npadvnc)/nptclstotal);
//	printf( "\nElapsed Time is: %f (ms)\n\n", cutGetTimerValue(*timer));


	cutDeleteTimer(*timer);
}




__host__
void XPlist::allocate(int nptcls_in)
{
	nptcls = nptcls_in;
	for(int i=0;i<nfloats_XPlist;i++)
	{
		CUDA_SAFE_CALL(cudaMalloc((void**)get_float_ptr(i),nptcls*sizeof(float)));
		CUDA_SAFE_CALL(cudaMemset(*get_float_ptr(i),0,nptcls*sizeof(float)));
	}

	for(int i=0;i<nints_XPlist;i++)
	{
		CUDA_SAFE_CALL(cudaMalloc((void**)get_int_ptr(i),nptcls*sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(*get_int_ptr(i),0,nptcls*sizeof(float)));
	}

	CUDA_SAFE_CALL(cudaMalloc((void**)&binid,nptcls*sizeof(ushort)));
	CUDA_SAFE_CALL(cudaMemset(binid,0,nptcls*sizeof(ushort)));
}

__global__
void reorder_particle_data(float* odata, float* idata,int* index_array,int nptcls)
{
	int idx = threadIdx.x;
	int gidx = idx + blockIdx.x*blockDim.x;



	while(gidx < nptcls)
	{
		int ogidx = index_array[gidx];
		//printf("particle %i now in slot %i\n",ogidx,gidx);
		odata[gidx] = idata[ogidx];

		gidx += blockDim.x*gridDim.x;
	}
}
__global__
void write_xpindex_array(int* index_array,int nptcls)
{
	int idx = threadIdx.x;
	int gidx = blockIdx.x*blockDim.x+idx;

	while(gidx < nptcls)
	{
		index_array[gidx] = gidx;
		gidx += blockDim.x*gridDim.x;
	}
}

__global__
void find_bin_boundaries(XPlist particles,Particlebin* bins)
{
	int idx = threadIdx.x;
	int gidx = idx+blockIdx.x*blockDim.x;

	int nptcls = particles.nptcls;

	uint binindex;
	uint binindex_left;
	uint binindex_right;

	while(gidx < nptcls)
	{
		if(gidx == 0)
		{
			binindex = particles.binid[gidx];
			bins[binindex].ifirstp = gidx;
			bins[binindex].binid = binindex;
		}
		else if(gidx == nptcls-1)
		{
			binindex = particles.binid[gidx];
			bins[binindex].ilastp = gidx;
			bins[binindex].binid = binindex;
		}
		else
		{
			binindex = particles.binid[gidx];
			binindex_left = particles.binid[max(gidx-1,0)];
			binindex_right = particles.binid[min((gidx+1),(nptcls-1))];

			if(binindex_left != binindex)
			{
				bins[binindex].ifirstp = gidx;
				bins[binindex_left].ilastp = gidx-1;
				bins[binindex].binid = binindex;
			}

			if(binindex_right != binindex)
			{
				bins[binindex].ilastp = gidx;
				bins[binindex_right].ifirstp = gidx+1;
				bins[binindex].binid = binindex;
			}

		}




		gidx += blockDim.x*gridDim.x;
	}
}

__host__
void XPlist::sort(Particlebin* bins)
{
	int cudaGridSize;
	int cudaBlockSize = SORT_BLOCK_SIZE;

	cudaGridSize = (nptcls+SORT_BLOCK_SIZE*SORT_NPTCLS_PER_THREAD-1)/
			(SORT_BLOCK_SIZE*SORT_NPTCLS_PER_THREAD);

	//CUDA_SAFE_CALL(cudaMemset(particle_id,0,nptcls*sizeof(int)));

	// Populate the particle index array
	CUDA_SAFE_KERNEL((write_xpindex_array<<<cudaGridSize,cudaBlockSize>>>
								 (particle_id,nptcls)));
#ifdef PROFILE_TIMERS
	int thrust_sort_timer = get_timer_int("thrust_sort");
	g_timers[thrust_sort_timer].start_timer();
#endif

#ifndef STUPID_SORT
	// wrap raw device pointers with a device_ptr
	thrust::device_ptr<ushort> thrust_keys(binid);
	thrust::device_ptr<int> thrust_values(particle_id);

	// Sort the data
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	thrust::sort_by_key(thrust_keys,thrust_keys+nptcls,thrust_values);
	cudaDeviceSynchronize();

	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
#else
	stupid_sort(binid,particle_id,nptcls);
#endif

#ifdef PROFILE_TIMERS
	g_timers[thrust_sort_timer].stop_timer();
	int reorder_pdata = get_timer_int("reorder_particle_list");
	g_timers[reorder_pdata].start_timer();
#endif


	// Reorder the particle data
	for(int i=0;i<8;i++)
	{
		float* idata = *(get_float_ptr(i));
		float* odata = buffer;
		CUDA_SAFE_KERNEL((reorder_particle_data<<<cudaGridSize,cudaBlockSize>>>
									 (odata,idata,particle_id,nptcls)));
		cudaDeviceSynchronize();
		*(get_float_ptr(i)) = odata;
		buffer = idata;

	}
/*
	for(int i=0;i<1;i++)
	{
		float* idata = (float*)(*(get_int_ptr(i)));
		float* odata = buffer;

		CUDA_SAFE_KERNEL((reorder_particle_data<<<cudaGridSize,cudaBlockSize>>>
									 (odata,idata,particle_id,nptcls)));
		cudaDeviceSynchronize();
		*(get_int_ptr(i)) = (int*)odata;
		buffer = idata;
	}
	*/
#ifdef PROFILE_TIMERS
	g_timers[reorder_pdata].stop_timer();
	int find_bin_boundaries_timer = get_timer_int("find_bin_boundaries");
	g_timers[find_bin_boundaries_timer].start_timer();
#endif

	// Find the cell-bin boundaries in the particle list
	CUDA_SAFE_KERNEL((find_bin_boundaries<<<cudaGridSize,cudaBlockSize>>>
								 (*this,bins)));

	cudaDeviceSynchronize();

#ifdef PROFILE_TIMERS
	g_timers[find_bin_boundaries_timer].stop_timer();
#endif

}

__global__
void find_cell_index_kernel(XPlist particles,Mesh_data mesh,int3 ncells)
{
	int idx = threadIdx.x;
	int gidx;
	int block_start = SORT_NPTCLS_PER_THREAD*blockIdx.x*blockDim.x;

	while(block_start < min(SORT_NPTCLS_PER_THREAD*blockDim.x*(blockIdx.x+1),particles.nptcls))
	{
		gidx = block_start+idx;
		if(gidx < particles.nptcls)
		{
			particles.calc_binid(&mesh,ncells,gidx);

		//	if(gidx >= 8388000)
			//	printf("particle %i is in bin %i\n",gidx,particles.binid[gidx]);
		}
		block_start += blockDim.x;
	}
}

extern "C" __host__
void xplist_sort_(long int* XP_ptr,long int* mesh_ptr,int* istep)
{
#ifdef time_run
	unsigned int timer = 0;
	cutCreateTimer(&timer);
	cutStartTimer(timer);
#endif


	XPlist* particles = (XPlist*)(*XP_ptr);
	Mesh_data mesh_d = *(Mesh_data*)(*mesh_ptr);
	//printf("Mesh parameters = %i, %i, %i\n",mesh_d.nr,mesh_d.nth,mesh_d.npsi);

	nptclstotal = particles->nptcls;

	int cudaBlockSize = SORT_BLOCK_SIZE;
	int cudaGridSize = (particles->nptcls+cudaBlockSize*SORT_NPTCLS_PER_THREAD-1)/(SORT_NPTCLS_PER_THREAD*cudaBlockSize);
	CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	cudaDeviceSynchronize();

#ifdef PROFILE_TIMERS
	int find_cell_index_timer = get_timer_int("find_cell_index_kernel");
	g_timers[find_cell_index_timer].start_timer();
#endif
	CUDA_SAFE_KERNEL((find_cell_index_kernel<<<cudaGridSize,cudaBlockSize>>>
								 (*particles,mesh_d,ncells_per_bin_g)));
#ifdef PROFILE_TIMERS
	g_timers[find_cell_index_timer].stop_timer();
#endif

	particles->sort(mesh_d.bins);

	*XP_ptr = (long int)(particles);
	cudaDeviceSynchronize();

#ifdef time_run
	cutStopTimer(timer);
	sort_timer += cutGetTimerValue( timer);
	nsorts++;
	cutDeleteTimer( timer);
#endif



	//if(*istep == 10)
	//{
	//	plot_particle_bins(mesh_d,ncells_per_bin_g);
	//}
}


__inline__ __device__
void write_to_nodes(float* data_out,float data_in,float4 cellf,int3 ncells)
{
	float* my_node;
	int bindimr = ncells.x + 1;
	int bindimth = ncells.y + 1;
	float weighted_data;


	for(int i=0;i<2;i++)
	{
		for(int j=0;j<2;j++)
		{
			for(int k=0;k<2;k++)
			{
				my_node = data_out+i+bindimr*(j+bindimth*k);
				weighted_data = (((1.0f-i)+(2*i-1)*cellf.x)*
						((1.0f-j)+(2*j-1)*cellf.y)*
						((1.0f-k)+(2*k-1)*cellf.z))*
						data_in;

				atomicAdd(my_node,weighted_data);


			}
		}
	}
}


__device__ int bin_census = 0;
int* bin_census_ptr = &bin_census;


__global__
void chargetomesh_kernel(XPlist particles,Mesh_data mesh,cudaMatrixf data_out,int3 ncells)
{
	int idx = threadIdx.x;
	int gidx = idx;
	int block_start;
	int bindimr = ncells.x + 1;
	int bindimth = ncells.y + 1;
	int bindimpsi = ncells.z + 1;
	Particlebin my_bin = mesh.bins[blockIdx.x];
	uint3 binindex;
	int4 my_cell;
	float4 cellfractions;
	float* my_data_out;
	float zetap;


	__shared__ float sdata[MAX_SMEM_PER_C2MESH];

	while(idx < bindimr*bindimth*bindimpsi)
	{
		sdata[idx] = 0;

		idx += CHARGE_ASSIGN_BLOCK_SIZE;
	}

	 idx = threadIdx.x;


	binindex = my_bin.get_bin_position(ncells);
/*
	if(idx == 0)
	{
	//	atomicAdd(&bin_census,my_bin.ilastp-my_bin.ifirstp+1);
	//	printf("Total number of particles = %i, %i bins contribution = %i\n",bin_census,blockIdx.x,my_bin.ilastp-my_bin.ifirstp+1);
		//printf("bin %i is at %i, %i, %i\n",blockIdx.x,binindex.x,binindex.y,binindex.z);
	}
*/
	__syncthreads();

	//my_cell.w = 0;

	block_start = my_bin.ifirstp;

	while(block_start <= my_bin.ilastp)
	{
		gidx = block_start+idx;

		if((gidx <= my_bin.ilastp)&&(gidx < particles.nptcls))
		{
			mesh.ptomesh<0>(particles.px[gidx],particles.py[gidx],particles.pz[gidx],&my_cell,&cellfractions,zetap);

			my_data_out = sdata + (my_cell.x-(int)(binindex.x)-1) + bindimr*(my_cell.y-binindex.y-1+bindimth*(my_cell.z-binindex.z-1));
			//my_cell.x -= 1;
			//my_cell.y -= 1;
			//my_cell.z -= 1;
/*
			rf = cellfractions.x;
			thf = cellfractions.y;
			pf = cellfractions.z;

			my_data_indexes[0] = my_cell.x+bindim*(my_cell.y+bindim*my_cell.z);
			my_data_indexes[1] = my_cell.x+1+bindim*(my_cell.y+bindim*my_cell.z);
			my_data_indexes[2] = my_cell.x+bindim*(my_cell.y+1+bindim*my_cell.z);
			my_data_indexes[3] = my_cell.x+1+bindim*(my_cell.y+1+bindim*my_cell.z);
			my_data_indexes[4] = my_cell.x+bindim*(my_cell.y+bindim*(my_cell.z+1));
			my_data_indexes[5] = my_cell.x+1+bindim*(my_cell.y+bindim*(my_cell.z+1));
			my_data_indexes[6] = my_cell.x+bindim*(my_cell.y+1+bindim*(my_cell.z+1));
			my_data_indexes[7] = my_cell.x+1+bindim*(my_cell.y+1+bindim*(my_cell.z+1));

			my_data[0] = (1.0-rf)*(1.0-thf)*(1.0-pf);
			my_data[1] = (rf)*(1.0-thf)*(1.0-pf);
			my_data[2] = (1.0-rf)*(thf)*(1.0-pf);
			my_data[3] = (rf)*(thf)*(1.0-pf);
			my_data[4] = (1.0-rf)*(1.0-thf)*(pf);
			my_data[5] = (rf)*(1.0-thf)*(pf);
			my_data[6] = (1.0-rf)*(thf)*(pf);
			my_data[7] = (rf)*(thf)*(pf);

			atomicAdd(&(data_out(my_cell.x,my_cell.y,my_cell.z)),my_data[0]);
			atomicAdd(&(data_out(my_cell.x+1,my_cell.y,my_cell.z)),my_data[1]);
			atomicAdd(&(data_out(my_cell.x,my_cell.y+1,my_cell.z)),my_data[2]);
			atomicAdd(&(data_out(my_cell.x+1,my_cell.y+1,my_cell.z)),my_data[3]);
			atomicAdd(&(data_out(my_cell.x,my_cell.y,my_cell.z+1)),my_data[4]);
			atomicAdd(&(data_out(my_cell.x+1,my_cell.y,my_cell.z+1)),my_data[5]);
			atomicAdd(&(data_out(my_cell.x,my_cell.y+1,my_cell.z+1)),my_data[6]);
			atomicAdd(&(data_out(my_cell.x+1,my_cell.y+1,my_cell.z+1)),my_data[7]);

			for(int i=0;i<8;i++)
			{
				//atomicAdd(sdata+my_data_indexes[i],my_data[i]);
			}
*/



			write_to_nodes(my_data_out,1.0f,cellfractions,ncells);
/*
			int3 lcell;
			lcell.x = my_cell.x-1;
			lcell.y = my_cell.y-1;
			lcell.z = my_cell.z-1;
			lcell.x -= (int)(binindex.x);
			lcell.y -= (int)(binindex.y);
			lcell.z -= (int)(binindex.z);

			if(((lcell.x) < ncells.x)&&((lcell.y) < ncells.y)&&((lcell.z) < ncells.z)&&
				((lcell.x) >= 0)&&((lcell.y) >= 0)&&((lcell.z) >= 0)&&
				(cellfractions.x >= 0.0f)&&(cellfractions.y >= 0.0f)&&
				(cellfractions.z >= 0.0f)&&
				(cellfractions.x <= 1.0f)&&(cellfractions.y <= 1.0f)&&
				(cellfractions.z <= 1.0f))
			{
				write_to_nodes(my_data_out,1.0f,cellfractions,ncells);
			}
			else
			{
				printf("Error particle %i in cell (%i, %i, %i) at position %f, %f, %f with cellf %f, %f, %f\n",gidx,my_cell.x,
							my_cell.y,my_cell.z,particles.px[gidx],particles.py[gidx],particles.pz[gidx],cellfractions.x,cellfractions.y,cellfractions.z);
			}
*/
		}

		block_start += CHARGE_ASSIGN_BLOCK_SIZE;
//		__syncthreads();

	}

	__syncthreads();

	while(idx < bindimr*bindimth*bindimpsi)
	{
		my_cell.z = idx/(bindimr*bindimth);
		my_cell.y = idx/bindimr - my_cell.z*bindimth;
		my_cell.x = idx - bindimr*(my_cell.y+bindimth*my_cell.z);

		my_cell.x += binindex.x;
		my_cell.y += binindex.y;
		my_cell.z += binindex.z;

		if(my_cell.z >= mesh.npsi)
		{
			my_cell.z = 0;
		}

	//	if((blockIdx.x) == 0)
		//	printf("thread %i contributing to node %i, %i, %i \n",idx,my_cell.x,my_cell.y,my_cell.z);

		//if((my_cell.x < mesh.nr)&&(my_cell.y < mesh.nth)&&(my_cell.z < mesh.npsi))
		atomicAdd(&(data_out(my_cell.x,my_cell.y,my_cell.z)),sdata[idx]);


		idx += CHARGE_ASSIGN_BLOCK_SIZE;
	}


}

extern "C" void gpu_chargeassign_(long int* XP_ptr,long int* mesh_ptr,float* psum)
{
#ifdef time_run
	unsigned int timer = 0;
	cutCreateTimer(&timer);
	cutStartTimer(timer);
#endif
	XPlist* particles = (XPlist*)(*XP_ptr);
	Mesh_data mesh_d = *(Mesh_data*)(*mesh_ptr);

	int cudaBlockSize = CHARGE_ASSIGN_BLOCK_SIZE;
	int cudaGridSize = mesh_d.nbins;

	//printf("nbins = %i\n",mesh_d.nbins);

	//CUDA_SAFE_CALL(cudaMemset(bin_census_ptr,0,sizeof(int)));

	mesh_d.psum.cudaMatrixSet(0);
	CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
#ifdef PROFILE_TIMERS
	int c2mesh_timer = get_timer_int("chargetomesh_kernel");
	g_timers[c2mesh_timer].start_timer();
#endif
	CUDA_SAFE_KERNEL((chargetomesh_kernel<<<cudaGridSize,cudaBlockSize>>>
								 (*particles,mesh_d,mesh_d.psum,ncells_per_bin_g)));
#ifdef PROFILE_TIMERS
	g_timers[c2mesh_timer].stop_timer();
#endif

	mesh_d.psum.cudaMatrixcpy(psum,cudaMemcpyDeviceToHost);
#ifdef time_run
	cutStopTimer(timer);
	chargetomesh_timer += cutGetTimerValue( timer);
	nchargeassign++;
	cutDeleteTimer( timer);
#endif

}

__inline__ __device__
float	Mesh_data::get_phi(const int& gidx,const int& gidy, const int& gidz)
const
{
#ifdef TEXTURE_PHI
	 return tex3D(phi_t,gidx,gidy,gidz);
#else
	 return phi(gidx,gidy,gidz);
#endif

}

__host__
void Mesh_data::phi_copy(float* src)
{
	#ifdef TEXTURE_PHI
	cudaMemcpy3DParms params = {0};
	cudaChannelFormatDesc channelDesc;

	CUDA_SAFE_CALL(cudaUnbindTexture(&phi_t));


	// Setup the copy params
	params.dstArray = phi;
	params.srcPtr.ptr = (void**)src;
	params.srcPtr.pitch = (nrfull+1)*sizeof(float);
	params.srcPtr.xsize = nrfull+1;
	params.kind = cudaMemcpyHostToDevice;

	params.srcPtr.ysize = nthfull+1;
	params.extent = make_cudaExtent( nrfull+1,nthfull+1,npsifull+1);
	// Do the copy
	CUDA_SAFE_CALL(cudaMemcpy3D(&params));

	CUDA_SAFE_CALL(cudaGetChannelDesc(&channelDesc, phi));

	// Bind the cudaArray to the texture reference
	CUDA_SAFE_CALL(cudaBindTextureToArray(&phi_t, phi, &channelDesc));



	#else
	phi.cudaMatrixcpy(src,cudaMemcpyHostToDevice);

	#endif
}

__global__
void xplist_advance_kernel(XPlist particles,const Mesh_data mesh,XPdiags diags,float dtin)
{
	int idx = threadIdx.x;
	int gidx;
	int block_start = blockIdx.x*blockDim.x*PADVNC_NPTCLS_PER_THREAD;

	// Time step stuff
	__shared__ float3 ptemp[PADVANCE_BLOCK_SIZE];
	__shared__ float3 vtemp[PADVANCE_BLOCK_SIZE];
	float rt,vr2;
	int didileave;

	//float dtl = dtin;

	__shared__ float3 momout;
	__shared__ float4 momprobe;
	__shared__ int ninner;

	if(idx == 0)
	{
		momout.x = 0;
		momout.y = 0;
		momout.z = 0;
		momprobe.x = 0;
		momprobe.y = 0;
		momprobe.z = 0;
		momprobe.w = 0;
		ninner = 0;
	}
	__syncthreads();

	while(block_start < (blockIdx.x+1)*blockDim.x*PADVNC_NPTCLS_PER_THREAD)
	{
		gidx = idx+block_start;
		if(gidx < particles.nptcls)
		{
			// Advance the particles
			particles.move(&mesh,ptemp[idx],vtemp[idx],dtin,gidx);

			// Now we need to see if we left the computational domain
			didileave = mesh.boundary_intersection(particles.px[gidx],particles.py[gidx],particles.pz[gidx],
															ptemp[idx].x,ptemp[idx].y,ptemp[idx].z);

			rt = ptemp[idx].x*ptemp[idx].x+ptemp[idx].y*ptemp[idx].y+ptemp[idx].z*ptemp[idx].z;
			vr2 = vtemp[idx].x*vtemp[idx].x+vtemp[idx].y*vtemp[idx].y+vtemp[idx].z*vtemp[idx].z;

			if(didileave)
			{
				if(rt < mesh.rmesh(mesh.nr)*mesh.rmesh(mesh.nr))
				{
					//dtl = sqrt((pow(particles.px[gidx]-ptemp.x,2)+pow(particles.py[gidx]-ptemp.y,2)+pow(particles.pz[gidx]-ptemp.z,2))/vr2);

					mesh.probe_diags(ptemp[idx],vtemp[idx]);

					atomicAdd(&momprobe.x,vtemp[idx].x);
					atomicAdd(&momprobe.y,vtemp[idx].y);
					atomicAdd(&momprobe.z,vtemp[idx].z);
					atomicAdd(&momprobe.w,0.5f*vr2);
					atomicAdd(&ninner,1);

				}
				else
				{
					atomicAdd(&momout.x,-vtemp[idx].x);
					atomicAdd(&momout.y,-vtemp[idx].y);
					atomicAdd(&momout.z,-vtemp[idx].z);

				}
			}


			// For now we'll just use half of the original time step as the time step for new particles
			particles.dt_prec[gidx] = dtin*(1-didileave);
			particles.px[gidx] = ptemp[idx].x;
			particles.py[gidx] = ptemp[idx].y;
			particles.pz[gidx] = ptemp[idx].z;
			particles.vx[gidx] = vtemp[idx].x;
			particles.vy[gidx] = vtemp[idx].y;
			particles.vz[gidx] = vtemp[idx].z;
			particles.didileave[gidx] = didileave;

			particles.particle_id[gidx] = gidx;


		}
		block_start += blockDim.x;
	}

	__syncthreads();

	if(idx == 0)
	{
		atomicAdd(&diags.momout->x,momout.x);
		atomicAdd(&diags.momout->y,momout.y);
		atomicAdd(&diags.momout->z,momout.z);

		atomicAdd(&diags.momprobe->x,momprobe.x);
		atomicAdd(&diags.momprobe->y,momprobe.y);
		atomicAdd(&diags.momprobe->z,momprobe.z);
		atomicAdd(&diags.momprobe->w,momprobe.w);

		atomicAdd(diags.ninner,ninner);
	}








}

template<typename T>
__global__
void scan_condense(T* data_out,T* data_in,int* scan_data,int* condition,int n_elements)
{
	// Copy data from data_in to a new location in data_out if condition is true
	int idx = threadIdx.x;
	int gidx = blockIdx.x*blockDim.x+idx;



	while(gidx < n_elements)
	{
		int oidxm = 0;
		int oidx = scan_data[gidx];

		if(gidx > 0)
			oidxm = scan_data[gidx-1];

		if(oidx != oidxm)
		{

			data_out[oidx-1] = data_in[gidx];
		}
		gidx += blockDim.x*gridDim.x;
	}
}

template<typename T>
__global__
void scan_broadcast(T* data_out,T* data_in,int* parent_ids,int n_elements)
{
	// Copy data from data_in to a new location in data_out if condition is true
	int idx = threadIdx.x;
	int gidx = blockIdx.x*blockDim.x+idx;

	while(gidx < n_elements)
	{

		int oidx = parent_ids[gidx];
		data_out[oidx] = data_in[gidx];

		gidx += blockDim.x*gridDim.x;
	}
}



__host__
void XPlist::advance(Mesh_data mesh,XPdiags diags,float* reinjlist_h,float dt,int &reinject_counter)
{

	int cudaBlockSize = PADVANCE_BLOCK_SIZE;
	int cudaGridSize = (nptcls+cudaBlockSize*PADVNC_NPTCLS_PER_THREAD-1)/(cudaBlockSize*PADVNC_NPTCLS_PER_THREAD);
	int nptcls_left;

	// Need to allocate space for the scan
	int* didileave_scan;
	//CUDA_SAFE_CALL(cudaMalloc((void**)&didileave_scan,nptcls*sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(didileave,0,nptcls*sizeof(int)));


	// Move the particles
	//printf("Moving the particles on the gpu\n");
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	cudaDeviceSynchronize();
#ifdef PROFILE_TIMERS
	int my_timer = get_timer_int("xplist_advance_kernel");
	g_timers[my_timer].start_timer();
#endif
	CUDA_SAFE_KERNEL((xplist_advance_kernel<<<cudaGridSize,cudaBlockSize>>>
								 (*this,mesh,diags,dt)));
#ifdef PROFILE_TIMERS
	g_timers[my_timer].stop_timer();
#endif
	// Copy the didileave array so that we can do a scan without messing up our earlier results
	//CUDA_SAFE_CALL(cudaMemcpy(didileave_scan,didileave,nptcls*sizeof(int),cudaMemcpyDeviceToDevice));

	// Copy the diags to the host

	// Figure out how many particles left the grid with a scan
	thrust::device_ptr<int> thrust_scan(didileave);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	thrust::inclusive_scan(thrust_scan,thrust_scan+nptcls,thrust_scan);

	CUDA_SAFE_CALL(cudaMemcpy(&nptcls_left,didileave+nptcls-1,sizeof(int),cudaMemcpyDeviceToHost));

	//printf("%i particles have left the domain\n",nptcls_left);

	if(nptcls_left > 0)
	{
		XPlist new_particles;

		new_particles.allocate(nptcls_left);
		float* dt_prec_dum = 0;
		float* vzinit_dum = 0;
		int* ipf_dum = 0;
		int* parent_id;
		int ndims = 6;
		int direction = 0;
		int xpdata_only = 1;

		long int Reinj_ptr = (long int)(&new_particles);

		CUDA_SAFE_CALL(cudaMalloc((void**)&parent_id,nptcls_left*sizeof(int)));



		// Copy the new particles from the pre calculated list
		xplist_transpose_(&Reinj_ptr,reinjlist_h+reinject_counter*6,dt_prec_dum,vzinit_dum,ipf_dum,&nptcls_left,&ndims,&direction,&xpdata_only);

		reinject_counter += nptcls_left;
		//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		// Copy dt_prec's of the particles that exited the grid
		((scan_condense<<<cudaGridSize,cudaBlockSize>>>
									 (new_particles.dt_prec,dt_prec,didileave,didileave,nptcls)));
		// Copy parent_id's of the new particles
		CUDA_SAFE_KERNEL((scan_condense<<<cudaGridSize,cudaBlockSize>>>
									 (parent_id,particle_id,didileave,didileave,nptcls)));

		// Advance the reinjected particles
		new_particles.advance(mesh,diags,reinjlist_h,0.5*dt,reinject_counter);

		// Copy the reinjected partilce data back to the parent list

		cudaGridSize = (nptcls_left+cudaBlockSize-1)/cudaBlockSize;
		for(int i=0;i<7;i++)
		{
			float* data_out = *get_float_ptr(i);
			float* data_in = *(new_particles.get_float_ptr(i));
			CUDA_SAFE_KERNEL((scan_broadcast<<<cudaGridSize,cudaBlockSize>>>
										 (data_out,data_in,parent_id,nptcls_left)));
		}

		// Get the momentum contribution from the reinjected particles
		thrust::device_ptr<float> xmout(new_particles.vx);
		thrust::device_ptr<float> ymout(new_particles.vy);
		thrust::device_ptr<float> zmout(new_particles.vz);

		float3 momout;

		CUDA_SAFE_CALL(cudaMemcpy(&momout,diags.momout,sizeof(float3),cudaMemcpyDeviceToHost));

		momout.x += thrust::reduce(xmout,xmout+nptcls_left);
		momout.y += thrust::reduce(ymout,ymout+nptcls_left);
		momout.z += thrust::reduce(zmout,zmout+nptcls_left);
		cudaDeviceSynchronize();
		CUDA_SAFE_CALL(cudaMemcpy(diags.momout,&momout,sizeof(float3),cudaMemcpyHostToDevice));



		new_particles.free();
		CUDA_SAFE_CALL(cudaFree(parent_id));

	}

	cudaDeviceSynchronize();
	//CUDA_SAFE_CALL(cudaFree(didileave_scan));
	//printf("Finished GPU PADVNC\n");

}

extern "C" void gpu_padvnc_(long int* XP_ptr,long int* mesh_ptr,
												float* phi,float* xpreinject,float* dt,int* reinject_counter,
												float* nincell,float* vrincell,float* vr2incell,
												float* xmout,float* ymout,float* zmout,
												float* xmomprobe,float* ymomprobe,float* zmomprobe,
												float* enerprobe,int* ninner)
{
#ifdef time_run
	unsigned int timer = 0;
	cutCreateTimer(&timer);
	cutStartTimer(timer);
#endif

#ifdef PROFILE_TIMERS
	g_timers[itimer_padvnc].start_timer();
#endif

	XPlist* particles = (XPlist*)(*XP_ptr);
	Mesh_data mesh_d = *(Mesh_data*)(*mesh_ptr);

	XPdiags diags_h(0);
	XPdiags diags_d(1);

	// Copy the potential to the gpu
	mesh_d.phi_copy(phi);

	// Reset Diagnostic Arrays
	mesh_d.nincell.cudaMatrixSet(0);
	mesh_d.vrincell.cudaMatrixSet(0);
	mesh_d.vr2incell.cudaMatrixSet(0);

	// Clear out dt_prec
	//CUDA_SAFE_CALL(cudaMemset(particles->dt_prec,0,(particles->nptcls)*sizeof(float)));

	particles->advance(mesh_d,diags_d,xpreinject,(*dt),*reinject_counter);

	mesh_d.nincell.cudaMatrixcpy(nincell,cudaMemcpyDeviceToHost);
	mesh_d.vrincell.cudaMatrixcpy(vrincell,cudaMemcpyDeviceToHost);
	mesh_d.vr2incell.cudaMatrixcpy(vr2incell,cudaMemcpyDeviceToHost);

	CUDA_SAFE_CALL(cudaMemcpy(diags_h.momout,diags_d.momout,sizeof(float3),cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(diags_h.momprobe,diags_d.momprobe,sizeof(float4),cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(diags_h.ninner,diags_d.ninner,sizeof(int),cudaMemcpyDeviceToHost));

	*xmout = diags_h.momout->x;
	*ymout = diags_h.momout->y;
	*zmout = diags_h.momout->z;

	*xmomprobe = diags_h.momprobe->x;
	*ymomprobe = diags_h.momprobe->y;
	*zmomprobe = diags_h.momprobe->z;
	*enerprobe = diags_h.momprobe->w;

	*ninner = *diags_h.ninner;

	diags_h.free();
	diags_d.free();

#ifdef PROFILE_TIMERS
	g_timers[itimer_padvnc].stop_timer();
#endif

#ifdef time_run
	cutStopTimer(timer);
	padvnc_timer += cutGetTimerValue( timer);
	npadvnc++;
	cutDeleteTimer( timer);
#endif

}















































#include "XPlist.cuh"
#include "/home/josh/CUDA/gnuplot_c/src/gnuplot_i.h"

#define concat(a, b)  a ## b


__device__
int gray_to_red(float gray)
{
	int result = (rint(((gray)/0.32f-0.78125f)*255.0f));
	return min(max(result,0),255);
}

__device__
int gray_to_green(float gray)
{
	int result = (rint(sin(3.14159265*(gray))*255.0f));
	return min(max(result,0),255);
}

__device__
int gray_to_blue(float gray)
{
	//int result = (rint(cos(3.14159265/2.0f*(gray))*255.0f));
	int result = (rint((3*gray-2)*255.0f));
	return  min(max(result,0),255);
}

__device__
float scale_edge(float cellf,float width_scale)
{
	float x = 1.0e-1/max(abs(cellf*(1.0-cellf)),1.0e-9);
	float result = pow(x,2) - x;

	return exp(-1.0*width_scale*result);
}

__global__
void particle_bin_image(Mesh_data mesh,
		pixel* pixels,
		int3 ncells,float2 xlims,float2 ylims,
		int minptcls,int maxptcls,int dim)
{
	int gthid = threadIdx.x+blockIdx.x*blockDim.x;

	float dx = (xlims.y-xlims.x)/((float)dim);
	float dy = (ylims.y-ylims.x)/((float)dim);


	while(gthid < dim*dim)
	{
		pixel my_pixel;
		int gidx = gthid%dim;
		int gidy = gthid/dim;

		float x = dx*gidx+xlims.x;
		float y = dy*gidy+ylims.x;
		float z = 0;

		float myr = sqrt(x*x+y*y+z*z);


		my_pixel.r = 255;
		my_pixel.g = 255;
		my_pixel.b = 255;
		my_pixel.a = 0;

		if(((x*x+y*y) < pow(mesh.rmesh(mesh.nr),2))&&
				((x*x+y*y) > pow(mesh.rmesh(0),2)))
		{
			my_pixel.a = 255;



			int4 icell;
			float4 cellf;
			float zetap;

			mesh.ptomesh<0>(x,y,z,&icell,&cellf,zetap);

			icell.x = (icell.x-1)/ncells.x;
			icell.y = (icell.y-1)/ncells.y;
			icell.z = (icell.z-1)/ncells.z;

			int binid = zorder(icell.x,icell.z,icell.y);
			float zval = (mesh.bins[binid].ilastp-mesh.bins[binid].ifirstp)+1;
			zval = (zval-minptcls)/((float)(maxptcls-minptcls));

			//zval = 1.5*(mesh.phi(icell.x,icell.y,icell.z)+0.5);

			my_pixel.r = gray_to_red(zval);
			my_pixel.g = gray_to_green(1-zval);
			my_pixel.b = gray_to_blue(1-zval);

			//int izval = 255*rint(binid/((float)mesh.nbins));
			//my_pixel.r = 255*(binid%4==5);
			//my_pixel.g = 255*((binid%4==1)||(binid%4 == 2));
			//my_pixel.b = 255*((binid%4==0)||(binid%4 == 3));


		//	float edge_scale = min(scale_edge(cellf.x,10.0f),
			//		min(scale_edge(cellf.y,1.0/myr),
			//		scale_edge(cellf.z,1.0/myr)));

		//	edge_scale = (edge_scale);

			//my_pixel.r *= edge_scale;
			//my_pixel.g *= edge_scale;
			//my_pixel.b *= edge_scale;

		}

		pixels[gthid] = my_pixel;

		gthid += blockDim.x*gridDim.x;

	}
}

__global__
void count_nptcls(Particlebin* bins,int* nptcls_out,int nbins)
{
	int gidx = threadIdx.x+blockIdx.x*blockDim.x;

	if(gidx < nbins)
	{
		nptcls_out[gidx] = bins[gidx].ilastp-bins[gidx].ifirstp+1;
	}
}

void plot_grid(gnuplot_ctrl* plot,Mesh_data mesh)
{
	gnuplot_cmd(plot,"set parametric");
	gnuplot_cmd(plot,"set trange [0:2*pi]");

	// Plot r grid
	char cmd[mesh.nr*80];
	for(int i=0;i<=mesh.nr;i++)
	{
		double r = mesh.rmesh_h2[i];
		char temp[80];
		sprintf(temp,"%10g*cos(t),%10g*sin(t) lc \"black\"",r,r);
		if(i == 0)
		{
			sprintf(cmd,"%s",temp);
		}
		else
		{
			sprintf(cmd,"%s,%s",cmd,temp);
		}
	}

	gnuplot_cmd(plot,"set title \"\"");
	gnuplot_cmd(plot,"set key off");
	//gnuplot_cmd(plot,"set style function lines; linecolor \"black\"");

	char line[mesh.nr*80];
	sprintf(line,"replot %s",cmd);
	gnuplot_cmd(plot,line);

	// Plot Theta grid
	char trange_cmd[50];

	double rmin = mesh.rmesh_h2[0];
	double rmax = mesh.rmesh_h2[mesh.nr];

	sprintf(trange_cmd,"set trange [%g:%g]",rmin,rmax);
	gnuplot_cmd(plot,trange_cmd);

	char cmd2[mesh.nth*80];

	for(int i=0;i<mesh.npsi;i++)
	{
		double theta = mesh.psimesh_h2[i];
		double xmin = rmin*cos(theta);
		double ymin = rmin*sin(theta);

		char temp[80];
		sprintf(temp,"%10g*t,%10g*t lc \"black\"",cos(theta),sin(theta));

		if(i == 0)
		{
			sprintf(cmd2,"%s",temp);
		}
		else
		{
			sprintf(cmd2,"%s, %s",cmd2,temp);
		}
	}

	//gnuplot_cmd(plot,"set title \"\"");
	//gnuplot_cmd(plot,"set style function lines linecolor \"black\"");
	char line2[mesh.nth*80];
	sprintf(line2,"replot %s",cmd2);
	gnuplot_cmd(plot,line2);


}


__host__
void plot_particle_bins(Mesh_data mesh,int3 ncells)
{
	float2 xlims;
	float2 ylims;

	int image_dim = 768;

	int cudaBlockSize = 512;
	int cudaGridSize = (mesh.nbins + cudaBlockSize - 1)/cudaBlockSize;

	int* nptcls_bin;
	int* nptcls_bin_temp;
	CUDA_SAFE_CALL(cudaMalloc((void**)&nptcls_bin,mesh.nbins*sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&nptcls_bin_temp,mesh.nbins*sizeof(int)));

	CUDA_SAFE_KERNEL((count_nptcls<<<cudaGridSize,cudaBlockSize>>>(mesh.bins,nptcls_bin,mesh.nbins)));

	CUDA_SAFE_CALL(cudaMemcpy(nptcls_bin_temp,nptcls_bin,mesh.nbins*sizeof(int),cudaMemcpyDeviceToDevice));

	thrust::device_ptr<int> nptcls_max_t(nptcls_bin);
	thrust::device_ptr<int> nptcls_min_t(nptcls_bin_temp);

	int nptcls_max = thrust::reduce(nptcls_max_t,nptcls_max_t+mesh.nbins,0,thrust::maximum<int>());
	int nptcls_min = thrust::reduce(nptcls_min_t,nptcls_min_t+mesh.nbins,nptcls_max,thrust::minimum<int>());

	printf("max / min particles per bin = %i / %i \n",nptcls_max,nptcls_min);

	CUDA_SAFE_CALL(cudaFree(nptcls_bin));
	CUDA_SAFE_CALL(cudaFree(nptcls_bin_temp));


	xlims.x = mesh.rmesh_h2[mesh.nr];
	xlims.y = -xlims.x;

	ylims = xlims;


	pixel* pixels_d;
	pixel* pixels_h = (pixel*)malloc(image_dim*image_dim*sizeof(pixel));

	CUDA_SAFE_CALL(cudaMalloc((void**)&pixels_d,image_dim*image_dim*sizeof(pixel)));

	cudaBlockSize = 512;
	cudaGridSize = 84;

	CUDA_SAFE_KERNEL((particle_bin_image<<<cudaGridSize,cudaBlockSize>>>
			(mesh,pixels_d,ncells,xlims,ylims,
			nptcls_min,nptcls_max,image_dim)));

	CUDA_SAFE_CALL(cudaMemcpy(pixels_h,pixels_d,image_dim*image_dim*sizeof(pixel),cudaMemcpyDeviceToHost));

	gnuplot_ctrl* plot;
	plot = gnuplot_init();

	gnuplot_cmd(plot,"set xlabel \"x (mm)\"");
	gnuplot_cmd(plot,"set ylabel \"y (mm)\"");

	gnuplot_plot_rbgaimage(plot,pixels_h,image_dim,image_dim,xlims.x,xlims.y,ylims.x,ylims.y);

	plot_grid(plot,mesh);



	printf("Press 'Enter' to continue\n");
	getchar();

	gnuplot_save_pdf(plot,"zorder_sceptic");


	gnuplot_close(plot);

	cudaFree(pixels_d);
	free(pixels_h);



}







































/*
float3 polar_to_cartiesian(float r, float th, float psi)
{
	float3 result;

	result.z = r;
	result.x = result.z*cos(psi);

	result.y = result.z*sin(psi);

	return result;

}

__host__
void plot_particle_bins(Particlebin* bins_d,
		float* rmesh,float* thmesh,float* pcc,
		int nr, int nth, int npsi,int nbins)
{

	Particlebin* bins_h = (Particlebin*)malloc(nbins*sizeof(Particlebin));

	cell cells_h[nr*npsi];
	float nptcls_bin[nr*npsi];

	float nptcls_max;
	float nptcls_min;

	int nbins_r = (nr+ncells_per_bin_g.x-1)/ncells_per_bin_g.x;
	int nbins_psi = (npsi+ncells_per_bin_g.z-1)/ncells_per_bin_g.z;

	CUDA_SAFE_CALL(cudaMemcpy(bins_h,bins_d,nbins*sizeof(Particlebin),cudaMemcpyDeviceToHost));

	nptcls_max = bins_h[0].ilastp-bins_h[0].ifirstp+1;
	nptcls_min = nptcls_max;

	// Loop over all bins at the psi=0 plane

	int k = 0;
	for(int ir = 0;ir < nr;ir++)
	{
		for(int ipsi = 0; ipsi< npsi;ipsi++)
		{
			int ith = 0;
			int ibin = zorder(ir,ith,ipsi);


			int ix[4] = {0,0,1,1};
			int iy[4] = {0,1,1,0};

			for(int i=0;i<4;i++)
			{
				float r = rmesh[ncells_per_bin_g.x*(ir+iy[i])];
				float theta = (thmesh[ncells_per_bin_g.y*(ith)]);
				float psi = pcc[ncells_per_bin_g.z*(ipsi+ix[i])];

				float3 position = polar_to_cartiesian(r,theta,psi);

				cells_h[k].x[i] = position.x;
				cells_h[k].y[i] = position.y;

			}

			nptcls_bin[k] = bins_h[ibin].ilastp-bins_h[ibin].ifirstp+1;

			nptcls_max = max(nptcls_max,nptcls_bin[k]);
			nptcls_min = min(nptcls_min,nptcls_bin[k]);



			k++;
		}
	}

	gnuplot_ctrl* plot;
	plot = gnuplot_init();

	float xdims[2] = {-rmesh[nr-1],rmesh[nr-1]};
	float ydims[2] = {-rmesh[nr-1],rmesh[nr-1]};

	gnuplot_setup_mesh(plot,cells_h,xdims,ydims,k);

	gnuplot_fill_mesh(plot,nptcls_bin,k,nptcls_min,nptcls_max);

	printf("Press 'Enter' to continue\n");
	getchar();
	gnuplot_close(plot);

	free(bins_h);


}
*/

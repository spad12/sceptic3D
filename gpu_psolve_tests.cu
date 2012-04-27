/*
 * "This software contains source code provided by NVIDIA Corporation."
 */

#include "gpu_solver.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>

extern "C" {void atimes_(int &n1,int &n2, int &n3,float* x, float* res,uint* itrnsp);}
extern "C" {void asolve_(int &n1,int &n2, int &n3,float* b, float* z,float* zerror);}
extern "C" {void cg3d_(int &n1,int &n2,int &n3,float* b,float* x,float &tol,int &iter,int&itmax);}
extern "C" {void shielding3d_(float &dt,int &n1);}

extern "C" void asolve_test_(long int* solverPtr,float* phi,float* bin,float* zin,
														float* gpc,int* n1,int* n2,int* n3)
{
	printf("Executing asolve test\n");
	PoissonSolver* solver;

	solver = ((PoissonSolver*)(*solverPtr));

	int nrsize = solver->nrsize;
	int nthsize = solver->nthsize;
	int npsisize = solver->npsisize;

	float tolerance = 1.0e-5;
	float zerror;
	float total_error = 0.0;

	int gidx;

	// Allocate temporary storage for the results.
	float* b_gpu = (float*)malloc((nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));
	float* b_cpu = (float*)malloc((nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));

	float* z_gpu = (float*)malloc((nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));
	float* z_cpu = (float*)malloc((nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));


	cudaMatrixf b_d = solver->b;
	cudaMatrixf z_d = solver->z;

	b_d.cudaMatrixcpy(bin,cudaMemcpyHostToDevice);
	b_d.cudaMatrixcpy(b_cpu,cudaMemcpyDeviceToHost);
	//z_d.cudaMatrixcpy(zin,cudaMemcpyHostToDevice);
	//z_d.cudaMatrixcpy(z_cpu,cudaMemcpyDeviceToHost);
	solver->phi.cudaMatrixcpy(phi,cudaMemcpyHostToDevice);
	solver -> gpc_copy(gpc);

	// Do cpu asolve
	asolve_(*n1,*n2,*n3,b_cpu,z_cpu,&zerror);

	// Do gpu asolve
	solver->asolve(*n1,*n2,*n3,b_d,z_d);



	// Copy results back to the host
	z_d.cudaMatrixcpy(z_gpu,cudaMemcpyDeviceToHost);
	b_d.cudaMatrixcpy(b_gpu,cudaMemcpyDeviceToHost);

	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	// Check the results with the results of the host routine
	for(int i=1;i<(*n1);i++)
	{
		for(int j=1;j<(*n2+1);j++)
		{
			for(int k=1;k<(*n3+1);k++)
			{
				gidx = i+(nrsize-1)*(j+(nthsize+1)*k);

				float gpu_data = z_gpu[gidx];
				float cpu_data = z_cpu[gidx];

				float terror = 2.0*abs(gpu_data-cpu_data)/max(1.0*tolerance,abs(cpu_data+gpu_data));

				if(terror > tolerance)
				{
					printf("Error asolve res %f != %f with error %f at %i, %i, %i\n",gpu_data,cpu_data,terror,i,j,k);
				}

				total_error += terror;
			}
		}
	}

	// Get the average error
	total_error /= (float)((*n1)*(*n2)*(*n3));
	printf("Total Asolve error = %g\n",total_error);



	free(b_gpu);
	free(b_cpu);
	free(z_gpu);
	free(z_cpu);



}

extern "C" void atimes_test_(long int* solverPtr,float* phi,float* xin,float* res,
														float* gpc,int* n1,int* n2,int* n3,uint* itrnsp)
{
	printf("Executing atimes test\n");
	PoissonSolver* solver;

	solver = ((PoissonSolver*)(*solverPtr));

	int nrsize = solver->nrsize;
	int nthsize = solver->nthsize;
	int npsisize = solver->npsisize;

	float tolerance = 1.0e-1;
	float total_error = 0.0;

	int gidx;

	// Allocate temporary storage for the results.
	float* res_gpu = (float*)malloc((nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));
	float* res_cpu = (float*)malloc((nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));

	float* x_gpu = (float*)malloc((nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));
	float* x_cpu = (float*)malloc((nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));

	printf("n1 = %i, n2= %i, n3 = %i\n",*n1,*n2,*n3);

	cudaMatrixf x_d = solver->x;
	cudaMatrixf res_d(nrsize-1,nthsize+1,npsisize+1);

	res_d.cudaMatrixcpy(res,cudaMemcpyHostToDevice);
	res_d.cudaMatrixcpy(res_cpu,cudaMemcpyDeviceToHost);
	solver->phi.cudaMatrixcpy(phi,cudaMemcpyHostToDevice);
	solver->x.cudaMatrixcpy(xin,cudaMemcpyHostToDevice);
	solver->x.cudaMatrixcpy(x_cpu,cudaMemcpyDeviceToHost);
	solver -> gpc_copy(gpc);


	// Do cpu atimes
	atimes_(*n1,*n2,*n3,x_cpu,res_cpu,itrnsp);

	// Do gpu atimes
	solver->atimes(*n1,*n2,*n3,x_d,res_d,0);



	// Copy results back to the host
	res_d.cudaMatrixcpy(res_gpu,cudaMemcpyDeviceToHost);
	solver->x.cudaMatrixcpy(x_gpu,cudaMemcpyDeviceToHost);

	// Check the results with the results of the host routine
	for(int i=0;i<(*n1);i++)
	{
		for(int j=1;j<(*n2+1);j++)
		{
			for(int k=1;k<(*n3+1);k++)
			{
				gidx = i+(nrsize-1)*(j+(nthsize+1)*k);

				float gpu_data = res_gpu[gidx];
				float cpu_data = res_cpu[gidx];

				float terror = 2.0*abs(gpu_data-cpu_data)/max(tolerance,abs(cpu_data+gpu_data));

				if((terror > tolerance)||(isnan(terror)))
				{
					printf("Error atimes (%i) res %f != %f with error %f at %i, %i, %i\n",*itrnsp,gpu_data,cpu_data,terror,i,j,k);
				}

				total_error += terror;
			}
		}
	}

	// Get the average error
	total_error /= (float)((*n1)*(*n2)*(*n3));
	printf("Total Atimes (%i) error = %g\n",*itrnsp,total_error);

	free(res_gpu);
	free(res_cpu);
	free(x_gpu);
	free(x_cpu);
	res_d.cudaMatrixFree();



}


extern "C" void cg3d_test_(long int* solverPtr,float* phi,int* lbcg,int* n1,int* n2,int* n3,
	    									 float* bin,float* xin,float* tol,float* gpc,int* iter,int* itmax)
{
	PoissonSolver* solver;

	solver = ((PoissonSolver*)(*solverPtr));

	int nrsize = solver->nrsize;
	int nthsize = solver->nthsize;
	int npsisize = solver->npsisize;

	int iter_gpu;
	int iter_cpu;
	float gpu_tol = 1.0e-5;

	float tolerance = 1.0e-2;
	float total_error = 0.0;

	uint t1,t2;
	float t_cpu,t_gpu;

	int gidx;

	// Allocate temporary storage for the results.
	float* b_gpu = (float*)malloc((nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));
	float* b_cpu = (float*)malloc((nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));

	float* x_gpu = (float*)malloc((nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));
	float* x_cpu = (float*)malloc((nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));

	memcpy(b_gpu,bin,(nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));
	memcpy(b_cpu,bin,(nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));

	memcpy(x_gpu,xin,(nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));
	memcpy(x_cpu,xin,(nrsize-1)*(nthsize+1)*(npsisize+1)*sizeof(float));

	// Do gpu atimes
	start_timer_(&t1);
	cg3d_gpu_(solverPtr,phi,lbcg,n1,n2,n3,b_gpu,x_gpu,&gpu_tol,gpc,&iter_gpu,itmax);
	stop_timer_(&t_gpu,&t1);


	// Do cpu cg3d
	start_timer_(&t2);
	cg3d_(*n1,*n2,*n3,b_cpu,x_cpu,*tol,iter_cpu,*itmax);
	stop_timer_(&t_cpu,&t2);

	printf("CPU took %i iterations and %f ms\n",iter_cpu,t_cpu/iter_cpu);



	printf("GPU took %i iterations and %f ms\n",iter_gpu, t_gpu/iter_gpu);

	// Check the results with the results of the host routine
	for(int i=1;i<(*n1);i++)
	{
		for(int j=1;j<(*n2+1);j++)
		{
			for(int k=1;k<(*n3+1);k++)
			{
				gidx = i+(nrsize-1)*(j+(nthsize+1)*k);

				float gpu_data = x_gpu[gidx];
				float cpu_data = x_cpu[gidx];

				float terror = 2.0*abs(gpu_data-cpu_data)/max(10000.0*tolerance,abs(cpu_data+gpu_data));

				if((terror > tolerance)||isnan(terror))
				{
					//printf("Error cg3d res %f != %f with error %f at %i, %i, %i\n",gpu_data,cpu_data,terror,i,j,k);
				}

				total_error += terror;
			}
		}
	}

	// Get the average error
	total_error /= (float)((*n1)*(*n2)*(*n3));
	printf("Total cg3d error = %g\n",total_error);

	free(b_gpu);
	free(b_cpu);
	free(x_gpu);
	free(x_cpu);
}

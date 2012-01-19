#include "gpu_solver.cuh"
#include "XPlist.cuh"



__global__
void innerbc_kernel(PoissonSolver solver, cudaMatrixf pcc,cudaMatrixf tcc,
								float Exext,float vprobe,int imin,int nthused,int npsiused)
{
	// No insulate, no lfloat
	int idx = threadIdx.x;
	int idy = threadIdx.y;
	int bidx = blockDim.x*blockIdx.x+1;
	int bidy = blockDim.y*blockIdx.y+1;
	int gidx = idx + bidx;
	int gidy = idy + bidy;

	float my_val = 0.0;


	while(gidy <= npsiused)
	{
		gidx = idx+bidx;
		while(gidx <= nthused)
		{
			/*
			float LS;
			if(mesh.Bz != 0.0)
			{
				float dpdr;
				dpdr = -solver.phi(3,gidx,gidy);
				dpdr += 4.0f*solver.phi(2,gidx,gidy);
				dpdr -= 3.0f*solver.phi(1,gidx,gidy)/(mesh.rccmesh(3)-mesh.rccmesh(1));
				dpdr *= 1.0/mesh.phi(1,gidx,gidy);

				LS = -1.0f/(min(dpdr,-1.01f)+1);
			}

			// Calculate the flux to each angular cell
			if(lcic)
			{
				mesh.fluxofangle(gidx,gidy) = mesh.fincellave(gidx,gidy)*(mesh.nthused-1.0)/
															(4.0f*pi_const*rhoinf*dt*pow(mesh.rmesh(1),2));

				if((gidx == 1)||(gidx == mesh.nthused))
					mesh.fluxofangle(gidx,gidy) *= mesh.fluxofangle(gidx,gidy);
			}
			else
			{
				mesh.fluxofangle(gidx,gidy) = mesh.fincellave(gidx,gidy)*(mesh.nthused)/
															(4.0f*pi_const*rhoinf*dt*pow(mesh.rmesh(1),2));
			}
			*/

			solver.phi(imin,gidx,gidy) = vprobe + Exext*cos(pcc(gidy))*sqrt(1-pow(tcc(gidx),2));

			gidx += blockDim.x*gridDim.x;
		}
		gidy += blockDim.y*gridDim.y;
	}

}


template<int bcphi>
__global__
void shieldingbc_kernel(PoissonSolver solver,Mesh_data mesh,int n1)
{
	int idx = threadIdx.x;
	int idy = threadIdx.y;
	int bidx = blockDim.x*blockIdx.x+1;
	int bidy = blockDim.y*blockIdx.y+1;
	int gidx = idx + bidx;
	int gidy = idy + bidy;

	if(bcphi == 0)
		float my_val = 0.0;

	while(gidy <= mesh.npsiused)
	{
		gidx = idx+bidx;
		while(gidx <= mesh.nthused)
		{
			switch(bcphi)
			{
			case 1:
				for(int i=n1+1;i<=mesh.nrused;i++)
				{
					solver.phi(i,gidx,gidy) = log(solver.rho(i,gidx,gidy));
				}
				solver.gpc(gidx,gidy,3) = solver.phi(n1+1,gidx,gidy);
				break;
			case 2:
				solver.phi(n1+1,gidx,gidy) = 0.0f;
				solver.gpc(gidx,gidy,3) = solver.phi(n1+1,gidx,gidy);
				break;
			case 0:
				float deficitj = 1-solver.phi(n1,gidx,gidy)/mesh.Ti - solver.rho(n1,gidx,gidy);
				break;

			default:
				break;
			}

			gidx += blockDim.x*gridDim.x;
		}
		gidy += blockDim.y*gridDim.y;
	}
}

__global__
void shielding3D_setup_kernel(PoissonSolver solver)
{
	int idx = threadIdx.x;
	int idy = threadIdx.y;
	int idz = threadIdx.z;
	int bidx = blockDim.x*blockIdx.x+1;
	int bidy = blockDim.y*blockIdx.y+1;
	int bidz = blockDim.z*blockIdx.z+1;
	int gidx = idx + bidx;
	int gidy = idy + bidy;
	int gidz = idz+bidz;


	while(gidz <= solver.n3)
	{
		gidy = idy+bidy;
		while(gidy <= solver.n2)
		{
			gidx = idx+bidx;
			while(gidx <= solver.n1)
			{
				if(gidy == 1)
					solver.phi(gidx,1,gidz) == solver.phiaxis(gidx,0,gidy);
				else if(gidy == solver.n2)
					solver.phi(gidx,solver.n2,gidz) == solver.phiaxis(gidx,1,gidy);

				float phi = solver.phi(gidx,gidy,gidz);
				float rho = solver.rho(gidx,gidy,gidz);
				if((gidx > 2)&&(gidx < solver.n1))
				{
					solver.b(gidx-1,gidy,gidz) = exp(phi)*(1-phi)-rho;
					solver.x(gidx-1,gidy,gidz) = phi;
				}
				else if(gidx == solver.n1)
				{
					solver.b(gidx-1,gidy,gidz) = exp(phi)*(1-phi)-rho-solver.apc(gidx)*solver.gpc(gidy,gidz,3);
					solver.x(gidx-1,gidy,gidz) = phi;
				}
				else if(gidx == 2)
				{
					solver.b(gidx-1,gidy,gidz) = exp(phi)*(1-phi)-rho-solver.bpc(gidx)*solver.phi(1,gidy,gidz);
					solver.x(gidx-1,gidy,gidz) = phi;
					solver.x(0,gidy,gidz) = 0;
				}

				gidx += blockDim.x*gridDim.x;
			}
			gidy += blockDim.y*gridDim.y;
		}
		gidz += blockDim.z*gridDim.z;
	}
}

__global__
void shielding3D_write_phi_kernel(PoissonSolver solver)
{
	int idx = threadIdx.x;
	int idy = threadIdx.y;
	int idz = threadIdx.z;
	int bidx = blockDim.x*blockIdx.x+1;
	int bidy = blockDim.y*blockIdx.y+1;
	int bidz = blockDim.z*blockIdx.z+1;
	int gidx = idx + bidx;
	int gidy = idy + bidy;
	int gidz = idz+bidz;


	while(gidz <= solver.n3)
	{
		gidy = idy+bidy;
		while(gidy <= solver.n2)
		{
			gidx = idx+bidx;
			while(gidx <= solver.n1)
			{
				solver.phi(gidx,gidy,gidz) = solver.x(gidx-1,gidy,gidz);

				if(gidx == 1)
				{
					solver.phi(gidx,gidy,gidz) = 2.5f*solver.x(0,gidy,gidz)-2.0f*solver.x(1,gidy,gidz)+0.5f*solver.x(2,gidy,gidz);
				}

				gidx += blockDim.x*gridDim.x;
			}
			gidy += blockDim.y*gridDim.y;
		}
		gidz += blockDim.z*gridDim.z;
	}
}

__global__
void shielding3D_finish_kernel(PoissonSolver solver,int nrused)
{
	int idx = threadIdx.x;
	int idy = threadIdx.y;
	int bidx = blockDim.x*blockIdx.x;
	int bidy = blockDim.y*blockIdx.y+1;
	int gidx = idx + bidx;
	int gidy = idy + bidy;

	int block_start = bidx;

	__shared__ float psisave1_s[66];
	__shared__ float psisave2_s[66];

	float psisave1 = 0.0;
	float psisave2 = 0.0;


	while(block_start <= nrused)
	{
		gidy = idy+bidy;

		// Set shared memory sums to 0
		if(gidx <= nrused)
		{
			if(idy == 0)
			{
				psisave1_s[idx] = 0;
				psisave2_s[idx] = 0;
			}
		}

		__syncthreads();

		// Each thread sums up a couple of elements in local memory
		if(gidx <= nrused)
		{
			while(gidy <= solver.n3)
			{
				psisave1 += solver.phi(gidx,1,gidy);
				psisave2 += solver.phi(gidx,solver.n2,gidy);

				gidy += blockDim.y*gridDim.y;
			}

			// Contribute local sum to shared sum
			atomicAdd(psisave1_s+idx,psisave1);
			atomicAdd(psisave2_s+idx,psisave2);
		}

		__syncthreads();

		// Modify shared Sum
		if(gidx <= nrused)
		{
			if(idy == 0)
			{
				psisave1_s[idx] /= solver.n3;
				psisave2_s[idx] /= solver.n3;
			}
		}

		__syncthreads();

		// Write Edge Nodes
		if(gidx <= nrused)
		{
			while(gidy <= solver.n3)
			{
				solver.phiaxis(gidx,1,gidy) = solver.phi(gidx,1,gidy);
				solver.phiaxis(gidx,2,gidy) = solver.phi(gidx,solver.n2,gidy);

				solver.phi(gidx,1,gidy) = psisave1_s[idx];
				solver.phi(gidx,solver.n2,gidy) = psisave2_s[idx];

				gidy += blockDim.y*gridDim.y;
			}
		}

		__syncthreads();

		// Set the shadow theta-cells
		if(gidx <= nrused)
		{
			while(gidy <= solver.n3)
			{
				int kk1 = (gidy+3*solver.n3/2-1)%solver.n3 + 1;
				int kk2 = (gidy+(3*solver.n3+1)/2-1)%solver.n3 + 1;

				solver.phi(gidx,0,gidy) = 0.5*(solver.phi(gidx,2,kk1)+solver.phi(gidx,2,kk2));
				solver.phi(gidx,solver.n2+1,gidy) = 0.5*(solver.phi(gidx,solver.n2-1,kk1)+solver.phi(gidx,solver.n2-1,kk2));

				gidy += blockDim.y*gridDim.y;
			}
		}

		__syncthreads();

		// Set the shadow psi-cells
		if(gidx <= nrused)
		{
			while(gidy <= solver.n2+2)
			{
				solver.phi(gidx,gidy-1,solver.n3+1) = solver.phi(gidx,gidy-1,1);
				solver.phi(gidx,gidy-1,0) = solver.phi(gidx,gidy,solver.n3);

				gidy += blockDim.y*gridDim.y;
			}
		}

		__syncthreads();

		block_start += blockDim.x*gridDim.x;
		gidx = block_start+idx;
	}
}

__host__
void PoissonSolver::shielding3D(float dt, int n1_in, int n2_in, int n3_in,int &iter,int nrused,int lbcg)
{
	n1 = n1_in;
	n2 = n2_in;
	n3 = n3_in;

	dim3 cudaBlockSize(1,1,1);
	dim3 cudaGridSize(1,1,1);

	int maxits = 2*pow((float)((n1+1)*n2*n3),0.3333);
	float dconverge = 1.0e-5;

	// Setup x and b
	cudaBlockSize.x = 32;
	cudaBlockSize.y = 8;
	cudaBlockSize.z = 1;

	cudaGridSize.x = 1;
	cudaGridSize.y = 1;
	cudaGridSize.z = (n3 + cudaBlockSize.z - 1)/cudaBlockSize.z;

	CUDA_SAFE_KERNEL((shielding3D_setup_kernel<<<cudaGridSize,cudaBlockSize>>>(*this)));

	// Run cg3d
	cg3D(n1,n2,n3,dconverge,iter,maxits,lbcg);

	// Write cg3D results to phi
	CUDA_SAFE_KERNEL((shielding3D_write_phi_kernel<<<cudaGridSize,cudaBlockSize>>>(*this)));

	// Write Shadow Cells
	cudaBlockSize.x = 32;
	cudaBlockSize.y = 8;
	cudaBlockSize.z = 1;

	cudaGridSize.x = (nrused + cudaBlockSize.x - 1)/cudaBlockSize.x;
	cudaGridSize.y = 1;
	cudaGridSize.z = 1;

	CUDA_SAFE_KERNEL((shielding3D_finish_kernel<<<cudaGridSize,cudaBlockSize>>>(*this,nrused)));

}



extern "C" void shielding3d_gpu_(long int* solverPtr,
														float* phi,float* rho,float* phiaxis,
														float* gpc,float* dt, int* lbcg,
														int* n1,int* n2,int* n3,int* nrused,int* iter)
{
	// Warning this has not yet been debugged, and does not contribute a significant amount to the run time
	PoissonSolver* solver;

	solver = ((PoissonSolver*)(*solverPtr));

	solver->phi.cudaMatrixcpy(phi,cudaMemcpyHostToDevice);
	solver->rho.cudaMatrixcpy(rho,cudaMemcpyHostToDevice);
	solver->phiaxis.cudaMatrixcpy(phiaxis,cudaMemcpyHostToDevice);
	solver->gpc.cudaMatrixcpy(gpc,cudaMemcpyHostToDevice);

	solver->shielding3D(*dt,*n1,*n2,*n3,*iter,*nrused,*lbcg);

	solver->phi.cudaMatrixcpy(phi,cudaMemcpyDeviceToHost);
	solver->phiaxis.cudaMatrixcpy(phiaxis,cudaMemcpyDeviceToHost);



}


extern "C" void shielding3d_test_(long int* solverPtr,float* phi,float* rho,float* phiaxis,float* gpc,
													  	int* lbcg,int* n1,int* n2,int* n3,int* nrused,float* dt)
{
	printf("\nTesting GPU shielding3D\n");
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
	float* phi_gpu = (float*)malloc((nrsize+1)*(nthsize+1)*(npsisize+1)*sizeof(float));
	float* phiaxis_gpu = (float*)malloc((nrsize+1)*(2)*(npsisize+1)*sizeof(float));

	memcpy(phi_gpu,phi,(nrsize+1)*(nthsize+1)*(npsisize+1)*sizeof(float));
	memcpy(phiaxis_gpu,phiaxis,(nrsize+1)*(2)*(npsisize+1)*sizeof(float));


	// Do gpu atimes
	start_timer_(&t1);
	shielding3d_gpu_(solverPtr,phi_gpu,rho,phiaxis_gpu,gpc,dt,lbcg,n1,n2,n3,nrused,&iter_gpu);
	stop_timer_(&t_gpu,&t1);


	// Do cpu cg3d
	start_timer_(&t2);
	shielding3d_(*dt,*n1);
	stop_timer_(&t_cpu,&t2);



	printf("GPU took %i iterations and %f ms\n",iter_gpu, t_gpu/iter_gpu);

	// Check the results with the results of the host routine
	for(int i=0;i<(*n1+1);i++)
	{
		for(int j=0;j<(*n2+1);j++)
		{
			for(int k=0;k<(*n3+1);k++)
			{
				gidx = i+(nrsize+1)*(j+(nthsize+1)*k);

				float gpu_data = phi_gpu[gidx];
				float cpu_data = phi[gidx];

				float terror = 2.0*abs(gpu_data-cpu_data)/max(10.0*tolerance,abs(cpu_data+gpu_data));

				if(terror > tolerance)
				{
					//printf("Error shielding3d res %f != %f with error %f at %i, %i, %i\n",gpu_data,cpu_data,terror,i,j,k);
				}

				total_error += terror;
			}
		}
	}

	// Get the average error
	total_error /= (float)((*n1)*(*n2)*(*n3));
	printf("Total shielding3d error = %g\n",total_error);

	free(phi_gpu);
	free(phiaxis_gpu);
}





















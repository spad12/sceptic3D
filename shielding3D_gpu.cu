#include "gpu_solver.cuh"
#include "XPlist.cuh"



__host__
void PoissonSolver::shielding3D(float dt, int n1, int n2, int n3,int lbcg)
{
	int maxits = 2*pow((float)((n1+1)*n2*n3),0.3333);
	float dconverge = 1.0e-5;

}



extern "C" void shielding3D_gpu_(long int* solverPtr,float* dt, int* lbcg,int* n1)
{
	PoissonSolver* solver;

	solver = ((PoissonSolver*)(*solverPtr));

}



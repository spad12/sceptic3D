#include "gpu_timing.cuh"
#include "XPlist.cuh"
#include <fstream>

extern "C" void output_timing_data_(int* nptcls,int* nr,int* nth,int* npsi,int* nsteps,
														float* dt,float* rmax,float* debyelen,
														float* Bz,float* vprobe,float* Ti,float* reinj_frac,
														float* times,float* total_times,int* ntimes,int* runid,int* gpurun)
{
	char filename[60];
	char* runtype;
	FILE* fp;

	int nparams = 13;
	char* run_params[nparams];
	char* time_names[*ntimes];

	int nbins;

	nbins = ((*nr+ncells_per_bin_g.x-1)/ncells_per_bin_g.x);
	nbins *= ((*nth+ncells_per_bin_g.y-1)/ncells_per_bin_g.y);
	nbins *= ((*npsi+ncells_per_bin_g.z-1)/ncells_per_bin_g.z);



	if(*gpurun)
	{
		runtype = "gpu";
	}
	else
	{
		runtype = "cpu";
	}

	run_params[0] = "nptcls";
	run_params[1] = "nr";
	run_params[2] = "nth";
	run_params[3] = "npsi";
	run_params[4] = "nbins";
	run_params[5] = "nsteps";
	run_params[6] = "dt";
	run_params[7] = "rmax";
	run_params[8] = "l_d";
	run_params[9] = "Bz";
	run_params[10] = "Vprobe";
	run_params[11] = "Ti";
	run_params[12] = "fReinj";


	time_names[0] = "ParticleSort*";
	time_names[1] = "ChargeAssign*";
	time_names[2] = "ChargeToMesh";
	time_names[3] = "PsumReduce";
	time_names[4] = "RhoCalc";
	time_names[5] = "cg3D*";
	time_names[6] = "PoissonSolve";
	time_names[7] = "FillReinject";
	time_names[8] = "PAdvance";
	time_names[9] = "PartReduce";
	time_names[10] = "TotalTime";

	// Setup the filename
	sprintf(filename,"./benchmarks/benchmark%i_%s.dat",*runid,runtype);

	// Check to see if the file exists
	fp = fopen(filename,"r+");

	// If the file doesn't exist yet, create it and write the top line
	if(fp == NULL)
	{

		fp = fopen(filename,"w");
		char header[nparams*9+35*(*ntimes)];
		for(int i=0;i<nparams;i++)
		{
			fprintf(fp,"%s,",run_params[i]);
		}

		for(int i=0;i<*ntimes;i++)
		{
			fprintf(fp,"%s(ms),%s(ns),",time_names[i],"adjusted");
		}

		fprintf(fp,"\n");
	}

	fclose(fp);

	fp = fopen(filename,"a");

	char lineout[nparams*9+35*(*ntimes)];



	fprintf(fp,"%i,%i,%i,%i,%i,%i,%f,%f,%f,%f,%f,%f,%f,",
				*nptcls,*nr,*nth,*npsi,nbins,*nsteps,*dt,*rmax,*debyelen,*Bz,*vprobe,*Ti,*reinj_frac);


	for(int i=0;i<*ntimes;i++)
	{
		fprintf(fp,"%f,%f,",times[i],total_times[i]);
	}

	fprintf(fp,"\n");


	fclose(fp);


	printf("\n");
	printf("reinjFrac = %f\n",*reinj_frac);
	for(int i=0;i<*ntimes;i++)
	{
		char temp[30];
		char temp2[20];

		sprintf(temp,"%s runtime: ",time_names[i]);
		sprintf(temp2,"%*f",(40-strlen(temp)),total_times[i]);
		printf("%s %s(ns)\n",temp,temp2);
	}



}


























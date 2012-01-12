

#include "XPlist.cuh"

__global__
void dummy_kernel2(void){;}


extern "C" void pick_gpu_(int* myid,int* mynpart)
{
	int ndevices;
	cudaGetDeviceCount(&ndevices);
	int my_device = *myid;

	if(*myid < ndevices)
	{
		CUDA_SAFE_CALL(cudaSetDevice(my_device));
	}
	else
	{
		*mynpart = 0;
	}
}

extern "C" void gpu_mesh_init_(long int* Mesh_ptr,
													float* phi,
													float* phiaxis,
													float* rho,
													float* rhoDiag,
													float* rmesh, // r(nrsize)
													float* rccmesh, // rcc(nrsize)
													float* thmesh, // th(nthsize)
													float* tccmesh, // tcc(nthsize)
													float* thang, // thang(nthsize)
													float* pcc, // pcc(npsisize)
													float* volinv, // volinv(nrsize)
													float* zeta, // zeta(nrsize+1)
													float* zetahalf, // zetahalf(nrsize+1)
													int* irpre, // irpre(nrpre)
													int* itpre, // itpre(ntpre)
													int* ippre, // ippre(nppre)
													float* Qcom, // Qcom(nQth)
													float* Gcom, // Gcom(nvel,nQth)
													float* Vcom, // Vcom(nvel))
													float* fparams,
													int* intparams,
													int* ierr
													)
{
	printf("sizeof ushort = %i\n",sizeof(ushort));
	int my_device;
	cudaGetDevice(&my_device);
	printf("My device is %i\n",my_device);
	Mesh_data Mesh_d;
	Mesh_data* Mesh = (Mesh_data*)malloc(sizeof(Mesh_data));

	int nr,nth,npsi;
	int nrpre,ntpre,nppre;
	int nQth,nvel;

	float pi = 3.1415927;

	nr = intparams[10]+1;
	nth = intparams[11]+1;
	npsi = intparams[12]+1;

	nrpre = intparams[13];
	ntpre = intparams[14];
	nppre = intparams[15];

	nQth = intparams[17];
	nvel = intparams[16];

	Mesh_d.nr = intparams[4];
	Mesh_d.nth = intparams[5];
	Mesh_d.npsi = intparams[6];
	Mesh_d.nrused = intparams[7];
	Mesh_d.nthused = intparams[8];
	Mesh_d.npsiused = intparams[9];
	Mesh_d.nrfull = nr-1;
	Mesh_d.nthfull = nth-1;
	Mesh_d.npsifull = npsi-1;
	Mesh_d.nrpre = nrpre;
	Mesh_d.ntpre = ntpre;
	Mesh_d.nppre = nppre;
	Mesh_d.nQth = nQth;
	Mesh_d.nvel = nvel;

	Mesh_d.lat0 = intparams[18];
	Mesh_d.lap0 = intparams[19];

	Mesh_d.rfac = fparams[0];
	Mesh_d.tfac = fparams[1];
	Mesh_d.pfac = fparams[2];

	Mesh_d.debyelen = fparams[3];

	Mesh_d.bdyfc = fparams[4];
	Mesh_d.Ti = fparams[5];
	Mesh_d.vd = fparams[6];
	Mesh_d.cd = fparams[7];
	Mesh_d.cB = fparams[8];
	Mesh_d.Bz = fparams[9];

	Mesh_d.dp = 2.0*pi/Mesh_d.npsi;
	Mesh_d.dth = pi/(Mesh_d.nth-1);
	Mesh_d.dpinv = Mesh_d.npsi/2.0/pi;
	Mesh_d.dthinv = (Mesh_d.nth-1)/pi;
	Mesh_d.dpsi = pcc[2]-pcc[1];

	//printf("Mesh parameters = %i, %i, %i, %i, %i, %i, %i, %i\n",Mesh_d.nr,Mesh_d.nth,Mesh_d.npsi,nrpre,ntpre,nppre,nQth,nvel);
	//printf("dims = %i, %i, %i\n",nr,nth,npsi);


/*
	// Populate host_commons mesh variables
	Mesh_h.phi = phi;
	Mesh_h.phiaxis = phiaxis;
	Mesh_h.rho = rho;
	Mesh_h.rhoDiag = rhoDiag;
	Mesh_h.rmesh = rmesh;
	Mesh_h.rccmesh = rccmesh;
	Mesh_h.thmesh = thmesh;
	Mesh_h.tccmesh = tccmesh;
	Mesh_h.thang = thang;
	Mesh_h.pcc = pcc;
	Mesh_h.volinv = volinv;
	Mesh_h.zeta = zeta;
	Mesh_h.zetahalf = zetahalf;
	Mesh_h.irpre = irpre;
	Mesh_h.itpre = itpre;
	Mesh_h.ippre = ippre;
	Mesh_h.Qcom = Qcom;
	Mesh_h.Gcom = Gcom;
	Mesh_h.Vcom = Vcom;
*/
	/*
	 * For the GPU mesh data:
	 * Most of these are only for general storage and latter reference.
	 * Currently all of the data is wrapped in a cudaMatrix so that I can use a template structure for
	 * both the host and device data. This should make it a lot easier to make changes or additions.
	 *
	 * I think the best approach is going to be to make a new object that can store data in either a
	 * cudaMatrix or a texture bound cudaArray. I need to determine whether or not we would have
	 * to rebind the cudaArray to a texture reference every time step. If we do, then it probably won't
	 * be worth it to use textures.
	 */

	// Allocate gpu memory
	Mesh_d.phi.cudaMatrix_allocate(nr,nth,npsi);
	Mesh_d.phiaxis.cudaMatrix_allocate(nr,2,npsi);
	Mesh_d.rho.cudaMatrix_allocate(nr,nth,npsi);
	Mesh_d.rhoDiag.cudaMatrix_allocate(nr,nth,npsi);
	Mesh_d.rmesh.cudaMatrix_allocate(nr,1,1);
	Mesh_d.rccmesh.cudaMatrix_allocate(nr,1,1);
	Mesh_d.thmesh.cudaMatrix_allocate(nth,1,1);
	Mesh_d.tccmesh.cudaMatrix_allocate(nth,1,1);
	Mesh_d.thang.cudaMatrix_allocate(nth,1,1);
	Mesh_d.pcc.cudaMatrix_allocate(npsi,1,1);
	Mesh_d.volinv.cudaMatrix_allocate(nr,1,1);
	Mesh_d.zeta.cudaMatrix_allocate(nr+1,1,1);
	Mesh_d.zetahalf.cudaMatrix_allocate(nr+1,1,1);
	Mesh_d.irpre.cudaMatrix_allocate(nrpre,1,1);
	Mesh_d.itpre.cudaMatrix_allocate(ntpre,1,1);
	Mesh_d.ippre.cudaMatrix_allocate(nppre,1,1);
	Mesh_d.Qcom.cudaMatrix_allocate(nQth,1,1);
	Mesh_d.Gcom.cudaMatrix_allocate(nvel,nQth,1);
	Mesh_d.Vcom.cudaMatrix_allocate(nvel,1,1);

	Mesh_d.psum.cudaMatrix_allocate(nr-2,nth-2,npsi-2);

	Mesh_d.nincell.cudaMatrix_allocate(nth-1,npsi-1,1);
	Mesh_d.vrincell.cudaMatrix_allocate(nth-1,npsi-1,1);
	Mesh_d.vr2incell.cudaMatrix_allocate(nth-1,npsi-1,1);

	Mesh_d.nbins = ((Mesh_d.nr)*(Mesh_d.nth)*(Mesh_d.npsi)+8*8*8-1)/(8*8*8);
	CUDA_SAFE_CALL(cudaMalloc((void**)&(Mesh_d.bins),(Mesh_d.nbins+8*8*8)*sizeof(Particlebin)));

	// Copy Mesh data to the GPU
	Mesh_d.rmesh.cudaMatrixcpy(rmesh,cudaMemcpyHostToDevice);
	Mesh_d.rccmesh.cudaMatrixcpy(rccmesh,cudaMemcpyHostToDevice);
	Mesh_d.thmesh.cudaMatrixcpy(thmesh,cudaMemcpyHostToDevice);
	Mesh_d.tccmesh.cudaMatrixcpy(tccmesh,cudaMemcpyHostToDevice);
	Mesh_d.thang.cudaMatrixcpy(thang,cudaMemcpyHostToDevice);
	Mesh_d.pcc.cudaMatrixcpy(pcc,cudaMemcpyHostToDevice);
	Mesh_d.volinv.cudaMatrixcpy(volinv,cudaMemcpyHostToDevice);
	Mesh_d.zeta.cudaMatrixcpy(zeta,cudaMemcpyHostToDevice);
	Mesh_d.zetahalf.cudaMatrixcpy(zetahalf,cudaMemcpyHostToDevice);
	Mesh_d.irpre.cudaMatrixcpy(irpre,cudaMemcpyHostToDevice);
	Mesh_d.itpre.cudaMatrixcpy(itpre,cudaMemcpyHostToDevice);
	Mesh_d.ippre.cudaMatrixcpy(ippre,cudaMemcpyHostToDevice);
	Mesh_d.Qcom.cudaMatrixcpy(Qcom,cudaMemcpyHostToDevice);
	Mesh_d.Gcom.cudaMatrixcpy(Gcom,cudaMemcpyHostToDevice);
	Mesh_d.Vcom.cudaMatrixcpy(Vcom,cudaMemcpyHostToDevice);




	// Store the pointer to the mesh data and wrap it as an int
	*Mesh = Mesh_d;

	*Mesh_ptr = (long int)Mesh;


}


extern "C" void gpu_particle_list_init_(long int* particles_out,int* nptcls)
{

	printf("Setting up particle list\n");
	XPlist* particles = (XPlist*)malloc(sizeof(XPlist));

	particles->allocate(*nptcls);

	*particles_out = (long int)particles;
	printf("Finished setting up particle list\n");

	size_t free = 0;
	size_t total = 0;
	// See how much memory is allocated / free
	cudaMemGetInfo(&free,&total);
	printf("Free Memory = %i mb\nUsed mememory = %i mb\n",(int)(free)/(1<<20),(int)(total-free)/(1<<20));

	printf("Setting up Mesh Arrays\n");

}

extern "C" void gpu_Diagnostics_init()
{

}

extern "C" void gpu_Distribution_fnct_init()
{

}
















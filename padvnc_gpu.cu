#include "cudamatrix_types.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <math.h>
#include "cuPrintf.cu"
#include <ctime>
#include <cstring>
#include "cuda.h"
#include "cutil.h"
#include "cuda_runtime.h"
#include "common_functions.h"
#include "sm_20_intrinsics.h"
#include "host_defines.h"
#include <iostream>
#include "math.h"




/*
float* xp, used in advancing.f, chargefield.f, maxreinject.f,
float* vzinit,
float* dtprec,
int* ipf,
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
float* irpre, // irpre(nrpre)
float* itpre, // itpre(ntpre)
float* ippre, // ippre(nppre)
float* Qcom, // Qcom(nQth)
float* Gcom, // Gcom(nvel,nQth)
float* Vcom, // Vcom(nvel)
float* nvdiag, // nvdiag(nvmax)
float* nvdiagave, // nvdiagave(nvmax)
float* vdiag, // vdiag(nvmax)
float* vrdiagin, // vrdiagin(nvmax)
float* vtdiagin, // vtdiagin(nvmax)
float* nincellstep, // nincellstep(nthsize,npsisize,nstepmax)
float* vrincellstep, // vrincellstep(nthsize,npsisize,nstepmax)
float* vr2incellstep, // vr2incellstep(nthsize,npsisize,nstepmax)
float* nincell, // nincell(nthsize,npsisize)
float* vrincell, // vrincell(nthsize,npsisize)
float* vr2incell, // vr2incell(nthsize,npsisize)
float* fincellave, // fincellave(nthsize,npsisize)
float* vrincellave, // vrincellave(nthsize,npsisize)
float* vr2incellave, // vr2incellave(nthsize,npsisize)
float* xorbit, // xorbit(nstepmax,nobsmax)
float* yorbit, // yorbit(nstepmax,nobsmax)
float* zorbit, // zorbit(nstepmax,nobsmax)
float* vxorbit, // vxorbit(nstepmax,nobsmax)
float* vyorbit, // vyorbit(nstepmax,nobsmax)
float* vzorbit, // vzorbit(nstepmax,nobsmax)
float* rorbit, // rorbit(nstepmax,nobsmax)
float* iorbitlen, // iorbitlen(nobsmax)
float* psum, // psum(nrsize-1,nthsize-1,npsisize-1)
float* vrsum, // the following have the same dimensions as above
float* vtsum,
float* vpsum,
float* vr2sum,
float* vt2sum,
float* vp2sum,
float* vrtsum,
float* vrpsum,
float* vtpsum,
float* vxsum,
float* vysum,

*/


extern "C" void padvnc_gpu_(float* xp,
										float* vzinit,
										float* dtprec,
										int* ipf,
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
										float* irpre, // irpre(nrpre)
										float* itpre, // itpre(ntpre)
										float* ippre, // ippre(nppre)
										float* Qcom, // Qcom(nQth)
										float* Gcom, // Gcom(nvel,nQth)
										float* Vcom, // Vcom(nvel)
										float* nvdiag, // nvdiag(nvmax)
										float* nvdiagave, // nvdiagave(nvmax)
										float* vdiag, // vdiag(nvmax)
										float* vrdiagin, // vrdiagin(nvmax)
										float* vtdiagin, // vtdiagin(nvmax)
										float* nincellstep, // nincellstep(nthsize,npsisize,nstepmax)
										float* vrincellstep, // vrincellstep(nthsize,npsisize,nstepmax)
										float* vr2incellstep, // vr2incellstep(nthsize,npsisize,nstepmax)
										float* nincell, // nincell(nthsize,npsisize)
										float* vrincell, // vrincell(nthsize,npsisize)
										float* vr2incell, // vr2incell(nthsize,npsisize)
										float* fincellave, // fincellave(nthsize,npsisize)
										float* vrincellave, // vrincellave(nthsize,npsisize)
										float* vr2incellave, // vr2incellave(nthsize,npsisize)
										float* xorbit, // xorbit(nstepmax,nobsmax)
										float* yorbit, // yorbit(nstepmax,nobsmax)
										float* zorbit, // zorbit(nstepmax,nobsmax)
										float* vxorbit, // vxorbit(nstepmax,nobsmax)
										float* vyorbit, // vyorbit(nstepmax,nobsmax)
										float* vzorbit, // vzorbit(nstepmax,nobsmax)
										float* rorbit, // rorbit(nstepmax,nobsmax)
										float* iorbitlen, // iorbitlen(nobsmax)
										float* psum, // psum(nrsize-1,nthsize-1,npsisize-1)
										float* vrsum, // the following have the same dimensions as above
										float* vtsum,
										float* vpsum,
										float* vr2sum,
										float* vt2sum,
										float* vp2sum,
										float* vrtsum,
										float* vrpsum,
										float* vtpsum,
										float* vxsum,
										float* vysum,
										)
{

}








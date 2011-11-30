
c Advance the particles
      subroutine padvnc(GPUXPlist,GPUMesh,
     $   		dtin,icolntype,colnwt,step,maccel,ierad)

			implicit none

			integer*8 GPUXPlist ! dummy variable to carry pointer to XPdata
      integer*8 GPUMesh ! dummy variable to carry pointer to mesh_data
      integer step,maccel,icolntype,ierad
      real dt,dtin,colnwt
c Common data:
      include 'piccom.f'
      include 'errcom.f'
      include 'colncom.f'




      if(step.eq.1)then
         ilastgen = 1
         icurrreinject = npreinject
      endif

      call orbitreinjectgen(xpreinject,npreinject,icurrreinject,
     $         ilastgen,dtin)

      if(step.eq.1)then
         ilastgen = 1
         icurrreinject = 1
      endif

      !call test_gpu_getaccel(GPUXPlist,GPUMesh,phi)





			if(gpurun)then
			  icurrreinject = icurrreinject  - 1
c Call the GPU particle advance
				call gpu_padvnc(GPUXPlist,GPUMesh,phi,xpreinject,
     $				dtin,icurrreinject,
     $				nincell,vrincell,vr2incell,
     $				xmout,ymout,zmout,
     $				xmomprobe,ymomprobe,zmomprobe,
     $				enerprobe,ninner)
				nrein = icurrreinject+1
				icurrreinject = icurrreinject + 1

			else

      	 call padvnc2(xp,vzinit,dtprec,ipf,
     $      phi,phiaxis,rho,rhoDiag,
     $      r,rcc,th,thang,pcc,irpre,itpre,ippre
     $     ,zeta,zetahalf
     $     ,nvdiag,vrdiagin,vtdiagin,nincell,vrincell,vr2incell
     $     ,iorbitlen,xorbit,yorbit,zorbit,rorbit
     $     ,vxorbit,vyorbit,vzorbit
     $     ,rfac,tfac,pfac
     $     ,bcr,bdyfc,debyelen,Bz,cB,cd,Ti,vneutral
     $     ,collcic,LCIC,ldist,lsubcycle,verlet
     $     ,lap0,lat0
     $     ,orbinit
     $     ,zmomprobe,xmomprobe,ymomprobe
     $     ,xmout,ymout,zmout
     $     ,enerprobe,fluxrein,spotrein,vrange
     $     ,npartmax,ndim,ninjcomp,npart,np,ntrapre
     $     ,nr,nth,npsi,nrsize,nthsize,npsisize,nrpre,ntpre,nppre
     $     ,nrfull,npsiused
     $     ,nvmax,nobsmax,nstepmax,nrein,nreintry,ninner,norbits
     $     ,dtin,icolntype,colnwt,step,maccel,ierad)

      endif



			!call test_gpu_padvnc(GPUXPlist,GPUMesh,xp,phi,xpreinject,dtin)
c			print*,'Fortran reinjected = ',icurrreinject



      end








































































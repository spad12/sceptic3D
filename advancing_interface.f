
c Advance the particles
      subroutine padvnc(GPUXPlist,GPUMesh,
     $   		dtin,icolntype,colnwt,step,maccel,ierad)

      integer step,maccel
      real dt,dtin
c Common data:
      include 'piccom.f'
      include 'errcom.f'
      include 'colncom.f'

      integer intparams(30)

      real fparams(10)

c Integer parameters that are in common blocks and need to be passed
      intparams(1) = npartmax
      intparams(2) = npart
      intparams(3) = ndim
      intparams(4) = np

c Grid dimensions
      intparams(5) = nr
      intparams(6) = nth
      intparams(7) = npsi
      intparams(8) = nrused
      intparams(9) = nthused
      intparams(10) = npsiused
      intparams(11) = nrfull
      intparams(12) = nthfull
      intparams(13) = npsifull

      intparams(8) = nrpre
      intparams(9) = ntpre
      intparams(10) = nppre
      intparams(11) = nvel
      intparams(12) = nQth

      fparams(1) = rfac
      fparams(2) = tfac
      fparams(3) = pfac




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



			call start_timer(t1)

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

			call stop_timer(t1)

			call test_gpu_padvnc(GPUXPlist,GPUMesh,xp,phi,xpreinject,dtin)

			!print*,'Fortran reinjected = ',icurrreinject
      end








































































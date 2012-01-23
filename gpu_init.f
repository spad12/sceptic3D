			subroutine initialize_gpu(GPUXPlist,GPUMesh,myid2)

c Common data:
      include 'piccom.f'
      include 'errcom.f'
      include 'colncom.f'

			integer*8 GPUXPlist
			integer*8 GPUMesh

			integer intparams(30)
			integer ierr
			integer lat0t,lap0t

      real fparams(10)

c	 We need to convert Fortran bool to an integer
			lat0t = 0
			lap0t = 0

			if(lat0) lat0t = 1
			if(lap0) lap0t = 1


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
      intparams(11) = nrsize
      intparams(12) = nthsize
      intparams(13) = npsisize
      intparams(14) = nrpre
      intparams(15) = ntpre
      intparams(16) = nppre
      intparams(17) = nvel
      intparams(18) = nQth
      intparams(19) = lat0t
      intparams(20) = lap0t

      !print*, "dims = ",nrsize,nthsize,npsisize

      fparams(1) = rfac
      fparams(2) = tfac
      fparams(3) = pfac
      fparams(4) = debyelen

      fparams(5) = bdyfc
      fparams(6) = Ti
      fparams(7) = vd
      fparams(8) = cd
      fparams(9) = cB
      fparams(10) = Bz

      test_atimes = 1
      test_atimesm = 1

			icg3dcall = 1

      call gpu_mesh_init(GPUMesh,phi,phiaxis,rho,rhoDiag,
     $						r,rcc,th,tcc,thang,pcc,volinv,zeta,zetahalf,
     $			irpre,itpre,ippre,Qcom,Gcom,Vcom,
     $					fparams,intparams,ierr)

			if(myid2.eq.0) then
           call gpu_psolver_init(GPUPsolve,GPUMesh,apc,bpc,cpc,dpc,
     $						epc,fpc,gpc,nrsize,nthsize,npsisize)
      endif

      call gpu_particle_list_init(GPUXPlist,npart)



			end








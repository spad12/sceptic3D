

      subroutine chargetomesh_gpu(GPUXPlist,GPUMesh,istep)

      implicit none
c Common data:
      include 'piccom.f'
      include 'errcom.f'

c      argument list
      integer*8 GPUXPlist ! dummy variable to carry pointer to XPdata
      integer*8 GPUMesh ! dummy variable to carry pointer to mesh_data
      integer istep
      real timeout
      integer timer,timer2,timer3




			if(gpurun) then
				if(istep.eq.1) then
C Transpose the Particle Array to the gpu
      		call XPlist_transpose(GPUXPlist,
     $      xp,dtprec,vzinit,ipf,npartmax,ndim,0,0)
				endif

				call start_timer(timer)

C Sort the particle list
					call start_timer(timer2)
					call XPlist_sort(GPUXPlist,GPUMesh,istep)
 				  call stop_timer(sortt,timer2)



C Transpose the sorted Particle Array back to the CPU
c				call XPlist_transpose(GPUXPlist,xp
c     $			,dtprec,vzinit,ipf,npartmax,ndim,1,0)

     		!call test_gpu_ptomesh(GPUXPlist,GPUMesh)

C Charge Assign
				call start_timer(timer3)
				call gpu_chargeassign(GPUXPlist,GPUMesh,psum)
				call stop_timer(chasst,timer3)
				!call test_chargetomesh(GPUXPlist,GPUMesh,psum)
 				call stop_timer(c2mesht,timer)
			else

c Assign charge to mesh
				call start_timer(timer)
      	call chargetomesh(xp,ipf,
     $      r,th,pcc,irpre,itpre,ippre,
     $      zeta,zetahalf,
     $      psum,vrsum,vr2sum,vtsum,vpsum,
     $      vt2sum,vp2sum,vrtsum,vrpsum,
     $      vtpsum,vxsum,vysum,vzsum,
     $      rfac,tfac,pfac,
     $      debyelen,
     $      diags,samp,iocprev,
     $      npartmax,ndim,
     $      nr,nth,npsi,
     $      nrsize,nthsize,npsisize,
     $      nrpre,ntpre,nppre,nrused,nthused,npsiused)

     	call stop_timer(c2mesht,timer)

      endif




      end



































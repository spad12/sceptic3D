

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
      integer timer

      call start_timer(timer)


			if(gpurun) then
				if(istep.eq.1) then
C Transpose the Particle Array to the gpu
      		call XPlist_transpose(GPUXPlist,
     $      xp,dtprec,vzinit,ipf,npartmax,ndim,0,0)
				endif

C Sort the particle list

					call XPlist_sort(GPUXPlist,GPUMesh,istep)




C Transpose the sorted Particle Array back to the CPU
c				call XPlist_transpose(GPUXPlist,xp
c     $			,dtprec,vzinit,ipf,npartmax,ndim,1,0)

     		!call test_gpu_ptomesh(GPUXPlist,GPUMesh)

C Charge Assign
				call gpu_chargeassign(GPUXPlist,GPUMesh,psum)

				!call test_chargetomesh(GPUXPlist,GPUMesh,psum)

			else

c Assign charge to mesh
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

      endif


      call stop_timer(c2mesht,timer)

      end



































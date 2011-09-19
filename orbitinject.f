
c***********************************************************************
c General version allows choice of reinjection scheme.
c***********************************************************************
      subroutine reinject(i,dt)

c Common data:
      include 'piccom.f'
      include 'errcom.f'

      integer j

      if(icurrreinject.gt.npreinject)then
         print*,'REINJECT icurr too big'
         icurrreinject = 1
      endif


      do j=1,6
         xp(j,i) = xpreinject(j,icurrreinject)
      enddo

      icurrreinject = icurrreinject + 1

c      call maxreinject(xp,npartmax,i,dt)


      end
c***********************************************************************
      subroutine injinit(icolntype,bcr)

      integer bcr

c      if(bcr.ne.0) then
c Injection from a maxwellian at boundary?
         call maxinjinit(bcr)
c      elseif(icolntype.eq.1.or.icolntype.eq.5) then
c Injection from fv distribution at the boundary.
c         call fvinjinit(icolntype)
c      elseif(icolntype.eq.2.or.icolntype.eq.6)then
c Injection from a general gyrotropic distribution at infinity
c         call ogeninjinit(icolntype)
c      else
c Injection from a shifted maxwellian at infinity
c         call oinjinit()
c      endif
      end

      subroutine orbitreinjectgen(xpreinject,npreinject,icurr,dt)

      real xpreinject(6,npreinject)

      if(icurr.gt.1)then

         do i=1,icurr
            call maxreinject(xpreinject,npreinject,i,dt)
         enddo

         icurr = 1

      endif

      end



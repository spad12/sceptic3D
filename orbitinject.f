
c***********************************************************************
c General version allows choice of reinjection scheme.
c***********************************************************************
      subroutine reinject(i)

      implicit none
c Common data:
      include 'piccom.f'
      include 'errcom.f'

      integer j,i
      real dt

      if(icurrreinject.gt.npreinject)then
         icurrreinject = 1
      endif


      do j=1,6
         xp(j,i) = xpreinject(j,icurrreinject)
      enddo

      nrein=nrein+1
      icurrreinject=icurrreinject+1

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


      subroutine orbitreinjectgen(xpreinject,npreinject,icurr,
     $      ilastgen,dt)

      implicit none

      integer npreinject,icurr,ilastgen
      real dt
      real xpreinject(6,npreinject)
      integer i,j


      do i=1,icurr
         call maxreinject(xpreinject,npreinject,i,dt)
      enddo

      icurr = 1

      end


      subroutine orbitreinjectgen2(xpreinject,npreinject,icurr,
     $      ilastgen,dt)

      implicit none

      integer npreinject,icurr,ilastgen
      real dt
      real xpreinject(6,npreinject)
      integer i,j


      if(((icurr-1)-ilastgen).lt.0) then

         if(ilastgen.lt.npreinject) then
            do i=ilastgen,npreinject
               call maxreinject(xpreinject,npreinject,i,dt)
            enddo
         endif

         if((icurr-1).gt.0) then
            do i=1,(icurr-1)
               call maxreinject(xpreinject,npreinject,i,dt)
            enddo
         endif


         ilastgen = icurr-1
      else if(((icurr-1)-(ilastgen)).gt.0) then

         do i=ilastgen,(icurr-1)
            call maxreinject(xpreinject,npreinject,i,dt)
         enddo


         ilastgen = icurr-1

      endif

      end



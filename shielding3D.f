c___________________________________________________________________________
c
c     SCEPTIC3D
c
c     This code is copyright (c)
c              Ian H Hutchinson    hutch@psfc.mit.edu.
c              Leonardo Patacchini patacchi@mit.edu
c
c     It may be used freely with the stipulation that any scientific or
c     scholarly publication concerning work that uses the code must give
c     an acknowledgement referring to the relevant papers
c
c     I.H. Hutchinson, Plasma Physics and Controlled Fusion, vol 44, p
c     1953 (2002), vol 45, p 1477 (2003).
c
c     L. Patacchini and I.H. Hutchinson, Plasma Physics and Controlled
c     Fusion, vol 49, p1193 (2007), vol 49, p 1719 (2007).
c
c     I.H. Hutchinson and L. Patacchini, Physics of Plasmas, vol 14,
c     p013505 (2007)
c
c     The code may not be redistributed except in its original package.
c
c     No warranty, explicit or implied, is given. If you choose to build
c     or run the code, you do so at your own risk.
c___________________________________________________________________________

c******************************************************************
      subroutine shielding3D(dt,n1,dconverge,niter,maxdphi)

      include 'piccom.f'
      real dt,dconverge,maxdphi
      integer maxits,niter
      integer n1
      real b(nrsize-1,0:nthsize,0:npsisize)
     $     ,x(nrsize-1,0:nthsize,0:npsisize)
      integer kk1,kk2

      maxits=2*(nrused*nthused*npsiused)**0.333

c Set the potential on axis to what the solver found at the previous timestep
      do i=2,n1
         do k=1,npsiused
            phi(i,1,k)=phiaxis(i,1,k)
            phi(i,nthused,k)=phiaxis(i,2,k)
         enddo
      enddo
         
      do k=1,npsiused
         do j=1,nthused
            do i=3,n1-1
               b(i,j,k)=exp(phi(i,j,k))*(1-phi(i,j,k))-rho(i,j,k)
               x(i,j,k)=phi(i,j,k)
            enddo
            
c Dirichlet part of the outer boundary condition
            i=n1
            b(i,j,k)=exp(phi(i,j,k))*(1-phi(i,j,k))-rho(i,j,k) 
     $           -apc(i)*gpc(j,k,4)  
            x(i,j,k)=phi(i,j,k)
            
c Inner boundary condition (Dirichlet since the probe potential has
c already been calculated in innerbc.f
            i=2
            b(i,j,k)=exp(phi(i,j,k))*(1-phi(i,j,k))-rho(i,j,k)-bpc(i)
     $           *phi(1,j,k)
            x(i,j,k)=phi(i,j,k)
            x(1,j,k)=0.
         enddo
      enddo


      call cg3D(n1,nthused,npsiused,b,x,dconverge,iter,maxits)


      maxdphi=0.
      do k=1,npsiused
         do j=1,nthused
            do i=2,n1
               maxdphi=max(maxdphi,abs(phi(i,j,k)-x(i,j,k)))
               phi(i,j,k)=x(i,j,k)
            enddo
         enddo
      enddo
      
c Output the number of iterations
c     In fact, have moved this output to Sceptic3D.F
c      write(*,'('':'',i3,$)')iter
      niter=iter

c     We set the potential on the inner shadow cell by second order
c     extrapolation from the potential at i=1,2,3
      do j=1,nthused
         do k=1,npsiused
c            phi(0,j,k)=2*phi(1,j,k)-phi(2,j,k) Only 1st order
            phi(0,j,k)=2.5*phi(1,j,k)-2*phi(2,j,k)+0.5*phi(3,j,k)
         enddo
      enddo
      


c     We must average the potential of each psi-cell at theta=0 or
c     theta=pi. The boundary conditions assume the cell center is on
c     axis, while what is indeed calculated is the potential at the
c     center of the first/last theta-cells.

      do i=0,nrused
         psiave1=0.
         psiave2=0.
         do k=1,npsiused
            psiave1=psiave1+phi(i,1,k)
            psiave2=psiave2+phi(i,nthused,k)
         enddo
         psiave1=psiave1/npsiused
         psiave2=psiave2/npsiused
         do k=1,npsiused
            phiaxis(i,1,k)=phi(i,1,k)
            phiaxis(i,2,k)=phi(i,nthused,k)
            phi(i,1,k)=psiave1
            phi(i,nthused,k)=psiave2
         enddo
      enddo

c We must set the potential of the shadow theta-cells to the
c physical value of the potential at psi+pi. The zero-derivative
c condition is not valid anymore in the 3D case
      do k=1,npsiused
         kk1=mod(k+3*npsi/2-1,npsi)+1
         kk2=mod(k+(3*npsi+1)/2-1,npsi)+1
         do i=0,nrused
            phi(i,0,k)=0.5*(phi(i,2,kk1)+phi(i,2,kk2))
            phi(i,nthused+1,k)=
     $           0.5*(phi(i,nthused-1,kk1)+phi(i,nthused-1,kk2))
         enddo
      enddo
         
c Set the psi shadow cells to their proper value to ensure periodicity
      do j=0,nthused+1
         do i=0,nrused
            phi(i,j,npsiused+1)=phi(i,j,1)
            phi(i,j,0)=phi(i,j,npsiused)
         enddo
      enddo
         

      end
      
c******************************************************************
c******************************************************************
      subroutine cg3D(n1,n2,n3,b,x,tol,iter,itmax)

c     Subroutine to solve Ax=b, where b is an array of dimensions
c     n1*n2*n3, considered here as a 3D matrix with dimensions
c     n1,n2,n3. A is the stiffness matrix of the Poisson solver, and x
c     is the potential. Because A is symmetric but not necessarily
c     definite positive, we use the minimum residual variant of the
c     algorithm

      include 'piccom.f'

      integer itmax,iter
      real eps,delta,deltamax
      parameter (eps=1.e-14)
      real b(nrsize-1,0:nthsize,0:npsisize),x(nrsize-1,0:nthsize
     $     ,0:npsisize),p(nrsize-1,0:nthsize,0:npsisize),res(nrsize-1
     $     ,0:nthsize ,0:npsisize),z(nrsize-1,0:nthsize,0:npsisize)
     $     ,pp(nrsize-1,0:nthsize ,0:npsisize),resr(nrsize-1,0:nthsize
     $     ,0:npsisize),zz(nrsize-1 ,0:nthsize,0:npsisize)
      integer n1,n2,n3
      real tol
      real bknum,bkden,aknum,akden

      iter=0
c Initialize the denominators to avoid warnings at compilation
      bkden=0
      akden=0

c     Calculate initial residual r=b-Ax, where x is the potential at the
c     previous time-step. With the conjugate gradient method, the first
c     search direction is the first residual


      call atimes(n1,n2,n3,x,res)
      
      do k=1,n3
         do j=1,n2
            do i=2,n1
               res(i,j,k)=b(i,j,k)-res(i,j,k)
            enddo
c     Inner bc lies in the rhs of poisson's equation (b)
            res(1,j,k)=0.
         enddo
      enddo

      call atimes(n1,n2,n3,res,resr)
      

      call asolve(n1,n2,n3,res,z,error0)

      

c     Main loop
 100  if(iter.lt.itmax) then
         iter=iter+1

         call asolve(n1,n2,n3,resr,zz,error)
         bknum=0.
         
         do k=1,n3
            do j=1,n2
               do i=2,n1
                  bknum=bknum+z(i,j,k)*resr(i,j,k)
               enddo
            enddo
         enddo

      

         if(iter.eq.1) then
            do k=1,n3
               do j=1,n2
                  do i=2,n1
                     p(i,j,k)=z(i,j,k)
                     pp(i,j,k)=zz(i,j,k)
                  enddo
                  p(1,j,k)=0.
                  pp(i,j,k)=0.
               enddo
            enddo
         else
            bk=bknum/bkden
            do k=1,n3
               do j=1,n2
                  do i=2,n1
                     p(i,j,k)=bk*p(i,j,k)+z(i,j,k)
                     pp(i,j,k)=bk*pp(i,j,k)+zz(i,j,k)
                  enddo
                  p(1,j,k)=0.
                  pp(1,j,k)=0.
               enddo
            enddo
         endif
         
         bkden=bknum
         call atimes(n1,n2,n3,p,z)
         akden=0.
         do k=1,n3
            do j=1,n2
               do i=2,n1
                  akden=akden+z(i,j,k)*pp(i,j,k)
               enddo
            enddo
         enddo
         ak=bknum/akden
         call atimes(n1,n2,n3,pp,zz)
         
         deltamax=0.
         do k=1,n3
            do j=1,n2
               do i=2,n1
                  delta=ak*p(i,j,k)
                  x(i,j,k)=x(i,j,k)+delta
                  if(abs(delta).gt.deltamax) then
                     deltamax=abs(delta)
                  endif
                  res(i,j,k)=res(i,j,k)-ak*z(i,j,k)
                  resr(i,j,k)=resr(i,j,k)-ak*zz(i,j,k)
               enddo
            enddo
         enddo

         if(deltamax.ge.tol) then
            call asolve(n1,n2,n3,res,z,error)
c            write(*,*) iter,deltamax,error
            goto 100
         endif

      endif


c The iteration number is larger than itmax, therefore leave the solver
      return
      
      end 


c **************************************

      subroutine asolve(n1,n2,n3,b,z,error)

c     Preconditioning subroutine. if Atilde is the preconditioning
c     matrix, returns z=Atilde^-1*b.

      include 'piccom.f'

      real b(nrsize-1,0:nthsize,0:npsisize), z(nrsize-1,0:nthsize
     $     ,0:npsisize)
      real error
      integer n1,n2,n3

      error=0.
      
      do k=1,n3
         do j=1,n2
            do i=2,n1-1
               z(i,j,k)=b(i,j,k)/(-fpc(i,j)-exp(phi(i,j,k)))
               error=error+b(i,j,k)**2
            enddo
         enddo
      enddo

      i=n1
      
      do k=1,n3
         do j=1,n2
            z(i,j,k)=b(i,j,k)/(-fpc(i,j)-exp(phi(i,j,k))
     $           +apc(i)*gpc(j,k,5))
            error=error+b(i,j,k)**2
         enddo
      enddo
         

      error=sqrt(error)
      end

c **************************************

      subroutine atimes(n1,n2,n3,x,res)

      include 'piccom.f'
c Outputs res=Ax, where A is the finite volumes stiffness matrix     
      real x(nrsize-1,0:nthsize,0:npsisize), res(nrsize-1
     $     ,0:nthsize ,0:npsisize)
      integer n1,n2,n3

c Bulk iteration
      
      do k=2,n3-1
         do j=1,n2
            do i=2,n1-1
               res(i,j,k)=apc(i)*x(i+1,j,k)+bpc(i)*x(i-1,j,k)+cpc(i,j)
     $              *x(i,j+1,k)+dpc(i,j)*x(i,j-1,k)+epc(i,j)*(x(i,j,k+1)
     $              +x(i,j,k-1))-(fpc(i,j)+exp(phi(i,j,k)))*x(i ,j,k)
            enddo
         enddo
      enddo

      k=1
      do j=1,n2
         do i=2,n1-1
            res(i,j,k)=apc(i)*x(i+1,j,k)+bpc(i)*x(i-1,j,k)+cpc(i,j) *x(i
     $           ,j+1,k)+dpc(i,j)*x(i,j-1,k)+epc(i,j)*(x(i,j,k+1) +x(i,j
     $           ,n3))-(fpc(i,j)+exp(phi(i,j,k)))*x(i,j,k)
         enddo
      enddo
      k=n3
      do j=1,n2
         do i=2,n1-1
            res(i,j,k)=apc(i)*x(i+1,j,k)+bpc(i)*x(i-1,j,k)+cpc(i,j) *x(i
     $           ,j+1,k)+dpc(i,j)*x(i,j-1,k)+epc(i,j)*(x(i,j,1) +x(i,j,k
     $           -1))-(fpc(i,j)+exp(phi(i,j,k)))*x(i,j,k)

         enddo
      enddo

c Outer boundary iteration

      i=n1
      do j=1,n2
         do k=2,n3-1
c The solution x one node further the boundary
            x(i+1,j,k)=gpc(j,k,1)*x(i-1,j,k)+gpc(j,k,2)*x(i,j-1,k)+gpc(j
     $           ,k,3)*x(i,j+1,k)+0*gpc(j,k,4)+gpc(j,k,5)*x(i,j,k)
            
            res(i,j,k)=apc(i)*x(i+1,j,k)+bpc(i)*x(i-1,j,k)+cpc(i,j) *x(i
     $           ,j+1,k)+dpc(i,j)*x(i,j-1,k)+epc(i,j)*(x(i,j,k+1) +x(i,j
     $           ,k-1))-(fpc(i,j)+exp(phi(i,j,k)))*x(i ,j,k)
         enddo
         k=1
         x(i+1,j,k)=gpc(j,k,1)*x(i-1,j,k)+gpc(j,k,2)*x(i,j-1,k)+gpc(j ,k
     $        ,3)*x(i,j+1,k)+0*gpc(j,k,4)+gpc(j,k,5)*x(i,j,k)
         
         res(i,j,k)=apc(i)*x(i+1,j,k)+bpc(i)*x(i-1,j,k)+cpc(i,j) *x(i,j
     $        +1,k)+dpc(i,j)*x(i,j-1,k)+epc(i,j)*(x(i,j,k+1) +x(i,j,n3))
     $        -(fpc(i,j)+exp(phi(i,j,k)))*x(i ,j,k)
         
         k=n3
         x(i+1,j,k)=gpc(j,k,1)*x(i-1,j,k)+gpc(j,k,2)*x(i,j-1,k)+gpc(j ,k
     $        ,3)*x(i,j+1,k)+0*gpc(j,k,4)+gpc(j,k,5)*x(i,j,k)
         
         res(i,j,k)=apc(i)*x(i+1,j,k)+bpc(i)*x(i-1,j,k)+cpc(i,j) *x(i,j
     $        +1,k)+dpc(i,j)*x(i,j-1,k)+epc(i,j)*(x(i,j,1) +x(i,j,k-1))
     $        -(fpc(i,j)+exp(phi(i,j,k)))*x(i,j,k)
         
      enddo


      end

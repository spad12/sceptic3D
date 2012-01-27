
			subroutine inittimes()
			include 'piccom.f'


			sumreducet = 0.0
			c2mesht = 0.0
			padvnct = 0.0
			fcalct = 0.0
			rhocalct =0.0
			totalt = 0.0
			offtime = 0.0
			preducet = 0.0
			chasst = 0.0
			sortt = 0.0
			cg3dt = 0.0
			reinjectt = 0.0


			end

			subroutine adjtimes(adjtime,timein,weight)

			include 'piccom.f'
c#ifdef MPI
      include 'mpif.h'
c#endif
			real temptime
			real weight
			real offset

			offset = offtime/maxsteps


			temptime=timein

			temptime = temptime/maxsteps

			temptime = temptime-offtime

c#ifdef MPI
c Weight each threads time by the number of particles it has to do
			temptime = temptime*weight

      call MPI_REDUCE(temptime,adjtime,1,MPI_REAL,MPI_SUM,0,
     $        MPI_COMM_WORLD,ierr)
c#else
     	adjtime=temptime
c#endif

c Get the time per particle
			adjtime=adjtime/nparttotal
c Get the average time per processor
			adjtime=adjtime
c Convert ms to ns
			adjtime=adjtime*1.0e6

			end

			subroutine output_times(fReinj,dt)
		  include 'piccom.f'

		  logical printtimes
		  logical writetimes
		  integer igpurun
		  integer ntimes

		  real fReinj

		  real times(11)
		  real total_times(11)

		  ntimes = 11


		  printtimes=.false.
		  writetimes=.true.

		  tweight=(1.0*npart)/(1.0*nparttotal)
			tweight = 1.0
			call adjtimes(padvncttot,padvnct,tweight)
			call adjtimes(reinjectttot,reinjectt,tweight)
			call adjtimes(c2meshttot,c2mesht,tweight)
			call adjtimes(chassttot,chasst,tweight)
			call adjtimes(sortttot,sortt,tweight)
			call adjtimes(fcalcttot,fcalct,tweight)
			call adjtimes(cg3dttot,cg3dt,1.0)
			call adjtimes(sumreducettot,sumreducet,tweight)
			call adjtimes(rhocalcttot,rhocalct,tweight)
			call adjtimes(preducettot,preducet,tweight)
			call adjtimes(totalttot,totalt,tweight)

      if(myid.eq.0.and.printtimes) then
				write(*,*)
				write(*,*)'Average Subroutine Run Times(ns) ',
     $				 'per particle per timestep'
      	write(*,*) "Average Padvnc Time : ",padvncttot
      	write(*,*) "Average Reinject Time : ",reinjectttot
      	write(*,*) "Average Chargetomesh Time : ",c2meshttot
      	write(*,*) "Average Sort Time : ",sortttot
      	write(*,*) "Average Chargeassign Time : ",chassttot
      	write(*,*) "Average fcalc Time : ",fcalcttot
      	write(*,*) "Average cg3d Time : ",cg3dttot
      	write(*,*) "Average sumreduce Time : ",sumreducettot
      	write(*,*) "Average rhocalc Time : ",rhocalcttot
      	write(*,*) "Average Total Time : ",totalttot
      endif

 			if(myid.eq.0.and.writetimes) then
      igpurun = 0
      if(gpurun) igpurun = 1



      times(1) = sortt
      times(2) = chasst
      times(3) = c2mesht
      times(4) = sumreducet
      times(5) = rhocalct
      times(6) = cg3dt
      times(7) = fcalct
      times(8) = reinjectt
      times(9) = padvnct
      times(10) = preducet
      times(11) = totalt

      total_times(1) = sortttot
      total_times(2) = chassttot
      total_times(3) = c2meshttot
      total_times(4) = sumreducettot
      total_times(5) = rhocalcttot
      total_times(6) = cg3dttot
      total_times(7) = fcalcttot
      total_times(8) = reinjectttot
      total_times(9) = padvncttot
      total_times(10) = preducettot
      total_times(11) = totalttot


      call output_timing_data(nparttotal,nr,nth,npsi,
     $		maxsteps,dt,rmax,debyelen,Bz,vprobe,
     $		Ti,fReinj,times,total_times,ntimes,runid,
     $		igpurun)

      endif

     	end
































			subroutine inittimes()
			include 'piccom.f'


			sumreducet = 0.0
			c2mesht = 0.0
			padvnct = 0.0
			fcalct = 0.0
			rhocalct =0.0
			totalt = 0.0
			offtime = 0.0


			end

			subroutine adjtimes(adjtime,timein,weight)

			include 'piccom.f'
#ifdef MPI
      include 'mpif.h'
#endif
			real temptime
			real weight
			real offset

			offset = offtime/maxsteps


			temptime=timein

			temptime = temptime/maxsteps

			temptime = temptime-offtime

#ifdef MPI
c Weight each threads time by the number of particles it has to do
			temptime = temptime*weight

      call MPI_REDUCE(temptime,adjtime,1,MPI_REAL,MPI_SUM,0,
     $        MPI_COMM_WORLD,ierr)
#else
     	adjtime=temptime
#endif

c Get the time per particle
			adjtime=adjtime/nparttotal
c Get the average time per processor
			adjtime=adjtime/numprocs
c Convert ms to ns
			adjtime=adjtime*1.0e6

			end

!====================================
! How alltoallv works.
! Each processor sends array
! of size of rank of the receiver
! to the receiver.
!====================================

program alltoallv

 use mpi

 implicit none

 integer, dimension(:), allocatable :: sendcount, reccount, sendbuf, recbuf, send_displ, rec_displ
 integer :: myid, ncpu, err, send_tot, rec_tot
 integer :: i, j, ind


 call MPI_INIT(err)

 call MPI_COMM_SIZE(MPI_COMM_WORLD, ncpu, err)
 call MPI_COMM_RANK(MPI_COMM_WORLD, myid, err)
 myid = myid  + 1




 allocate(sendcount(1:ncpu))
 allocate(reccount(1:ncpu))
 allocate(send_displ(1:ncpu))
 allocate(rec_displ(1:ncpu))
 sendcount = 0; reccount = 0; send_displ = 0; rec_displ = 0;


 do i = 1, ncpu
   sendcount(i) = i
 enddo


 call MPI_ALLTOALL(sendcount, 1, MPI_INT, reccount, 1, MPI_INT, MPI_COMM_WORLD, err)


 ! write(*,*) "ID", myid, "S", sendcount, "R", reccount


 do i = 2, ncpu
   send_displ(i) = send_displ(i-1) + sendcount(i-1)
   rec_displ(i) = rec_displ(i-1) + reccount(i-1)
 enddo

 send_tot = sum(sendcount)
 rec_tot = sum(reccount)
  
 allocate(sendbuf(1:send_tot))
 sendbuf = myid
 allocate(recbuf(1:rec_tot))
 recbuf = 0


 ! write(*,*) "ID", myid, "SD", send_displ, "RD", rec_displ
 

 call MPI_ALLTOALLV(sendbuf,sendcount,send_displ, MPI_INT, recbuf, reccount, rec_displ, MPI_INT, MPI_COMM_WORLD, err)

 write(*,*) "ID", myid, "RECEIVED", recbuf

 call MPI_FINALIZE(err)





end program alltoallv

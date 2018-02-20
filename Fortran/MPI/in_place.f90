!==========================================
! How to use in-place reductions with MPI.
!==========================================



program in_place


  use mpi

  implicit none

  integer :: ierr
  integer :: myid, ncpu
  integer :: buf1, buf2
  


  !---------------------
  ! Initialise
  !---------------------


  call MPI_INIT(ierr)
  call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
  call MPI_COMM_SIZE(MPI_COMM_WORLD, ncpu, ierr)
  myid = myid + 1
  


  !--------------------
  ! ALLREDUCE in place
  !--------------------

  buf1 = myid
  call MPI_ALLREDUCE(MPI_IN_PLACE, buf1, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD, ierr)

  if (myid==1) write(*,*) "Lowest MPI rank is", buf1-1





  !---------------------
  ! REDUCE in place
  !---------------------

  buf2 = myid
  if (myid == 1) then
    ! reduce value at rank 0 (= only for myid 1)
    call MPI_REDUCE(MPI_IN_PLACE, buf2, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD, ierr)
    write(*,*) "Highest MPI rank is", buf2-1
  else
    call MPI_REDUCE(buf2, buf2, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD, ierr)
    write(*,*) "ID", myid, "buf2", buf2, "is unchanged."
  endif


  !-----------------
  ! Finish
  !-----------------

  call MPI_FINALIZE(ierr)



end program in_place

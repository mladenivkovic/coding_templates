!===================================================
! A short demonstration of allgather vs alltoall.
!===================================================

program reducing
  use mpi
  implicit none
  real, dimension(5) :: somearray, somearray_sum
  integer :: error_number, myid, nproc
  integer :: i

  ! initialize MPI
  call mpi_init(error_number)
  
  ! write how many processes are used in nproc
  call mpi_comm_size(mpi_comm_world, nproc, error_number)
  
  !write which process this is in myid
  call mpi_comm_rank(mpi_comm_world, myid, error_number)
  
  if(myid==0) then
    somearray=[(i, i=1, 5)]
  else
    somearray=[(3*i, i=1,5)]
  end if
   

  write(*, '(A5,I3,A6,5F8.3)') "myid", myid, "array", somearray
  
  call MPI_ALLREDUCE(somearray, somearray_sum,size(somearray),MPI_REAL, MPI_SUM, MPI_COMM_WORLD, error_number)
    

  if(myid==0)write(*, '(A5,I3,A6,5F8.3)') "myid", myid, "array", somearray_sum
    ! mpi_allreduce(sendbuf, recvbuf, count, datatype, op, comm, code)
    ! "reduce n_processes of result, then distribute result"
    ! sendbuf:	address of send buffer = partial_pi
    !		= partial pi; what will be sent to be reduced
    ! recbuf: 	address of receive buffer
    !		= total_pi, where it will be reduced to, where the variable will be stored
    ! count:	number of elements in send buffer
    !		= how many elements will you be sending
    ! datatype:	type of elements in send buffer
    !		= what type will be sent. Here: mpi_double_precision
    ! op:	reduce operation
    !		= how will the values be reduced. Here: by sum.
    !		= sum them all together into one.
    ! root:	rank of root process
    !		= which process number is the main process
    ! comm:	communicator
    ! ierror:	error integer
    
    ! possible operations:
    !MPI_SUM     Sum of elements
    !MPI_PROD    Product of elements
    !MPI_MAX     Maximum of elements
    !MPI_MIN     Minimum of elements
    !MPI_MAXLOC  Maximum of elements and location
    !MPI_MINLOC  Minimum of elements and location
    !MPI_LAND    Logical AND
    !MPI_LOR     Logical OR
    !MPI_LXOR    Logical exclusive OR

     call mpi_finalize(error_number)
end program reducing

! This program calculates pi with the MPI reduce operation.
! This is done by integrating  INTEGRAL 4 * 1 / (1 + x^2) FROM 0 TO 1 = 4 * atan(1) = pi.
! It also times the process.



include '../timing_module.f90'

program reducing
  use timing_module
  use mpi
  implicit none
  integer, parameter :: dp = selected_real_kind(selected_real_kind(15, 307))
  real (dp) :: fortran_internal_pi
  real (dp) :: partial_pi
  real (dp) :: total_pi
  real (dp) :: width
  real (dp) :: partial_sum
  real (dp) :: x
  integer :: n
  integer :: this_process
  integer :: n_processes
  integer :: i
  integer :: j
  integer :: error_number

  ! initialize MPI
  call mpi_init(error_number)
  
  ! write how many processes are used in n_processes
  call mpi_comm_size(mpi_comm_world, n_processes, error_number)
  
  !write which process this is in this_process
  call mpi_comm_rank(mpi_comm_world, this_process, error_number)
  
  n = 100000 ! n * width = 1; n = number of separations
  fortran_internal_pi = 4.0_dp*atan(1.0_dp)
  
  if (this_process==0) then
    call start_timing()
    print *, ' fortran_internal_pi = ', fortran_internal_pi
  end if
  
  do j = 1, 5 ! loop to go through n = 100000 * 10 ^ j. So do it for 5 different separations.

    width = 1.0_dp/n
    partial_sum = 0.0_dp
    
    ! calculation for every process: 
    ! Go through loop from your process number to highest n in steps of total process number
    ! Take the value in the middle between two x: x(i + 1/2), where x(i) = width * i, since x_max = n * width
    do i = this_process + 1, n, n_processes
      x = width*(real(i,dp)-0.5_dp)
      partial_sum = partial_sum + f(x)
    end do
    partial_pi = width*partial_sum
    
    
    
    ! mpi_reduce(sendbuf, recvbuf, count, datatype, op, root, comm)
    ! "reduce n_processes of result to only one"
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
    
    call mpi_reduce(partial_pi, total_pi, 1, mpi_double_precision, mpi_sum, 0, mpi_comm_world, error_number)
    if (this_process==0) then
      print 100, n, time_difference()
      print 110, total_pi, abs(total_pi-fortran_internal_pi)
    end if
    
    n = n*10 ! This is about the max that it can do without overflowing
  end do
  
  
  call mpi_finalize(error_number)
100 format (' N intervals = ', i12, ' time = ', f8.3)
110 format (' pi = ', f20.16, /, ' difference = ', f20.16)

contains

  real (dp) function f(x) ! the integral function
    implicit none
    real (dp), intent (in) :: x

    f = 4.0_dp/(1.0_dp+x*x)
  end function f

end program reducing

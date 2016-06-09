! Do not compile with gfortran, but with mpif90 compiler!
! mpif90 helloworld_mpi.f90
! then mpiexec -n 8 ./a.out

program helloworldmpi

    use mpi
    implicit none
    integer :: error_number
    integer :: this_process_number
    integer :: number_of_processes

    ! This must be the first MPI routine called:
    call mpi_init(error_number) 
    ! It sets up the MPI environment.

    ! Typically this is the second MPI routine called. 
    ! All MPI communication is associated with a communicator
    ! that describes the communication context and an associated 
    ! set of processes.
    ! Here we use the default communicator, mpi_comm_world.
    ! The number of processes available is returned via
    ! The second argument.
    call mpi_comm_size(mpi_comm_world, number_of_processes, error_number)

    ! Returns the process number for this process/copy of the
    ! program.
    call mpi_comm_rank(mpi_comm_world, this_process_number, error_number)

    write(*,'(A, I2, A, I2)') " Hello from process ", this_process_number, " of ", number_of_processes, " processes!"


    !last call to the MPI system. 
    call mpi_finalize(error_number)
end program helloworldmpi

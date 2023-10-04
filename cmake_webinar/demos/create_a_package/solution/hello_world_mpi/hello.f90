program main
    use mpi
    implicit none
    integer :: ierr, rank, nRanks

    call MPI_Init(ierr)

    call MPI_Comm_Rank(MPI_COMM_WORLD,rank, ierr)

    print *, "Hello world! from rank", rank, " ! "
    
    call MPI_Finalize(ierr)

end program main

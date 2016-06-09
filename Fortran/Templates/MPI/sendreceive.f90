!!!!!!!!!!!!!
DOESNT WORK!!!!



program sendreceive

    use mpi
    implicit none    
    integer :: error_number
    integer :: this_process_number
    integer :: number_of_processes

    call mpi_init(error_number)
    call mpi_comm_size(mpi_comm_world, number_of_processes, error_number)
    call mpi_comm_rank(mpi_comm_world, this_process_number, error_number)

!call MPI_SENDRECV(sendbuf,sendcount,sendtype, dest,sendtag, recvbuf,recvcount,recvtype,source,recvtag, comm, status, code)


    call mpi_finalize(error_number)
end program sendreceive

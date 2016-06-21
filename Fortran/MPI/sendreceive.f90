program sendreceive

    use mpi
    implicit none
    integer :: error_number
    integer :: this_process_number
    integer :: number_of_processes

    integer :: i
    integer, dimension(mpi_status_size) :: status

    ! initialise MPI 
    call mpi_init(error_number)

    ! specify communicator; 
    call mpi_comm_size(mpi_comm_world, number_of_processes, error_number)

    ! returns process number to variable this_process_number
    call mpi_comm_rank(mpi_comm_world, this_process_number, error_number)
    
    if (this_process_number==0) then
        write(*, *) "Hello from process", this_process_number, " of ", number_of_processes, "processes"
        
        do i = 1, number_of_processes -1
            
            ! mpi_recv( <> buf(*), integer count, datatype, source, tag, comm, status(mpi_status_size, ierror)
            ! <> buf(*) :   initial address of receive buffer.
            !               Here: = 0 = this_process_number
            !               = which process is receiving
            ! integer count:number of elements in the receive buffer
            !               = 1 item will be transferred
            ! datatype:     data type of each receive buffer element
            !               = the received data will be an (mpi) integer
            ! source:       rank of source
            !               = receive from this source/process nr 
            ! tag:          message tag
            ! comm:         communicator
            !               = mpi_comm_world 
            ! status:
            ! ierror:       Error integer
            
            call mpi_recv(this_process_number, 1, mpi_integer, i, 1, mpi_comm_world, status, error_number)

            write(*,*) "Hello from process", this_process_number, " of ", number_of_processes, "processes"
        end do
    else
        ! if not main process (= process 0), send:
        ! mpi_send( <> buf (*), integer count, datatype, dest, tag, comm, ierror)
        ! Only difference to mpi_recv:
        ! dest instead of source
        ! dest:     rank of destination
        !           process number to which to send
        call mpi_send(this_process_number, 1, mpi_integer, 0, 1, mpi_comm_world, error_number)
    end if
    
    ! end MPI
    call mpi_finalize(error_number)

end program sendreceive

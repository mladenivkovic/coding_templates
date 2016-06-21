program writewithmpi

    ! initialise MPI
    ! write a message to a single file
    ! write output to screen from only 1 processor
    ! finalize MPI

    use mpi 
    implicit none
    integer :: dummy_io, errnr, myid, ncpu

    ! Never finalize MPI in a subroutine, otherwise no other
    ! subroutine can restart MPI

    call mpi_init(errnr)
    call mpi_comm_size(mpi_comm_world, ncpu, errnr)
    call mpi_comm_rank(mpi_comm_world, myid, errnr)
    myid = myid + 1


    call mladen_writetoonefile("my message")
    call mladen_message("Hello! This is the message for stdout. Other output was written to mladen_output.txt")

    call mpi_finalize(errnr)

contains
subroutine mladen_writetoonefile(somemessage)
    use mpi
    implicit none
    integer, parameter :: tag_mladen = 666
    character(len=*), intent(in) :: somemessage

    ! local vars
    integer :: i
    character (len=80) :: filename

    filename=TRIM('mladen_output.txt')

    dummy_io = myid

    if (myid == 1) then
        !write some stuff
        write(*, *) "myid = 1"
        write(*, *) "ncpu", ncpu
        open(unit=666, file=filename, form='formatted')
        write(666, *) somemessage, dummy_io, 1
        
        !loop over all IDs
        ! receive their ID
        ! write it along with message

        do i = 2, ncpu
            call mpi_recv(dummy_io, 1, mpi_integer, i-1, tag_mladen, mpi_comm_world, MPI_STATUS_IGNORE, errnr)
            write(666, *) somemessage, dummy_io, i
        end do
        close(666)
    else
        ! send your ID (dummy_IO) to ID 1
        call mpi_send(dummy_io, 1, mpi_integer, 0, tag_mladen, mpi_comm_world,MPI_STATUS_IGNORE, errnr)      
        write(6, *) "myid /= 1"
    end if
    

end subroutine mladen_writetoonefile


subroutine mladen_message(somemessage)
    ! a subroutine to pass on messages when i need them to.
    ! only myid=1 writes.
    implicit none
    character(len=*), intent(in), optional :: somemessage
    if (myid ==1) then 
        write(6, '(A)')
        write(6, '(A)') "#############################################"
        write(6, '(A)') "This is a message for Mladen."
        if(present(somemessage)) then
            if (len(somemessage) /= 0) then
                write(6, '(A)')
                write(6, '(A)') "Your message is: "
                write(6, '(A)') somemessage
            end if
        else
            write(6, '(A)') "You left no specific message."
        end if
        write(6, '(A)') "#############################################"
        write(6, '(A)')
    end if

end subroutine mladen_message


end program writewithmpi

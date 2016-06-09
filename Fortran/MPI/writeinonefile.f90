program writeinonefile

    implicit none

    call mladen_writetoonefile("my message")




contains
subroutine mladen_writetoonefile(somemessage)
    use mpi
    implicit none
    integer, parameter :: tag_mladen = 666
    integer :: dummy_io, errnr, myid, ncpu
    character(len=*), intent(in) :: somemessage
    

    ! local vars
    integer :: i
    character (len=80) :: filename
    character (len=5) :: nchar
    character (len=5) :: somechar


    call mpi_init(errnr)
    call mpi_comm_size(mpi_comm_world, ncpu, errnr)
    call mpi_comm_rank(mpi_comm_world, myid, errnr)
    myid = myid + 1

    filename=TRIM('mladen_output.txt')

    dummy_io = myid

    if (myid == 1) then
        !call mladen_message("Myid=1 found")
        !write(somechar, '(I2)') ncpu
        !call mladen_message("ncpu = "//somechar)
        write(*, *) "myid = 1"
        write(*, *) "ncpu", ncpu
        open(unit=666, file=filename, form='formatted')
        write(666, *) somemessage, dummy_io, 1
        do i = 2, ncpu
            call mpi_recv(dummy_io, 1, mpi_integer, i-1, tag_mladen, mpi_comm_world, MPI_STATUS_IGNORE, errnr)
            write(666, *) somemessage, dummy_io, i
        end do
        close(666)
    else
        call mpi_send(dummy_io, 1, mpi_integer, 0, tag_mladen, mpi_comm_world,MPI_STATUS_IGNORE, errnr)      
        write(6, *) "myid /= 1"
    end if
    

end subroutine mladen_writetoonefile


subroutine mladen_message(somemessage)
    ! a subroutine to pass on messages when i need them to.
    implicit none
    character(len=*), intent(in), optional :: somemessage
    
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

end subroutine mladen_message


end program writeinonefile

program IO

    implicit none
    integer :: someinteger


    open (unit=1, file='data.txt')
    
    write(1, '(2A8)' ) "Column1", "Column2"
    do someinteger = 1, 10
        write(1, '(2i8)' ) someinteger * 2, someinteger*50
    end do
    close(1)


end program IO

program  createinput
    implicit none
    integer :: someinteger
    real (kind=selected_real_kind(20, 291)) :: somereal


! Create a simple 1 column  integer containing  file
    open (unit=1, file='1columninteger.dat')
    
    do someinteger = 1, 10
        write(1, '(i8)' ) someinteger
    end do
    
    close(1)





! Create a 2 column integer containing file with a title
    open (unit=1, file='2columninteger.dat')
    
    do someinteger = 1, 10
        write(1, '(2i8)' ) someinteger * 2, someinteger*50
    end do

    close(1)



! Create a 2 column integer containing file with a title
    open (unit=1, file='2columnintegerwithtitle.dat')
    
    write(1, '(2A8)' ) "Column1", "Column2"
    do someinteger = 1, 10
        write(1, '(2i8)' ) someinteger * 2, someinteger*50
    end do
    
    close(1)


! Create a 4 column integer containing file with a title

    open (unit=1, file='4columnintegerwithtitle.dat')
    
    write(1, '(4A8)' ) "Column1", "Column2", "Column3", "Column4"

    do someinteger = 1, 10
        write(1, '(4i8)' ) someinteger, someinteger*2, someinteger*3, someinteger*4
    end do
    
    close(1)

! Create a 4 column integer containing file without  title

    open (unit=1, file='4columninteger.dat')
    
    do someinteger = 1, 10
        write(1, '(4i8)' ) someinteger, someinteger*2, someinteger*3, someinteger*4
    end do
    
    close(1)


! Create a 3 column reals containing file with a title

    open (unit=1, file='3columnrealswithtitle.dat')
    
    write(1, '(3A10)' ) "Column1", "Column2", "Column3"

    somereal = 3.14159
    do someinteger = 1, 10
        somereal = somereal * someinteger
        write(1, '(3ES10.3E2)' ) somereal, somereal * 10.0**someinteger, somereal*10.0**(-1 * someinteger)
    end do
    
    close(1)

! Create a 3 column reals containing file without a title

    open (unit=1, file='3columnreals.dat')
    
    somereal = 3.14159
    do someinteger = 1, 10
        somereal = somereal * someinteger
        write(1, '(3ES10.3E2)' ) somereal, somereal * 10.0**someinteger, somereal*10.0**(-1 * someinteger)
    end do
    
    close(1)

end program createinput

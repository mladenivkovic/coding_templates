program writing
! A script to demonstrate how to use write(*, *) with formatted output
! See http://www.cs.mtu.edu/~shene/COURSES/cs201/NOTES/format.html
    implicit none
    integer :: someint = 123456
    real :: somereal = 123.456789
    character(len=20) :: somechar = 'your text here'
    logical :: somelogical = .true.

    !------------------
    ! Integers:
    !------------------

    ! formatting: rIw.m
    ! r: repeat
    ! I: integer
    ! w: width
    ! m: minimum digits
    write(*, *) "Integers"
    write(*, *) "--------"
    write(*, *) ""
    write(*, *) "write(*, *) someint"
    write(*, *) someint
    write(*, *) ""
    write(*, *) "write(*, '(I6)') someint"
    write(*, '(I6)') someint
    write(*, *) ""
    write(*, *) "write(*, '(I4)') someint" ! width to small, result = ****
    write(*, '(I4)') someint
    write(*, *) ""
    write(*, *) "write(*, '(I12)') someint"
    write(*, '(I12)') someint
    write(*, *) ""
    write(*, *) "write(*, '(I8.8)') someint"
    write(*, '(I8.8)') someint
    write(*, *) ""
    write(*, *) "write(*, '(2I9.8)') someint someint"
    write(*, '(2I9.8)') someint, someint
    write(*, *) ""
    write(*, *) "write(*, '(I8.8, I12.9)') someint someint"
    write(*, '(I8.8, I12.9)') someint, someint
    write(*, *) ""
    write(*, *) ""

    !------------------
    ! REALS:
    !------------------

    write(*, *) "Reals"
    write(*, *) "-----"
    write(*, *) ""
    write(*, *) "MIND THE ROUNDING!"
    WRITE(*, *) ""
    write(*, *) "Point form"

    ! rFw.d
    ! repeat F width . digits_after_comma
    ! w >= digits_before_comma + d + 1
    write(*, *) "rFw.d"
    write(*, *) " "
    write(*, '(F7.3)') somereal




    write(*, *) "Exponential forms"
    write(*, *) ""
    ! rEw.d
    ! repeat E width . digits
    ! Therefore condition: w >= d + 7
    ! x is a space.

    write(*,*) "rEw.d"
    write(*, *) "E6.7, E20.7,  x, E6.7, E14.7 (x = space)"
    write(*, '(E20.7,E20.7, x, E6.7, E14.7)') somereal, somereal, somereal, somereal
    write(*, *) ""



    ! rEw.dEe
    ! repeat E width . digits E exponent
    ! Therefore condition: w >= d + 7
    ! x is a space.
    write(*,*) "rEw.dEe"
    write(*, *) "E6.7E2, E20.7E3,  x, E6.7E3, E14.7E2 (x = space)"
    write(*, '(E20.7E2,E20.7E3, x, E6.7E3, E14.7E2)') somereal, somereal, somereal, somereal
    write(*, *) ""

    ! ESw.dEe
    ! w = d + e + 4
    write(*,*) "ESw.dEe"
    write(*, *) "ES12.3E3, ES12.3E4, ES14.5E3"
    write(*, '(ES12.3E3, ES12.3E4, ES14.5E3)') somereal, somereal, somereal








    !------------------
    ! LOGICALS
    !------------------

    write(*, *) "Logicals"
    write(*, *) "-----"
    write(*, *) ""

    !  rLw 
    ! repead L width
    ! Output is only T or F; so there are w-1 spaces
    write(*, *) "L1, L4"
    write(*, '(L1, L4)') somelogical, somelogical




    !----------------------
    !Characters
    !----------------------

    print *, ""
    print *, "----------"
    print *, "Characters"
    print *, "----------"
    print *, ""
    print *, "A edit descriptor: rAw - repeat A width" 
    print *, " "

    ! use write(*, '(rAw)') somechar for writing characters.
    ! If w is not specified, the length of somechar is taken.
    ! If somechar is longer than w, the first w chars are
    ! printed.
    print *, "print *, somechar                              ", somechar
    write(*, *) "write(*,*) somechar                            ", somechar
    write(*, '(A, A)'),  " write(*, '(A)') somechar                       ", somechar
    write(*, '(A, A6)'),  " write(*, '(A6)') somechar                      ", somechar
    write(*, '(A, A, A3)') " write(*, '(A, A3)') somechar, somechar         ", somechar, somechar
    write(*, '(4A)') " write(*, '(3A4)') somechar, somechar, somechar ", somechar, somechar, somechar



end program writing


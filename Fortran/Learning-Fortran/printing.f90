program printing

!A program to demonstrate formatted printing with fortran
!An Edit descriptor is more or less a formatting operator, as 
!far as I can tell.

    implicit none
    integer :: someinteger
    real :: somereal, someotherreal
    character(len=15) :: somechar

    !----------------------
    ! Integers
    !----------------------

    print *, ""
    print *, "--------"
    print *, "Integers"
    print *, "--------"
    print *, ""
    print *, "i * 120 * (-1)^i"
    do someinteger = 1, 10
        print 100, someinteger, someinteger, someinteger*120*(-1)**(someinteger)
        !print 100: 100 is a statement label. There must be a format statement with this label in the program.
    end do
    100 format ( ' ', i3, ' * 120 * (-1)^', i3, ' = ', i4) 
    ! format statements labelled 100. i3: print out first variable
    ! with maximal 3 columns ("digits"). i4: print out second 
    ! variable with maximal 4 columns.
    ! Commas are item separators.
    ! You must leave a column for the minus sign.
    ! If too few columns were specified for the data, the output
    !will be in asterisks (*)

    

    !----------------------
    !Reals
    !----------------------
    print *, ""
    print *, "-----"
    print *, "Reals"
    print *, "-----"
    print *, ""
    print *, "f edit descriptor: fw.d - fwidth.digitsaftercomma"
    print *, "e edit descriptor: ew.s - ewidth.significantdigits"
    print *, "g edit descriptor: gw.s - gwidth.sigificantdigits"
    print *, " "


    ! fw.d : w = total column width, d = digits after comma
    ! The comma needs a column as well!!!!!
    ! The numbers will be rounded:


    ! ew.s: w = total width, s = significant digits.
    ! gives a real in scientific notation: 0.1234E-09


    ! gw.s: w = total width, s = significant digits.
    ! gives a real in scientific notation: 0.1234E-09

    print *, "Calculation        ", "f                     ", "e           ", "g      "
    print *, "                   ", "f19.12                ", "e12.3       ", "g12.3      "
    somereal = 1.2345
    someotherreal = somereal
    do someinteger = -5, 5
        someotherreal = somereal*10.0**someinteger
        print 200, somereal, someinteger, someotherreal, someotherreal, someotherreal 
    end do

    print *, ""
    200 format (f7.5, ' * 10^(', i2, ') = ', f19.12, e12.3, g12.3)
    ! chose g and e 12.3 because: 3 significant digits + 0 in the
    ! beginning + . + E(+/-)XY + 3 spaces before previous
    ! table column
   
    print *, ""
    print *, "Mind the rounding!"





    !----------------------
    !Characters
    !----------------------

    print *, ""
    print *, "----------"
    print *, "Characters"
    print *, "----------"
    print *, ""
    print *, "A edit descriptor: rAw - repeatAwidth" 
    print *, " "

    ! use write(*, '(rAw)') somechar for writing characters.
    ! If w is not specified, the length of somechar is taken.
    ! If somechar is longer than w, the first w chars are
    ! printed.
    somechar='your text here'
    print *, "print *, somechar                              ", somechar
    write(*, *) "write(*,*) somechar                            ", somechar
    write(*, '(A, A)'),  " write(*, '(A)') somechar                       ", somechar
    write(*, '(A, A6)'),  " write(*, '(A6)') somechar                      ", somechar
    write(*, '(A, A, A3)') " write(*, '(A, A3)') somechar, somechar         ", somechar, somechar
    write(*, '(4A)') " write(*, '(3A4)') somechar, somechar, somechar ", somechar, somechar, somechar

end program printing

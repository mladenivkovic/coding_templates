program ifelse

    implicit none
    logical :: true = .TRUE., false = .FALSE.
    integer :: int1 = 1, int2 = 2    

    

    !---------------------
    !- RELATIONAL OPERATORS
    !---------------------

    ! There are six relational operators:

    !    < : less than
    !    <= : less than or equal to
    !    > : greater than
    !    >= : greater than or equal to
    !    == : equal to
    !    /= : not equal to


    !   FORTRAN LOGICAL OPERATORS
    !    .NOT. : logical not
    !    .AND. : logical and
    !    .OR. : logical or
    !    .EQV. : logical equivalence
    !    .NEQV. : logical not equivalence


    write(*, *) ""
    write(*, *) "RELATIONAL OPERATORS"
    write(*, *) ""

    write(*, '(A8, I2, 4x, A7, I2)') "int1 = ", int1, "int2 = ", int2

    write(*, *) "int1 < int2, int1 <= int2, int1 > int2, int1 >= int2, int1 == int2, int1 /= int2"


    write(*,'(6L4)') int1 < int2, int1 <= int2, int1 > int2, int1 >= int2, int1 == int2, int1 /= int2







    !---------------------
    !- IF - STATEMENTS
    !---------------------


    write(*, *) ""
    write(*, *) ""
    write(*, *) "IF STATEMENTS"
    write(*, *) ""


    ! if - then - else - end if: if (logical expr) then statement1 else statement 2 end if
    write(*, *) "If - then - else - end if"
    if (true) then
        write(*, '(A8)') "True"
    else
        write(*, '(A8)') "False"
    end if


    !if - then - end if: if (logical expr) then statement end if
    write(*, *) ""
    write(*, *) "If - then - end if"
    if (true) then
        write(*, '(A8)') "False"
    end if

    ! logical if: if (logical expr) statement
    ! only one statement allowed! It can't be another if.
    write(*, *) ""
    write(*, *) "logical if"
    if (int1 < int2) write(*, *) 'true'
     



    ! if - then - else if then - else - end if: if (logical expr) then statement1 else statement 2 end if
    write(*,*) " "
    write(*, *) "If - then - else if - else - end if"
    if (int1 < int2) then
        write(*, '(A20)') " int1 < int2"
    else if (int1 /= int2) then
        write(*, '(A20)') " int1 /= int2"
    else 
        write(*,*) " nope"
    end if


    ! Logical Operators
    !   FORTRAN LOGICAL OPERATORS
    !    .NOT. : logical not
    !    .AND. : logical and
    !    .OR. : logical or
    !    .EQV. : logical equivalence
    !    .NEQV. : logical not equivalence   write(*,*) " "



    write(*,*) " "
    write(*,*) " "
    write(*,*) " "
    write(*, *) "LOGICAL OPERATORS"

    write(*,*) " "
    write(*,*) " "
    write(*, *) "true = .TRUE. ; false = .FALSE. "
    write(*, *) ""
    write(*, *) ""
    if (true.AND.false) then
        write(*, *) "true.AND.false = true"
    else
        write(*, *) "true.AND.false = false"
    end if

    write(*,*) " "
    if (.NOT.false) then
        write(*, *) ".NOT.false = true"
    else
        write(*, *) ".NOT.false = false"
    end if


    write(*,*) " "
    if (true.AND..NOT.false) then
        write(*, *) "true.AND.NOT.false = true"
    else
        write(*, *) "true.AND.NOT.false = false"
    end if


    write(*,*) " "
    if (true.OR.false) then
        write(*, *) "true.OR.false = true"
    else
        write(*, *) "true.OR.false = false"
    end if


    
    write(*, *) ""
    if (true.EQV.false) then
        write(*, *) "true.EQV.false = true"
    else
        write(*, *) "true.EQV.false = false"
    end if
    


    write(*,*) " "
    if (true.NEQV.false) then
        write(*, *) "true.NEQV.false = true"
    else
        write(*, *) "true.NEQV.false = false"
    end if




end program ifelse

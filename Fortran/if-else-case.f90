program ifelsecase


! How to use if, if else, and case



    implicit none
    logical :: true = .TRUE., false = .FALSE.
    integer :: int1 = 1, int2 = 2    
    character :: casechar 
    

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




    !-----------------
    ! CASE
    !----------------

    write(*, *) ""
    write(*, *) ""
    write(*, *) "CASE OPERATOR"
    write(*, *) ""

    casechar='g'
   
    ! can be done without "somename"
    somename: select case (casechar)
        case ('a', 'b', 'c') somename
            write(*, *) "Got a, b or c"
        case ('d', 'e', 'f') somename
            write(*, *) "Got d, e or f"
        case default somename
            write(*, *) "Got into default here - so no case was matched."
    end select somename

    

end program ifelsecase

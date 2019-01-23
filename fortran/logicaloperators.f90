program logicaloperators

    !=================================
    ! Fortran logical operators
    !=================================

    implicit none
    logical :: true = .TRUE., false = .FALSE.

    !-----------------------------------------------------------
    !   FORTRAN LOGICAL OPERATORS
    !    .NOT. : logical not
    !    .AND. : logical and
    !    .OR. : logical or
    !    .EQV. : logical equivalence
    !    .NEQV. : logical not equivalence   write(*,*) " "
    !-----------------------------------------------------------



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




end program logicaloperators

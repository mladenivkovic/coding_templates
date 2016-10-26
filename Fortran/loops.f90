program loops

! All kinds of do loops

    implicit none

    integer :: start = 1, finish = 10, step = 1, controlvar

    integer :: counter
    logical :: somelogical



!DO control-var = initial-value, final-value, [step-size]
!   statements
!END DO

    write(*, *) "Simple do loop"
    do controlvar = start, finish, step
        write(*, '(I4)') controlvar
    end do


    write(*, *) ""
    write(*, *) "start = -5, finish = 21, step = 3"
    start = -5
    finish = 21
    step = 3
    do controlvar = start, finish, step
        write(*, '(I4)') controlvar
    end do



    write(*, *) ""
    write(*, *) "start = 21, finish = -5, step = -3"
    start = 21
    finish = -5
    step = -3
    do controlvar = start, finish, step
        write(*, '(I4)') controlvar
    end do



! General DO

! DO
!   statements 1
!   exit
!   statements 2
!END DO



    write(*, *) ""
    write(*, *) ""
    write(*, *) "General do"

    counter = 0
    somelogical = .FALSE.
    do 
        if (somelogical) then
            write(*, *) "Ending loop."
            write(*, *) "Last counter:", counter
            exit
        else 
            if (counter < 5) then
                write(*, '(A, I2)') "counter is less than 5. It is", counter
                counter = counter + 1
            else
                somelogical = .TRUE.
            end if
        end if
    end do


! DO WHILE (logical expressions)
!   expressions
! enddo

    write(*, *) ""
    write(*, *) ""
    write(*, *) "Do while"
    write(*, *) 
    
    counter = 0
    do while (counter < 5)
        write(*, *) "Counter is now", counter
        counter = counter + 1
    end do











end program loops

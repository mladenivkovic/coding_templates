program loops

    !--------------------------
    ! All kinds of do loops
    !--------------------------

    implicit none

    integer :: start = 1, finish = 10, step = 1, controlvar

    integer :: counter, i, ii
    logical :: somelogical

    integer :: len=10, wid=5;
    integer, allocatable, dimension(:,:) :: z



    !----------------------------------------------------------
    !DO control-var = initial-value, final-value, [step-size]
    !   statements
    !END DO
    !----------------------------------------------------------

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





    !--------------------
    ! General DO
    !
    ! DO
    !   statements 1
    !   exit
    !   statements 2
    !END DO
    !--------------------



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







    !----------------------------------
    ! DO WHILE (logical expressions)
    !   expressions
    ! enddo
    !----------------------------------

    write(*, *) ""
    write(*, *) ""
    write(*, *) "Do while"
    write(*, *) 
    
    counter = 0
    do while (counter < 5)
        write(*, *) "Counter is now", counter
        counter = counter + 1
    end do


    write(*,*)
    write(*,*)
    write(*,*)  "do while does not run even once if "
    write(*,*)  "the condition is not met at the first time:"

    counter=10
    do while(counter<0)
        write(*,*) "counter is now:", counter
        counter = counter + 1
    end do

    write(*,*) "Counter is after loop:", counter
    write(*,*) "Counter was before loop:   10"







    !-----------------------------
    ! Using construct names
    !-----------------------------

    write(*, *) ""
    write(*, *) ""
    write(*, *) "Loop control with construct names"
    write(*, *) 

    
   
    allocate(z(1:len, 1:wid))
    z = 0

    outer: do i=1, len
      do ii=1,wid
        if ( mod(i,2) == 0 ) cycle outer
        if (mod(ii,2)==1) z(i, ii) = 1
      enddo
    enddo outer


    do i = 1, len
      do ii = 1, wid
        write(*, '(I3,x)', advance='no') z(i, ii)
      end do
      write(*,*)
    end do




end program loops

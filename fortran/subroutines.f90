program subroutines

    !------------------------------------------------------------------------
    ! A fortran subroutine does not return any value with its name.
    ! subroutines use formal arguments to receive values and to pass 
    ! results back.
    !
    ! Syntax:
    !
    ! subroutine subroutine-name(arg1, ..., argn)
    !   implicit none
    !   [specification part]
    !   [execution part]
    !   [subprogram part]
    ! end subroutine subroutine-name
    !
    !
    !
    ! INTENT
    !
    ! intent(in) :
    !   the argument is expected to have a value, but its value will not
    !   be changed inside the subroutine.
    !
    ! intent(out) :
    !   the parameter does not have a value and is given one in the
    !   called routine
    !
    ! intent(inout):
    !   parameter already has a value and is changed inside the routine
    !
    !
    !
    ! To use a subroutine, the CALL statement is needed.
    !------------------------------------------------------------------------

    


    !-------------------------------------------------------
    ! simple example of the use of a main program
    ! and two subroutines.
    ! one interacts with the user and the
    ! second solves a quadratic equation,
    ! based on the user input.
    !
    ! It calculates the solutions for the quadratic eqn
    ! of the form ax**2  + bx + c = 
    !
    ! It calculates the solutions for the quadratic eqn
    ! of the form ax**2  + bx + c = 0
    !-------------------------------------------------------


    implicit none
    real :: p, q, r, root1, root2
    integer :: ifail = 0
    logical :: ok 
    integer :: someint, arraylength

    integer, parameter :: n = 100
    real, dimension(1:n) :: myarray = [(real(someint), someint = 1, n)]
    real, dimension(1:10) :: anotherarray = [(someint, someint = 1, 20, 2)]
    real :: mean, std_dev




    write(*, *) "Subroutine interact and solve"
    call interact(p, q, r, ok)
    if (ok) then
          call solve(p, q, r, root1, root2, ifail)
          if (ifail==1) then
               write(*, *) ' complex roots'
               write(*, *) ' calculation abandoned'
          else
               write(*, *) ' roots are ', root1, ' ', root2
          end if
    else
         write(*, *) ' error in data input program ends'
    end if



    ! Trying the save attribute
    
    write(*, *)
    write(*, *)
    write(*, *) "Subroutine counting"
    do
      call counting(someint)
      if (someint == 5) exit
    end do


    ! 

    write(*, *)
    write(*, *)
    write(*, *) "Subroutine subrwitharrays"

    arraylength = size(myarray)
    call subrwitharrays(myarray, arraylength, mean, std_dev)
    write(*, '(A10, F6.3, A15, F6.3)') "Mean = ", mean, " Std dev = ",std_dev


    write(*, *)
    write(*, *)
    write(*, *) "Subroutine optionalandkeywords"
    write(*, *)

    write(*, '(A)') " Anotherarray"
    write(*, '(10F6.2)') anotherarray 
    write(*, *)

    call optionalandkeywords(anotherarray)
    write(*, *) "optionalandkeywords(anotherarray)"
    write(*, '(10F6.2)') anotherarray
    write(*, *)

    call optionalandkeywords(anotherarray, 2.0)
    write(*, *) "optionalandkeywords(anotherarray, 2.0)"
    write(*, '(10F6.2)') anotherarray
    write(*, *)

    call optionalandkeywords(anotherarray, 2.0, 3.0)
    write(*, *) "optionalandkeywords(anotherarray, 2.0, 3.0)"
    write(*, '(10F6.2)') anotherarray
    write(*, *)

    call optionalandkeywords(anotherarray, another_opt_var=3.0)
    write(*, *) "optionalandkeywords(anotherarray, 2.0, another_opt_var=3.0)"
    write(*, '(10F6.2)') anotherarray
    write(*, *)
!###############################################
!###############################################




contains


    subroutine interact(a, b, c, ok)
        !------------------------------------------------------------
        ! reads in 3 values, checks if no problems while reading in
        !------------------------------------------------------------
        implicit none
        real, intent (out) :: a
        real, intent (out) :: b
        real, intent (out) :: c
        logical, intent (out) :: ok
        integer :: io_status = 0

        ! fuck reading in, let's just do -1, 2 and 3
        ! write(*, *) ' type in the coefficients a, b and c (e.g. -1, 2, 3)'
        ! read (unit=*, fmt=*, iostat=io_status) a, b, c
        ! if (io_status==0) then
        !   ok = .true.
        ! else
        !   ok = .false.
        ! end if
        a = -1
        b = 2
        c = 3
        ok = .true.
        io_status = 1 ! useless, just to keep from 'unused variable' warnings
    end subroutine interact





!###############################################






    subroutine solve(e, f, g, root1, root2, ifail)
        implicit none
        real, intent (in) :: e
        real, intent (in) :: f
        real, intent (in) :: g
        real, intent (out) :: root1
        real, intent (out) :: root2
        integer, intent (inout) :: ifail
    !   local variables
        real :: term
        real :: a2

        term = f*f - 4.*e*g
        a2 = e*2.0
    !   if term < 0, roots are complex
        if (term<0.0) then
              ifail = 1
        else
              term = sqrt(term)
              root1 = (-f+term)/a2
              root2 = (-f-term)/a2
        end if
    end subroutine solve




!###############################################




    subroutine counting(state)
        !------------------------------------------------------------
        ! a subroutine to demonstrate the save option
        !------------------------------------------------------------
        implicit none
        integer, save :: i = 1  ! This local  variable will be 
                                ! saved in between calls.
        integer, intent(out) :: state
        
        write(*, *) "Called subroutine counting."
        write(*, *) "i is now: ", i
        state = i
        i = i + 1
    end subroutine counting






!###############################################








    subroutine optionalandkeywords(somearray, opt_var, another_opt_var)
        !------------------------------------------------------------
        ! a subroutine to demonstrate optional arguments
        !------------------------------------------------------------
        implicit none
        real, intent(in), optional :: opt_var
        real, intent(in), optional :: another_opt_var
        real, dimension(:), intent(inout) :: somearray
        real :: somereal, someotherreal
        integer i, length

        if (present(opt_var)) then
            write(*, *) "Subroutine optionalandkeywords: opt_var was defined"
            somereal = opt_var
        else
            somereal = 1.0
        end if

        if (present(another_opt_var)) then
            write(*, *) "Subroutine optionalandkeywords: another_opt_var was defined"
            someotherreal = another_opt_var
        else
            someotherreal = 0.0
        end if

        length = size(somearray)
        do i = 1, length
            somearray(i) = somearray(i) * somereal + someotherreal
        end do
    end subroutine optionalandkeywords 











!###############################################






    subroutine subrwitharrays(x, n, mean, std_dev)
        !------------------------------------------------------------
        !x : array
        !n : array length
        !mean: mean
        !std_dev: standard deviation
        !------------------------------------------------------------
        
        implicit none
        integer, intent(in) :: n
        real, intent(in), dimension(:) :: x
        real, intent(out) :: mean, std_dev
        real :: variance
        real :: sumxi= 0.0, sumxisquared = 0.0
        integer :: i
        do i = 1, n
            sumxi = sumxi + x(i)
            sumxisquared = sumxisquared + x(i) * x(i)
        end do
        
        mean = sumxi/n
        variance = (sumxisquared - sumxi*sumxi/n)/(n-1)
        std_dev = sqrt(variance)
    end subroutine subrwitharrays






!###############################################






end program subroutines

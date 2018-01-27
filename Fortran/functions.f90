program functions

  !=====================================
  ! Dealing with functions.
  !=====================================


  ! Syntax:
  !
  ! type FUNCTION function-name (arg1, arg2, ..., argn)
  !    IMPLICIT NONE
  !    [specification part] 
  !    [execution part] 
  !    [subprogram part] 
  !    END FUNCTION function-name


  ! "type" is a Fortran type: logical, integer, real...
  ! function-name : a fortran identifier. Just name it.
  ! arg1, ..., argn : formal arguments

  ! Somewhere in a function there has to be one or more 
  ! assignment statements like " function-name = expression"
  ! function-name cannot appear in the right-hand side of 
  ! any expression.

  ! In a type specification, formal arguments shold have a new
  ! Attribute: INTENT(IN)
  ! Its meaning is that the function only takes the calue from
  ! a formal agrument and does not change its content.

  ! Functions can have no formal argument, but () is still required.




  !--------------------
  ! INTERNAL FUNCTIONS
  !--------------------

  ! Internal functions are inside of a program, the main program:
  ! Program program-name 
  !   implicit none
  !   [specification part]
  !   [execution part]
  ! contains
  !   [functions]
  ! End program program-name

  ! Common problems:
  ! - forget function type
  ! - forget INTENT (IN)
  ! - change an INTENT(IN) argument
  ! - forget to return a value

    implicit none

real :: a = 14.0, b = 7.7
    integer :: i=5

    write(*, *) "a = ", a, "b = ", b, "i = ", i
    
    write(*, *) "divide(a,b) ", divide(a, b)
    write(*, *) "factorial(i) ", factorial(i)





contains
    real FUNCTION divide(a, b)
        implicit none
        real, intent(in) :: a, b
        write(*, *) "This is an output of the divide function."
        divide = a/b
    end FUNCTION divide




    recursive integer function factorial(i) result (answer)
        implicit none
        integer , intent (in) :: i

        if (i == 0) then 
            answer = 1
        else
            answer = i * factorial(i - 1)
        end if
    end function factorial

end program functions

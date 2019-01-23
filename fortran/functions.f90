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
  ! OR: use type FUNCTION function-name (args) result(resultvar)
  ! then the result is resultvar

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
  integer, dimension(2) :: res

  write(*, *) "a = ", a, "b = ", b, "i = ", i
  
  write(*, *) "divide(a,b)              ", divide(a, b)
  write(*, *) "divide(a,b) with results ", divide_results(a, b)
  write(*, *) "factorial(i)             ", factorial(i)
  res = fibo(i)
  write(*, *) "fibonacci(i)             ", res(1)





contains
  !=================
  ! 0815 function
  !=================
  real FUNCTION divide(a, b)
    implicit none
    real, intent(in) :: a, b
    divide = a/b      ! <- here you tell the function what to return
  end FUNCTION divide




  !===============================================================
  ! Use "result" to return another variable, not function name
  !===============================================================
  real FUNCTION divide_results(a, b) result(r)
    implicit none
    real, intent(in) :: a, b
    ! real :: r   ! <= no need to re-declare r: it already is real
    r = a/b
  end FUNCTION divide_results




  !===============================================================
  ! Recursive Functions
  !===============================================================
  recursive integer function factorial(i) result (answer)
    implicit none
    integer , intent (in) :: i

    if (i == 0) then 
      answer = 1
    else
      answer = i * factorial(i - 1)
    end if
  end function factorial



  !===============================================================
  ! Return arrays
  !===============================================================
  recursive function fibonacci(i) result(r)
    ! Compute fibonacci numbers
    implicit none
    integer, intent(in) :: i
    integer, dimension(2) :: r, temp
    ! this function returns r = (n-1, n-2)

    if (i==1) then
      r = (/1, 1/)
    else
      temp = fibonacci(i-1)
      r = (/temp(1)+temp(2), temp(1)/)
    endif
  end function fibonacci

end program functions

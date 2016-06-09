program array_basics_frage_quadruple
! Take input from inputdata.txt: cmd line ./array_basics < inputdata.txt
!
! The number value of a quadruple precision real is not the same as the value in an array
!
    implicit none
    integer, parameter :: ndim = 2 ! Number of Elements to read/import

!Defining precision parameters
    integer, parameter :: dp = selected_real_kind(15, 307) !Double Precision
    integer, parameter :: sp = selected_real_kind(6,37) ! Single Precision
    integer, parameter :: qp = selected_real_kind(30,291) ! quadruple precision

!Defining Reals with specified precision
    real(kind=dp) :: DoublePrecisionReal
    real(kind=sp) :: SinglePrecisionReal
    real(kind=qp) :: QuadruplePrecisionReal

!Defining Arrays with specified precision
    real(kind=dp), dimension (1:ndim):: DoublePrecisionArray
    real(kind=sp), dimension (1:ndim):: SinglePrecisionArray
    real(kind=qp), dimension (1:ndim):: QuadruplePrecisionArray
    real, dimension(1:ndim, 1:ndim):: TwoDarray
    integer :: counter
    integer :: counter2

!Reading from Data, assigning to arrays and printing each value out
    print*, 'Did you take data from <inputdata.txt? If not, give some (2) noninteger numbers.'
    do counter = 1, ndim
        read *, QuadruplePrecisionReal
        DoublePrecisionReal = QuadruplePrecisionReal
        SinglePrecisionReal = DoublePrecisionReal

        print*, 'SP', SinglePrecisionReal 
        print*, 'DP', DoublePrecisionReal
        print*, 'QP', QuadruplePrecisionReal

        SinglePrecisionArray(counter) = QuadruplePrecisionReal  
        DoublePrecisionArray(counter) = QuadruplePrecisionReal
        QuadruplePrecisionArray(counter) = QuadruplePrecisionReal 
    end do

    print*, '----------------------'
    print*, ' '
    print*, 'SinglePrecisionArray', SinglePrecisionArray
    print*, 'DoublePrecisionArray', DoublePrecisionArray
    print*, 'QuadruplePrecisionArray', QuadruplePrecisionArray
    print*, ' '
    print*, '----------------------'

end program array_basics_frage_quadruple

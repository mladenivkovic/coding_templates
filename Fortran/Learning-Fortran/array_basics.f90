program array_basics
! Take input from inputdata.txt: cmd line ./array_basics <
! inputdata.txt A program to demonstrate some array basics in Fortran
! (90).    
    implicit none

    !Defining amount of input
    integer, parameter :: ndim = 3
    !Defining integer precision
    integer, parameter :: sp = selected_real_kind(6,37) ! Single Precision 
    integer, parameter :: dp = selected_real_kind(15, 307) !Double Precision 
    integer, parameter :: qp = selected_real_kind(30,291) ! quadruple precision

    !Defining Reals with specified precision
    real(kind=sp) :: SinglePrecisionReal 
    real(kind=dp) :: DoublePrecisionReal 
    real(kind=qp) :: QuadruplePrecisionReal


    !Defining 1D-Arrays with specified precision
    real(kind=sp), dimension (1:ndim):: SinglePrecisionArray
    real(kind=dp), dimension (1:ndim):: DoublePrecisionArray
    real(kind=qp), dimension (1:ndim):: QuadruplePrecisionArray

!   Defining 2D-Arrays with specified precision
    real(kind=sp), dimension (1:ndim, 1:ndim):: SinglePrecisionArray2D
    real(kind=dp), dimension (1:ndim, 1:ndim):: DoublePrecisionArray2D
    real(kind=qp), dimension (1:ndim, 1:ndim):: QuadruplePrecisionArray2D

!    Defining an allocatable Array (= dimensions can be defined later)
    integer, dimension(:), allocatable :: AllocatableArray 
    integer, dimension(:), allocatable :: AllocatableArray2D

!   Defining others
    integer :: zeile 
    integer :: spalte 
    integer :: randomint = 15

    !Reading input. Assigning it to Reals with specified precision
    !first, then to Arrays with corresponding precision. All assigned
    !Reals are printed.
    print*, 'Assigned Numbers' 
    do zeile = 1, ndim 
        read *, QuadruplePrecisionReal  !asks you for number. Can read from file if cmd line includes <inputfile 
        DoublePrecisionReal = QuadruplePrecisionReal 
        SinglePrecisionReal = QuadruplePrecisionReal

        print*, 'SP', SinglePrecisionReal 
        print*, 'DP', DoublePrecisionReal 
        print*, 'QP', QuadruplePrecisionReal

        SinglePrecisionArray(zeile) = QuadruplePrecisionReal
        DoublePrecisionArray(zeile) = QuadruplePrecisionReal
        QuadruplePrecisionArray(zeile) = QuadruplePrecisionReal 
        end do

    print*, '----------------------' 
    print*, ' ' 
    print*, '1D-Arrays'
    print*, ' ' 
    print*, 'SinglePrecisionArray', SinglePrecisionArray
    print*, 'DoublePrecisionArray', DoublePrecisionArray 
    print*, 'QuadruplePrecisionArray', QuadruplePrecisionArray 
    print*, ' '
    print*, '----------------------'

    print*, ' ' 
    print*, '2D-Arrays' 
    print*, ' '
    
    !Assigning 2D-Arrays    
    do zeile=1, ndim 
        do spalte=1, ndim 
        SinglePrecisionArray2D(zeile, spalte) = SinglePrecisionArray(spalte)*zeile
        DoublePrecisionArray2D(zeile, spalte) = DoublePrecisionArray(spalte)*zeile
        QuadruplePrecisionArray2D(zeile, spalte) = QuadruplePrecisionArray(spalte)*zeile 
        end do 
    end do
    
    print*, 'Bad way to print it' 
    print*, DoublePrecisionArray2D

    print*, ' ' 
    print*, 'Better way to print it'

    do zeile = 1, ndim 
        print*, (DoublePrecisionArray2D(zeile, spalte), spalte=1,ndim)
    end do
    
    
    !printing 2D-Arrays with specified precision
    print*, ' ' 
    print*, '----------' 
    print*, ' '
    
    print*, 'SinglePrecision 3D - Array' 
    do zeile = 1, ndim 
        print*, (SinglePrecisionArray2D(zeile, spalte), spalte=1,ndim)
    end do
    
    print*, ' ' 
    print*, 'DoublePrecision 3D - Array' 
    do zeile = 1, ndim 
        print*, (DoublePrecisionArray2D(zeile, spalte), spalte=1,ndim)
    end do
    
    print*, ' ' 
    print*, 'QuadruplePrecision 3D - Array' 

    do zeile = 1, ndim 
        print*, (QuadruplePrecisionArray2D(zeile, spalte), spalte=1, ndim)
    end do
   

    print*, ' ' 
    print*, '----------------------' 
    print*, ' ' 
    print*, 'Allocateable Arrays' 
    print*, ' '
    
    ! Allocating Arrays
    allocate(AllocatableArray(1:randomint))
    
    !Assigning them just simple integers so they won't be empty
    do zeile=1, randomint 
        AllocatableArray(zeile) = zeile 
    end do
   
   !printing every element on a new line
    do zeile=1, randomint 
        print*, AllocatableArray(zeile) 
    end do 
end program array_basics

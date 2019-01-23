program array_init

!=======================================================================
! A quick template to create arrays.
!=======================================================================


  !---------------------------------------------------------------------
  ! The rank of an array is its number of dimensions.
  ! The extent of an array is the number of elements in one dimension.
  ! The shape of an array is a vector for which each dimension equals 
  ! the extent.
  !---------------------------------------------------------------------


    implicit none

    integer, parameter :: nelements = 10
    integer, parameter :: xdim = 10
    integer, parameter :: ydim = 5
    integer, parameter :: cubexdim=3, cubeydim = 3, cubezdim = 3 ! to define 3 x 3 x 3 cube

    !3D-Array
    integer, dimension (1:cubexdim, 1:cubeydim, 1:cubezdim) :: CubeArray

    
    !Allocatable Array
    integer, dimension(:), allocatable :: AllocatableArray
    integer, dimension(:,:), allocatable :: AllocatableArray2D

    
    !Initiating others that we might need
    integer :: ArrayElement
    integer :: x, y, z

    ! Initiate an array with implied do loop
    integer :: i
    real, dimension(1:nelements) :: ImpliedDoArray 






    !-----------------------------
    ! ARRAY INITIATION
    !-----------------------------

    ! Initiate an array with initial values:
    real, dimension(4:8) :: InitialArray = (/1.0, 1.2, 1.4, 1.6, 1.8/)
    real, dimension(5) :: InitialArrayTwo = (/1.0, 1.2, 1.4, 1.6, 1.8/)


    write(*,*) "-------------------------"
    write(*,*) "Array with initial values"
    write(*,*) "-------------------------"
    write(*,*) ' '
    write(*, '(5F6.2)') InitialArray
    write(*, '(5F6.2)') InitialArrayTwo
    write(*,*) ' '
    write(*,*) ' '








    !------------------------------
    ! Implies Do Array
    !------------------------------

    ImpliedDoArray = [ (i**2, i=1, nelements) ]

    write(*,*) "------------------------"
    write(*,*) "    Implied Do Loop     "
    write(*,*) "------------------------"
    write(*,*) ' '   
    write(*, '(10F6.1)'  ) ImpliedDoArray 
    write(*,*) ' '   
    write(*,*) ' '   








    !----------------------------
    ! Allocatable array
    !----------------------------

    allocate(AllocatableArray(1:xdim))
        Allocatablearray = [(i, i=1, xdim) ]
    
    allocate(AllocatableArray2D(1:xdim, 1:ydim))
        do x = 1, xdim
            do y = 1, ydim
            AllocatableArray2D(x, y) = 10*x + y
            end do
        end do



    write(*,*) ' '   
    write(*,*) "------------------------"
    write(*,*) "    Allocated Arrays    "
    write(*,*) "------------------------"
    write(*,*) ' '  
    
    write(*,*) "Allocatable Array"
    write(*,'(10I3)') AllocatableArray 
    write(*,*) ' '  
    
    write(*,*) "Allocatable Array 2D"
    do y = 1, ydim
        write(*,'(10I4)') (AllocatableArray2D(x,y), x=1, xdim)
    end do
    write(*,*) ' '  
    write(*,*) ' '   
    









    !-------------------
    ! Cube Array
    !-------------------

    do x = 1, cubexdim
        do y = 1, cubeydim
            do z = 1, cubezdim
            ArrayElement = 100*x + 10*y + z
            CubeArray(x, y, z) = ArrayElement
            end do
        end do
    end do


    write(*,*) ' '

    write(*,*) "------------------------"
    write(*,*) '       Cube array       '
    write(*,*) "------------------------"
    write(*,*) ' '
    write(*,*) ' Printing coordinates as (x y z)'

    do z = 1, cubezdim
        write(*,*) ' '
        write(*,*) "Printing for z = ", z
        write(*,*) ""
        do x = 1, cubexdim
            write(*,*) (CubeArray(x, y, z), y = 1, cubeydim)
        end do
    end do



end program array_init

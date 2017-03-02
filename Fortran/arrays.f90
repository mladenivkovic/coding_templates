program array_quickie
! A quick template to create and print arrays and allocatable arrays.
! IMPORTANT!!!!!!!!
! Execute with a.out < ./inputfiles/arrayinput.txt


! The rank of an array is its number of dimensions.
! The extent of an array is the number of elements in one dimension.
! The shape of an array is a vector for which each dimension equals 
! the extent.


    implicit none

    integer, parameter :: nelements = 10
    integer, parameter :: xdim = 10
    integer, parameter :: ydim = 5
    integer, parameter :: cubexdim=3, cubeydim = 3, cubezdim = 3 ! to define 3 x 3 x 3 cube

    !1D-Array
    integer, dimension(1:nelements) :: OneDimArray
    integer, dimension(1:nelements) :: CalculatingOneDimArray

    !2D-Array
    integer, dimension(1:xdim, 1:ydim) :: TwoDimArrayOne, TwoDimArrayTwo

    !3D-Array
    integer, dimension (1:cubexdim, 1:cubeydim, 1:cubezdim) :: CubeArray

    
    !Allocatable Array
    integer, dimension(:), allocatable :: AllocatableArray
    integer, dimension(:,:), allocatable :: AllocatableArray2D

    !Reshaping Arrays
    integer, dimension(1:12) :: ArrayToReshape
    integer, dimension(1:3, 1:4) :: ReshapedArray
    
    !Initiating others that we might need
    integer :: ArrayElement
    integer :: zeile, spalte
    integer :: x, y, z
    integer :: counter = 0

!###################################
! ARRAY INITIATION
!###################################
    ! Initiate an array with initial values:
    real, dimension(4:8) :: InitialArray = (/1.0, 1.2, 1.4, 1.6, 1.8/)
    real, dimension(5) :: InitialArrayTwo = (/1.0, 1.2, 1.4, 1.6, 1.8/)

! Initiate an array with implied do loop
    integer :: i
    real, dimension(1:nelements) :: ImpliedDoArray = [ (i**2, i=1, nelements) ]

! allocatable array

    allocate(AllocatableArray(1:xdim))
        Allocatablearray = [(i, i=1, xdim) ]
    
    allocate(AllocatableArray2D(1:xdim, 1:ydim))
        do x = 1, xdim
            do y = 1, ydim
            AllocatableArray2D(x, y) = 10*x + y
            end do
        end do


! Initiate via read from input

    !1-dim array
    do spalte = 1, nelements
        read *, ArrayElement
        OneDimArray(spalte) = ArrayElement
    end do

    !2-dim array: Two ways of reading in values
    do zeile  = 1, ydim
        do spalte = 1, xdim 
            read *, ArrayElement
            !print*, zeile, spalte
            TwoDimArrayOne(spalte, zeile) = ArrayElement
        end do
    end do

    do spalte = 1, xdim
        do zeile = 1, ydim
            read *, ArrayElement
            TwoDimArrayTwo(spalte, zeile) = ArrayElement
        end do
    end do

    ! Cube Array
    do x = 1, cubexdim
        do y = 1, cubeydim
            do z = 1, cubezdim
            ArrayElement = 100*x + 10*y + z
            CubeArray(x, y, z) = ArrayElement
            end do
        end do
    end do
!#############################################
! PRINTING ARRAYS
!#############################################

    ! formatting for nicer output:

    print*, "------------------------"
    print*, "Array with initial values"
    print*, "------------------------"
    print*, ' '
    write(*, '(5F6.2)') InitialArray
    write(*, '(5F6.2)') InitialArrayTwo
    print*, 'Accessing single elements: ', InitialArray(8), InitialArrayTwo(5)
    print*, ' '

    print*, "------------------------"
    print*, "----Implied Do Loop-----"
    print*, "------------------------"
    print*, ' '   
    write(*, '(10F6.1)'  ) ImpliedDoArray 


    print*, ' '   
    print*, "------------------------"
    print*, "----Allocated Arrays----"
    print*, "------------------------"
    print*, ' '  
    
    print*, "Allocatable Array"
    write(*,'(10I3)') AllocatableArray 
    print*, ' '  
    
    print*, "Allocatable Array 2D"
    do y = 1, ydim
        write(*,'(10I4)') (AllocatableArray2D(x,y), x=1, xdim)
    end do
    print*, ' '  
    
    print*, "------------------------"
    print*, "---Read from input------"
    print*, "------------------------"

    print*, ' '
    print*,  '1dim-array'
    print*, ' '

    write(*,'(10I3)')  OneDimArray
    print*, ' '
    do spalte = 1, nelements !print spalten als zeilen. Hier: jedes element
        print*, OneDimArray(spalte)
    end do

    print*, "------------------------"
    print*, "------------------------"

    print*, ' '
    print*, '2dim-array - 1st verision'
    print*, ' '
    print*, 'First print (as "intended")'
    print*, ' '
    do zeile = 1, ydim ! Print array als 5 Zeilen und 10 Spalten
        write(*,'(10I3)') (TwoDimArrayOne(spalte, zeile), spalte = 1, xdim)
    end do

    print*, ' '
    print*, "------------------------"
    print*, ' '
    print*, 'Second print'
    print*, ' '
    do spalte = 1, xdim ! Print array als 5 Zeilen und 10 Spalten
        write(*,'(10I3)') (TwoDimArrayOne(spalte, zeile), zeile = 1, ydim)
    end do


    print*, "------------------------"
    print*, "------------------------"

    print*, ' '
    print*, "2dim-array, 2nd version"
    print*, ' '
    print*, 'First print (as "intended")'
    print*, ' '

    do zeile = 1, ydim ! Print array als 5 Zeilen und 10 Spalten
    write(*, '(10I4)') (TwoDimArrayTwo( spalte, zeile), spalte = 1, xdim)
    end do

    print*, ' '
    print*, "------------------------"
    print*, ' '
    print*, 'Second print'
    print*, ' '

    do zeile = 1, xdim ! Print array als 10 zeilen und 5 spalten
        write(*, '(10I4)')  (TwoDimArrayTwo(zeile, spalte), spalte = 1, ydim)
    end do

    print*, "------------------------"
    print*, ""
    print*, "Accessing elements"
    write(*, '(A, I4)') "TwoDimArrayTwo(2, 1) = ", TwoDimArrayTwo(2, 1)
    write(*, '(A, 10I4)') "TwoDimArrayTwo(:,1) =", TwoDimArrayTwo(:, 1)
    write(*, '(A, 10I4)')  "TwoDimArrayTwo(1, :) =", TwoDimArrayTwo(1, :)

    print*, ""
    print*, "------------------------"
    print*, "------------------------"

    print*, ' '
    print*, 'Cube array'
    print*, ' '
    print*, ' Printing coordinates as (x y z)'

    do z = 1, cubezdim
        print*, ' '
        print*, "Printing for z = ", z
        print*, ""
        do x = 1, cubexdim
            print*, (CubeArray(x, y, z), y = 1, cubeydim)
        end do
    end do

    print*, "Accessing single elements:"
    print*, "CubeArray(1, 2, 3) =", CUbearray(1,2,3)
    print*, ' ' 
    print*, "------------------------"
    print*, "---------RESHAPE--------"
    print*, "------------------------"
    print*, ' '

    ArrayToReshape =[ (i, i=1, 12) ]



    print*, "Array to reshape"
    write(*, '(12I3)') ArrayToReshape
    print*, ""


    ReshapedArray = reshape(ArrayToReshape, (/3, 4/))
    print*, "reshaped array"
    do x=1, 3
        write(*, '(4I3)')  (ReshapedArray(x, y), y=1, 4)
    end do
    
    print*, ' ' 
    print*, ' ' 
    print*, "------------------------"
    print*, "----ARRAY OPERATIONS----"
    print*, "------------------------"
    print*, ' '

    print*, " Using OneDimArray for demonstration."
    write(*, '(A, 10I3)') "  OneDimArray =", OneDimArray
    print*, ' '

    write(*, '(A, I3)') "  Length of an array: size(OneDimArray) = ", size(OneDimArray)
    print*, ' '

    write(*, '(A, I3)') "  Maximal value of an array: maxval(OneDimArray) = ", maxval(OneDimArray)
    print*, ' '


    write(*, *) " Calculations:"
    print*, ' '

    CalculatingOneDimArray = OneDimArray + 3 * OneDimArray
    write(*,*) " Resultarray = OneDimArray + 3 * OneDimArray : "
    write(*, '(10I3)')  CalculatingOneDimArray

    print*, ' '
    write(*,*) " Itrinistic functions: e.g. sin(real(OneDimArray)):"
    write(*, '(10F7.3)') sin(real(OneDimArray))

    print*, ' '
    write(*,*) " Sum all elements: "
    write(*,*) " sum(OneDimArray) :", sum(OneDimArray) ! all elements
    print*, ' '
    write(*, *) " Sum over all odd elements:" 
    write(*,*) " sum(OneDimArray, mask=mod(OneDimArray, 2) == 1):", sum(OneDimArray, mask=mod(OneDimArray, 2) == 1)


end program array_quickie

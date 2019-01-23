program array_quickie

!=======================================================================
! A quick template on array operations.
!=======================================================================


  !---------------------------------------------------------------------
  ! The rank of an array is its number of dimensions.
  ! The extent of an array is the number of elements in one dimension.
  ! The shape of an array is a vector for which each dimension equals 
  ! the extent.
  !---------------------------------------------------------------------


    implicit none

    integer, parameter :: xdim = 3
    integer, parameter :: ydim = 2
    integer, parameter :: nelements = xdim * ydim

    !1D-Array
    real, dimension(1:nelements) :: array1D, calcarray1D
    integer, dimension(1:nelements) :: array1D_int
    logical, dimension(1:nelements) :: cond_satisfied

    !2D-Array
    real, dimension(1:xdim, 1:ydim) :: array2D


    !Reshaping Arrays
    integer, dimension(1:nelements) :: ArrayToReshape
    integer, dimension(1:xdim, 1:ydim) :: ReshapedArray
    
    !Initiating others that we might need
    integer :: x, y, i







    !-----------------------------
    ! ARRAY INITIATION
    !-----------------------------

    
    array1D = [(x, x=1, nelements)]

    do y = 1, ydim
      array2D(:,y) = [(10*y+x, x=1, xdim)]
    enddo







    write(*,*) ""
    write(*,*) "------------------------"
    write(*,*) " Accessing elements"
    write(*,*) "------------------------"
    write(*,*) ""
    write(*,*) "array2D(2, 1) =", array2D(2, 1)
    write(*,*) "array2D(:, 1) =", array2D(:, 1)
    write(*,*) "array2D(1, :) =", array2D(1, :)

    write(*,*) ""
    write(*,*) ""




    write(*,*) ""
    write(*,*) "------------------------"
    write(*,*) " Reshaping Arrays"
    write(*,*) "------------------------"
    write(*,*) ""


    ArrayToReshape = [ (i, i=1, nelements) ]
    write(*,*) "Array to reshape:" 
    write(*,*) ArrayToReshape
    write(*,*) ""


    ReshapedArray = reshape(ArrayToReshape, (/xdim, ydim/))
    write(*,*) "reshaped array:"
    do x=1, xdim
        write(*, *)  (ReshapedArray(x, y), y=1, ydim)
    end do




    
    write(*,*) ""
    write(*,*) ""
    write(*,*) "------------------------"
    write(*,*) " Array Properties       "
    write(*,*) "------------------------"
    write(*,*) ""

    write(*,*) " Using array1D for demonstration."
    write(*,*) " array1D =", array1D
    write(*,*) ""
    write(*,*) ""

    write(*,*) " Length of an array: size(array1D) = ", size(array1D)
    write(*,*) ""

    write(*,*) " Maximal value of an array: maxval(array1D) = ", maxval(array1D)
    write(*,*) ""

    write(*,*) " Minimal value of an array: minval(array1D) = ", minval(array1D)
    write(*,*) ""
    write(*,*) ""




    write(*,*) ""
    write(*,*) ""
    write(*,*) "------------------------"
    write(*,*) " Array Operations       "
    write(*,*) "------------------------"
    write(*,*) ""

    write(*,*) " Calculations:"
    write(*,*) ""

    calcarray1D = array1D + 3 * array1D
    write(*,*) " Resultarray = array1D + 3 * array1D : "
    write(*,*)  calcarray1D

    write(*,*) ""
    write(*,*) " Itrinistic functions: e.g. sin(array1D):"
    write(*,*) sin(array1D)

    write(*,*) ""
    write(*,*) " Sum all elements: "
    write(*,*) " sum(array1D) :", sum(array1D) ! all elements
    write(*,*) ""

    array1D_int = [(i, i=1, nelements)]
    write(*,*) " Sum over all odd elements: "
    write(*,*) " First initialise array_1D_int =", array1D_int
    write(*,*) " sum(array1D, mask=mod(array1D_int, 2) == 1):", sum(array1D, mask=mod(array1D_int, 2) == 1)



    write(*,*) ""
    write(*,*) ""
    write(*,*) " Get booleans for some condition en masse: value at index > 3.9?"
    cond_satisfied = array1D > 3.9
    write(*,*) " ", cond_satisfied


end program array_quickie

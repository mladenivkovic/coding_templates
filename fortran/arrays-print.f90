program array_print

!=======================================================================
! A quick template how to print arrays.
!=======================================================================

    implicit none

    integer, parameter :: nelements = 10
    integer, parameter :: xdim = 10
    integer, parameter :: ydim = 5
    integer :: x, y

    integer, dimension(:,:), allocatable :: AllocatableArray2D

  
    allocate(AllocatableArray2D(1:xdim, 1:ydim))
        do x = 1, xdim
            do y = 1, ydim
            AllocatableArray2D(x, y) = 10*x + y
            end do
        end do


    write(*,*) ""
    write(*,*) "------------------------"
    write(*,*) " Array Printing"
    write(*,*) "------------------------"
    write(*,*) ""
    write(*,*) ""
    


    write(*,*) "No formatting"

    write(*,*) AllocatableArray2D 
    write(*,*) ""
    write(*,*) ""
    






    write(*,*) "Implied Do Loop"

    do y = 1, ydim
        write(*,*) (AllocatableArray2D(x,y), x=1, xdim)
    end do
    write(*,*) ""
    write(*,*) ""





    write(*,*) "Formatting"

    do y = 1, ydim
      do x = 1, xdim
        write(*,'(I3,x)', advance='no') AllocatableArray2D(x,y) ! don't make new line after writing
      enddo
      write(*,*) !make new line here, after a whole line is written
    end do
    write(*,*) ""
    write(*,*) ""

    
  
end program array_print

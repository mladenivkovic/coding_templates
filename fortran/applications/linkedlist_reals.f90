program linkedreals

    !====================================================================
    ! This program reads in an unknown number of reals from 
    ! a file specified with fname.
    ! It is assumed here that the file contains 3 columns of reals
    ! with the format 3ES10.3E2 (as it was created with that format...)
    !====================================================================


    implicit none

    type link
        real :: x
        real :: y
        real :: z
        type(link), pointer :: next => null()
    end type link

    type (link), pointer :: first, current
    real, dimension(:, :), allocatable :: arrayofreals
    character (len=80) :: fname="inputfiles/3columnreals.dat"
    integer :: io_stat_number = 0
    integer :: n, i = 0, j

    open(unit=1, file = fname, status='old')




    !--------------------------------------------
    ! assigning first row
    !--------------------------------------------
    allocate(first)
    read(1, fmt='(3ES10.3E2)', iostat=io_stat_number) first%x, first%y, first%z
    if (io_stat_number == 0) then !if document not finished, create a new space in memory
        allocate(first%next)
        i = i+1
    end if




    !--------------------------------------------
    ! assigning all other rows
    !--------------------------------------------

    current => first !point the current real as the first

    do while (associated(current%next))
        ! if there is a next one: current points to the next
        current=> current%next 

        ! read in values to current
        read(1, fmt='(3ES10.3E2)', iostat=io_stat_number) current%x, current%y, current%z 
        
        ! if file isn't finished, create a target for the next link object
        if (io_stat_number /= -1) then          
            allocate(current%next)
            i = i+1
        end if
    end do

    write(*, *) i, "lines à 3 reals read"





    !--------------------------------------------
    ! Placing all these reals in an array now
    !--------------------------------------------

    ! store the amount of read rows
    n = i

    ! allocate array
    allocate (arrayofreals(1:n, 1:3))
    i = 0

    !point the current at the first again
    current => first

    do while (associated(current%next))
        !while there is a next:
        i = i + 1
        ! assign values
        arrayofreals(i, 1) = current%x
        arrayofreals(i, 2) = current%y
        arrayofreals(i, 3) = current%z
        !set pointer to the next:
        current=>current%next
    end do

    ! print
    do i = 1, n
        write(*, '(3ES10.3E2)') (arrayofreals(i, j), j= 1, 3)
    end do







end program linkedreals

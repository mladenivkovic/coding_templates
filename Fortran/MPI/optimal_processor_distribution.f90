program distribution

    implicit none
    integer :: nx = 100
    integer :: ny = 100
    integer :: nproc = 8
    integer :: nvar = 4

    
    integer :: nproc_x, nproc_y
    real :: lattency = 0.001
    real :: real_datasize=64.0
    real :: communication_speed=100000.0

    call distribute_processors(nproc_x, nproc_y)

    write(*, *) "nproc_x", nproc_x
    write(*, *) "nproc_y", nproc_y



contains 

subroutine distribute_processors(nproc_x, nproc_y)

    implicit none
    integer, intent(out) :: nproc_x, nproc_y
    real, dimension(:), allocatable:: calcarray
    integer, dimension(:,:), allocatable :: procarray
    integer :: i, ind
    real :: mintime

    

    allocate(procarray(1:nproc, 1:2),calcarray(1:nproc))
    calcarray=0.0
    procarray=0

    write(*, *) "allocated"
    do i = 1, nproc
        if( mod(nproc, i) == 0) then
            write(*, *) "Inside loop. i=", i
            procarray(i, 1) = i
            procarray(i, 2) = nproc/i
            calcarray(i) = calculate_communications(i, nproc/i)
        end if
    end do

    mintime=calcarray(1)
    ind=1
    do i=1, nproc
        if (calcarray(i) /= 0.0) then
            if (mintime > calcarray(i)) then
            mintime=calcarray(i)
            ind=i
            end if
        end if
    end do

    nproc_x=procarray(ind,1)
    nproc_y=procarray(ind,2)

end subroutine distribute_processors

real function calculate_communications(proc_x, proc_y)
    implicit none
    integer, intent(in) :: proc_x, proc_y
    real :: domainsize_x, domainsize_y
    real :: talking_time
    real :: waiting_time
    integer :: communications
    write(*, *) "Inside function. x, y=", proc_x, proc_y
    domainsize_x = nx/proc_x
    domainsize_y = ny/proc_y

    communications = 4*nproc-2*(proc_x+proc_y) ! The number of communications

    talking_time = real(communications)/2 * real_datasize / communication_speed * (domainsize_x + domainsize_y)*nvar

    waiting_time = communications*lattency

    calculate_communications=talking_time + waiting_time


end function calculate_communications

end program distribution

! main:     line 
! twodim:   line 23
! threedim: line 201
! globalvars: line 7. (Are you blind...?)
! WRITING: ID = 1
!-----------------------------------------------------------------------------------
module globalvars

    implicit none
    integer :: code, myid, nproc
    integer, parameter :: rows = 5, columns = 12, depth = 3

end module globalvars


!#############################################################################
!#############################################################################
!#############################################################################
!#############################################################################
!#############################################################################


module twodim
    use globalvars
    implicit none
    real, dimension(1:rows, 1:columns) :: array2D
    integer :: domainwidth
    integer :: domainstart
    integer :: domainend

contains
subroutine fill2Darray(id, array)

    implicit none
    integer, intent(in) :: id
    real, dimension(1:rows, 1:columns), intent(inout) :: array

    integer :: domainwidth, x, y
    real:: value


    value = real(id) / 10

    do x = domainstart, domainend 
        do y = 1, rows 
            array(y, x) = value + real(100*y + x)
        end do
    end do
    

end subroutine fill2Darray


!###################
!###################
!###################



subroutine write2Darray(array, message, id)
! prints 2d array.

    implicit none
    integer, intent(in) :: id
    real, intent(in), dimension(:,:) :: array
    character(len=*), intent(in) :: message
    integer :: counter, another_counter

    if (id == 1) then
        write(*,*)
        write(*, *) message
        write(*, *)
        do counter=1, rows
            write(*, '(20(F8.1))') (array(counter, another_counter), another_counter=1, columns)
        end do
        write(*,*)
    end if

end subroutine write2Darray


!###################
!###################
!###################


subroutine copylefttoright2D(id, array)
    implicit none
    integer, intent(in) :: id
    real, intent(inout), dimension(1:rows, 1:columns) :: array

    integer :: x, y

    if (id /= nproc-1) then
        do x = domainend -1 , domainend
            do y = 1, rows
                array(y, x+2) = array(y, x)
            end do
        end do
    end if
end subroutine copylefttoright2D

!###################
!###################
!###################

subroutine copyrighttoleft2D(id, array)
    implicit none
    integer, intent(in) :: id
    real, intent(inout), dimension(1:rows, 1:columns) :: array

    integer :: x, y

    if (id /= 0) then
        do x = domainstart, domainstart + 1
            do y = 1, rows
                array(y, x-2) = array(y, x)
            end do
        end do
    end if
end subroutine copyrighttoleft2D

!###################
!###################
!###################


subroutine communicate2D(id, array)

    use mpi
    implicit none
    integer, intent(in) :: id
    real, intent(inout), dimension(1:rows, 1:columns) :: array

    real, dimension(1:rows, 1:2) :: ghostcells_send_left, ghostcells_send_right, cells_receive_left, cells_receive_right
    integer :: x, y, code
    integer, dimension(MPI_STATUS_SIZE) :: status
    
    ghostcells_send_left = 0 
    ghostcells_send_right = 0
    cells_receive_left = 0
    cells_receive_right = 0

    do x = 1, 2
        do y = 1, rows
            !copy all ghost cells left of the domain
            if (id /= 0) ghostcells_send_left(y, x) = array(y, domainstart -2 + x)
            !copy all ghost cells right of the domain
            if (id /= nproc -1) ghostcells_send_right(y, x) = array(y, domainend + x)
        end do
    end do

    !send ghostcells_send_left (ghost cells left of domain), receive cells of the domain that are left

    if (id /= 0) then 
        call MPI_SENDRECV(ghostcells_send_left(1:rows, 1:2), 2*rows, MPI_REAL, id-1, 100, cells_receive_left(1:rows, 1:2), 2*rows, MPI_REAL, id-1, 200, MPI_COMM_WORLD, status, code)
    end if


    if (id /= nproc-1) then
        call MPI_SENDRECV(ghostcells_send_right(1:rows,1:2), 2*rows, MPI_REAL, id+1, 200, cells_receive_right(1:rows, 1:2), 2*rows, MPI_REAL, id+1, 100, MPI_COMM_WORLD, status, code)
    end if

! This works as well:

!    if (id /= 0) then 
!        call MPI_SENDRECV(ghostcells_send_left(1,1), 2*rows, MPI_REAL, id-1, 100, cells_receive_left(1, 1), 2*rows, MPI_REAL, id-1, 200, MPI_COMM_WORLD, status, code)
!    end if


!    if (id /= nproc-1) then
!        call MPI_SENDRECV(ghostcells_send_right(1,1), 2*rows, MPI_REAL, id+1, 200, cells_receive_right(1,1), 2*rows, MPI_REAL, id+1, 100, MPI_COMM_WORLD, status, code)
!    end if


    ! sum up new cells with old cells

    do x = 1, 2
        do y = 1, rows
        array(y, domainstart - 1 + x) = array(y, domainstart -1 + x) + cells_receive_left(y, x)
        array(y, domainend + 1 - x) = array(y, domainend + 1 -x) + cells_receive_right(y, 3-x)
        end do
    end do

end subroutine communicate2D



end module twodim



!#############################################################################
!#############################################################################
!#############################################################################
!#############################################################################
!#############################################################################



module threedim
    use globalvars
    implicit none
    real, dimension(1:rows, 1:columns, 1:depth) :: array3D
    integer :: domainwidth
    integer :: domainstart
    integer :: domainend

contains
subroutine fill3Darray(id, array)

    implicit none
    integer, intent(in) :: id
    real, dimension(1:rows, 1:columns, 1:depth), intent(inout) :: array

    integer :: domainwidth, x, y, z
    real:: value


    value = real(id) / 10

    do x = domainstart, domainend 
        do y = 1, rows 
            do z = 1, depth
                array(y, x, z) = value + real(1000*z+100*y + x)
            end do
        end do
    end do
    

end subroutine fill3Darray


!###################
!###################
!###################



subroutine write3Darray(array, message, id)
! prints 2d array.

    implicit none
    integer, intent(in) :: id
    real, intent(in), dimension(:,:, :) :: array
    character(len=*), intent(in) :: message
    integer :: counter, another_counter, third_counter

    if (id == 1) then
        write(*,*)
        write(*, *) message
        write(*, *)
        do third_counter = 1, depth
            write(*, *) "z = ", third_counter
            do counter=1, rows
                write(*, '(20(F8.1))') (array(counter, another_counter, third_counter), another_counter=1, columns)
            end do
            write(*,*)
        end do
    end if

end subroutine write3Darray


!###################
!###################
!###################


subroutine copylefttoright3D(id, array)
    implicit none
    integer, intent(in) :: id
    real, intent(inout), dimension(1:rows, 1:columns, 1:depth) :: array

    integer :: x, y, z

    if (id /= nproc-1) then
        do x = domainend -1 , domainend
            do y = 1, rows
                do z = 1, depth
                    array(y, x+2, z) = array(y, x, z)
                end do
            end do
        end do
    end if
end subroutine copylefttoright3D

!###################
!###################
!###################

subroutine copyrighttoleft3D(id, array)
    implicit none
    integer, intent(in) :: id
    real, intent(inout), dimension(1:rows, 1:columns, 1:depth) :: array

    integer :: x, y, z

    if (id /= 0) then
        do x = domainstart, domainstart + 1
            do y = 1, rows
                do z = 1, depth
                    array(y, x-2, z) = array(y, x, z)
                end do
            end do
        end do
    end if
end subroutine copyrighttoleft3D

!###################
!###################
!###################


subroutine communicate3D(id, array)

    use mpi
    implicit none
    integer, intent(in) :: id
    real, intent(inout), dimension(1:rows, 1:columns, 1:depth) :: array

    real, dimension(1:rows, 1:2, 1:depth) :: ghostcells_send_left, ghostcells_send_right, cells_receive_left, cells_receive_right
    integer :: x, y, z, code
    integer, dimension(MPI_STATUS_SIZE) :: status
    
    ghostcells_send_left = 0 
    ghostcells_send_right = 0
    cells_receive_left = 0
    cells_receive_right = 0

    do x = 1, 2
        do y = 1, rows
            do z = 1, depth
                !copy all ghost cells left of the domain
                if (id /= 0) ghostcells_send_left(y, x, z) = array(y, domainstart -2 + x, z)
                !copy all ghost cells right of the domain
                if (id /= nproc -1) ghostcells_send_right(y, x, z) = array(y, domainend + x, z)
            end do
        end do
    end do

    !send ghostcells_send_left (ghost cells left of domain), receive cells of the domain that are left

    if (id /= 0) then 
        call MPI_SENDRECV(ghostcells_send_left(1:rows, 1:2, 1:depth), 2*rows*depth, MPI_REAL, id-1, 100, cells_receive_left(1:rows, 1:2, 1:depth), 2*rows*depth, MPI_REAL, id-1, 200, MPI_COMM_WORLD, status, code)
    end if


    if (id /= nproc-1) then
        call MPI_SENDRECV(ghostcells_send_right(1:rows,1:2, 1:depth), 2*rows*depth, MPI_REAL, id+1, 200, cells_receive_right(1:rows, 1:2, 1:depth), 2*rows*depth, MPI_REAL, id+1, 100, MPI_COMM_WORLD, status, code)
    end if

    ! sum up new cells with old cells

    do x = 1, 2
        do y = 1, rows
            do z = 1, depth
                array(y, domainstart - 1 + x, z) = array(y, domainstart -1 + x, z) + cells_receive_left(y, x, z)
                array(y, domainend + 1 - x, z) = array(y, domainend + 1 -x, z) + cells_receive_right(y, 3-x, z)
            end do
        end do
    end do

end subroutine communicate3D



end module threedim




!#############################################################################
!#############################################################################
!#############################################################################
!#############################################################################
!#############################################################################



program sendarrays

    ! Initiate arrays of same size for each processor.
    ! Proc0: fill first half.
    ! Proc1: fill second half.
    ! Copy first two columns of Proc1-array into position
    ! of last two columns of proc0-array, then
    ! send them via mpi to proc0 and sum them.
    use globalvars
    use mpi

    implicit none
    

    call MPI_INIT(code)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myid, code)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, nproc, code)

    call run2d()
    !call run3d()


    call MPI_FINALIZE(code)
!##################################





contains

subroutine run2d()
    use twodim
    use globalvars
    implicit none

    if (myid == 1) write(*, *) "#########################"
    if (myid == 1) write(*, *) "#####     RUN 2D    #####"
    if (myid == 1) write(*, *) "#########################"

    array2D = 0
    
    domainwidth = columns/nproc
    domainstart = myid * domainwidth+1
    domainend = (myid + 1) * domainwidth

    call fill2Darray(myid, array2D)
    call write2Darray(array2D, 'before send', myid)
    call copylefttoright2D(myid, array2D)   ! can be ignored in hydro; values in ghost cells happen by themselves
    call copyrighttoleft2D(myid, array2D)   ! can be ignored in hydro; values in ghost cells happeb by themselves
    call write2Darray(array2D, 'after copyleftright', myid)
    call communicate2D(myid, array2D)
    call write2Darray(array2D, 'after mpi', myid)

end subroutine run2d


subroutine run3d()
    use threedim
    use globalvars
    implicit none

    if (myid == 1) write(*, *) "#########################"
    if (myid == 1) write(*, *) "#####     RUN 3D    #####"
    if (myid == 1) write(*, *) "#########################"

    array3D = 0
    
    domainwidth = columns/nproc
    domainstart = myid * domainwidth+1
    domainend = (myid + 1) * domainwidth

    call fill3Darray(myid, array3D)
    call write3Darray(array3D, 'before send', myid)
    call copylefttoright3D(myid, array3D)   ! can be ignored in hydro; values in ghost cells happen by themselves
    call copyrighttoleft3D(myid, array3D)   ! can be ignored in hydro; values in ghost cells happeb by themselves
    call write3Darray(array3D, 'after copyleftright', myid)
    call communicate3D(myid, array3D)
    call write3Darray(array3D, 'after mpi', myid)

end subroutine run3d


end program sendarrays



program mapdomain2d

    use mpi
    implicit none
    integer :: error_number
    integer :: myid
    integer :: nproc

    integer :: i, j
    integer :: nproc_x = 2, nproc_y = 4
    integer :: leftofme, rightofme, aboveme, belowme
    integer :: wall=0


    ! map a "domain" into nproc_x parts on x axis and nproc_y parts
    ! on y_axis.
    ! Map starts on the upper left corner with myid=1 and goes from
    ! left to right. Example:
    !   1   2   3
    !   4   5   6
    !   7   8   9
    !   10  11  12
    ! Then "teach" them who their neighbours are and if the neighbours
    ! are walls.


    ! initialise MPI 
    call mpi_init(error_number)

    ! specify communicator; 
    call mpi_comm_size(mpi_comm_world, nproc, error_number)

    ! returns process number to variable this_process_number
    call mpi_comm_rank(mpi_comm_world, myid, error_number)

    myid = myid+1


    do i=1, nproc_x
        if (myid==i) aboveme=wall ! create upper wall
        if (myid==nproc-i+1) belowme=wall !create lower wall
        do j=1, nproc_y

            if (myid==1+(j-1)*nproc_x) leftofme=wall !create left wall
            if (myid==nproc_x+(j-1)*nproc_x) rightofme=wall !create right wall

            !get neighbours above
            if (j /= 1) then
                if (myid==i+(j-1)*nproc_x) aboveme=i+(j-2)*nproc_x 
            end if

            !get neighbours below
            if (j/=nproc_y) then 
                if (myid==i+(j-1)*nproc_x) belowme=i+j*nproc_x 
            end if

            !get neighbours to the right 
            if (i/=nproc_x) then
                if(myid==i+(j-1)*nproc_x) rightofme=i+1+(j-1)*nproc_x
            end if

            !get neightbours to the left
            if (i/=1) then
                if (myid==i+(j-1)*nproc_x) leftofme=i-1+(j-1)*nproc_x
            end if
        end do
    end do

    do i=1, nproc
        if (myid==i) then
        write(*,'(A12, I4)') "Myid", myid
        write(*,'(A12, I4)') "Left of me", leftofme
        write(*,'(A12, I4)') "Right of me", rightofme
        write(*,'(A12, I4)') "Below me", belowme
        write(*,'(A12, I4)') "Above me", aboveme
        write(*,*)
        end if
        call mpi_barrier(mpi_comm_world, error_number)
    end do


    ! end MPI
        call mpi_finalize(error_number)


end program mapdomain2d

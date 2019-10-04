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


    ! Tells each processors who their neighbours to the left, 
    ! to the right, above and below are, or if there is a wall.
    ! Processors are mapped in the following way:
    ! myid=1 is always the lower left corner
    ! with nproc_x = 3 and nproc_y = 4:
    ! 10 11  12
    ! 7  8   9
    ! 4  5   6
    ! 1  2   3


    ! initialise MPI 
    call mpi_init(error_number)

    ! specify communicator; 
    call mpi_comm_size(mpi_comm_world, nproc, error_number)

    ! returns process number to variable this_process_number
    call mpi_comm_rank(mpi_comm_world, myid, error_number)

    myid = myid+1

    ! create procmap
    do i=1, nproc_x
        do j=1, nproc_y

            if (myid==1+(j-1)*nproc_x) leftofme=wall !create left wall
            if (myid==nproc_x+(j-1)*nproc_x) rightofme=wall !create right wall

            !get neighbours below
            if (myid==i+j*nproc_x) belowme=myid-nproc_x

            !get neighbours above
            if (j/=nproc_y) then
                if (myid==((j-1)*nproc_x)+i) aboveme=myid+nproc_x 
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
        if (myid==i) belowme=wall ! create upper wall
        if (myid==nproc-i+1) aboveme=wall !create lower wall
    end do



    ! write everything
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

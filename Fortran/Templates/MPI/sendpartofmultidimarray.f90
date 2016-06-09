! compile: mpiifort sendpartofmultidimarray.f90
! execution: mpirun -n 2 ./a.out
!
! program sendarray             line 16
! subroutine sendcolumn         line 36
! subroutine sendline           line 102
! subroutine sendmatrixblock    line 161
! subroutine send3dimsubarray   line 218
!
! subroutine makematrix         line 299
! subroutine writearray         line 326
! subroutine make3Dmatrix       line 356
! subroutine write3Darray       line 385
!
!
program sendarray

    use mpi
    implicit none
    integer :: code, rank

    call MPI_INIT (code)
    call MPI_COMM_RANK ( MPI_COMM_WORLD ,rank,code)

    !call sendcolumn(rank)
    !call sendline(rank)
    !call sendmatrixblock(rank)
    call send3dimsubarray(rank)
    call send3dimsubarray2(rank)
    
    call MPI_FINALIZE(code)

contains



subroutine sendcolumn(rank)
! sends first column of matrix in proc0 to last column of matrix in proc1

    use mpi
    implicit none
    integer, intent(in) :: rank

    integer, parameter:: linenumber=5,columnnumber=6
    integer, parameter:: tag=100
    real, dimension(linenumber,columnnumber):: a
    integer, dimension( MPI_STATUS_SIZE):: status
    integer::code,type_column



    if (rank == 1) then
        write(*, *) "############################"
        write(*, *) "subroutine sendcolumn"
        write(*, *)
    end if

    call makematrix(linenumber, columnnumber, rank, a)

    call writearray(linenumber, columnnumber, a, 'before send', rank)


    ! Definition of the type_column datatype
    call MPI_TYPE_CONTIGUOUS(linenumber, MPI_REAL ,type_column,code)
    ! how many elements, what kind, new type name, errorcode

    ! Validation of the type_column datatype
    call MPI_TYPE_COMMIT(type_column,code)

    ! Sending of the first column
    if ( rank == 0 ) then
        call MPI_SEND (a(1,1),1,type_column,1,tag, MPI_COMM_WORLD ,code)
        ! send 1 element of type type_column to proc 1 starting with a(1, 1)
    ! Reception in the last column
    elseif ( rank == 1 ) then
        call MPI_RECV (a(1,columnnumber),linenumber, MPI_REAL ,0,tag,&
        MPI_COMM_WORLD,status,code)
        ! receive starting at a(1, last column) linenumber amount of elements of kind REAL
        ! from processor 0
    end if


    call writearray(linenumber, columnnumber, a, 'after send', rank)


    ! Free the datatype
    call MPI_TYPE_FREE (type_column,code)


end subroutine sendcolumn





!###########################################################################
!###########################################################################
!###########################################################################




subroutine sendline(rank)
! send first line of one proc to last line of second proc
    use mpi
    implicit none
    integer, intent(in) :: rank

    integer, parameter:: linenumber=5,columnnumber=6
    integer, parameter:: tag=100
    real, dimension(linenumber,columnnumber):: a
    integer, dimension( MPI_STATUS_SIZE):: status
    integer:: code,type_line


    
 ! Initialization of the matrix on each process
    call makematrix(linenumber, columnnumber, rank, a)

    if (rank == 1) then
        write(*, *) "############################"
        write(*, *) "subroutine sendline"
        write(*, *)
    end if

    call writearray(linenumber, columnnumber, a, 'before send', rank)
    
    ! Creation of the datatype type_bloc
    call MPI_TYPE_VECTOR(columnnumber,1,linenumber,&
    MPI_REAL ,type_line,code)
    
    ! Validation of the datatype type_block
    call MPI_TYPE_COMMIT(type_line,code)

    ! Sending of a block
    if ( rank == 0 ) then
        call MPI_SEND (a(1,1),1,type_line,1,tag, MPI_COMM_WORLD,code)
    
    ! Reception of the block
    elseif ( rank == 1 ) then
        call MPI_RECV (a(linenumber,1),1,type_line,0,tag,&
        MPI_COMM_WORLD,status,code)
    end if


    call writearray(linenumber, columnnumber, a, 'after send', rank)
    
    ! Freeing of the datatype type_block
    call MPI_TYPE_FREE (type_line,code)


end subroutine sendline



!###########################################################################
!###########################################################################
!###########################################################################



subroutine sendmatrixblock(rank)
! send a block of a matrix across processors.
    use mpi
    implicit none
    integer, intent(in) :: rank
    integer, parameter:: linenumber=5,columnnumber=6
    integer, parameter:: tag=100
    integer, parameter:: linenumber_block=2,columnnumber_block=3
    real, dimension(linenumber,columnnumber):: a
    integer, dimension( MPI_STATUS_SIZE):: status
    integer:: code,type_block


    
 ! Initialization of the matrix on each process
    call makematrix(linenumber, columnnumber, rank, a)

    if (rank == 1) then
        write(*, *) "############################"
        write(*, *) "subroutine sendmatrixblock"
        write(*, *)
    end if

    call writearray(linenumber, columnnumber, a, 'before send', rank)
    
    ! Creation of the datatype type_bloc
    call MPI_TYPE_VECTOR(columnnumber_block,linenumber_block,linenumber,&
    MPI_REAL ,type_block,code)
    
    ! Validation of the datatype type_block
    call MPI_TYPE_COMMIT(type_block,code)

    ! Sending of a block
    if ( rank == 0 ) then
        call MPI_SEND (a(1,1),1,type_block,1,tag, MPI_COMM_WORLD,code)
    
    ! Reception of the block
    elseif ( rank == 1 ) then
        call MPI_RECV (a(linenumber-1,columnnumber-2),1,type_block,0,tag,&
        MPI_COMM_WORLD,status,code)
    end if


    call writearray(linenumber, columnnumber, a, 'after send', rank)
    
    ! Freeing of the datatype type_block
    call MPI_TYPE_FREE (type_block,code)


end subroutine sendmatrixblock


!###########################################################################
!###########################################################################
!###########################################################################

subroutine send3dimsubarray(rank)
! sends all rows and depth (y and z values) for the last 2 x values to other proc
! The rank of an array is its number of dimensions.
! The extent of an array is the number of elements in one dimension.
! The shape of an array is a vector for which each dimension equals 
! the extent.
! For example, the T(10,0:5,-10:10) array: Its rank is 3; its extent 
! in the first dimension is 10, in the second 6 and in the third 21; 
! so its shape is the (10,6,21) vector.

    use mpi
    implicit none
    integer, intent(in) :: rank

    integer, parameter:: linenumber=5,columnnumber=6, depth =3
    integer, parameter:: tag=100
    real, dimension(1:linenumber,1:columnnumber, 1:depth):: a
    integer, dimension( MPI_STATUS_SIZE):: status
    integer::code,type_subarray
    integer, dimension(3) :: shape_array, shape_subarray, start_coord

    shape_array(:) = (/linenumber, columnnumber, depth /)
    shape_subarray(:) = (/linenumber,2, depth/) ! von jeder Zeile 2 Spalten
    start_coord(:) = (/0, columnnumber-2, 0/)
    !!! ATTENTION!!!!
    ! the start_coord(:) array contains the indexes that MPI_TYPE_CREATE_SUBARRAY
    ! needs. BUT: Fortran usually starts with index 1, this MPI subroutine 
    ! starts with index 0 !!!!


!    if (rank ==1) write(*, '(3(I2, x))') shape_subarray, shape_array, start_coord

    if (rank == 1) then
        write(*, *) "############################"
        write(*, *) "subroutine send3dimarray"
        write(*, *)
    end if

    call make3Dmatrix(linenumber, columnnumber, depth, rank, a)
    call write3Darray(linenumber, columnnumber, depth, a, 'before send', rank)

    call MPI_TYPE_CREATE_SUBARRAY(3, shape_array, shape_subarray, start_coord, MPI_ORDER_FORTRAN, MPI_REAL, type_subarray, code)
    call MPI_TYPE_COMMIT(type_subarray, code)
    ! call MPI_TYPE_CREATE_SUBARRAY(nb_dims,shape_array,shape_sub_array,coord_start, order,old_type,new_type,code)
    ! nb_dims : rank of the array
    ! shape_array : shape of the array from which a subarray will be extracted
    ! shape_sub_array : shape of the subarray
    ! coord_start : start coordinates if the indices of the array start at 0.
    ! For example, if we want the start coordinates of the subarray to be 
    ! array(2,3), we must have coord_start(:)=(/ 1,2 /)
    ! order : storage order of elements

    call MPI_SENDRECV_REPLACE(a,1,type_subarray,mod(rank+1,2),tag, mod(rank+1,2),tag, MPI_COMM_WORLD,status,code)
    call MPI_TYPE_FREE (type_subarray,code)

    call write3Darray(linenumber, columnnumber, depth, a, 'after send', rank)

end subroutine send3dimsubarray


















subroutine send3dimsubarray2(rank)
! sends all rows and depth (y and z values) for the last 2 x values to other proc
! The rank of an array is its number of dimensions.
! The extent of an array is the number of elements in one dimension.
! The shape of an array is a vector for which each dimension equals 
! the extent.
! For example, the T(10,0:5,-10:10) array: Its rank is 3; its extent 
! in the first dimension is 10, in the second 6 and in the third 21; 
! so its shape is the (10,6,21) vector.

    use mpi
    implicit none
    integer, intent(in) :: rank

    integer, parameter:: linenumber=5,columnnumber=6, depth =3
    integer, parameter:: tag=100
    real, dimension(1:linenumber,1:columnnumber, 1:depth):: a
    integer, dimension( MPI_STATUS_SIZE):: status
    integer::code,type_subarray
    integer, dimension(3) :: shape_array, shape_subarray, start_coord

    shape_array(:) = (/linenumber, columnnumber, depth /)
    shape_subarray(:) = (/linenumber,2, depth/) ! von jeder Zeile 2 Spalten
    start_coord(:) = (/0, columnnumber-2, 0/)
    !!! ATTENTION!!!!
    ! the start_coord(:) array contains the indexes that MPI_TYPE_CREATE_SUBARRAY
    ! needs. BUT: Fortran usually starts with index 1, this MPI subroutine 
    ! starts with index 0 !!!!


!    if (rank ==1) write(*, '(3(I2, x))') shape_subarray, shape_array, start_coord

    if (rank == 1) then
        write(*, *) "############################"
        write(*, *) "subroutine send3dimarray"
        write(*, *)
    end if

    call make3Dmatrix(linenumber, columnnumber, depth, rank, a)
    call write3Darray(linenumber, columnnumber, depth, a, 'before send', rank)

    !call MPI_TYPE_CREATE_SUBARRAY(3, shape_array, shape_subarray, start_coord, MPI_ORDER_FORTRAN, MPI_REAL, type_subarray, code)
    !call MPI_TYPE_COMMIT(type_subarray, code)
    ! call MPI_TYPE_CREATE_SUBARRAY(nb_dims,shape_array,shape_sub_array,coord_start, order,old_type,new_type,code)
    ! nb_dims : rank of the array
    ! shape_array : shape of the array from which a subarray will be extracted
    ! shape_sub_array : shape of the subarray
    ! coord_start : start coordinates if the indices of the array start at 0.
    ! For example, if we want the start coordinates of the subarray to be 
    ! array(2,3), we must have coord_start(:)=(/ 1,2 /)
    ! order : storage order of elements

    !call MPI_SENDRECV(a, 1, type_subarray, 1, tag, a(1, columnnumber-1, 1), 1, type_subarray, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE, code)
  
    call MPI_SENDRECV(a(1, columnnumber-1, 1), 2*linenumber*depth,MPI_REAL, 1, tag, a(1, 1, 1), 2*linenumber*depth, MPI_REAL, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE, code) 

    !call MPI_TYPE_FREE (type_subarray,code)

    call write3Darray(linenumber, columnnumber, depth, a, 'after send', rank)

end subroutine send3dimsubarray2




!###########################################################################
!###########################################################################
!###########################################################################



!###########################################################################
!###########################################################################
!###########################################################################

!###########################################################################
!###########################################################################
!###########################################################################

!###########################################################################
!###########################################################################
!###########################################################################

!###########################################################################
!###########################################################################
!###########################################################################




subroutine makematrix(rows, columns, id, matrix)
! makes a rows x columns matrix where each element is rowscolumns (e.g. index (1, 1) will be 11).
! + if rank = 1, - if rank = 0
    implicit none
    integer, intent(in) :: rows, columns, id
    real, intent(out), dimension(:, :) :: matrix
    integer:: counter, another_counter, sign

    if (id ==0) then
        sign = -1
    else
        sign = 1
    end if
    
    do counter = 1, rows
        do another_counter = 1, columns
        matrix(counter, another_counter) = real((10* counter + another_counter )*sign)
        end do
    end do
end subroutine makematrix


!###########################################################################
!###########################################################################
!###########################################################################


subroutine writearray(rows, columns, array, message, id)
! prints matrix.

    implicit none
    integer, intent(in) :: rows, columns, id
    real, intent(in), dimension(:,:) :: array
    character(len=*), intent(in) :: message
    integer :: counter, another_counter

    if (id == 1) then
        write(*,*)
        write(*, *) message
        write(*, *)
        do counter=1, rows
            write(*, '(10(F6.2,x))') (array(counter, another_counter), another_counter=1, columns)
        end do
        write(*,*)
    end if



end subroutine writearray


!###########################################################################
!###########################################################################
!###########################################################################



subroutine make3Dmatrix(rows, columns, depth, id, matrix)
! makes a rows x columns x depth  matrix where each element is rowscolumns (e.g. index (1, 2, 3) will be 123).
! + if rank = 1, - if rank = 0
    implicit none
    integer, intent(in) :: rows, columns, depth, id
    real, intent(out), dimension(1:rows, 1:columns, 1:depth) :: matrix
    integer:: counter, another_counter, third_counter, sign

    if (id ==0) then
        sign = -1
    else
        sign = 1
    end if
    
    do counter = 1, rows
        do another_counter = 1, columns
            do third_counter = 1, depth
                matrix(counter, another_counter, third_counter) = real((100* counter + 10*another_counter + third_counter )*sign)
            end do
        end do
    end do
end subroutine make3Dmatrix


!###########################################################################
!###########################################################################
!###########################################################################


subroutine write3Darray(rows, columns, depth, array, message, id)
! prints matrix.

    implicit none
    integer, intent(in) :: rows, columns, id, depth
    real, intent(in), dimension(1:rows, 1:columns, 1:depth) :: array
    character(len=*), intent(in) :: message
    integer :: counter, another_counter, third_counter

    if (id == 1) then
        write(*,*)
        write(*, *) message
        write(*, *)
        do third_counter=1, depth
            write(*, *)
            write(*, '(A,x,I2)') "z = ", third_counter
            do counter = 1, rows
                write(*, '(10(F6.1,x))') (array(counter, another_counter, third_counter), another_counter=1, columns)
            end do
        end do
        write(*,*)
    end if



end subroutine write3Darray




end program sendarray

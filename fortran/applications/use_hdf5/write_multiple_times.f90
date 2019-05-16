!----------------------------------------------
! Examples on how to use hdf5.
! Write multiple times at different positions
! in a dataspace.
!----------------------------------------------


program use_hdf5


  use hdf5

  implicit none


  call write_dataspace()
  call read_simple_file()


contains

  !=============================================
  subroutine write_dataspace()
  !=============================================
    ! Write a simple file containing a simple
    ! dataset.
    !-------------------------------------------


    CHARACTER(LEN=18), PARAMETER :: filename = "multiple_writes.h5" ! File name
    CHARACTER(LEN=4),  PARAMETER :: dsetname = "dset"               ! Dataset name

    ! Use hdf5 types for portability
    INTEGER(HID_T) :: file_id       ! File identifier
    INTEGER(HID_T) :: dset_id       ! Dataset identifier
    INTEGER(HID_T) :: dspace_id     ! Dataspace identifier
    INTEGER(HID_T) :: memspace_id   ! Memory space (subdataspace) identifier


    INTEGER(HSIZE_T), DIMENSION(2) :: dims = (/6,6/) ! Dataset dimensions
    INTEGER(HSIZE_T), DIMENSION(2) :: dimsm = (/2,6/) ! Memory space dimensions
    INTEGER(HSIZE_T), DIMENSION(1:2) :: offset = (/0,0/) ! Hyperslab offset
    INTEGER(HSIZE_T), DIMENSION(1:2) :: stride = (/1,1/) ! Hyperslab stride 
    INTEGER(HSIZE_T), DIMENSION(1:2) :: block = (/1,1/)  ! Hyperslab block size 
    INTEGER(HSIZE_T), DIMENSION(1:2) :: count = (/2,6/)  ! Data count block size 
    INTEGER     ::   rank= 2                         ! Dataset rank


    INTEGER(HID_T), dimension(1:2, 1:6) :: dset_data      ! data to read/write

    INTEGER(HID_T), dimension(6:2, 1:6) :: fulldata      ! data to read/write
    INTEGER(HSIZE_T), DIMENSION(2) :: data_dims = (/6, 6/)

    INTEGER     ::   error ! Error flag

    INTEGER     :: i, j, r


    ! Initialize FORTRAN interface.
    CALL h5open_f(error)

    ! Create a new file using default properties.
    CALL h5fcreate_f(filename, H5F_ACC_TRUNC_F, file_id, error)

    ! Create the dataspace: a 4 x 6 matrix 
    CALL h5screate_simple_f(rank, dims, dspace_id, error)

    ! Create the dataset with default properties.
    CALL h5dcreate_f(file_id, dsetname, H5T_NATIVE_INTEGER, dspace_id, &
         dset_id, error)


    do r=1, 3

      ! create some data to store
      do i=1, 2
        do j=1, 6
          dset_data(i,j) = r 
        enddo
      enddo

      offset = (/(r-1)*2, 0/)

      CALL h5sselect_hyperslab_f(dspace_id, H5S_SELECT_SET_F, &
           offset, count, error, stride, BLOCK) 

      ! Create memory dataspace.
      dimsm = (/2, 6/)
      CALL h5screate_simple_f(rank, dimsm, memspace_id, error)

      ! write data
      call h5dwrite_f(dset_id, H5T_NATIVE_INTEGER, dset_data, dimsm, error, &
        memspace_id, dspace_id)


  data_dims = (/6,6/)
  CALL h5dread_f(dset_id, H5T_NATIVE_INTEGER, fulldata, data_dims, error)

  !
  ! Read entire dataset back 
  !
  WRITE(*,'(A)') "Data in File after Subset Written:"
  DO i = 1, 6 
    WRITE(*,'(100(1X,I0,1X))') fulldata(i,1:6)
  END DO
  PRINT *, " "


    enddo

    ! End access to the dataset and release resources used by it.
    CALL h5dclose_f(dset_id, error)

    ! Terminate access to the data space.
    CALL h5sclose_f(dspace_id, error)

    ! Close the file.
    CALL h5fclose_f(file_id, error)

    ! Don't forget to close FORTRAN interface.
    CALL h5close_f(error)

    write(*, '(A)') "Finished writing simple file."

  end subroutine


  !=============================================
  subroutine read_simple_file()
  !=============================================
    ! Read a simple file containing a simple
    ! dataset. Read in the file you wrote in
    ! subroutine write_simple_file and print
    ! the data to screen.
    !-------------------------------------------


    CHARACTER(LEN=18), PARAMETER :: filename = "multiple_writes.h5" ! File name
    CHARACTER(LEN=4), PARAMETER :: dsetname = "dset"                ! Dataset name

    ! Use hdf5 types for portability
    INTEGER(HID_T) :: file_id       ! File identifier
    INTEGER(HID_T) :: dset_id       ! Dataset identifier

    INTEGER(HSIZE_T), DIMENSION(2) :: dims = (/6,6/) ! Dataset dimensions

    INTEGER(HID_T), dimension(1:4, 1:6) :: dset_data      ! data to read/write

    INTEGER     ::   error ! Error flag
    INTEGER     ::   i, j


    ! Initialize FORTRAN interface.
    CALL h5open_f(error)

    ! Open an existing file.
    CALL h5fopen_f (filename, H5F_ACC_RDWR_F, file_id, error)

    ! Open an existing dataset.
    CALL h5dopen_f(file_id, dsetname, dset_id, error)

    ! Read the dataset.
    CALL h5dread_f(dset_id, H5T_NATIVE_INTEGER, dset_data, dims, error)

    ! Close the dataset.
    CALL h5dclose_f(dset_id, error)

    ! Close the file.
    CALL h5fclose_f(file_id, error)

    ! Close FORTRAN interface.
    CALL h5close_f(error)

    write(*, '(A)') "Finished reading simple file. Read in:"
    do i = 1, 6
      do j = 1, 6
        write(*, '(I5,x)', advance='no') dset_data(i, j)
      enddo
      write(*,*)
    enddo

  end subroutine


end program use_hdf5

!----------------------------------------------
! Examples on how to use hdf5.
! Write a hdf5 file, then read it back in 
! and print out what you read.
!----------------------------------------------


program use_hdf5


  use hdf5

  implicit none


  call write_simple_file()
  call read_simple_file()


contains

  !=============================================
  subroutine write_simple_file()
  !=============================================
    ! Write a simple file containing a simple
    ! dataset.
    !-------------------------------------------


    CHARACTER(LEN=14), PARAMETER :: filename = "simple_file.h5" ! File name
    CHARACTER(LEN=4),  PARAMETER :: dsetname = "dset"           ! Dataset name

    ! Use hdf5 types for portability
    INTEGER(HID_T) :: file_id       ! File identifier
    INTEGER(HID_T) :: dset_id       ! Dataset identifier
    INTEGER(HID_T) :: dspace_id     ! Dataspace identifier


    INTEGER(HSIZE_T), DIMENSION(2) :: dims = (/4,6/) ! Dataset dimensions
    INTEGER     ::   rank= 2                         ! Dataset rank


    INTEGER(HID_T), dimension(1:4, 1:6) :: dset_data      ! data to read/write

    INTEGER     ::   error ! Error flag

    INTEGER     :: i, j


    ! Initialize FORTRAN interface.
    CALL h5open_f(error)

    ! Create a new file using default properties.
    CALL h5fcreate_f(filename, H5F_ACC_TRUNC_F, file_id, error)

    ! Create the dataspace: a 4 x 6 matrix 
    CALL h5screate_simple_f(rank, dims, dspace_id, error)

    ! Create the dataset with default properties.
    CALL h5dcreate_f(file_id, dsetname, H5T_NATIVE_INTEGER, dspace_id, &
         dset_id, error)

    ! create some data to store
    do i=1, 4
      do j=1, 6
        dset_data(i,j) = 10*i + j
      enddo
    enddo

    ! write data
    call h5dwrite_f(dset_id, H5T_NATIVE_INTEGER, dset_data, dims, error)

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


    CHARACTER(LEN=14), PARAMETER :: filename = "simple_file.h5" ! File name
    CHARACTER(LEN=4), PARAMETER :: dsetname = "dset"            ! Dataset name

    ! Use hdf5 types for portability
    INTEGER(HID_T) :: file_id       ! File identifier
    INTEGER(HID_T) :: dset_id       ! Dataset identifier

    INTEGER(HSIZE_T), DIMENSION(2) :: dims = (/4,6/) ! Dataset dimensions

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
    do i = 1, 4
      do j = 1, 6
        write(*, '(I5,x)', advance='no') dset_data(i, j)
      enddo
      write(*,*)
    enddo

  end subroutine


end program use_hdf5

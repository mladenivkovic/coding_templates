program concatenate_arrays

  use hdf5

  implicit none

  character(len=26), parameter :: frows = "concatenated_array_rows.h5" ! Output file name
  character(len=26), parameter :: fcols = "concatenated_array_cols.h5" ! Output file name

  call concatenate_rows(frows)
  call read_result(frows)
  call concatenate_columns(fcols)
  call read_result(fcols)

  contains

  !=============================================
  subroutine concatenate_rows(outfilename)
  !=============================================

    !--------------------------------------------------------
    ! Concatenate 2d-arrays with same y-dimension
    ! along x axis in a loop [array(1:x, 1:y)]
    ! NOTE: fortran and C arrays are stored in different
    ! order. (5x3) array in fortran is (3x5) in C.
    ! Effectively in the file, the rows are concatenated.
    ! a1 a2 a3
    ! +
    ! b1 b2 b3
    ! +
    ! c1 c2 c3
    ! +
    ! d1 d2 d3
    !--------------------------------------------------------

    implicit none

    ! HDF5 file and path names
    character(len=*) :: outfilename
    character(len=8), parameter :: dsetname = "dsetname"

    ! Declare HDF5 IDs
    integer(HID_T) :: outfile_id      ! File identifier
    integer(HID_T) :: dset_id         ! dataset identifier    
    integer(HID_T) :: dataspace_id    ! dataspace identifier
    integer(HID_T) :: memspace_id     ! memory space identifier
    integer(HID_T) :: crp_list        ! dataset creation property identifier

    ! Dataset dimensions in the file
    integer(HSIZE_T), dimension(1:2) :: dset_dims ! dataset dimensions at creation time
    integer(HSIZE_T), dimension(1:2) :: chunk_dims
    integer(HSIZE_T), dimension(1:2) :: memory_dims ! dimensions of memory space; In our case = count(:, :)

    ! maximum dimensions
    integer(HSIZE_T), dimension(1:2) :: maxdims
    integer(HSIZE_T), dimension(1:2) :: offset, count

    ! data
    integer, dimension(:,:), allocatable :: my_data

    integer(HSIZE_T), dimension(1:2) :: data_dims
    integer(HSIZE_T), dimension(1:2) :: size


    integer(HSIZE_T) :: i, j, r, rep
    integer(HSIZE_T) :: rows, columns
    integer(HSIZE_T) :: zero = 0
    integer :: error, rank=2 ! HDF5 error flag


    ! Initialize FORTRAN interface
    call h5open_f(error)

    ! Create a new file using the default properties.
    call h5fcreate_f(outfilename, H5F_ACC_TRUNC_F, outfile_id, error)

    ! Create the data space with unlimited dimensions.
    ! create some junk data to initialize dataset
    rows = 3
    columns = 1
    dset_dims = (/rows, columns/)
    maxdims = (/rows, H5S_UNLIMITED_F/)
    call h5screate_simple_f(rank, dset_dims, dataspace_id, error, maxdims)

    ! Modify dataset creation properties, i.e. enable chunking
    call h5pcreate_f(H5P_DATASET_CREATE_F, crp_list, error)
    chunk_dims = (/rows, columns/)
    call h5pset_chunk_f(crp_list, rank, chunk_dims, error) 


    ! create initial dataset
    call h5dcreate_f(outfile_id, dsetname, H5T_NATIVE_INTEGER, dataspace_id, &
      dset_id, error, crp_list)
    call h5sclose_f(dataspace_id, error)

    ! fill initial data array and write it to dataset
    data_dims = (/rows, columns/)
    allocate(my_data(1:rows, 1:columns))
    do i = 1, data_dims(1)
      do j = 1, data_dims(2)
        my_data(i, j) = int(-1)
      enddo
    enddo

    call h5dwrite_f(dset_id, H5T_NATIVE_INTEGER, my_data, data_dims, error)
    deallocate(my_data)


    offset(1:2) = (/zero, zero/) ! set initial offset to zero; overwrite initial junk
    count(1:2) = (/zero, zero/)
    size(1:2) = (/rows, zero/)

    rep = 5
    do r = 1, rep
      ! for every loop, add <loop counter> rows containing <loop counter> as value


      ! extend the dataset
      size(2) = size(2) + r
      call h5dset_extent_f(dset_id, size, error)

      ! Create the memory space for the selection
      count = (/rows, r/)
      memory_dims = count
      call h5screate_simple_f(rank, memory_dims, memspace_id, error)

      ! create data and write to extended part of dataset
      call h5dget_space_f(dset_id, dataspace_id, error)
      offset(2) = offset(2) + r - 1
      call h5sselect_hyperslab_f(dataspace_id, H5S_SELECT_SET_F, offset, count, error)

      data_dims = (/rows, r/)
      allocate(my_data(1:rows, 1:r))
      do i = 1, data_dims(1)
        do j = 1, data_dims(2)
          my_data(i,j) = int(r)
        enddo
      enddo
      call h5dwrite_f(dset_id, H5T_NATIVE_INTEGER, my_data, data_dims, error, memspace_id, dataspace_id)
      deallocate(my_data)

    enddo

    ! Close dataspace, dataset, and file
    call h5dclose_f(dset_id, error)
    call h5sclose_f(memspace_id, error)
    call h5fclose_f(outfile_id, error)

    call h5close_f(error)

  end subroutine concatenate_rows




  !=============================================
  subroutine concatenate_columns(outfilename)
  !=============================================

    !--------------------------------------------------------
    ! Concatenate 2d-arrays with same x-dimension
    ! along y axis in a loop
    ! NOTE: fortran and C arrays are stored in different
    ! order. (5x3) array in fortran is (3x5) in C.
    ! Effectively in the file, the columns are concatenated.
    ! a1 + b1 + c1 + d1
    ! a2 + b2 + c2 + d2
    ! a3 + b3 + c3 + d3
    !--------------------------------------------------------

    implicit none

    ! HDF5 file and path names
    character(len=*) :: outfilename
    character(len=8), parameter :: dsetname = "dsetname"

    ! Declare HDF5 IDs
    integer(HID_T) :: outfile_id      ! File identifier
    integer(HID_T) :: dset_id         ! dataset identifier    
    integer(HID_T) :: dataspace_id    ! dataspace identifier
    integer(HID_T) :: memspace_id     ! memory space identifier
    integer(HID_T) :: crp_list        ! dataset creation property identifier

    ! Dataset dimensions in the file
    integer(HSIZE_T), dimension(1:2) :: dset_dims ! dataset dimensions at creation time
    integer(HSIZE_T), dimension(1:2) :: chunk_dims
    integer(HSIZE_T), dimension(1:2) :: memory_dims ! dimensions of memory space; In our case = count(:, :)

    ! maximum dimensions
    integer(HSIZE_T), dimension(1:2) :: maxdims
    integer(HSIZE_T), dimension(1:2) :: offset, count

    ! data
    integer, dimension(:,:), allocatable :: my_data

    integer(HSIZE_T), dimension(1:2) :: data_dims
    integer(HSIZE_T), dimension(1:2) :: size


    integer(HSIZE_T) :: i, j, r, rep
    integer(HSIZE_T) :: rows, columns
    integer(HSIZE_T) :: one = 1, zero = 0
    integer :: error, rank=2 ! HDF5 error flag


    ! Initialize FORTRAN interface
    call h5open_f(error)

    ! Create a new file using the default properties.
    call h5fcreate_f(outfilename, H5F_ACC_TRUNC_F, outfile_id, error)

    ! Create the data space with unlimited dimensions.
    ! create some junk data to initialize dataset
    rows = 1
    columns = 3
    dset_dims = (/rows, columns/)
    maxdims = (/H5S_UNLIMITED_F, columns/)
    call h5screate_simple_f(rank, dset_dims, dataspace_id, error, maxdims)

    ! Modify dataset creation properties, i.e. enable chunking
    call h5pcreate_f(H5P_DATASET_CREATE_F, crp_list, error)
    chunk_dims = (/one, columns/)
    call h5pset_chunk_f(crp_list, rank, chunk_dims, error) 


    ! create initial dataset
    call h5dcreate_f(outfile_id, dsetname, H5T_NATIVE_INTEGER, dataspace_id, &
      dset_id, error, crp_list)
    call h5sclose_f(dataspace_id, error)

    ! fill initial data array and write it to dataset
    allocate(my_data(1:rows, 1:columns))
    data_dims = (/rows, columns/)
    do i = 1, data_dims(1)
      do j = 1, data_dims(2)
        my_data(i, j) = int(-1)
      enddo
    enddo

    call h5dwrite_f(dset_id, H5T_NATIVE_INTEGER, my_data, data_dims, error)
    deallocate(my_data)


    offset(1:2) = (/zero, zero/) ! set initial offset to zero; overwrite initial junk
    count(1:2) = (/zero, zero/)
    size(1:2) = (/zero, columns/)

    rep = 5
    do r = 1, rep
      ! for every loop, add <loop counter> rows containing <loop counter> as value


      ! extend the dataset
      size(1) = size(1) + r
      call h5dset_extent_f(dset_id, size, error)

      ! Create the memory space for the selection
      count = (/r, columns/)
      memory_dims = count
      call h5screate_simple_f(rank, memory_dims, memspace_id, error)

      ! create data and write to extended part of dataset
      call h5dget_space_f(dset_id, dataspace_id, error)
      offset(1) = offset(1) + r - 1
      call h5sselect_hyperslab_f(dataspace_id, H5S_SELECT_SET_F, offset, count, error)

      data_dims = (/r, columns/)
      allocate(my_data(1:r, 1:columns))
      do i = 1, data_dims(1)
        do j = 1, data_dims(2)
          my_data(i,j) = int(r)
        enddo
      enddo
      call h5dwrite_f(dset_id, H5T_NATIVE_INTEGER, my_data, data_dims, error, memspace_id, dataspace_id)
      deallocate(my_data)

    enddo

    ! Close dataspace, dataset, and file
    call h5dclose_f(dset_id, error)
    call h5sclose_f(memspace_id, error)
    call h5fclose_f(outfile_id, error)

    call h5close_f(error)

  end subroutine concatenate_columns





  !=====================================
  subroutine read_result(outfilename)
  !=====================================
    ! Read in data from file written by
    ! append_x routine
    !----------------------------------

    character(len=*) :: outfilename
    character(len=8), parameter :: dsetname = "dsetname"

    ! Use hdf5 types for portability
    INTEGER(HID_T) :: file_id                   ! File identifier
    INTEGER(HID_T) :: dset_id                   ! Dataset identifier
    INTEGER(HID_T) :: dspace_id                 ! Dataspace ID

    INTEGER(HSIZE_T), DIMENSION(:), allocatable :: dset_dims    ! Dataset dimensions
    INTEGER(HSIZE_T), DIMENSION(:), allocatable :: dset_maxdims ! Maximal dataset dimensions

    INTEGER, dimension(:,:), allocatable :: read_in_data  ! data to read

    INTEGER          ::   error, rank
    INTEGER(HSIZE_T) :: i, j


    ! Initialize FORTRAN interface.
    CALL h5open_f(error)

    ! Open an existing file.
    CALL h5fopen_f(outfilename, H5F_ACC_RDWR_F, file_id, error)

    ! Open an existing dataset.
    CALL h5dopen_f(file_id, dsetname, dset_id, error)

    ! Get dataspace of opened dataset
    CALL h5dget_space_f(dset_id, dspace_id, error)

    ! Get dataset rank
    CALL h5sget_simple_extent_ndims_f(dspace_id, rank, error)

    ! Get dataset dimensions
    allocate(dset_dims(1:rank))
    allocate(dset_maxdims(1:rank))
    CALL h5sget_simple_extent_dims_f(dspace_id, dset_dims, dset_maxdims, error)

    ! allocate space to read in file
    allocate(read_in_data(1:dset_dims(1), 1:dset_dims(2)))

    ! Read the dataset.
    CALL h5dread_f(dset_id, H5T_NATIVE_INTEGER, read_in_data, dset_dims, error)

    ! Close the dataset.
    CALL h5dclose_f(dset_id, error)

    ! Close the file.
    CALL h5fclose_f(file_id, error)

    ! Close FORTRAN interface.
    CALL h5close_f(error)



    write(*, '(3A)') "Finished reading file ", outfilename, ". Read in:"
    do i = 1, dset_dims(1)
      do j = 1, dset_dims(2)
        write(*, '(I5,x)', advance='no') read_in_data(i, j)
      enddo
      write(*,*)
    enddo



  end subroutine read_result 
  
end program concatenate_arrays

program append_datasets

  use hdf5

  implicit none


  call append_y
  call read_result("appended_datasets-y.h5")

  call append_x
  call read_result("appended_datasets-x.h5")

  contains



  !===============================
  subroutine append_y()
  !===============================

    !-------------------------------------------
    ! Concatenate arrays of equal x-dimension
    ! along y axis in hdf5 file
    !-------------------------------------------

    implicit none

    ! HDF5 file and path names
    character(len=22), parameter :: outfilename = "appended_datasets-y.h5" ! Output file name
    character(len=8), parameter :: dsetname = "dsetname"

    ! Declare HDF5 IDs
    integer(HID_T) :: outfile_id       ! File identifier
    integer(HID_T) :: dset_id            
    integer(HID_T) :: dataspace_id 
    integer(HID_T) :: memspace_id 

    ! Dataset dimensions in the file
    integer(HSIZE_T), dimension(2) :: chunk_dims = (/7,1/)        ! Chunk dimensions
    integer(HSIZE_T), dimension(2) :: full_dset_dims = (/7,9/)  ! File dimensions

    ! Data buffers
    integer, dimension(7,1) :: data_chunk  ! Chunk dimension

    ! Chunk parameters
    integer(HSIZE_T), dimension(2) :: data_offset, data_chunkcount

    integer :: rank = 2 
    integer(HSIZE_T) :: i, j, k
    integer :: error ! HDF5 error flag

    ! Initialize FORTRAN interface
    call h5open_f(error)

    ! Create a new file using the default properties.
    call h5fcreate_f(outfilename, H5F_ACC_TRUNC_F, outfile_id, error)

    ! Create the data space for the dataset.
    call h5screate_simple_f(rank, full_dset_dims, dataspace_id, error)

    ! Create the chunked dataset.
    call h5dcreate_f(outfile_id, dsetname, H5T_NATIVE_INTEGER, dataspace_id, dset_id, error)

    ! Create the memory space for the selection
    call h5screate_simple_f(rank, chunk_dims, memspace_id, error)

    ! Select hyperslab
    data_offset(1) = 0
    data_chunkcount = (/7,1/)


    DO i=1, full_dset_dims(2) 

      ! fill up data
      do j = 1, chunk_dims(1) 
        do k = 1, chunk_dims(2) 
          data_chunk(j,k) = int(100*i+10*j+k)
        end do
      end do


      data_offset(2) = i-1
      call h5sselect_hyperslab_f(dataspace_id, H5S_SELECT_SET_F, data_offset, data_chunkcount, error)

      ! Write the data to the dataset.
      call h5dwrite_f(dset_id, H5T_NATIVE_INTEGER, data_chunk, chunk_dims, error, memspace_id, dataspace_id)
      call h5sclose_f(dataspace_id, error)
      call h5dget_space_f(dset_id, dataspace_id, error)
    END DO

    ! Close dataspace, dataset, and file
    call h5dclose_f(dset_id, error)
    call h5sclose_f(memspace_id, error)
    call h5fclose_f(outfile_id, error)

    call h5close_f(error)

  end subroutine append_y






  !===============================
  subroutine append_x()
  !===============================

    !------------------------------------------------
    ! Concatenate 2d-arrays with same y-dimension
    ! along x axis
    !------------------------------------------------

    implicit none

    ! HDF5 file and path names
    character(len=22), parameter :: outfilename = "appended_datasets-x.h5" ! Output file name
    character(len=8), parameter :: dsetname = "dsetname"

    ! Declare HDF5 IDs
    integer(HID_T) :: outfile_id      ! File identifier
    integer(HID_T) :: dset_id         ! dataset identifier    
    integer(HID_T) :: dataspace_id    ! dataspace identifier
    integer(HID_T) :: memspace_id     ! memory space identifier
    integer(HID_T) :: crp_list        ! dataset creation property identifier

    ! Dataset dimensions in the file
    integer(HSIZE_T), dimension(1:2) :: dset_init_dims = (/3, 3/) ! dataset dimensions at creation time
    integer(HSIZE_T), dimension(1:2) :: dimsc = (/2, 5/)          ! todo: ???
    integer(HSIZE_T), dimension(1:2) :: dimsm = (/3, 7/)          ! todo: ???

    ! maximum dimensions
    integer(HSIZE_T), dimension(1:2) :: maxdims
    integer(HSIZE_T), dimension(1:2) :: offset, count

    ! data
    integer, dimension(1:3, 1:3) :: data_init
    integer, dimension(1:3, 1:7) :: data_new

    integer(HSIZE_T), dimension(1:2) :: data_dims
    integer(HSIZE_T), dimension(1:2) :: size


    integer(HSIZE_T) ::  i, j, k
    integer :: error, rank=2 ! HDF5 error flag

    ! Initialize FORTRAN interface
    call h5open_f(error)

    ! Create a new file using the default properties.
    call h5fcreate_f(outfilename, H5F_ACC_TRUNC_F, outfile_id, error)

    ! Create the data space with unlimited dimensions.
    maxdims = (/H5S_UNLIMITED_F, H5S_UNLIMITED_F/)
    call h5screate_simple_f(rank, dset_init_dims, dataspace_id, error, maxdims)

    ! Modify dataset creation properties, i.e. enable chunking
    call h5pcreate_f(H5P_DATASET_CREATE_F, crp_list, error)
    call h5pset_chunk_f(crp_list, rank, dimsc, error) ! TODO: dimsc????

    ! create initial dataset
    call h5dcreate_f(outfile_id, dsetname, H5T_NATIVE_INTEGER, dataspace_id, &
      dset_id, error, crp_list)
    call h5sclose_f(dataspace_id, error)

    ! fill initial data array and write it to dataset
    do i = 1, dset_init_dims(1)
      do j = 1, dset_init_dims(2)
        data_init(i, j) = 1
      enddo
    enddo

    data_dims(1:2) = (/3, 3/)
    call h5dwrite_f(dset_id, H5T_NATIVE_INTEGER, data_init, data_dims, error)

    ! extend the dataset
    size(1:2) = (/3, 10/)
    call h5dset_extent_f(dset_id, size, error)
    offset(1:2) = (/0, 3/)
    count(1:2) = (/3, 7/)

    ! Create the memory space for the selection
    call h5screate_simple_f(rank, dimsm, memspace_id, error)

    ! create data and write to extended part of dataset
    call h5dget_space_f(dset_id, dataspace_id, error)
    call h5sselect_hyperslab_f(dataspace_id, H5S_SELECT_SET_F, offset, count, error)

    data_dims = (/3, 7/)
    do i = 1, data_dims(1)
      do j = 1, data_dims(2)
        data_new(i,j) = 2
      enddo
    enddo
    call h5dwrite_f(dset_id, H5T_NATIVE_INTEGER, data_new, data_dims, error, memspace_id, dataspace_id)

    ! Close dataspace, dataset, and file
    call h5dclose_f(dset_id, error)
    call h5sclose_f(memspace_id, error)
    call h5fclose_f(outfile_id, error)

    call h5close_f(error)

  end subroutine append_x





  !=====================================
  subroutine read_result(outfilename)
  !=====================================
    ! Read in data from file written by
    ! append_x routine
    !----------------------------------

    character(len=*) :: outfilename
    ! character(len=22), parameter :: outfilename = "appended_datasets-x.h5" ! Output file name
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



    write(*, '(A)') "Finished reading appended-y file. Read in:"
    do i = 1, dset_dims(1)
      do j = 1, dset_dims(2)
        write(*, '(I5,x)', advance='no') read_in_data(i, j)
      enddo
      write(*,*)
    enddo



  end subroutine read_result 
  
end program append_datasets

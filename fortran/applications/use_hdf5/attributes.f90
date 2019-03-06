program creating_attributes

  !---------------------------------
  ! Reading and writing attributes.
  !---------------------------------

  use hdf5
  implicit none

  character(len=24), parameter :: filename = "creating_attributes.h5" ! File name
  character(len=4), parameter  :: dsetname = "dset"                   ! Dataset name
  character(len=9), parameter  :: aname = "attr_long"                 ! Attribute name

  integer(HID_T) :: file_id       ! File identifier
  integer(HID_T) :: dset_id       ! Dataset identifier
  integer(HID_T) :: dspace_id     ! Dataspace identifier
  integer(HID_T) :: attr_id       ! Attribute identifier
  integer(HID_T) :: aspace_id     ! Attribute Dataspace identifier
  integer(HID_T) :: atype_id      ! Attribute Dataspace identifier
  integer(HSIZE_T), dimension(1)      :: adims = (/2/)  ! Attribute dimension
  integer                             :: arank = 1      ! Attribure rank
  integer(SIZE_T)                     :: attrlen        ! Length of the attribute string

  character(len=80), dimension(2)     :: attr_data      ! Attribute data

  integer(HSIZE_T), dimension(1)      :: data_dims
  integer(HSIZE_T), dimension(2)      :: dims = (/4,6/) ! Dataset dimensions

  integer(HID_T), dimension(1:4, 1:6) :: dset_data      ! data to read/write
  integer                             ::   rank= 2      ! Dataset rank
  integer     :: i, j
  integer     :: error ! Error flag





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
  ! CALL h5dclose_f(dset_id, error)







  ! Open an existing dataset.
  ! CALL h5dopen_f(file_id, dsetname, dset_id, error)

  ! Initialize attribute's data
  attr_data(1) = "Dataset character attribute"
  attr_data(2) = "Some other string here     "
  attrlen = 80

  ! Create scalar data space for the attribute.
  CALL h5screate_simple_f(arank, adims, aspace_id, error)

  ! Create datatype for the attribute.
  CALL h5tcopy_f(H5T_NATIVE_CHARACTER, atype_id, error)
  CALL h5tset_size_f(atype_id, attrlen, error)

  ! Create dataset attribute.
  CALL h5acreate_f(dset_id, aname, atype_id, aspace_id, attr_id, error)

  ! Write the attribute data.
  data_dims(1) = 2
  CALL h5awrite_f(attr_id, atype_id, attr_data, data_dims, error)

  ! Close the attribute.
  CALL h5aclose_f(attr_id, error)

  ! Terminate access to the data space.
  CALL h5sclose_f(aspace_id, error)

  ! End access to the dataset and release resources used by it.
  CALL h5dclose_f(dset_id, error)

  ! Close the file.
  CALL h5fclose_f(file_id, error)

  ! Close FORTRAN interface.
  CALL h5close_f(error)




end program creating_attributes

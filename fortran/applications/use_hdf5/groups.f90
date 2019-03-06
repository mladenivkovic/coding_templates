program creating_groups

  !--------------------------------------
  ! Create multiple groups and datasets
  ! within those groups.
  !--------------------------------------

  use hdf5
  implicit none

  character(len=9),  parameter :: filename = "group.h5"     ! File name
  character(len=8),  parameter :: group1name = "MyGroup1"   ! Group name
  character(len=9),  parameter :: group2name = "MyGroup2"   ! Group name
  character(len=13), parameter :: dsetname1 = "dset1"       ! Dataset name
  character(len=9), parameter  :: aname1 = "attr_char"      ! Attribute name
  character(len=9), parameter  :: aname2 = "attr_real"      ! Attribute name

  integer(HID_T) :: file_id       ! File identifier
  integer(HID_T) :: group_id      ! Group identifier
  integer(HID_T) :: dataset_id    ! Dataset identifier
  integer(HID_T) :: dataspace_id  ! Data space identifier
  integer(HID_T) :: attr_id       ! Attribute identifier
  integer(HID_T) :: aspace_id     ! Attribute Dataspace identifier
  integer(HID_T) :: atype_id      ! Attribute Dataspace identifier


  integer(HSIZE_T), dimension(1)      :: adims          ! Attribute dimension
  integer                             :: arank = 1      ! Attribure rank
  integer(SIZE_T)                     :: attrlen        ! Length of the attribute string
  character(len=80), dimension(2)     :: attr1_data     ! Attribute data
  real(kind=kind(1.d0))               :: attr2_data     ! Attribute data

  integer ::   error ! Error flag
  integer ::   i, j
   
  integer, dimension(3,3)  :: dset1_data  ! Data arrays

  integer(HSIZE_T), dimension(2) :: dims1 = (/3,3/) ! Datasets dimensions
  integer(HSIZE_T), dimension(2) :: data_dims

  integer     ::   rank = 2 ! Datasets rank


  ! Initialize dset1_data array
  do i = 1, 3
     do j = 1, 3
        dset1_data(i,j) = j;
     end do
  end do



  ! Initialize FORTRAN interface.
  CALL h5open_f(error)

  ! Create a new file using default properties.
  CALL h5fcreate_f(filename, H5F_ACC_TRUNC_F, file_id, error)





  !-------------------------------------------------
  ! Create a group named "/MyGroup1" in the file.
  !-------------------------------------------------
  CALL h5gcreate_f(file_id, group1name, group_id, error)
  CALL h5gclose_f(group_id, error)

  ! Open an existing group in the specified file.
  CALL h5gopen_f(file_id, group1name, group_id, error)



  ! Create the data space for the first dataset.
  CALL h5screate_simple_f(rank, dims1, dataspace_id, error)

  ! Create a dataset in group "MyGroup" with default properties.
  CALL h5dcreate_f(GROUP_ID, dsetname1, H5T_NATIVE_INTEGER, dataspace_id, &
       dataset_id, error)

  ! Write the first dataset.
  data_dims(1) = 3
  data_dims(2) = 3
  CALL h5dwrite_f(dataset_id, H5T_NATIVE_INTEGER, dset1_data, data_dims, error)

  ! Close the dataspace for the first dataset.
  CALL h5sclose_f(dataspace_id, error)

  ! Close the first dataset.
  CALL h5dclose_f(dataset_id, error)

  ! Close the group.
  CALL h5gclose_f(group_id, error)




  !-------------------------------------------------
  ! Create a group named "/MyGroup2" in the file.
  !-------------------------------------------------
  CALL h5gcreate_f(file_id, group2name, group_id, error)

  ! Initialize attribute 1's data
  attr1_data(1) = "Group character attribute"
  attr1_data(2) = "Some other string here     "
  attrlen = 80
  adims = (/2/)

  ! Create scalar data space for the attribute.
  CALL h5screate_simple_f(arank, adims, aspace_id, error)

  ! Create datatype for the attribute.
  CALL h5tcopy_f(H5T_NATIVE_CHARACTER, atype_id, error)
  CALL h5tset_size_f(atype_id, attrlen, error)

  ! Create group attribute.
  CALL h5acreate_f(group_id, aname1, atype_id, aspace_id, attr_id, error)

  ! Write the attribute data.
  data_dims(1) = 2
  CALL h5awrite_f(attr_id, atype_id, attr1_data, data_dims, error)

  ! Close the attribute.
  CALL h5aclose_f(attr_id, error)

  ! Terminate access to the data space.
  CALL h5sclose_f(aspace_id, error)


  attr2_data = 3.1415926
  adims = (/1/)
  ! Create scalar data space for the attribute.
  CALL h5screate_simple_f(arank, adims, aspace_id, error)

  ! Create group attribute.
  CALL h5acreate_f(group_id, aname2, H5T_NATIVE_DOUBLE, aspace_id, attr_id, error)
  ! Write the attribute data.
  data_dims(1) = 1
  CALL h5awrite_f(attr_id, H5T_NATIVE_DOUBLE, attr2_data, data_dims, error)

  ! Close the attribute.
  CALL h5aclose_f(attr_id, error)

  ! Terminate access to the data space.
  CALL h5sclose_f(aspace_id, error)



  ! Close the group.
  CALL h5gclose_f(group_id, error)
  ! Terminate access to the file.
  CALL h5fclose_f(file_id, error)

  ! Close FORTRAN interface.
  CALL h5close_f(error)



end program creating_groups

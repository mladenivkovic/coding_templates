!======================================================
! A short demonstration of alltoall vs allreduce.
! Meant to be executed with mpirun -n 4
!======================================================

program alltoall
  use mpi
  implicit none
  integer,allocatable, dimension(:) :: send_array, receive_array_1
  integer,allocatable, dimension(:) :: receive_array_2
  integer :: error_number, myid, nproc
  integer :: i, j, k


  !--------------
  ! Setup
  !--------------

  ! initialize MPI
  call mpi_init(error_number)
  
  ! write how many processes are used in nproc
  call mpi_comm_size(mpi_comm_world, nproc, error_number)
  
  !write which process this is in myid
  call mpi_comm_rank(mpi_comm_world, myid, error_number)


  myid = myid+1

  allocate(send_array(1:nproc))
  allocate(receive_array_1(1:nproc))
  allocate(receive_array_2(1:nproc))

  send_array = -1
  receive_array_1 = -1
  receive_array_2 = -1

  do i = 1, nproc
    send_array(i) = 100*myid+i
    if (myid == i) then
      send_array(i) = 100*myid+i + 10
    end if
  end do





  !---------------------
  ! Print before send
  !---------------------

  call MPI_BARRIER(MPI_COMM_WORLD, error_number)

  if (myid == 1) then
    write(*,*) "Before sending"
  end if

  do i = 1, nproc
    if (myid == i) then
      write(*,'(A3,x,I2,x,A3,x,4(I4,x),A12,x,4(I4,x),A12,x,4(I4,x))') "ID", myid, "S:", send_array, &
        "R gather:",receive_array_1, "R toall:", receive_array_2 
    else
      !waste some time to get ordered printing
      k = 0
      do j = 1, 500000
        k = k*j
      end do
    end if
  end do



  !-----------------
  ! Send 
  !-----------------

  call MPI_ALLTOALL(send_array, 1, MPI_INT, receive_array_1, 1, MPI_INT, MPI_COMM_WORLD, error_number)
  call MPI_ALLGATHER(send_array(myid), 1, MPI_INT, receive_array_2, 1, MPI_INT, MPI_COMM_WORLD, error_number)





  !---------------------
  ! Print after send
  !---------------------

  call MPI_BARRIER(MPI_COMM_WORLD, error_number)

  if (myid == 1) then
    write(*,*)
    write(*,* ) "After sending"
  end if

  call MPI_BARRIER(MPI_COMM_WORLD, error_number)

  do i = 1, nproc
    if (myid == i) then
      write(*,'(A3,x,I2,x,A3,x,4(I4,x),A12,x,4(I4,x),A12,x,4(I4,x))') "ID", myid, "S:", send_array, &
        "R gather:",receive_array_1, "R toall:", receive_array_2
    else
      !waste some time to get ordered printing
      k = 0
      do j = 1, 500000
        k = k*j
      end do
    end if
  end do

  call mpi_finalize(error_number)


end program alltoall

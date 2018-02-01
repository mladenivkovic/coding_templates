!=======================================
! A program to demonstrate reading and
! writing to/from the same file with MPI
!=======================================



program readwrite_mpi

  use mpi

  implicit none

  integer :: myid, ncpu
  integer :: mpi_err, filehandle
  integer, dimension(MPI_STATUS_SIZE) :: state

  integer, dimension(1:2) :: readbuffer
  integer, dimension(:), allocatable :: writebuffer_shared, readbuffer_shared
  integer :: i

  character(len=80) :: filename


  !-----------------------------
  ! Start MPI
  !-----------------------------


  call MPI_INIT(mpi_err)

  call MPI_COMM_RANK(MPI_COMM_WORLD, myid, mpi_err)

  call MPI_COMM_SIZE(MPI_COMM_WORLD, ncpu, mpi_err)

  myid = myid + 1






  !-----------------------------------
  ! Write file for all to read in
  ! Need to use MPI to write, MPI read
  ! doesn't understand fortran dumps!
  !------------------------------------

  filename = TRIM("io_files/inputfile.dat")

  !open and close are blocking, must be done outside if construct!
  call MPI_FILE_OPEN(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY+MPI_MODE_CREATE, MPI_INFO_NULL, filehandle, mpi_err)

  if (myid == 1) then
    call MPI_FILE_WRITE(filehandle, [123, 5234], 2, MPI_INTEGER, MPI_STATUS_IGNORE, mpi_err )
  endif
  
  call MPI_FILE_CLOSE(filehandle, mpi_err)


  if (myid == 1) write(*, *) "Everybody reading same file"
  call MPI_BARRIER(MPI_COMM_WORLD, mpi_err)








  !-------------------------------
  ! Everybody read the same file
  !-------------------------------

  
  call MPI_FILE_OPEN(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, filehandle, mpi_err)

  call MPI_FILE_READ(filehandle, readbuffer, 2, MPI_INTEGER, state, mpi_err)

  call MPI_FILE_CLOSE(filehandle, mpi_err)


  write(*,'(A3,x,I3,x,A10,x,2(I6,x))') "ID", myid, "read in", readbuffer
  if (myid == 1) write(*,*)








  !--------------------------------
  ! Write file for shared reading
  !--------------------------------



  filename = TRIM("io_files/inputfile_shared.dat")

  allocate(writebuffer_shared(1:ncpu))
  writebuffer_shared = 0
  allocate(readbuffer_shared(1:ncpu))
  readbuffer_shared = 0

  !open and close are blocking, must be done outside if construct!
  call MPI_FILE_OPEN(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY+MPI_MODE_CREATE, MPI_INFO_NULL, filehandle, mpi_err)

  if (myid == 1) then
    do i = 1, ncpu
      writebuffer_shared(i) = i
    enddo
    call MPI_FILE_WRITE(filehandle, writebuffer_shared, ncpu, MPI_INTEGER, MPI_STATUS_IGNORE, mpi_err )
  endif
  
  call MPI_FILE_CLOSE(filehandle, mpi_err)


  call MPI_BARRIER(MPI_COMM_WORLD, mpi_err)









  !---------------------------
  ! Shared reading
  !---------------------------



  if (myid == 1) write(*, *) "Shared reading from same file from array", writebuffer_shared


  call MPI_FILE_OPEN(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, filehandle, mpi_err)

  call MPI_FILE_READ_SHARED(filehandle, readbuffer_shared(myid), 1, MPI_INTEGER, state, mpi_err)

  call MPI_FILE_CLOSE(filehandle, mpi_err)


  write(*,'(A3,x,I3,x,A10,x)', advance='no') "ID", myid, "read in"
  do i = 1, ncpu
    write(*, '(I6,x)', advance='no') readbuffer_shared(i)
  enddo
  write(*,*)




  !-----------------
  ! Finish
  !-----------------

 
  call MPI_FINALIZE(mpi_err)







end program readwrite_mpi

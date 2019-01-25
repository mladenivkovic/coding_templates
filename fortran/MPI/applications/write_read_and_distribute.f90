program writeread

!----------------------------------------------------
! replace a MPI_FILE_WRITE_ORDERED:
! Every cpu does a fortran unformatted dump
! Then reads it back in
! then communicates it in a common array
!----------------------------------------------------

  use mpi

  implicit none

  integer :: myid, ncpu
  integer :: mpi_err

  character(len=5) :: id_to_string
  character(len=80):: filename
  integer, allocatable, dimension(:) :: writedata, readdata, alldata
  integer, allocatable, dimension(:) :: recvcount, displacements
  integer :: i, n, nr, n_all
  integer :: f=1 ! how much data per proc rank

  call MPI_INIT(mpi_err)

  call MPI_COMM_SIZE(MPI_COMM_WORLD, ncpu, mpi_err)
  call MPI_COMM_RANK(MPI_COMM_WORLD, myid, mpi_err)
  myid = myid+1

  n = f*myid


  !--------------------------------------
  ! Write data
  !--------------------------------------

  allocate(writedata(1:n))
  writedata=myid

  call title(myid, id_to_string)
  filename = TRIM('proc_'//id_to_string//'.dat')
  open(unit=666,file=filename, form='unformatted')
  write(666) n
  write(666) writedata
  close(666)

  if (myid==1) write(*, *) "Finished writing data."
  call MPI_BARRIER(MPI_COMM_WORLD, mpi_err)



  !-----------------------------------------
  ! Read data
  !-----------------------------------------

  call title(myid, id_to_string)
  filename = TRIM('proc_'//id_to_string//'.dat')
  open(unit=666,file=filename, form='unformatted')
  read(666) nr

  allocate(readdata(1:nr))
  if (nr>0) then
    read(666) readdata
  endif
  close(666)

  ! first gather receivecounts
  allocate(recvcount(1:ncpu))
  write(*,*) "recvcount before comm", recvcount
  call MPI_ALLGATHER(nr, 1, MPI_INT, recvcount, 1, MPI_INT, MPI_COMM_WORLD, mpi_err)

  write(*,*) "recvcount after comm", recvcount



  ! now communicate total data
  n_all = sum(recvcount)
  allocate(displacements(1:ncpu))
  displacements = 0
  do i=1, ncpu-1
    displacements(i+1) = displacements(i) + recvcount(i) 
  enddo

  write(*,*) "displacements", displacements

  allocate(alldata(1:n_all))

  call MPI_ALLGATHERV(readdata, nr, MPI_INT, alldata, recvcount, displacements, MPI_INT, MPI_COMM_WORLD, mpi_err)

  write(*,*)
  write(*,*) "alldata", alldata
  

  call MPI_FINALIZE(mpi_err)




contains

subroutine title(n,nchar)
  implicit none
  integer::n
  character(LEN=5)::nchar

  character(LEN=1)::nchar1
  character(LEN=2)::nchar2
  character(LEN=3)::nchar3
  character(LEN=4)::nchar4
  character(LEN=5)::nchar5

  if(n.ge.10000)then
     write(nchar5,'(i5)') n
     nchar = nchar5
  elseif(n.ge.1000)then
     write(nchar4,'(i4)') n
     nchar = '0'//nchar4
  elseif(n.ge.100)then
     write(nchar3,'(i3)') n
     nchar = '00'//nchar3
  elseif(n.ge.10)then
     write(nchar2,'(i2)') n
     nchar = '000'//nchar2
  else
     write(nchar1,'(i1)') n
     nchar = '0000'//nchar1
  endif


end subroutine title


end program writeread

!=======================================
! A program to demonstrate reading and
! writing to/from the same file with MPI
! Don't run with more than 10 processors
!=======================================



program readwrite_mpi

  use mpi

  implicit none

  integer :: myid, ncpu




  integer :: mpi_err, filehandle
  integer, dimension(MPI_STATUS_SIZE) :: state

  character(len=80) :: filename
  character(len=1)  :: id_char
  integer, dimension(1:2) :: readbuffer

  !-----------------------------
  ! Start MPI
  !-----------------------------


  call MPI_INIT(mpi_err)
  call MPI_COMM_RANK(MPI_COMM_WORLD, myid, mpi_err)
  call MPI_COMM_SIZE(MPI_COMM_WORLD, ncpu, mpi_err)

  myid = myid + 1




  ! all processors need same file
  call all_read_same_file()
  call sleep(1)


  ! shared reading from same file
  call shared_reading()
  call sleep(1)

  
  ! reading and writing to shared file
  call filesharing()
  call sleep(1)




  ! reading/writing to individual files
  call individual_readwrite()
  call sleep(1)












  !-----------------
  ! Finish
  !-----------------

 
  call MPI_FINALIZE(mpi_err)







contains
  !===================================
  subroutine all_read_same_file()
  !===================================


    implicit none
    character(len=80) :: filename
    integer :: mpi_err, filehandle
    integer, dimension(MPI_STATUS_SIZE) :: state
    integer, dimension(1:2) :: readbuffer




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

  end subroutine all_read_same_file








  !===================================
  subroutine shared_reading()
  !===================================


    implicit none
    character(len=80) :: filename
    integer :: mpi_err, filehandle
    integer, dimension(:), allocatable :: writebuffer_shared, readbuffer_shared
    integer, dimension(MPI_STATUS_SIZE) :: state
    integer :: i



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




  end subroutine shared_reading











  !===================================
  subroutine filesharing()
  !===================================


    implicit none
    character(len=80) :: filename
    integer :: mpi_err, filehandle
    ! use dimension 1:ncpu for no particular reason.i
    ! Just a variable so it's not hardcoded.
    integer, dimension(1:ncpu) :: writebuffer_shared, readbuffer_shared
    integer, dimension(MPI_STATUS_SIZE) :: state
    integer :: i


    ! make up stuff to write 
    do i = 1, ncpu
      writebuffer_shared(i) = 10*myid + i
    enddo
    


    !------------------
    ! Ordered writing
    !------------------

    filename = TRIM("io_files/shared_file.dat")

    !open and close are blocking, must be done outside if construct!
    call MPI_FILE_OPEN(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY+MPI_MODE_CREATE, MPI_INFO_NULL, filehandle, mpi_err)

    call MPI_FILE_WRITE_ORDERED(filehandle, writebuffer_shared, ncpu, MPI_INTEGER, MPI_STATUS_IGNORE, mpi_err )
    
    call MPI_FILE_CLOSE(filehandle, mpi_err)











    !---------------------------
    ! Shared reading
    !---------------------------

    if (myid == 1) then
      write(*,*)
      write(*, *) "Shared reading from shared file"
    endif

    call MPI_FILE_OPEN(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, filehandle, mpi_err)

    call MPI_FILE_READ_ORDERED(filehandle, readbuffer_shared, ncpu, MPI_INTEGER, state, mpi_err)

    call MPI_FILE_CLOSE(filehandle, mpi_err)


    write(*,'(A3,x,I3,x,A10,x)', advance='no') "ID", myid, "read in"
    do i = 1, 4
      write(*, '(I6,x)', advance='no') readbuffer_shared(i)
    enddo
    write(*,*)




  end subroutine filesharing








  !===================================
  subroutine individual_readwrite()
  !===================================

    !-------------------------------------
    ! Every processor writes own file
    !-------------------------------------

    if (myid == 1) write(*,*) 
    if (myid == 1) write(*,*) "Reading and writing to individual files"



    write(id_char, '(i1)') myid
    filename = TRIM("io_files/outputfile_"//id_char//".dat")
    


    ! A separate communicator is needed for every unique file. 
    ! "MPI_FILE_OPEN is a collective routine: all processes must 
    ! provide the same value for amode, and all processes must provide 
    ! filenames that reference the same file
    ! => use MPI_COMM_SELF instead of MPI_COMM_WORLD

    call MPI_FILE_OPEN(MPI_COMM_SELF, filename, MPI_MODE_WRONLY+MPI_MODE_CREATE, MPI_INFO_NULL, filehandle, mpi_err)

    call MPI_FILE_WRITE_ALL(filehandle, [10*myid, 10*myid+1], 2, MPI_INTEGER, MPI_STATUS_IGNORE, mpi_err )

    call MPI_FILE_CLOSE(filehandle, mpi_err)







    !----------------------------------------
    ! Now let every thread read its own file
    ! back in
    !----------------------------------------

    call MPI_FILE_OPEN(MPI_COMM_SELF, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, filehandle, mpi_err)

    call MPI_FILE_READ_ALL(filehandle, readbuffer, 2, MPI_INTEGER, state, mpi_err )

    call MPI_FILE_CLOSE(filehandle, mpi_err)

    write(*,'(A3,x,I3,x,A10,x,2(I6,x))') "ID", myid, "read in", readbuffer

    call MPI_BARRIER(MPI_COMM_WORLD, mpi_err)
    if (myid == 1) write(*,*)

  end subroutine individual_readwrite


end program readwrite_mpi

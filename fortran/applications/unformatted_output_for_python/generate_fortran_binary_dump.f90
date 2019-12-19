!=============================================================================
! Create various fortran binary dumps, so it can be read in with 
! scipy.io.FortranFile
!
! compile with e.g.
!   gfortran generate_fortran_binary_dump.f90 -o generate_binary_dump.o
! then run it with
!   ./generate_binary_dump.o
!
! It will create a file called 'fortran_unformatted_dump.dat'
!============================================================================




program generate_unformatted_dump

  implicit none

  character (len=30) :: fname = 'fortran_unformatted_dump.dat'

  integer, parameter :: dp = selected_real_kind(15, 307)
  integer, parameter :: qp = selected_real_kind(30, 291)
  
  integer i, j

  character           :: somechar = 'c'
  integer             :: someint = 10
  real                :: somefloat = 2.345
  real(dp)            :: somedouble = 1.2345d-306
  real(kind=8)        :: somereal8 = 2.345e34
  ! real(qp)            :: somequad = 2.34567q900 
  ! don't use quads. They are nonstandard, and scipy.io.FortranFile can't do it properly.


  character (len=40)            :: somestring
  integer, dimension(1:10)      :: intarr
  integer, dimension(1:3, 1:9)  :: intarr2d

  somestring = "Hello world! I was written and read in."

  do i = 1, 10
    intarr(i) = i
  enddo

  do i=1, 3
    do j = 1, 9
      intarr2d(i, j) = 10*i + j
    enddo
  enddo







  open(unit=666, form='unformatted', file=fname)
  write(666) someint
  write(666) somefloat
  write(666) somedouble
  write(666) somereal8
  ! write(666) somequad
  write(666) somechar
  write(666) somestring
  write(666) intarr
  write(666) intarr2d
  close(666)

  write(*, '(2A)') "Finished writing to file", fname
  write(*, '(A)') "what I wrote:"
  write(*, '(I10)') someint
  write(*, '(F7.3)') somefloat
  write(*, '(E20.8)') somedouble
  write(*, '(E20.8)') somereal8
  ! write(*, '(E60.30)') somequad
  write(*, '(A)') somechar
  write(*, '(A)') somestring
  do i = 1, 10
    write(*, '(I4)', advance='no') intarr(i)
  enddo
  write(*,*)
  write(*,*)

  do i=1, 3
    do j = 1, 9
      write(*, '(I4)', advance='no') intarr2d(i, j)
    enddo
    write(*,*)
  enddo





end program generate_unformatted_dump

! Demonstrating how to use FFTW.
! Compile with gfortran use_fftw.f90 -o use_fftw.o -I/home/mivkov/.local/include
! I installed fftw3 with ./configure --prefix=/home/mivkov/.local --enable-threads --enable-openmp --enable-mpi; make; make install

program use_fftw

  use,intrinsic :: iso_c_binding
  implicit none
  include 'fftw3.f03'


  integer, parameter :: N = 1000
  real, parameter    :: pi = 3.1415926
  real, parameter    :: physical_length = 10.0
  real, parameter    :: lambda1 = 0.5
  real, parameter    :: lambda2 = 0.7
  real, parameter    :: dx = physical_length/real(N)
  real, parameter    :: dk = 2.0 * pi / physical_length

  integer :: i, j


  ! for double precision: use double complex & call dfftw_plan_dft_1d
  double complex, allocatable, dimension(:)      :: arr_in, arr_out
  real(kind(1.d0)), allocatable, dimension(:)    :: P_k
  integer*8 :: my_plan


  allocate(arr_in(1:N))
  allocate(arr_out(1:N))


  call dfftw_plan_dft_1d(my_plan, N, arr_in, arr_out, FFTW_FORWARD, FFTW_ESTIMATE)

  arr_in = [(cmplx(cos(2.0*pi/lambda1*i*dx)+sin(2.0*pi/lambda2*i*dx), 0), i=1, N)]
  arr_out = [(cmplx(0, 0), i=1, N)]

  call dfftw_execute_dft(my_plan, arr_in, arr_out)

  allocate(P_k(1:N))
  P_k = [(abs(arr_out(i))**2, i=1, N)]

  open(unit=666,file='./fftw_output_1d_complex.txt', form='formatted')
  do i = 1, N
    if (i>N/2+1) then
      j=-N+i
    else
      j=i
    endif
    write(666, '(2E14.5,x)') (j-1)*dk, P_k(i)
  enddo
  close(666)


  deallocate(arr_in, arr_out, P_k)
  call dfftw_destroy_plan(my_plan)

  write(*,*) "Finished! Written results to fftw_output_1d_complex.txt"

end program use_fftw

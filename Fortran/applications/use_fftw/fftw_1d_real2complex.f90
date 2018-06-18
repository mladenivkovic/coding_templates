! Demonstrating how to use FFTW.
! I installed fftw3 with ./configure --prefix=/home/mivkov/.local --enable-threads --enable-openmp --enable-mpi; make; make install

program use_fftw

  use,intrinsic :: iso_c_binding
  implicit none
  include 'fftw3.f03'


  integer, parameter   :: N = 200
  integer, parameter   :: dp = kind(1.d0)
  real(dp), parameter  :: pi = 3.1415926
  real(dp), parameter  :: physical_length = 20.0
  real(dp), parameter  :: lambda1 = 0.5
  real(dp), parameter  :: lambda2 = 0.7
  real(dp), parameter  :: dx = physical_length/real(N)
  real(dp), parameter  :: dk = 2.0 * pi / physical_length

  integer :: i


  ! for double precision: use double complex & call dfftw_plan_dft_1d
  double complex, allocatable, dimension(:)      :: arr_out
  real(dp), allocatable, dimension(:)            :: P_k, arr_in
  integer*8                                      :: my_plan


  allocate(arr_in(1:N))
  allocate(arr_out(1:N/2+1))


  call dfftw_plan_dft_r2c_1d(my_plan, N, arr_in, arr_out, FFTW_ESTIMATE)

  arr_in = [(cos(2.0*pi/lambda1*i*dx)+sin(2.0*pi/lambda2*i*dx), i=1, N)]

  ! IMPORTANT: _dft_r2c, not just _dft !!!!
  call dfftw_execute_dft_r2c(my_plan, arr_in, arr_out)

  allocate(P_k(1:N/2+1))
  ! P_k = [(abs(arr_out(i))**2, i=1, N/2+1)]
  P_k = [(arr_out(i)*conjg(arr_out(i)), i=1, N/2+1)]

  open(unit=666,file='./fftw_output_1d_real.txt', form='formatted')
  do i = 1, N/2+1
    write(666, '(2E14.5,x)') (i-1)*dk, P_k(i)
  enddo
  close(666)


  deallocate(arr_in, arr_out, P_k)
  call dfftw_destroy_plan(my_plan)

  write(*,*) "Finished! Written results to fftw_output_1d_real.txt"

end program use_fftw

! Demonstrating how to use FFTW.
! Compile with gfortran use_fftw.f90 -o use_fftw.o -I/usr/local/include -lfftw3

program use_fftw

  use,intrinsic :: iso_c_binding
  implicit none
  include 'fftw3.f03'
 

  integer, parameter   :: N = 1000
  integer, parameter   :: dp = kind(1.d0)
  real(dp), parameter  :: pi = 3.1415926
  real(dp), parameter  :: physical_length = 100.0
  real(dp), parameter  :: lambda1 = 0.5
  real(dp), parameter  :: lambda2 = 0.7
  real(dp), parameter  :: dx = physical_length/real(N)
  real(dp), parameter  :: dk = 2.d0*pi / physical_length

  integer :: i

  complex(C_DOUBLE_COMPLEX), allocatable, dimension(:) :: arr_out
  real(C_DOUBLE),    allocatable, dimension(:)         :: arr_in
  real(kind(1.d0)),  allocatable, dimension(:)         :: P_k
  type(C_PTR)                                          :: my_plan


  allocate(arr_in(1:N))
  allocate(arr_out(1:N/2+1))


  my_plan = fftw_plan_dft_r2c_1d(N, arr_in, arr_out, FFTW_ESTIMATE)

  arr_in = [(cos(2.0*pi/lambda1*i*dx)+sin(2.0*pi/lambda2*i*dx), i=1, N)]

  ! IMPORTANT: _dft_r2c, not just _dft !!!!
  call fftw_execute_dft_r2c(my_plan, arr_in, arr_out)

  allocate(P_k(1:N/2+1))
  P_k = [(abs(arr_out(i))**2, i=1, N/2+1)]

  open(unit=666,file='./fftw_output_1d_real.txt', form='formatted')
  do i = 1, N/2+1
    write(666, '(2E14.5,x)') (i-1)*dk, P_k(i)
  enddo
  close(666)


  deallocate(arr_in, arr_out, P_k)
  call fftw_destroy_plan(my_plan)

  write(*,*) "Finished! Written results to fftw_output_1d_real.txt"

end program use_fftw

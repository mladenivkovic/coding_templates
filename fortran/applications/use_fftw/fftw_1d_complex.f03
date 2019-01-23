! Demonstrating how to use FFTW.
!                                      /        i k x
! results gauged for convention F[k] = | f(x) e       dx
!                                      /
! Compile with gfortran use_fftw.f90 -o use_fftw.o -I/usr/local/include -lfftw3

program use_fftw

  use,intrinsic :: iso_c_binding
  implicit none
  include 'fftw3.f03'


  integer, parameter     :: dp = kind(1.d0)
  integer, parameter     :: N = 1000
  real(dp), parameter    :: pi = 3.1415926
  real(dp), parameter    :: physical_length = 100.d0
  real(dp), parameter    :: lambda1 = 0.5d0
  real(dp), parameter    :: lambda2 = 0.7d0
  real(dp), parameter    :: dx = physical_length/real(N)
  real(dp), parameter    :: dk = 1.d0 / physical_length

  integer :: i, j

  complex(C_DOUBLE_COMPLEX), allocatable, dimension(:) :: arr_in, arr_out
  real(dp),                  allocatable, dimension(:) :: P_k
  type(C_PTR)                                          :: plan_forward


  allocate(arr_in(1:N))
  allocate(arr_out(1:N))


  plan_forward = fftw_plan_dft_1d(N, arr_in, arr_out, FFTW_FORWARD, FFTW_ESTIMATE)

  arr_in = [(cmplx(cos(2*pi*i*dx/lambda1)+sin(2*pi*i*dx/lambda2), 0.d0, kind=dp), i=1, N)]
  arr_out = [(cmplx(0, 0), i=1, N)]

  call fftw_execute_dft(plan_forward, arr_in, arr_out)

  allocate(P_k(1:N))
  P_k = [(abs(arr_out(i))**2, i=1, N)]

  open(unit=666,file='./fftw_output_1d_complex.txt', form='formatted')
  do i = 1, N
    if (i>N/2+1) then; j=-N+i; else; j=i; endif;
    write(666, '(2E14.5,x)') (j-1)*dk*2*pi, P_k(i)
  enddo
  close(666)


  deallocate(arr_in, arr_out, P_k)
  call fftw_destroy_plan(plan_forward)

  write(*,*) "Finished! Written results to fftw_output_1d_complex.txt"

end program use_fftw

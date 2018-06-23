! Demonstrating how to use FFTW.
!                            /                                           /
! Convert convention: F[f] = | f(x) e^{-ixk} dx wanted; FFTW uses F[f] = | f(x) e^{-2pi*ikx} dx
!                            /                                           /
! Here for r2c FFTW.
! In essence: k = (i,j,k) * dk.
! In FFTW convention:   dk =   1/physical_length;
! In no-2pi-convention: dk = 2pi/physical_length;
! So all you need to do is re-scale dk when you compute stuff with it.
! Compile with gfortran use_fftw.f90 -o use_fftw.o -I/usr/local/include -lfftw3

program use_fftw

  use,intrinsic :: iso_c_binding
  implicit none
  include 'fftw3.f03'


  integer, parameter   :: N = 1000
  integer, parameter   :: dp = kind(1.d0)
  real(dp), parameter  :: pi = 3.1415926d0
  real(dp), parameter  :: physical_length = 100
  real(dp), parameter  :: dx = physical_length/real(N)
  real(dp), parameter  :: dk = 2.d0*pi / physical_length

  integer :: i, ind1, ind2

  complex(C_DOUBLE_COMPLEX), allocatable, dimension(:) :: arr_out
  real(C_DOUBLE),    allocatable, dimension(:)         :: arr_in
  type(C_PTR)                                          :: plan_forward, plan_backward



  allocate(arr_in(1:N))
  allocate(arr_out(1:N/2+1))


  plan_forward = fftw_plan_dft_r2c_1d( N, arr_in, arr_out, FFTW_ESTIMATE)
  plan_backward = fftw_plan_dft_c2r_1d( N, arr_out, arr_in, FFTW_ESTIMATE)


  !----------------------
  ! Setup
  !----------------------

  ! add +1: index = 1 corresponds to x=0
  ind1 = int(1.d0/dx)+1
  ind2 = int((physical_length-1.d0)/dx)+1

  arr_in = 0
  arr_in(ind1) =  0.5d0
  arr_in(ind2) = -0.5d0


  !----------------------
  ! Forward
  !----------------------
  call fftw_execute_dft_r2c(plan_forward, arr_in, arr_out)

  write(*,*) "Verification: Max real part of arr_out:", maxval(real(arr_out))

  open(unit=666,file='./fftw_output_convert_convention_real_fft.txt', form='formatted')
  do i = 1, N/2+1
    write(666, '(2E14.5,x)') (i-1)*dk, aimag(arr_out(i))
  enddo
  close(666)

  write(*,*) "Finished! Written results to fftw_output_convert_convention_real_fft.txt"


  !----------------------
  ! Backward
  !----------------------
  call fftw_execute_dft_c2r(plan_backward, arr_out, arr_in)

  arr_in = arr_in/N

  open(unit=666,file='./fftw_output_convert_convention_real_real.txt', form='formatted')
  do i = 1, N
    write(666, '(2E14.5,x)') (i-1)*dx, arr_in(i)
  enddo
  close(666)

  write(*,*) "Finished! Written results to fftw_output_convert_convention_real_real.txt"

  deallocate(arr_in, arr_out)
  call fftw_destroy_plan(plan_forward)
  call fftw_destroy_plan(plan_backward)

end program use_fftw

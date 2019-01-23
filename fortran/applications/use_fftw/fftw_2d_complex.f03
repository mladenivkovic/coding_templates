! Demonstrating how to use FFTW.
! Compile with gfortran use_fftw.f90 -o use_fftw.o -I/usr/local/include -lfftw3

program use_fftw

  use,intrinsic :: iso_c_binding
  implicit none
  include 'fftw3.f03'

  integer, parameter     :: dp = kind(1.0d0)
  integer, parameter     :: Nx = 400
  integer, parameter     :: Ny = 200
  integer, parameter     :: nsamples = 200
  real(dp), parameter    :: pi = 3.1415926d0
  real(dp), parameter    :: physical_length_x = 40.d0
  real(dp), parameter    :: physical_length_y = 20.d0
  real(dp), parameter    :: lambda1 = 0.5d0
  real(dp), parameter    :: lambda2 = 0.7d0
  real(dp), parameter    :: dx = physical_length_x/real(Nx,dp)
  real(dp), parameter    :: dy = physical_length_y/real(Ny,dp)
  real(dp), parameter    :: dkx = 1.d0/ physical_length_x
  real(dp), parameter    :: dky = 1.d0/ physical_length_y

  integer :: i, j, ix, iy, ik
  real(dp):: kmax, d

  real(dp), allocatable, dimension(:)                     :: Pk, distances_k
  complex(C_DOUBLE_COMPLEX),  allocatable, dimension(:,:) :: arr_out, arr_in
  type(C_PTR)                                             :: my_plan
  ! integer                                                 :: n(2) = (/Ny, Nx/)


  allocate(arr_in( 1:Nx,1:Ny))
  allocate(arr_out(1:Nx,1:Ny))

  !!!!!!!! IMPORTANT: IN MODERN FORTRAN, YOU NEED TO REVERSE ORDER: N x M x L array => L x M x N
  my_plan = fftw_plan_dft_2d(Ny,Nx, arr_in, arr_out, FFTW_FORWARD, FFTW_ESTIMATE)
  ! my_plan = fftw_plan_dft(2, n, arr_in, arr_out, FFTW_FORWARD, FFTW_ESTIMATE)

  do i = 1, Nx
    do j = 1, Ny
      arr_in(i,j) =cmplx(cos(2.0*pi/lambda1*(i-1)*dx)+sin(2.0*pi/lambda2*(j-1)*dy) , 0.d0, kind=dp)
      arr_out(i,j) = cmplx(0.d0,0.d0,kind=dp)
    enddo
  enddo

  call fftw_execute_dft(my_plan, arr_in, arr_out)

  allocate(distances_k(0:nsamples))
  allocate(Pk(1:nsamples))
  distances_k = 0
  Pk = 0

  ! Get bin distances
  kmax = sqrt(((Nx/2+1)*dkx)**2+((Ny/2+1)*dky)**2)
  do i = 0, nsamples
    distances_k(i) = i*kmax/nsamples 
  enddo

  ! Compute P(k) field, distances field
  ! move all looping indexes one back:
  ! k starts with (0,0), but array indices start with (1, 1)
  do i = 1, Nx
    if (i-1<Nx/2+1) then; ix = i-1; else; ix = -Nx+i-1; endif;
    do j = 1, Ny
      if (j-1<Ny/2+1) then; iy = j-1; else; iy = -Ny+j-1; endif;

      d = sqrt((ix*dkx)**2+(iy*dky)**2)

      do ik=1, nsamples
        if (d<=distances_k(ik) .or. ik==nsamples) exit
      enddo

      Pk(ik) = Pk(ik)+abs(arr_out(i,j))**2
    enddo
  enddo



  ! write file
  open(unit=666,file='./fftw_output_2d_complex.txt', form='formatted')
  do i = 1, nsamples
    ! (dist(i-1)-dist(i))/2 * 2pi => 2's cancel out
    write(666, '(2E14.5,x)') pi*(distances_k(i-1)+distances_k(i)), Pk(i)
  enddo
  close(666)


  deallocate(arr_in, arr_out, Pk, distances_k)
  call fftw_destroy_plan(my_plan)

  write(*,*) "Finished! Written results to fftw_output_2d_complex.txt"

end program use_fftw

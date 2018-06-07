! Demonstrating how to use FFTW.
! Compile with gfortran use_fftw.f90 -o use_fftw.o -I/home/mivkov/.local/include
! I installed fftw3 with ./configure --prefix=/home/mivkov/.local --enable-threads --enable-openmp --enable-mpi; make; make install

program use_fftw

  use,intrinsic :: iso_c_binding
  implicit none
  include 'fftw3.f03'

  integer, parameter     :: dp=kind(1.0d0)
  integer, parameter     :: Nx = 200
  integer, parameter     :: Ny = 100
  integer, parameter     :: nsamples = 200
  real(dp), parameter    :: pi = 3.1415926d0
  real(dp), parameter    :: physical_length_x = 20.d0
  real(dp), parameter    :: physical_length_y = 10.d0
  real(dp), parameter    :: lambda1 = 0.5d0
  real(dp), parameter    :: lambda2 = 0.7d0
  real(dp), parameter    :: dx = physical_length_x/real(Nx,dp)
  real(dp), parameter    :: dy = physical_length_y/real(Ny,dp)
  real(dp), parameter    :: dkx = 2.d0 * pi / physical_length_x
  real(dp), parameter    :: dky = 2.d0 * pi / physical_length_y

  integer :: i, j, ix, iy, ik
  real(dp):: kmax, d


  ! for double precision: use double complex & call dfftw_plan_dft_1d
  complex(dp), allocatable, dimension(:,:)    :: arr_out, Pk_field
  real(dp), allocatable, dimension(:,:)       :: arr_in
  real(dp), allocatable, dimension(:)         :: Pk, distances_k
  integer*8                                   :: my_plan
  integer, dimension(2)                       :: n(2) = (/Nx, Ny/)


  allocate(arr_in( 1:Nx,1:Ny))
  arr_in=0
  allocate(arr_out(1:Nx/2+1,1:Ny))
  arr_out = 0


  call dfftw_plan_dft_r2c(my_plan, 2, n, arr_in, arr_out, FFTW_ESTIMATE)

  do i = 1, Nx
    do j = 1, Ny
      arr_in(i,j) =cos(2.0*pi/lambda1*i*dx)+sin(2.0*pi/lambda2*j*dy) 
    enddo
  enddo

  call dfftw_execute_dft_r2c(my_plan, arr_in, arr_out)


  allocate(Pk_field(1:Nx/2+1, 1:Ny))
  allocate(distances_k(0:nsamples))
  allocate(Pk(1:nsamples))
  distances_k = 0
  Pk = 0

  ! Get bin distances
  kmax = sqrt(((Nx/2+1)*dkx)**2+((Ny/2+1)*dky)**2)
  do i = 1, nsamples
    distances_k(i) = i*kmax/nsamples*1.001d0 ! add a little more so max value will fit
  enddo

  ! Compute P(k) field, distances field
  ! move all looping indexes one back:
  ! k starts with (0,0), but array indices start with (1, 1)
  do i = 1, Nx/2+1

    ix = i-1

    do j = 1, Ny

      if (j-1<Ny/2+1) then
        iy = j-1
      else
        iy = -Ny+j-1
      endif

      Pk_field(i,j) = arr_out(i,j)*conjg(arr_out(i,j))
      d = sqrt((ix*dkx)**2+(iy*dky)**2)

      do ik=1, nsamples
        if (d<=distances_k(ik) .or. ik==nsamples) exit
      enddo

      Pk(ik) = Pk(ik)+real(Pk_field(i,j))
    enddo
  enddo



  ! write file
  open(unit=666,file='./fftw_output_2d_real.txt', form='formatted')
  do i = 1, nsamples
    write(666, '(2E14.5,x)') 0.5*(distances_k(i-1)+distances_k(i)), Pk(i)
  enddo
  close(666)


  deallocate(arr_in, arr_out, Pk, Pk_field, distances_k)
  call dfftw_destroy_plan(my_plan)

  write(*,*) "Finished! Written results to fftw_output_2d_real.txt"

end program use_fftw

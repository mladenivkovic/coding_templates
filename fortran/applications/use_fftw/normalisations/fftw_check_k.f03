! Demonstrating how to use FFTW.
! computes power spectrum of real 3D array, then transforms it back.
! Compile with gfortran use_fftw.f90 -o use_fftw.o -I/usr/local/include -lfftw3

program use_fftw

  use,intrinsic :: iso_c_binding
  implicit none
  include 'fftw3.f03'

  integer, parameter     :: dp=kind(1.0d0)
  integer, parameter     :: Nx = 128
  integer, parameter     :: Ny = 128
  integer, parameter     :: Nz = 128
  integer, parameter     :: nsamples = 200
  real(dp), parameter    :: pi = 3.1415926d0
  real(dp), parameter    :: physical_length_x = 30.d0
  real(dp), parameter    :: physical_length_y = 30.d0
  real(dp), parameter    :: physical_length_z = 30.d0
  real(dp), parameter    :: k1 = 10.d0/1
  real(dp), parameter    :: k2 = 10.d0/2
  real(dp), parameter    :: k3 = 10.d0/3
  real(dp), parameter    :: dx = physical_length_x/real(Nx,dp)
  real(dp), parameter    :: dy = physical_length_y/real(Ny,dp)
  real(dp), parameter    :: dz = physical_length_z/real(Nz,dp)
  real(dp), parameter    :: dkx = 2*pi / physical_length_x
  real(dp), parameter    :: dky = 2*pi / physical_length_y
  real(dp), parameter    :: dkz = 2*pi / physical_length_z


  integer :: i, j, k, ix, iy, iz, ik
  real(dp):: kmax, d

  complex(C_DOUBLE_COMPLEX), allocatable, dimension(:,:,:)  :: arr_out
  real(C_DOUBLE), allocatable, dimension(:,:,:)             :: arr_in
  real(dp), allocatable, dimension(:)                       :: Pk, distances_k
  type(C_PTR)                                               :: plan_forward, plan_backward
  ! integer, dimension(3)                                     :: n(3) = (/Nz, Ny, Ny/)


  allocate(arr_in( 1:Nx, 1:Ny, 1:Nz));      arr_in=0;
  allocate(arr_out(1:Nx/2+1, 1:Ny, 1:Nz));  arr_out=0;


  plan_forward = fftw_plan_dft_r2c_3d(Nz, Ny, Nx, arr_in, arr_out, FFTW_ESTIMATE)
  plan_backward = fftw_plan_dft_c2r_3d(Nz, Ny, Nx, arr_out, arr_in, FFTW_ESTIMATE)

  ! Fill array with wave
  ! NOTE: wave only depends on x so you can plot it later.
  do i = 1, Nx
    do j = 1, Ny
      do k = 1, Nz
        arr_in(i,j,k) = cos((i-1)*dx*k1)+sin((j-1)*dy*k2)+cos((k-1)*dz*k3) 
      enddo
    enddo
  enddo




  !----------------------------
  ! Forward transform
  !----------------------------

  call fftw_execute_dft_r2c(plan_forward, arr_in, arr_out)


  allocate(distances_k(0:nsamples));        distances_k = 0;
  allocate(Pk(1:nsamples));                 Pk = 0;

  ! Get bin distances
  kmax = sqrt(((Nx/2)*dkx)**2+((Ny/2)*dky)**2 + ((Nz/2)*dkz)**2)
  do i = 1, nsamples
    distances_k(i) = i*kmax/nsamples
  enddo

  ! Compute P(k) field, distances field
  ! move all looping indexes one back:
  ! k starts with (0,0), but array indices start with (1, 1)
  do i = 1, Nx/2+1
    ix = i-1
    do j = 1, Ny
      if (j-1<Ny/2+1) then; iy = j-1; else; iy = -Ny+j-1; endif;
      do k = 1, Nz
        if (k-1<Ny/2+1) then; iz = k-1; else; iz = -Nz+k-1;  endif;

        d = sqrt((ix*dkx)**2+(iy*dky)**2 + (iz*dkz)**2)

        ik = int(d/kmax*nsamples)+1
        if (ik==nsamples+1) ik=nsamples
        Pk(ik) = Pk(ik)+abs(arr_out(i,j,k))**2
      enddo
    enddo
  enddo



  ! write file
  open(unit=666,file='./fftw_output_3d_Pk.txt', form='formatted')
  do i = 1, nsamples
    write(666, '(2E22.10,x)') (distances_k(i-1)+distances_k(i))/2, Pk(i)
  enddo
  close(666)

  write(*,*) "P(k) Finished! Written results to fftw_output_3d_Pk.txt"


  deallocate(arr_in, arr_out, Pk, distances_k)
  call fftw_destroy_plan(plan_forward)
  call fftw_destroy_plan(plan_backward)

end program use_fftw

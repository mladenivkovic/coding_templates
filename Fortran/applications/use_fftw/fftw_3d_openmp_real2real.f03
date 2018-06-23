! Demonstrating how to use FFTW.
! computes power spectrum of real 3D array, then transforms it back, with openMP.
! IMPORTANT: Compile with
! gfortran fftw_3d_openmp_real2real.f03 -o fftw_3d_openmp_real2real.o -O3 -I/usr/local/include -lfftw3 -fopenmp -lfftw3_omp

program use_fftw

  use,intrinsic :: iso_c_binding
  use omp_lib
  implicit none
  include 'fftw3.f03'

  integer, parameter     :: dp=kind(1.0d0)
  integer, parameter     :: Nx = 400
  integer, parameter     :: Ny = 200
  integer, parameter     :: Nz = 200
  integer, parameter     :: nsamples = 200
  real(dp), parameter    :: pi = 3.1415926d0
  real(dp), parameter    :: physical_length_x = 40.d0
  real(dp), parameter    :: physical_length_y = 20.d0
  real(dp), parameter    :: physical_length_z = 20.d0
  real(dp), parameter    :: lambda1 = 0.5d0
  real(dp), parameter    :: lambda2 = 0.7d0
  real(dp), parameter    :: lambda3 = 0.9d0
  real(dp), parameter    :: dx = physical_length_x/real(Nx,dp)
  real(dp), parameter    :: dy = physical_length_y/real(Ny,dp)
  real(dp), parameter    :: dz = physical_length_z/real(Nz,dp)
  real(dp), parameter    :: dkx = 1 / physical_length_x
  real(dp), parameter    :: dky = 1 / physical_length_y
  real(dp), parameter    :: dkz = 1 / physical_length_z

  integer :: void

  integer :: i, j, k, ix, iy, iz, ik, guess
  real(dp):: kmax, d

  complex(C_DOUBLE_COMPLEX), allocatable, dimension(:,:,:)  :: arr_out
  real(C_DOUBLE), allocatable, dimension(:,:,:)             :: arr_in
  real(dp), allocatable, dimension(:)                       :: Pk, distances_k
  type(C_PTR)                                               :: plan_forward, plan_backward
  ! integer, dimension(3)                                     :: n(3) = (/Nz, Ny, Ny/)


  allocate(arr_in( 1:Nx, 1:Ny, 1:Nz));      arr_in=0
  allocate(arr_out(1:Nx/2+1, 1:Ny, 1:Nz));  arr_out = 0

  allocate(distances_k(0:nsamples));        distances_k = 0
  allocate(Pk(1:nsamples));                 Pk = 0

  kmax = sqrt(((Nx/2+1)*dkx)**2+((Ny/2+1)*dky)**2 + ((Nz/2+1)*dkz)**2)



  void = fftw_init_threads()
  if (void==0) then
    write(*,*) "Error in fftw_init_threads, quitting"
    stop
  endif
  ! Call before any FFTW routine is called outside of parallel region
  call fftw_plan_with_nthreads(omp_get_max_threads())


  ! plan execution is thread-safe, but plan creation and destruction are not:
  ! you should create/destroy plans only from a single thread
  plan_forward = fftw_plan_dft_r2c_3d(Nz, Ny, Nx, arr_in, arr_out, FFTW_ESTIMATE)
  plan_backward = fftw_plan_dft_c2r_3d(Nz, Ny, Nx, arr_out, arr_in, FFTW_ESTIMATE)


  ! Fill array with wave
  ! NOTE: wave only depends on x so you can plot it later.
  !$OMP PARALLEL DO PRIVATE(i, j, k, d)
    do i = 1, Nx
      d = 2.0*pi*i*dx
      do j = 1, Ny
        do k = 1, Nz
          arr_in(i,j,k) = cos(d/lambda1)+sin(d/lambda2)+cos(d/lambda3) 
        enddo
      enddo
    enddo
  !$OMP END PARALLEL DO


  !---------------------------------------
  ! transforms: outside parallel region!
  ! Also: transform, then histogrammise,
  ! then transform back: transformations
  ! may change input arrays too!
  !---------------------------------------

  call fftw_execute_dft_r2c(plan_forward, arr_in, arr_out)


  !$OMP PARALLEL
    ! Get bin distances
    !$OMP DO
      do i = 1, nsamples
        distances_k(i) = i*kmax/nsamples
      enddo
    !$OMP END DO

    ! Compute P(k) field, distances field
    ! move all looping indexes one back:
    ! k starts with (0,0), but array indices start with (1, 1)

    !$OMP DO COLLAPSE(3) PRIVATE(i, ix, j, iy, k, iz, d, ik, guess)
      do i = 1, Nx/2+1
        do j = 1, Ny
          do k = 1, Nz
            ix = i-1
            if (j-1<Ny/2+1) then; iy = j-1; else; iy = -Ny+j-1; endif;
            if (k-1<Nz/2+1) then; iz = k-1; else; iz = -Nz+k-1; endif;
            d = sqrt((ix*dkx)**2+(iy*dky)**2 + (iz*dkz)**2)
            guess = int(d/kmax*nsamples) - 1
            guess = max(guess, 1)
            do ik=guess, nsamples
              if (d<=distances_k(ik) .or. ik==nsamples) exit
            enddo
            !$OMP ATOMIC
            Pk(ik) = Pk(ik)+real(arr_out(i,j,k)*conjg(arr_out(i,j,k)))
          enddo
        enddo
      enddo
    !$OMP END DO
  !$OMP END PARALLEL


  ! write file
  open(unit=666,file='./fftw_output_3d_omp_Pk.txt', form='formatted')
  do i = 1, nsamples
    ! 1/2 for distance * 2 * pi => * pi remains
    write(666, '(2E22.10,x)') pi*(distances_k(i-1)+distances_k(i)), Pk(i)
  enddo
  write(*,*) "P(k) Finished! Written results to fftw_output_3d_omp_Pk.txt"
  close(666)


  call fftw_execute_dft_c2r(plan_backward, arr_out, arr_in)



  ! write file
  open(unit=667,file='./fftw_output_3d_omp_real.txt', form='formatted')
  do i = 1, Nx
    write(667, '(2E22.10,x)') i*dx, arr_in(i,1,1)/(Nx*Ny*Nz)
  enddo
  write(*,*) "P(k) Finished! Written results to fftw_output_3d_omp_real.txt"
  close(667)




  deallocate(arr_in, arr_out, Pk, distances_k)
  ! destroy plans is not thread-safe; do only with single
  call fftw_destroy_plan(plan_forward)
  call fftw_destroy_plan(plan_backward)

end program use_fftw

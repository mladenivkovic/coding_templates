!------------------------------------------------------------
! Check the normalisation by using Parseval's theorem:
! int_space |f(x)|^2 dx = int_k-space |F(k)|^2 dk
! should hold for functions that are periodic with period of
! the domain you've given as the input array.
!
! Notation used: sigma^2 = int_space |f(x)|^2 dx 
!                Pk^2    = int_k-space |F(k)|^2 dk
!------------------------------------------------------------


program use_fftw

  use,intrinsic :: iso_c_binding
  implicit none
  include 'fftw3.f03'

  integer, parameter   :: Nx = 128
  integer, parameter   :: Ny = 128
  integer, parameter   :: Nz = 128
  ! integer, parameter   :: Nx = 128
  ! integer, parameter   :: Ny = 64
  ! integer, parameter   :: Nz = 32
  integer, parameter   :: nsamples = 200
  integer, parameter   :: dp = kind(1.d0)
  real(dp), parameter  :: pi = 3.1415926d0

  integer :: i,j,k
  real(dp):: temp1, temp2, temp3

  real(dp):: s2dr = 0.d0, pk2dr = 0.d0
  real(dp):: s3dr = 0.d0, pk3dr = 0.d0
  real(dp):: s2dc = 0.d0, pk2dc = 0.d0
  real(dp):: s3dc = 0.d0, pk3dc = 0.d0

  real(C_DOUBLE),             allocatable, dimension(:,:)   :: arr_in_2d_r
  real(C_DOUBLE),             allocatable, dimension(:,:,:) :: arr_in_3d_r
  complex(C_DOUBLE_COMPLEX),  allocatable, dimension(:,:)   :: arr_in_2d_c, arr_out_2d_c, arr_out_2d_r
  complex(C_DOUBLE_COMPLEX),  allocatable, dimension(:,:,:) :: arr_in_3d_c, arr_out_3d_c, arr_out_3d_r
  type(C_PTR)                                               :: p2d_r2c, p3d_r2c, p2d_cf, p3d_cf
  type(C_PTR)                                               :: p2d_c2r, p3d_c2r, p2d_cb, p3d_cb

  integer :: n
  integer, dimension(:), allocatable :: seed
  
  
  !=================================================
  ! PART0: SETUP
  !=================================================

  ! pick a seed for random nrs
  call random_seed(size=n)
  allocate(seed(n))
  seed = 73* [(n-i, i=1, n)]
  call random_seed(put=seed)
  deallocate(seed)

  ! Set up everything you need
  allocate(arr_in_2d_r(1:Nx, 1:Ny))
  allocate(arr_in_2d_c(1:Nx, 1:Ny))
  allocate(arr_out_2d_r(1:Nx/2+1, 1:Ny))
  allocate(arr_out_2d_c(1:Nx, 1:Ny))
  allocate(arr_in_3d_r(1:Nx, 1:Ny, 1:Nz))
  allocate(arr_in_3d_c(1:Nx, 1:Ny, 1:Nz))
  allocate(arr_out_3d_r(1:Nx/2+1, 1:Ny, 1:Nz))
  allocate(arr_out_3d_c(1:Nx, 1:Ny, 1:Nz))

  p2d_cf  =     fftw_plan_dft_2d(Ny, Nx, arr_in_2d_c, arr_out_2d_c, FFTW_FORWARD, FFTW_ESTIMATE)
  p2d_r2c = fftw_plan_dft_r2c_2d(Ny, Nx, arr_in_2d_r, arr_out_2d_r, FFTW_ESTIMATE)
  p2d_cb  =     fftw_plan_dft_2d(Ny, Nx, arr_out_2d_c, arr_in_2d_c, FFTW_BACKWARD, FFTW_ESTIMATE)
  p2d_c2r = fftw_plan_dft_c2r_2d(Ny, Nx, arr_out_2d_r, arr_in_2d_r, FFTW_ESTIMATE)

  p3d_cf  =     fftw_plan_dft_3d(Nz, Ny, Nx, arr_in_3d_c, arr_out_3d_c, FFTW_FORWARD, FFTW_ESTIMATE)
  p3d_r2c = fftw_plan_dft_r2c_3d(Nz, Ny, Nx, arr_in_3d_r, arr_out_3d_r, FFTW_ESTIMATE)
  p3d_cb  =     fftw_plan_dft_3d(Nz, Ny, Nx, arr_out_3d_c, arr_in_3d_c, FFTW_BACKWARD, FFTW_ESTIMATE)
  p3d_c2r = fftw_plan_dft_c2r_3d(Nz, Ny, Nx, arr_out_3d_r, arr_in_3d_r, FFTW_ESTIMATE)


  !----------------------
  ! Fill arrays
  !----------------------

  do i=1, Nx
    do j=1, Ny
      call random_number(temp1)
      call random_number(temp2)
      call random_number(temp3)
      arr_in_2d_c(i,j) = cmplx(temp1, temp2, kind=dp)
      arr_in_2d_r(i,j) = temp3
      ! arr_in_2d_c(i,j) = cmplx(i,0,kind=dp)
      ! arr_in_2d_r(i,j) = i
      do k=1, Nz
        call random_number(temp1)
        call random_number(temp2)
        call random_number(temp3)
        arr_in_3d_c(i,j,k) = cmplx(temp1, temp2, kind=dp)
        arr_in_3d_r(i,j,k) = temp3
        ! arr_in_3d_c(i,j,k) = cmplx(i,0,kind=dp)
        ! arr_in_3d_r(i,j,k) = i
      enddo
    enddo
  enddo





  !=================================================
  ! PART1: FORWARD TRANSFORM
  !=================================================

  !---------------------------------------------------
  ! FFTW may change input arrays:
  ! Compue integrals of squares of input arrays before
  ! the transform
  !---------------------------------------------------
  do i=1, Nx
    do j=1, Ny
      s2dc  = s2dc  + abs(arr_in_2d_c(i,j))**2
      s2dr  = s2dr  +     arr_in_2d_r(i,j)**2
      do k=1, Nz
        s3dc  = s3dc + abs(arr_in_3d_c(i,j,k))**2
        s3dr  = s3dr +     arr_in_3d_r(i,j,k)**2
      enddo
    enddo
  enddo


  !----------------------
  ! Do Forward FFTs
  !----------------------
  call     fftw_execute_dft(p2d_cf,  arr_in_2d_c, arr_out_2d_c)
  call fftw_execute_dft_r2c(p2d_r2c, arr_in_2d_r, arr_out_2d_r)
  call     fftw_execute_dft(p3d_cf,  arr_in_3d_c, arr_out_3d_c)
  call fftw_execute_dft_r2c(p3d_r2c, arr_in_3d_r, arr_out_3d_r)



  !-----------------------------------
  ! Compute integrals of squares
  !-----------------------------------
  do i=1, Nx
    do j=1, Ny 
      pk2dc = pk2dc + abs(arr_out_2d_c(i,j))**2
      do k=1, Nz
        pk3dc = pk3dc + abs(arr_out_3d_c(i,j,k))**2
      enddo
    enddo
  enddo

  do i=1, Nx/2+1
    if (i==1.or.i==Nx/2+1) then; temp1=1.d0; else; temp1=2.d0; endif;
    do j=1, Ny
      pk2dr = pk2dr + temp1*abs(arr_out_2d_r(i,j))**2
      do k=1, Nz
        pk3dr = pk3dr + temp1*abs(arr_out_3d_r(i,j,k))**2
      enddo
    enddo
  enddo

  


  write(*,*) "------------------------"
  write(*,*) "   FORWARD TRANSFORM    "
  write(*,*) "------------------------"
  write(*,'(A21, 4A18)')   "",                         &
    "2d r2c",               "2d c2c",                 "3d r2c",                "3d c2c"
  write(*,'(A21, 4E18.6)') "s^2",                      & 
    s2dr,                    s2dc,                    s3dr,                    s3dc
  write(*,'(A21, 4E18.6)') "Pk^2",                     &
    pk2dr,                   pk2dc,                   pk3dr,                   pk3dc
  write(*,'(A21, 4E18.6)') "Pk^2/s^2",                 &
    ps(pk2dr,s2dr),          ps(pk2dc,s2dc),          ps(pk3dr,s3dr),          ps(pk3dc,s3dc)
  write(*,'(A21, 4E18.6, A20)') "Pk^2/s^2/full volume",&
    ps_fav2(pk2dr,s2dr,'d'), ps_fav2(pk2dc,s2dc,'d'), ps_fav3(pk3dr,s3dr,'d'), ps_fav3(pk3dc, s3dc,'d'), "should be 1.0!"








  !=================================================
  ! PART2: BACKWARD TRANSFORM
  !=================================================

  s2dc=0;  pk2dc=0;
  s2dr=0;  pk2dr=0;
  s3dc=0;  pk3dc=0;
  s3dr=0;  pk3dr=0;


  ! First normalize arr_out correctly:
  ! FFTW is unnormalised: F^-1[F(f(x))] = Volume*f(x). To recover original values, you need 
  ! to divide 1/Volume somewhere. However, each transformation adds a factor 1/sqrt(volume), 
  ! but since you're computing squares of transforms, you'll need to normalize with 1/volume 
  ! each time anyway.
  arr_out_2d_c=arr_out_2d_c/(Nx*Ny)
  arr_out_2d_r=arr_out_2d_r/(Nx*Ny)
  arr_out_3d_c=arr_out_3d_c/(Nx*Ny*Nz)
  arr_out_3d_r=arr_out_3d_r/(Nx*Ny*Nz)


  !---------------------------------------------------
  ! FFTW may change input arrays:
  ! Compue integrals of squares of input arrays before
  ! the transform
  !---------------------------------------------------
  do i=1, Nx
    do j=1, Ny
      pk2dc = pk2dc + abs(arr_out_2d_c(i,j))**2
      do k=1, Nz
        pk3dc = pk3dc + abs(arr_out_3d_c(i,j,k))**2
      enddo
    enddo
  enddo

  do i=1, Nx/2+1
    if (i==1.or.i==Nx/2+1) then; temp1=1.d0; else; temp1=2.d0; endif;
    do j=1, Ny
      pk2dr = pk2dr + temp1*abs(arr_out_2d_r(i,j))**2
      do k=1, Nz
        pk3dr = pk3dr + temp1*abs(arr_out_3d_r(i,j,k))**2
      enddo
    enddo
  enddo



  !----------------------
  ! Do backward FFTs
  !----------------------
  call     fftw_execute_dft(p2d_cb,  arr_out_2d_c, arr_in_2d_c)
  call fftw_execute_dft_c2r(p2d_c2r, arr_out_2d_r, arr_in_2d_r)
  call     fftw_execute_dft(p3d_cb,  arr_out_3d_c, arr_in_3d_c)
  call fftw_execute_dft_c2r(p3d_c2r, arr_out_3d_r, arr_in_3d_r)


  !-----------------------------------
  ! Compute integrals of squares
  !-----------------------------------
  do i=1, Nx
    do j=1, Ny
      s2dc  = s2dc  + abs(arr_in_2d_c(i,j))**2
      s2dr  = s2dr  +     arr_in_2d_r(i,j)**2
      do k=1, Nz
        s3dc  = s3dc + abs(arr_in_3d_c(i,j,k))**2
        s3dr  = s3dr +     arr_in_3d_r(i,j,k)**2
      enddo
    enddo
  enddo



  

  write(*,*)
  write(*,*) "------------------------"
  write(*,*) "  BACKWARD TRANSFORM    "
  write(*,*) "------------------------"
  write(*,'(A21, 4A18)')   "",                          &
    "2d r2c",              "2d c2c",                 "3d r2c",                "3d c2c"
  write(*,'(A21, 4E18.6)') "s^2",                       &
    s2dr,                    s2dc,                    s3dr,                    s3dc
  write(*,'(A21, 4E18.6)') "Pk^2",                      &
    pk2dr,                   pk2dc,                   pk3dr,                   pk3dc
  write(*,'(A21, 4E18.6)') "Pk^2/s^2",                  &
    ps(pk2dr,s2dr),          ps(pk2dc,s2dc),          ps(pk3dr,s3dr),          ps(pk3dc,s3dc)
  write(*,'(A21, 4E18.6, A20)') "Pk^2/s^2/full volume", &
    ps_fav2(pk2dr,s2dr,'m'), ps_fav2(pk2dc,s2dc,'m'), ps_fav3(pk3dr,s3dr,'m'), ps_fav3(pk3dc, s3dc,'m'), "should be 1.0!"

  ! Note that here the transform goes from Pk -> sigma, but to keep the 
  ! layout of the printing, multiply the source instead of dividing the result.




  deallocate(arr_in_2d_r, arr_out_2d_r)
  deallocate(arr_in_2d_c, arr_out_2d_c)
  deallocate(arr_in_3d_r, arr_out_3d_r)
  deallocate(arr_in_3d_c, arr_out_3d_c)
  call fftw_destroy_plan(p2d_r2c)
  call fftw_destroy_plan(p2d_cf)
  call fftw_destroy_plan(p3d_r2c)
  call fftw_destroy_plan(p3d_cf)



  contains 
    real(dp) function ps(pk, sig)
      ! returns Pk^2/s^2
      real(dp), intent(in) ::pk, sig
      ps = pk/sig
    end function

    real(dp) function ps_fav2(pk, sig, which)
      ! returns Pk^2/s^2/Full array volume in 2D
      real(dp), intent(in) ::pk, sig
      character(len=1), intent(in):: which
      if (which=='d') then
        ps_fav2 = pk/sig / (Nx*Ny)
      else if (which=='m') then
        ps_fav2 = pk/sig * (Nx*Ny)
      endif
    end function

    real(dp) function ps_fav3(pk, sig, which)
      ! returns Pk^2/s^2/Full array volume in 3D
      real(dp), intent(in) ::pk, sig
      character(len=1), intent(in):: which
      if (which=='d') then
        ps_fav3 = pk/sig / (Nx*Ny*Nz)
      else if (which=='m') then
        ps_fav3 = pk/sig * (Nx*Ny*Nz)
      endif
    end function

end program use_fftw

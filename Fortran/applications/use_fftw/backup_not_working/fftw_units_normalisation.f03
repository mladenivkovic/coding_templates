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

  ! integer, parameter   :: Nx = 128
  ! integer, parameter   :: Ny = 128
  ! integer, parameter   :: Nz = 128
  integer, parameter   :: Nx = 64
  integer, parameter   :: Ny = 64
  integer, parameter   :: Nz = 64
  ! integer, parameter   :: Nx = 128
  ! integer, parameter   :: Ny = 64
  ! integer, parameter   :: Nz = 32
  integer, parameter   :: nsamples = 100
  integer, parameter   :: dp = kind(1.d0)
  real(dp), parameter  :: pi = 3.1415926d0

  integer :: i,j,k,ind,ind2
  real(dp):: ix, iy, iz
  real(dp):: temp1

  real(dp):: s3dr = 0.d0, pk3dr = 0.d0
  real(dp):: s3dc = 0.d0, pk3dc = 0.d0
  real(dp):: vol, volk

  real(C_DOUBLE),             allocatable, dimension(:,:,:) :: arr_in_3d_r
  complex(C_DOUBLE_COMPLEX),  allocatable, dimension(:,:,:) :: arr_in_3d_c, arr_out_3d_c, arr_out_3d_r
  type(C_PTR)                                               :: p3d_r2c, p3d_c2r, p3d_cf, p3d_cb

  integer :: n
  integer, dimension(:), allocatable :: seed
  


  ! invent some physical quantities for your array
  real(dp), parameter  :: lx = Nx/2
  real(dp), parameter  :: ly = Ny/2
  real(dp), parameter  :: lz = Nz/2
  real(dp), parameter  :: dx = lx/Nx
  real(dp), parameter  :: dy = ly/Ny
  real(dp), parameter  :: dz = lz/Nz
  real(dp), parameter  :: dkx = 1.d0/lx
  real(dp), parameter  :: dky = 1.d0/ly
  real(dp), parameter  :: dkz = 1.d0/lz

  real(dp), dimension(:), allocatable :: distances_k, distances_k2pi, distances
  real(dp), dimension(:), allocatable :: Pk_3dc, Pk_3dr
  real(dp), dimension(:), allocatable :: sigma_3dc, sigma_3dr
  integer, dimension(:), allocatable  :: pk3d_counts, sigma3d_counts
  real(dp)                            :: lmax, kmax, d2, d3


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
  allocate(arr_in_3d_r(1:Nx, 1:Ny, 1:Nz))
  allocate(arr_in_3d_c(1:Nx, 1:Ny, 1:Nz))
  allocate(arr_out_3d_r(1:Nx/2+1, 1:Ny, 1:Nz))
  allocate(arr_out_3d_c(1:Nx, 1:Ny, 1:Nz))

  p3d_cf  =     fftw_plan_dft_3d(Nz, Ny, Nx, arr_in_3d_c, arr_out_3d_c, FFTW_FORWARD, FFTW_ESTIMATE)
  p3d_r2c = fftw_plan_dft_r2c_3d(Nz, Ny, Nx, arr_in_3d_r, arr_out_3d_r, FFTW_ESTIMATE)
  p3d_cb  =     fftw_plan_dft_3d(Nz, Ny, Nx, arr_out_3d_c, arr_in_3d_c, FFTW_BACKWARD, FFTW_ESTIMATE)
  p3d_c2r = fftw_plan_dft_c2r_3d(Nz, Ny, Nx, arr_out_3d_r, arr_in_3d_r, FFTW_ESTIMATE)


  !----------------------
  ! Fill arrays
  !----------------------

  do i=1, Nx
    do j=1, Ny
      do k=1, Nz
        ! call random_number(temp1)
        ! call random_number(temp2)
        ! call random_number(temp3)
        ! arr_in_3d_c(i,j,k) = cmplx(temp1, temp2, kind=dp)
        ! arr_in_3d_r(i,j,k) = temp3
        arr_in_3d_c(i,j,k) = cmplx(i,0,kind=dp)
        arr_in_3d_r(i,j,k) = i
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
      do k=1, Nz
        s3dc  = s3dc + abs(arr_in_3d_c(i,j,k))**2
        s3dr  = s3dr +     arr_in_3d_r(i,j,k)**2
      enddo
    enddo
  enddo


  !----------------------
  ! Do Forward FFTs
  !----------------------
  call     fftw_execute_dft(p3d_cf,  arr_in_3d_c, arr_out_3d_c)
  call fftw_execute_dft_r2c(p3d_r2c, arr_in_3d_r, arr_out_3d_r)



  !-----------------------------------
  ! Compute integrals of squares
  !-----------------------------------
  do i=1, Nx
    do j=1, Ny 
      do k=1, Nz
        pk3dc = pk3dc + abs(arr_out_3d_c(i,j,k))**2
      enddo
    enddo
  enddo

  do i=1, Nx/2+1
    if (i==1.or.i==Nx/2+1) then; temp1=1.d0; else; temp1=2.d0; endif;
    do j=1, Ny
      do k=1, Nz
        pk3dr = pk3dr + temp1*abs(arr_out_3d_r(i,j,k))**2
      enddo
    enddo
  enddo

  


  write(*,*) "------------------------"
  write(*,*) "   FORWARD TRANSFORM    "
  write(*,*) "------------------------"
  write(*,'(A21, 2A18)')   "",                         &
  "3d r2c",                "3d c2c"
  write(*,'(A21, 2E18.6)') "s^2",                      & 
  s3dr,                    s3dc
  write(*,'(A21, 2E18.6)') "Pk^2",                     &
  pk3dr,                   pk3dc
  write(*,'(A21, 2E18.6)') "Pk^2/s^2",                 &
  ps(pk3dr,s3dr),          ps(pk3dc,s3dc)
  write(*,'(A21, 2E18.6, A20)') "Pk^2/s^2/full volume",&
  ps_fav3(pk3dr,s3dr,'d'), ps_fav3(pk3dc, s3dc,'d'), "should be 1.0!"











  !=================================================
  ! PART2: CHECK CHOICE OF k BY COMPUTING int P(|k|)
  !=================================================
 
  !------------------------------------------------
  ! Now normalize your arrays in Fourier space
  !------------------------------------------------

  ! temp1 = sqrt(real(Nx*Ny*Nz,kind=dp))
  arr_out_3d_c=arr_out_3d_c/temp1
  arr_out_3d_r=arr_out_3d_r/temp1

  !---------------------
  ! Get distances
  !---------------------
  allocate(distances(0:nsamples));      distances=0.d0
  allocate(distances_k(0:nsamples));    distances_k=0.d0
  allocate(distances_k2pi(0:nsamples)); distances_k2pi=0.d0
  lmax = sqrt(((Nx/2.d0))**2 + ((Ny/2.d0))**2 + ((Nz/2.d0))**2)
  kmax = sqrt(((Nx/2.d0))**2 + ((Ny/2.d0))**2 + ((Nz/2.d0))**2)
  ! lmax = sqrt(((Nx/2+1)*dx)**2 + ((Ny/2+1)*dy)**2 + ((Nz/2+1)*dz)**2)
  ! kmax = sqrt(((Nx/2+1)*dkx)**2 + ((Ny/2+1)*dky)**2 + ((Nz/2+1)*dkz)**2)
  do i=0, nsamples
    distances(i)=i*lmax/nsamples
    distances_k(i) = i*kmax/nsamples
    distances_k2pi(i) = i*kmax/nsamples*2*pi
  enddo


  !-------------------------
  ! Histogram P(k)
  !-------------------------
  allocate(Pk_3dc(0:nsamples));         Pk_3dc=0;
  allocate(Pk_3dr(0:nsamples));         Pk_3dr=0;
  allocate(pk3d_counts(0:nsamples));    pk3d_counts = 0;
  allocate(sigma_3dc(0:nsamples));      sigma_3dc=0;
  allocate(sigma_3dr(0:nsamples));      sigma_3dr=0;
  allocate(sigma3d_counts(0:nsamples)); sigma3d_counts=0;
  

  do i=1, Nx
    if ((i-1)<Nx/2+1) then; ix=i-1; else; ix=-Nx+i-1; endif
    do j=1, Ny
      if ((j-1)<Ny/2+1) then; iy=j-1; else; iy=-Ny+j-1; endif
      do k=1, Nz
        if ((k-1)<Nz/2+1) then; iz=k-1; else; iz=-Nz+k-1; endif

        d3 = sqrt((ix)**2+(iy)**2+(iz)**2)
        ! d3 = sqrt((ix*dx)**2+(iy*dy)**2+(iz*dz)**2)
        ind = int(d3/lmax*nsamples)
        do ind2=ind, nsamples
          if (d3<distances(ind2)) then
            sigma_3dc(ind) = sigma_3dc(ind) + abs(arr_in_3d_c(i,j,k))
            sigma_3dr(ind) = sigma_3dr(ind) + abs(arr_in_3d_r(i,j,k))
            sigma3d_counts(ind) = sigma3d_counts(ind) + 1
            exit
          endif
        enddo

        d3 = sqrt((ix)**2+(iy)**2+(iz)**2)
        ! d3 = sqrt((ix*dkx)**2+(iy*dky)**2+(iz*dkz)**2)
        ind = int(d3/kmax*nsamples)
        do ind2=ind, nsamples
          if (d3<distances_k(ind2)) then
            Pk_3dc(ind) = Pk_3dc(ind) + abs(arr_out_3d_c(i,j,k))**2
            pk3d_counts(ind) = pk3d_counts(ind)+1
            exit
          endif
        enddo
      enddo
    enddo
  enddo

  do i=1, Nx/2+1
    ix=i-1
    if (i==1 .or. i==Nx/2+1) then; temp1=1.d0; else; temp1=2.d0; endif
    do j=1, Ny
      if ((j-1)<Ny/2+1) then; iy=j-1; else; iy=-Ny+j-1; endif
      do k=1, Nz
        if ((k-1)<Nz/2+1) then; iz=k-1; else; iz=-Nz+k-1; endif
        d3 = sqrt((ix)**2+(iy)**2+(iz)**2)
        ! d3 = sqrt((ix*dkx)**2+(iy*dky)**2+(iz*dkz)**2)
        ind = int(d3/kmax*nsamples)
        do ind2=ind, nsamples
          if (d3<distances_k(ind2)) then
            Pk_3dr(ind) = Pk_3dr(ind) + temp1*abs(arr_out_3d_r(i,j,k))**2
            exit
          endif
        enddo
      enddo
    enddo
  enddo

  ! compute average
  do i=0, nsamples
    if (sigma3d_counts(i)>0) then
      sigma_3dc(i) = sigma_3dc(i)/sigma3d_counts(i)
      sigma_3dr(i) = sigma_3dr(i)/sigma3d_counts(i)
    endif
    if (pk3d_counts(i)>0) then
      Pk_3dc(i) = Pk_3dc(i)/pk3d_counts(i)
      Pk_3dr(i) = Pk_3dr(i)/pk3d_counts(i)
    endif
  enddo


  ! volk = (Nx*dkx*Ny*dky*Nz*dkz)
  ! vol = (dx*dy*dz)
  ! volk = (dkx*dky*dkz)
  ! vol = (lx*ly*lz)
  ! volk=1
  ! vol=1
  vol=Nx*Ny*Nz
  volk=vol

  write(*,*)
  write(*,*) "------------------------"
  write(*,*) "  Testing choice of k   "
  write(*,*) "------------------------"


  write(*,'(A31, 2A18)')   "",                                            &
    "3d r2c",                         "3d c2c"
  write(*,'(A31, 2E18.6)') "s^2",                              &
    integ(distances,sigma_3dr),        integ(distances,sigma_3dc)
  write(*,'(A31, 2E18.6)') "s^2/s_orig ",                              &
    integ(distances,sigma_3dr)/ s3dr,        integ(distances,sigma_3dc)/s3dc
  write(*,'(A31, 2E18.6)') "s^2/s_orig/vol ",                              &
    integ(distances,sigma_3dr)/ s3dr/vol,        integ(distances,sigma_3dc)/s3dc/vol

  write(*,*)
  write(*,*)
  write(*,*)
  write(*,*)
  write(*,*)
  ! write(*,'(A31, 2E18.6)') "Pk^2 for k=1/l",                              &
  !   integ(distances_k,Pk_3dr),        integ(distances_k,Pk_3dc)
  ! write(*,'(A31, 2E18.6)') "Pk^2 for k=2pi/l",                              &
  !   integ(distances_k2pi,Pk_3dr),        integ(distances_k2pi,Pk_3dc)
  ! write(*,'(A31, 2E18.6)') "Pk^2/s^2 for k=1/l",                          &
  !   integ(distances_k,Pk_3dr)/s3dr,   integ(distances_k,Pk_3dc)/s3dc
  ! write(*,'(A31, 2E18.6)') "Pk^2/s^2 for k=2pi/l",                        &
  !   integ(distances_k2pi,Pk_3dr)/s3dr, integ(distances_k2pi,Pk_3dc)/s3dc
  ! write(*,'(A31, 2E18.6)') "Pk^2/Pk_orig^2 k=1/l",                          &
  !   integ(distances_k,Pk_3dr)/ pk3dr,   integ(distances_k,Pk_3dc)/pk3dc
  ! write(*,'(A31, 2E18.6)') "Pk^2/Pk_orig^2 k=2pi/l",                        &
  !   integ(distances_k2pi,Pk_3dr)/pk3dr, integ(distances_k2pi,Pk_3dc)/pk3dc
  !
  ! write(*,*)
  ! write(*,'(A31, 2E18.6)') "Pk^2/Pk_orig^2/vol_k k=1/l",                          &
  !   integ(distances_k,Pk_3dr)/ pk3dr/volk,   integ(distances_k,Pk_3dc)/pk3dc/volk
  ! write(*,'(A31, 2E18.6)') "Pk^2/Pk_orig^2/vol_k2pi k=1/l",                          &
  !   integ(distances_k,Pk_3dr)/ pk3dr/volk/(2*pi)**3,   integ(distances_k,Pk_3dc)/pk3dc/volk/(2*pi)**3
  !
  ! write(*,*)
  ! write(*,'(A31, 2E18.6)') "Pk^2/Pk_orig^2/vol_k k=2pi/l",                        &
  !   integ(distances_k2pi,Pk_3dr)/pk3dr/volk, integ(distances_k2pi,Pk_3dc)/pk3dc/volk
  ! write(*,'(A31, 2E18.6)') "Pk^2/Pk_orig^2/vol_k2pi k=2pi/l",                        &
  !   integ(distances_k2pi,Pk_3dr)/pk3dr/volk/(2*pi)**3, integ(distances_k2pi,Pk_3dc)/pk3dc/volk/(2*pi)**3
  !
  ! write(*,*)
  ! write(*,*)
  ! write(*,'(A31, 2E18.6)') "s^2/vol",                              &
  !   integ(distances,sigma_3dr)*vol,        integ(distances,sigma_3dc)*vol
  !
  ! write(*,*)
  ! write(*,'(A31, 2E18.6)') "Pk^2/vol_k k=1/l",                          &
  !   integ(distances_k,Pk_3dr)/ volk,   integ(distances_k,Pk_3dc)/volk
  ! write(*,'(A31, 2E18.6)') "Pk^2/vol_k2pi k=1/l",                          &
  !   integ(distances_k,Pk_3dr)/ volk/(2*pi)**3,   integ(distances_k,Pk_3dc)/volk/(2*pi)**3
  !
  ! write(*,*)
  ! write(*,'(A31, 2E18.6)') "Pk^2/Pk_orig^2/vol_k k=2pi/l",                        &
  !   integ(distances_k2pi,Pk_3dr)/volk, integ(distances_k2pi,Pk_3dc)/volk
  ! write(*,'(A31, 2E18.6)') "Pk^2/Pk_orig^2/vol_k2pi k=2pi/l",                        &
  !   integ(distances_k2pi,Pk_3dr)/volk/(2*pi)**3, integ(distances_k2pi,Pk_3dc)/volk/(2*pi)**3
  !


  deallocate(distances_k, distances_k2pi, distances)
  deallocate(Pk_3dc, Pk_3dr, pk3d_counts)
  deallocate(sigma_3dc, sigma_3dr, sigma3d_counts)







  !=================================================
  ! PART3: BACKWARD TRANSFORM
  !=================================================

  s3dc=0;  pk3dc=0;
  s3dr=0;  pk3dr=0;


  !---------------------------------------------------
  ! FFTW may change input arrays:
  ! Compue integrals of squares of input arrays before
  ! the transform
  !---------------------------------------------------
  do i=1, Nx
    do j=1, Ny
      do k=1, Nz
        pk3dc = pk3dc + abs(arr_out_3d_c(i,j,k))**2
      enddo
    enddo
  enddo

  do i=1, Nx/2+1
    if (i==1.or.i==Nx/2+1) then; temp1=1.d0; else; temp1=2.d0; endif;
    do j=1, Ny
      do k=1, Nz
        pk3dr = pk3dr + temp1*abs(arr_out_3d_r(i,j,k))**2
      enddo
    enddo
  enddo



  !----------------------
  ! Do Forward FFTs
  !----------------------
  call     fftw_execute_dft(p3d_cb,  arr_out_3d_c, arr_in_3d_c)
  call fftw_execute_dft_c2r(p3d_c2r, arr_out_3d_r, arr_in_3d_r)


  !-----------------------------------
  ! Compute integrals of squares
  !-----------------------------------
  do i=1, Nx
    do j=1, Ny
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
  write(*,'(A21, 2A18)')   "",                          &
  "3d r2c",                "3d c2c"
  write(*,'(A21, 2E18.6)') "s^2",                       &
  s3dr,                    s3dc
  write(*,'(A21, 2E18.6)') "Pk^2",                      &
  pk3dr,                   pk3dc
  write(*,'(A21, 2E18.6)') "Pk^2/s^2",                  &
  ps(pk3dr,s3dr),          ps(pk3dc,s3dc)
  write(*,'(A21, 2E18.6, A20)') "Pk^2/s^2/full volume", &
  ps_fav3(pk3dr,s3dr,'m'), ps_fav3(pk3dc, s3dc,'m'), "should be 1.0!"

  ! Note that here the transform goes from Pk -> sigma, but to keep the 
  ! layout of the printing, multiply the source instead of dividing the result.




  deallocate(arr_in_3d_r, arr_out_3d_r)
  deallocate(arr_in_3d_c, arr_out_3d_c)
  call fftw_destroy_plan(p3d_r2c)
  call fftw_destroy_plan(p3d_cf)



  contains 
    real(dp) function ps(pk, sig)
      ! returns Pk^2/s^2
      real(dp), intent(in) ::pk, sig
      ps = pk/sig
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

    real(dp) function integ(distances, pk)
      ! Integrate P(k) over all fourier space, assume spherical symmetry
      real(dp),dimension(0:nsamples), intent(in) :: distances, pk
      integer  :: i
      real(dp) :: temp

      temp = 0

      do i=1, nsamples
        temp = temp + 0.5*(pk(i)*distances(i)**2 + pk(i-1)*distances(i-1)**2) * (distances(i)-distances(i-1))
      enddo

      integ = 4*pi*temp

    end function

end program use_fftw



  
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


  real(dp), dimension(:), allocatable :: distances_k2d, distances_k2pi2d, distances_2d
  real(dp), dimension(:), allocatable :: distances_k3d, distances_k2pi3d, distances_3d
  real(dp), dimension(:), allocatable :: Pk_2dc, Pk_2dr, Pk_3dc, Pk_3dr
  real(dp), dimension(:), allocatable :: sigma_2dc, sigma_2dr, sigma_3dc, sigma_3dr
  integer, dimension(:), allocatable  :: pk2d_counts, pk3d_counts, sigma2d_counts, sigma3d_counts
  real(dp)                            :: lmax, kmax, d2, d3


  integer :: ff2d, ff3d, ff2dk, ff3dk

  !=================================================
  ! PART2: CHECK CHOICE OF k BY COMPUTING int P(|k|)
  !=================================================
 
  !------------------------------------------------
  ! Now normalize your arrays in Fourier space
  !------------------------------------------------

  temp1 = sqrt(real(Nx*Ny,kind=dp))
  temp2 = sqrt(real(Nx*Ny*Nz,kind=dp))
  arr_out_2d_c=arr_out_2d_c/temp1
  arr_out_2d_r=arr_out_2d_r/temp1
  arr_out_3d_c=arr_out_3d_c/temp2
  arr_out_3d_r=arr_out_3d_r/temp2

  !---------------------
  ! Get distances
  !---------------------
  allocate(distances2d(0:nsamples));      distances2d=0.d0
  allocate(distances_k2d(0:nsamples));    distances_k2d=0.d0
  allocate(distances_k2pi2d(0:nsamples)); distances_k2pi2d=0.d0
  allocate(distances3d(0:nsamples));      distances3d=0.d0
  allocate(distances_k3d(0:nsamples));    distances_k3d=0.d0
  allocate(distances_k2pi3d(0:nsamples)); distances_k2pi3d=0.d0
  ! lmax = sqrt(((Nx/2+1)*dx)**2 + ((Ny/2+1)*dy)**2 + ((Nz/2+1)*dz)**2)
  ! kmax = sqrt(((Nx/2+1)*dkx)**2 + ((Ny/2+1)*dky)**2 + ((Nz/2+1)*dkz)**2)
  ! do i=0, nsamples
  !   distances(i)=i*lmax/nsamples
  !   distances_k(i) = i*kmax/nsamples
  !   distances_k2pi(i) = i*kmax/nsamples*2*pi
  ! enddo

  ff2d = 1
  ff3d = 1
  ff2dk = 1
  ff3dk = 1
  do i=0, Nx/2
    do j=0, Ny/2
      d2 = sqrt((i*dx)**2+(j*dy)**2)
      if (d>distances_2d(ff2d-1)) then
        distances_2d(ff2d) = d2
        ff2d = ff2d + 1
      else
        do ind=1, nsamples
          if (d2<distances_2(ind)) exit
        enddo
        if (ind < nsamples) then
          do ind2=ff2d, ind, -1
            distances_2d(ind2+1) = distances_2d(ind2)
          enddo
          distances_2d(ind) = d2
        endif
      endif
        
        
      do z=0, Nz/2


  !-------------------------
  ! Histogram P(k)
  !-------------------------
  allocate(Pk_2dc(0:nsamples)); Pk_2dc=0;
  allocate(Pk_2dr(0:nsamples)); Pk_2dr=0;
  allocate(Pk_3dc(0:nsamples)); Pk_3dc=0;
  allocate(Pk_3dr(0:nsamples)); Pk_3dr=0;
  allocate(pk2d_counts(0:nsamples)); pk2d_counts=0;
  allocate(pk3d_counts(0:nsamples)); pk3d_counts=0;
  allocate(sigma_2dc(0:nsamples)); sigma_2dc=0;
  allocate(sigma_2dr(0:nsamples)); sigma_2dr=0;
  allocate(sigma_3dc(0:nsamples)); sigma_3dc=0;
  allocate(sigma_3dr(0:nsamples)); sigma_3dr=0;
  allocate(sigma2d_counts(0:nsamples)); sigma2d_counts=0;
  allocate(sigma3d_counts(0:nsamples)); sigma3d_counts=0;
  

  do i=1, Nx
    if ((i-1)<Nx/2+1) then; ix=i-1; else; ix=-Nx+i-1; endif
    do j=1, Ny
      if ((j-1)<Ny/2+1) then; iy=j-1; else; iy=-Ny+j-1; endif
      
      d2 = sqrt((ix*dx)**2+(iy*dy)**2)
      ind = int(d2/lmax*nsamples)
      sigma_2dc(ind) = sigma_2dc(ind) + abs(arr_in_2d_c(i,j))**2
      sigma_2dr(ind) = sigma_2dr(ind) + abs(arr_in_2d_r(i,j))**2
      sigma2d_counts(ind) = sigma2d_counts(ind) + 1

      d2 = sqrt((ix*dkx)**2+(iy*dky)**2)
      ind = int(d2/kmax*nsamples)
      Pk_2dc(ind) = Pk_2dc(ind) + abs(arr_out_2d_c(i,j))**2
      pk2d_counts(ind) = pk2d_counts(ind)+1

      do k=1, Nz
        if ((k-1)<Nz/2+1) then; iz=k-1; else; iz=-Nz+k-1; endif

        d3 = sqrt((ix*dx)**2+(iy*dy)**2+(iz*dz)**2)
        ind = int(d3/lmax*nsamples)
        sigma_3dc(ind) = sigma_3dc(ind) + abs(arr_in_3d_c(i,j,k))
        sigma_3dr(ind) = sigma_3dr(ind) + abs(arr_in_3d_r(i,j,k))
        sigma3d_counts(ind) = sigma3d_counts(ind) + 1

        d3 = sqrt((ix*dkx)**2+(iy*dky)**2+(iz*dkz)**2)
        ind = int(d3/kmax*nsamples)
        Pk_3dc(ind) = Pk_3dc(ind) + abs(arr_out_3d_c(i,j,k))**2
        pk3d_counts(ind) = pk3d_counts(ind)+1
      enddo
    enddo
  enddo

  do i=1, Nx/2+1
    ix=i-1
    if (i==1 .or. i==Nx/2+1) then; temp1=1.d0; else; temp1=2.d0; endif
    do j=1, Ny
      if ((j-1)<Ny/2+1) then; iy=j-1; else; iy=-Ny+j-1; endif
      d2 = sqrt((ix*dkx)**2+(iy*dky)**2)
      ind = int(d2/kmax*nsamples)
      Pk_2dr(ind) = Pk_2dr(ind) + temp1*abs(arr_out_2d_r(i,j))**2
      do k=1, Nz
        if ((k-1)<Nz/2+1) then; iz=k-1; else; iz=-Nz+k-1; endif
        d3 = sqrt((ix*dkx)**2+(iy*dky)**2+(iz*dkz)**2)
        ind = int(d3/kmax*nsamples)
        Pk_3dr(ind) = Pk_3dr(ind) + temp1*abs(arr_out_3d_r(i,j,k))**2
      enddo
    enddo
  enddo


  do i=0, nsamples
    if (sigma2d_counts(i)>0) then
      sigma_2dc(i) = sigma_2dc(i)/sigma2d_counts(i)
      sigma_2dr(i) = sigma_2dr(i)/sigma2d_counts(i)
    endif
    if (sigma3d_counts(i)>0) then
      sigma_3dc(i) = sigma_3dc(i)/sigma3d_counts(i)
      sigma_3dr(i) = sigma_3dr(i)/sigma3d_counts(i)
    endif
    if (pk2d_counts(i)>0) then
      Pk_2dc(i) = Pk_2dc(i)/pk2d_counts(i)
      Pk_2dr(i) = Pk_2dr(i)/pk2d_counts(i)
    endif
    if (pk3d_counts(i)>0) then
      Pk_3dc(i) = Pk_3dc(i)/pk3d_counts(i)
      Pk_3dr(i) = Pk_3dr(i)/pk3d_counts(i)
    endif
  enddo


  write(*,*)
  write(*,*) "------------------------"
  write(*,*) "  Testing choice of k   "
  write(*,*) "------------------------"
  write(*,'(A21, 4A18)')   "",                                            &
    "2d r2c",                         "2d c2c",                           &
    "3d r2c",                         "3d c2c"
  write(*,'(A21, 4E18.6)') "s^2",                              &
    integ(distances,sigma_2dr),        integ(distances,sigma_2dc),          &
    integ(distances,sigma_3dr),        integ(distances,sigma_3dc)
  write(*,'(A21, 4E18.6)') "Pk^2 for k=1/l",                              &
    integ(distances_k,Pk_2dr),        integ(distances_k,Pk_2dc),          &
    integ(distances_k,Pk_3dr),        integ(distances_k,Pk_3dc)
  write(*,'(A21, 4E18.6)') "Pk^2 for k=2pi/l",                              &
    integ(distances_k2pi,Pk_2dr),        integ(distances_k2pi,Pk_2dc),          &
    integ(distances_k2pi,Pk_3dr),        integ(distances_k2pi,Pk_3dc)

  write(*,'(A21, 4E18.6)') "Pk^2/s^2 for k=1/l",                          &
    integ(distances_k,Pk_2dr)/s2dr,   integ(distances_k,Pk_2dc)/s2dc,     &
    integ(distances_k,Pk_3dr)/s3dr,   integ(distances_k,Pk_3dc)/s3dc
  write(*,'(A21, 4E18.6)') "Pk^2/s^2 for k=2pi/l",                        &
    integ(distances_k2pi,Pk_2dr)/s2dr, integ(distances_k2pi,Pk_2dc)/s2dc, &
    integ(distances_k2pi,Pk_3dr)/s3dr, integ(distances_k2pi,Pk_3dc)/s3dc



  deallocate(distances_k, distances_k2pi)
  deallocate(Pk_2dc, Pk_2dr)
  deallocate(Pk_3dc, Pk_3dr)



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




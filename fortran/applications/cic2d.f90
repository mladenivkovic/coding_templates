!---------------------------------------------------------
! An implementation of the Cloud-In-Cell interpolation
! method on a 2d grid with periodical boundary conditions.
!---------------------------------------------------------

program cic

  implicit none

  integer, parameter :: N = 4                   ! Number of cells per side
  integer, parameter :: np = 1                  ! Number of particles to distribute
  ! integer, parameter :: np = 2                  ! Number of particles to distribute
  real, parameter    :: boxlen = real(N)        ! size of a side of a box
  real, parameter    :: dx = N/boxlen           ! size of a cell
  real, parameter    :: hdx = dx/2              ! half size of a cell
  
  real, dimension(1:N, 1:N) :: density = 0 ! density field
  real, dimension(1:np)     :: x, y, m     ! x, y and mass arrays of particles
  

  integer :: i,j,p
  integer :: iup, idown, jup, jdown
  real    :: rho  
  real    :: xup, yup

  !--------------------------------------------------
  ! Setup
  ! Don't forget to set np in parameters properly!
  !--------------------------------------------------

  ! 1 Particle, inside grid
  ! x = (/1.2/)
  ! y = (/1.2/)
  ! m = (/1.0/)

  ! 1 Particle, on edge
  ! x = (/3.7/)
  ! y = (/3.7/)
  ! m = (/1.0/)

  ! 2 Particles, on edge
  ! x = (/1.7, 3.7/)
  ! y = (/3.7, 2.2/)
  ! m = (/1.0, 1.0/)

  ! 1 Particle, center of cell 
  x = (/1.5/)
  y = (/1.5/)
  m = (/1.0/)



  ! normalize mass
  m = m/sum(m)


  !----------------------
  ! CIC
  !----------------------

  do p=1, np
    ! get upper and lower cell indices where mass will be deposited
    iup   = int((x(p)+hdx)/boxlen*N)+1
    idown = iup - 1
    jup   = int((y(p)+hdx)/boxlen*N)+1
    jdown = jup - 1

    rho = m(p)/dx**3
    xup = x(p) + hdx - (iup-1)*dx
    yup = y(p) + hdx - (jup-1)*dx

    if (iup>N)   iup   = iup -N
    if (idown<1) idown = N+idown
    if (jup>N)   jup   = jup -N
    if (jdown<1) jdown = N+jdown

    write(*,*)

    density(iup,   jup)   = density(iup,   jup)   + xup      * yup      * rho
    density(idown, jup)   = density(idown, jup)   + (dx-xup) * yup      * rho
    density(iup,   jdown) = density(iup,   jdown) + xup      * (dx-yup) * rho
    density(idown, jdown) = density(idown, jdown) + (dx-xup) * (dx-yup) * rho

  enddo



  !--------------------------------
  ! Print results
  !--------------------------------

  ! Reminder: (0,0) corner is bottom left corner in print

  write(*,'(A5)') "y= |"

  do j=N, 1, -1
    write(*,'(I3, A2)', advance='no') j, "|"
    do i=1, N
        write(*,'(F6.3,x)', advance='no') density(i,j)
    enddo
    write(*,*)
  enddo

  do i=1, N
    write(*,'(A7)', advance='no') "-------"
  enddo
  write(*,'(A6)') "------"

  write(*,'(A5)', advance='no') "x= |"
  do i=1, N
    write(*,'(I6,x)', advance='no') i
  enddo
  write(*,*)

  write(*,*)
  write(*,*) "Total mass:", sum(density)

end program cic

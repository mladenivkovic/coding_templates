program preprocessing

! a program to demonstrate preprocessing and preprocessing
! directives.

! Compile with ifort -fpp -D limit=X preprocessingouptions.f90

! See 
! https://software.intel.com/en-us/node/524750#AE98EA34-DCDC-4EEA-B0AE-80ACF4326CA9
! (Using the fpp directive)
! and
! https://software.intel.com/en-us/node/525101#308122D4-3830-4AAD-AA98-722470D4B280
! (the D compiler option)
! for more informations.

! This program will calculate the integral of sin(x) from 0 to
! pi/2  if limit = 1
! pi    if limit = 2
! 2 pi  if limit = 3
! send out an error message if no directive was specified.



! preprocessing directives must always be at the beginning of the line.
! No whitespace before them is allowed!


    implicit none
    integer, parameter :: n = 1000000
    integer :: i
    real (8) , dimension(0:n) :: array
    real (8),  parameter :: pihalf = 1.5707963, pi = 3.14159265, nul=0

#ifdef limit
    write(*, *) "You selected the limit ", limit
    write(*, *)
#endif

#if limit==1
    write(*, *) "Integrating sin(x) from 0 to pi/2"
    write(*, '(A, F5.3)') " Result: ", integratesin(nul, pihalf)

#elif limit==2
    write(*, *) "Integrating sin(x) from 0 to pi"
    write(*, '(A, F5.3)') " Result: ", integratesin(nul, pi)

#elif limit==3
    write(*, *) "Integrating sin(x) from 0 to 2pi"
    write(*, '(A, F6.3)') " Result: ", integratesin(nul, 2.000*pi)

#else
    write(*, *) "Option", limit, "not found."
#endif

contains 
    real (8) function integratesin(start, finish)

        implicit none
        real  (8), intent (in) :: start, finish
        real  (8):: calc = 0.0, stepsize, step

        stepsize = 1.000/n
        
        step = start
        
        do while (step <= finish)
            calc = calc + sin(step)
            step = step + stepsize
        end do

        integratesin = calc * stepsize
    end function integratesin


end program preprocessing

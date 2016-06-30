program random

! A program to demonstrate generating random numbers with Fortran.

    implicit none
    real :: randomreal, anotherrandomreal

    ! This produces a random real in the interval [0,1)
    call random_number(randomreal)
    write(*, *) randomreal
    
    !random_seed() makes sure, that the random number is different
    !every time the program is executed.
    !The first random number will always remain the same,
    !while the one below (after random_seed) will differ every
    !time the program is executed.
    call random_seed()
    call random_number(anotherrandomreal)
    write(*,*) anotherrandomreal 


end program random

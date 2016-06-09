program smallnamelist

    implicit none

    real :: somerealone, somerealtwo, somerealthree
    integer :: someint, someotherint
    real, dimension(4) :: somerealarray
    integer, dimension(3) :: someintarray


    ! declare what part of the namelists will be stored where
    namelist /nmlint/ someint, someotherint
    namelist /nmlreal/ somerealone, somerealtwo, somerealthree
    namelist /nmlarray/ somerealarray, someintarray




    open(1, file='input.nml', status='old')

    write(6, *)
    write(6, *) "Reading in NMLINT"
    read(1, nml=nmlint)
    write(6, *) someint, someotherint


    write(6, *)
    write(6, *) "Reading in NMLREAL"
    read(1, nml=nmlreal)
    write(6, *) somerealone, somerealtwo, somerealthree

    write(6, *)
    write(6, *) "Reading in NMLARRAY"
    read(1, nml=nmlarray) 
    write(6, *) somerealarray
    write(6, *) someintarray



end program smallnamelist

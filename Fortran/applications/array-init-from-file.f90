program array_from_file

!=======================================================================
! A quick template to create and print arrays and allocatable arrays.
! IMPORTANT!!!!!!!!
! Execute with a.out < ./inputfiles/arrayinput.txt
!=======================================================================


 
    implicit none

    integer, parameter :: nelements = 10
    integer, parameter :: xdim = 10
    integer, parameter :: ydim = 5
    integer, parameter :: cubexdim=3, cubeydim = 3, cubezdim = 3 ! to define 3 x 3 x 3 cube

    !1D-Array
    integer, dimension(1:nelements) :: OneDimArray

    !2D-Array
    integer, dimension(1:xdim, 1:ydim) :: TwoDimArrayOne, TwoDimArrayTwo


    !Initiating others that we might need
    integer :: ArrayElement
    integer :: zeile, spalte





    write(*,*) "-----------------------"
    write(*,*) "--- Read from input ---"
    write(*,*) "-----------------------"
    write(*,*) "(If the program hangs - did you execute with ./a.out < ../inputfiles/arrayinput.txt ?)"



    write(*,*) ' '
    write(*,*)  '1dim-array'
    write(*,*) ' '


    !1-dim array
    do spalte = 1, nelements
        read *, ArrayElement
        OneDimArray(spalte) = ArrayElement
    end do


    write(*,'(10I3)')  OneDimArray
    write(*,*) ""
    write(*,*) ""











    write(*,*) ' '
    write(*,*) '2dim-array - 1st verision'
    write(*,*) ' '
    
    !2-dim array: Two ways of reading in values
    do zeile  = 1, ydim
        do spalte = 1, xdim 
            read *, ArrayElement
            !write(*,*) zeile, spalte
            TwoDimArrayOne(spalte, zeile) = ArrayElement
        end do
    end do


    do zeile = 1, ydim ! Print array als 5 Zeilen und 10 Spalten
        write(*,'(10I3)') (TwoDimArrayOne(spalte, zeile), spalte = 1, xdim)
    end do










    write(*,*) ""
    write(*,*) ""
    write(*,*) ' '
    write(*,*) "2dim-array, 2nd version"
    write(*,*) ' '


    do spalte = 1, xdim
        do zeile = 1, ydim
            read *, ArrayElement
            TwoDimArrayTwo(spalte, zeile) = ArrayElement
        end do
    end do


    do zeile = 1, ydim ! Print array als 5 Zeilen und 10 Spalten
        write(*, '(10I4)') (TwoDimArrayTwo( spalte, zeile), spalte = 1, xdim)
    end do



end program array_from_file

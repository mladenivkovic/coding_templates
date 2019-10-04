program readfromfile

    !================================
    ! How to read input from files.
    !================================


    implicit none
    integer :: i,j, linecounter 
    integer, dimension(:), allocatable :: onedimarray
    integer, dimension(:, :), allocatable :: twodimarray, unknownlengtharray
    real, dimension(:,:), allocatable :: twodimrealsarray, skippedlinearray

    !---------------
    ! WARNING
    !---------------
    ! Unit numbers 0, 5, and 6 are associated with the standard error, standard input, and standard output files. 
    ! (e.g.  open(unit = x), you mustn't use x = 0, 5 or 6.






    !=====================
    ! INTEGERS
    !=====================


    write(*, *) ""
    write(*, *) ""
    write(*, *) "INTEGERS"
    write(*, *) "--------"



    !-----------------------------
    ! One column integer input
    !-----------------------------
    
    allocate(onedimarray(1:10))
    
    open(unit=1, file='inputfiles/1columninteger.dat')
    ! for more information: see 
    ! https://docs.oracle.com/cd/E19957-01/805-4939/6j4m0vnaf/index.html
    do i = 1, 10
        read (1, '(I8)') onedimarray(i)
    end do
    close(1)

    write(*, *) "1 column integer input"
    write(*, *) ""
    write(*, '(I4)') onedimarray







    !-----------------------------
    ! Four column integer input
    !-----------------------------
    
    allocate(twodimarray(1:10, 1:4))
    
    open(unit=2, file='inputfiles/4columninteger.dat')
    do i = 1, 10
        read (2, '(4I8)') twodimarray(i, 1), twodimarray(i, 2), twodimarray(i, 3), twodimarray(i, 4)
    end do
    close(2)

    write(*, *) ""
    write(*, *) ""
    write(*, *) "4 column integer input"
    write(*, *) ""
    
    do i = 1, 10 
        write(*, '(10I4)') (twodimarray(i, j), j = 1, 4)
    end do









    !=====================
    ! REALS
    !=====================


    write(*, *) ""
    write(*, *) ""
    write(*, *) "REALS"
    write(*, *) "-----"



    write(*, *) ""
    write(*, *) "3 column reals input"
    allocate(twodimrealsarray(1:10, 1:3))

    open(unit=7, file='inputfiles/3columnreals.dat')
    do i = 1, 10
        read(7, '(3ES10.3E2)') twodimrealsarray(i, 1), twodimrealsarray(i, 2), twodimrealsarray(i, 3)
    end do
    
    close(unit=7)


    do i = 1, 10
        write(*,  '(3ES12.3E2)') (twodimrealsarray(i, j), j = 1, 3)
    end do











    !=========================
    ! Skipping the first line
    !=========================

    write(*, *) ""
    write(*, *) ""
    write(*, *) "SKIPPING THE FIRST LINE"
    write(*, *) "-----------------------"



    write(*, *) ""
    allocate(skippedlinearray(1:10, 1:3))

    open(unit=7, file='inputfiles/3columnrealswithtitle.dat') 
    read(7,*)   ! Just execute read once, fortran will then
                ! go on to the next line.
    do i = 1, 10
        read(7, '(3ES10.3E2)') skippedlinearray(i, 1),skippedlinearray(i, 2), skippedlinearray(i, 3)
    end do
    
    close(unit=7)


    do i = 1, 10
        write(*,  '(3ES12.3E2)') (skippedlinearray(i, j), j = 1, 3)
    end do









    !=========================
    ! Unknown number of lines 
    !=========================

    write(*, *) ""
    write(*, *) ""
    write(*, *) "DETERMINING THE NUMBER OF LINES FIRST"
    write(*, *) "-------------------------------------"


    !Assuming unknown number of lines, but 3 columns with 1 line for title
    write(*, *) ""

    open(unit=7, file='inputfiles/unknownlengthintegers.dat') 
    linecounter = 0
    do
        read(7,*, end=10) 
        linecounter = linecounter + 1
    end do
    10 close(7)

    write(*,*) "linecounter", linecounter 
    write(*,*) 



    allocate(unknownlengtharray(1:linecounter-1, 1:3)) 
    ! allocating array now. 
    ! Limit is linecounter -1 because first line = title 

    ! now reading in the values
    open(unit=8, file='inputfiles/unknownlengthintegers.dat') 
    
    read(8,*)   !skipping title

    do i = 1, linecounter-1
        read(8, '(3I8)') unknownlengtharray(i, 1), unknownlengtharray(i, 2), unknownlengtharray(i, 3)
    end do
    close(8)

    ! Printing
    do i = 1, linecounter-1
        write(*,  '(3I8)') (unknownlengtharray(i, j), j = 1, 3)
    end do


end program readfromfile
           

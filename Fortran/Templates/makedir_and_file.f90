program makedir

    implicit none
    character(20) :: dirname='test_directory'
    character(10) :: filename='testfile'
    character(80) :: cmnd

    cmnd = 'mkdir -p '//TRIM(dirname)
    call system(cmnd)
    
    open(unit=1, file=TRIM(dirname)//'/'//TRIM(filename), form='formatted')
    write(1, '(A)') "Hello!"
    close(1)

end program makedir

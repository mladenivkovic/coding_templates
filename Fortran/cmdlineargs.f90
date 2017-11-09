program cmdlineargs

  integer :: i
  character(len=32) :: arg


  write(*, *) "Arguments given: ", iargc()

  do i = 1, iargc()
    call getarg(i, arg)
    write(*,*) arg
  end do



end program cmdlineargs


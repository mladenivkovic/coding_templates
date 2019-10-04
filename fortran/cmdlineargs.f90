program cmdlineargs

  !=============================
  ! Reading cmd line args
  !=============================

  integer :: i
  character(len=32) :: arg


  write(*,*) "Give me cmd line args to show you what I can do."

  ! GNU extension
  !-------------------

  ! write(*, *) "Arguments given: ", iargc()
  !
  ! do i = 1, iargc()
  !   call getarg(i, arg)
  !   write(*,*) arg
  ! end do


  ! 2003 Fortran standard
  do i=1, command_argument_count()
    call get_command_argument(i, arg)
    write(*,*) arg
  enddo


end program cmdlineargs


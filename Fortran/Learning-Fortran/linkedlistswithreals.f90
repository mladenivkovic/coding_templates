program linkedlistwithreals

  type link
    real :: n
    type (link), pointer :: next => null()
  end type link

  implicit none
  type (link), pointer :: root, current
  integer :: i = 0, m
  integer :: io_stat_number = 0
  real, allocatable, dimension (:) :: x
  character (len=80) :: fname

  print *, 'file containing real numbers?'
  read '(a)', fname
  open (unit=1, file=fname, status='old')
  allocate (root)
! read in 1st number from file
  read (unit=1, fmt=*, iostat=io_stat_number) root%n
  if (io_stat_number==0) then !     not end of file
    i = i + 1
    allocate (root%next)
  end if
  current => root
! read in remaining numbers (1 per line)
  do while (associated(current%next))
    current => current%next
    read (unit=1, fmt=*, iostat=io_stat_number) current%n
    if (io_stat_number==0) then
      i = i + 1
      allocate (current%next)
    end if
  end do
  m = i
  allocate (x(1:m))
  i = 1
  current => root
  do while (associated(current%next))
    x(i) = current%n
    i = i + 1
    current => current%next
  end do
  print *, m, ' numbers read from file'
  print *, m, ' elements of array x are:'
  do i = 1, m
    print *, x(i)
  end do
end program linkedlistwithreals

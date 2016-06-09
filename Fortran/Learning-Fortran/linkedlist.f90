program ch2201

  !use link_module
  
  implicit none
  
  ! definitions for singly linked list
    type link
        character(len=1) :: c
        type (link), pointer :: next => null()
    end type link
  
  type (link), pointer :: root, current
  character (len=:), allocatable :: word
  character (len=80) :: fname="input.txt"
  integer :: io_stat_number = 0
  integer :: n, i = 0
! reads an arbitrary amount of text from a file
! - eof terminates it
  !print *, ' Type in the file name'
  !read '(a)', fname='input.txt'
  open (unit=1, file=fname, status='old')
! reads in a character at a time until eof
  allocate (root)

  read (unit=1, fmt='(a)', advance='no', iostat=io_stat_number) root%c

  if (io_stat_number/=-1) then
    allocate (root%next)
    i = i + 1
  end if

  current => root

  do while (associated(current%next))
    current => current%next
    read (unit=1, fmt='(a)', advance='no', iostat=io_stat_number) current%c
    if (io_stat_number/=-1) then
      allocate (current%next)
      i = i + 1
    end if
  end do

  print *, i, ' characters read'
! Allocate the deferred length character variable
! to the correct size and copy from the linked list.
  n = i
  allocate (character(len=n) :: word)
  i = 0
  current => root

  do while (associated(current%next))
    i = i + 1
    word(i:i) = current%c
    current => current%next
  end do

  print *, 'text read in was:'
  print *, word

end program ch2201


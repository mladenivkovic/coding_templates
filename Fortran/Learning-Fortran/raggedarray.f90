module ragged_module
  implicit none
  type ragged
    real, dimension (:), allocatable :: ragged_row
  end type ragged
end module ragged_module

program ch2203
  use ragged_module
  implicit none
  integer :: i
  integer, parameter :: n = 3
  type (ragged), dimension (1:n) :: lower_diag
  do i = 1, n
    allocate (lower_diag(i)%ragged_row(1:i))
    print *, ' type in the values for row ', i
    read *, lower_diag(i)%ragged_row(1:i)
  end do
  do i = 1, n
    print *, lower_diag(i)%ragged_row(1:i)
  end do
end program ch2203


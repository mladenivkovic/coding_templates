program check_if_array_in_array

  !--------------------------------------------
  ! Check element by element of target array
  ! which elements are in source array.
  ! Assume both arrays are sorted by value.
  !--------------------------------------------


  implicit none

  integer, dimension(1:10) :: target_arr = (/ 2, 4, 6, 8, 10, 12, 14, 16, 18, 20/) !represents progenitor particles
  integer, dimension(1:10) :: found
  integer, dimension(1:4)  :: source_arr = (/3, 4, 6, 9/)                    !represents particles on CPU
  integer :: nt = 10, ns = 4, it, is 
  integer :: nfound = 0




  it = 1; is = 1;

  do while (it < nt .and. is < ns); 
    if (target_arr(it) < source_arr(is)) then
      it = it + 1
    else if (target_arr(it) > source_arr(is)) then
      is = is + 1
    else
      nfound = nfound + 1
      found(nfound) = target_arr(it)
      it = it + 1
      is = is + 1
    endif

  enddo



  write(*, '(A13,x,10(I3,x))') "target array:", target_arr
  write(*, '(A13,x,4(I3,x))') "source array:", source_arr
  write(*, '(A31)', advance='no') "Target elements in source array: "
  do it = 1, nfound
    write(*, '(I3,x)', advance='no') found(it)
  enddo
  write(*,*)





end program

!===================================
! Some openmp basics
! compile with -fopenmp
!===================================


program omp_basics

  use omp_lib
  implicit none

  integer, parameter       :: n=12
  integer                  :: i, j
  integer, dimension(n,n)  :: arr
  integer, dimension(n)    :: oned
  integer, dimension(10*n) :: indexes

  arr=0
  oned=0
  ! get dummy indexes for writing in same spot multiple times
  do i=1,10
    indexes((i-1)*n+1:i*n) = [(j, j=1, n)]
  enddo



  !$OMP PARALLEL DEFAULT(SHARED)

    !$OMP DO
      do j=1, n
        do i=1, n
          arr(i,j) = arr(i,j)+10*i+j
        enddo
      enddo
    !$OMP END DO

    !$OMP DO
      do i=1, 10*n
        !$OMP ATOMIC
        oned(indexes(i)) = oned(indexes(i)) + 1
      enddo
    !$OMP END DO


  !$OMP END PARALLEL


  do j=1,n
    do i=1,n
      write(*,'(I5,x)', advance='no') arr(i,j)
    enddo
    write(*,*)
  enddo

  write(*,*)

  do i=1, n
    write(*,'(I5,x)', advance='no') oned(i)
  enddo
  write(*,*)


end program omp_basics

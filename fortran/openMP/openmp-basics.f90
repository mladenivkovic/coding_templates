!===================================
! Some openmp basics
! compile with -fopenmp
!===================================


program omp_basics

  use omp_lib
  implicit none

  integer, parameter :: n=10000
  integer            :: id, nthreads
  integer            :: i
  real               :: mysum = 0.235234


  ! set number of threads during runtime
  call omp_set_num_threads(3) 


  !------------------------------
  ! start parallel region
  !------------------------------

  !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(id, nthreads)

    !----------------------
    ! get runtime params
    !----------------------
    nthreads = omp_get_num_threads()
    id = omp_get_thread_num()

    write(*,*) "Hello world from thread ", id, "out of", nthreads

    !------------------------------
    !$OMP DO REDUCTION(+:mysum)
    !------------------------------
    do i=1, n 
      mysum = mysum+1
    enddo
    !$OMP END DO

    
    !------------------------
    !$OMP MASTER
    !------------------------
    write(*,*) "Sum:", mysum
    !$OMP END MASTER
      


  !$OMP END PARALLEL

end program omp_basics

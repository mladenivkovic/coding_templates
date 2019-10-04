program random

  !==================================================================
  ! A program to demonstrate generating random numbers with Fortran.
  !==================================================================

  use omp_lib
  
  implicit none
  integer :: i, id, nthreads
  logical, dimension(:), allocatable :: seed_set ! for independent multithreading



  write(*,*) "Generate new random numbers"
  call get_new_random_numbers()



  write(*,*)
  write(*,*) "Generate same random numbers"
  call get_same_random_numbers()



  write(*,*)
  write(*,*) "Multithreading test"
  call test_multithreading()




  write(*,*)
  write(*,*) "Multithreading random numbers"
  allocate(seed_set(1:omp_get_max_threads()))
  seed_set = .false.
  !$OMP PARALLEL DEFAULT(PRIVATE) SHARED(seed_set)
    id = omp_get_thread_num()
    nthreads = omp_get_num_threads()

    !$OMP CRITICAL (indep)
      write(*, '(A7,I3,A7)', advance='no') "Thread", id, "got:"
      do i=1, 10
        write(*, '(F10.6)', advance='no') independent_multithreading(id)
      enddo
      write(*,*)
    !$OMP END CRITICAL (indep)
  !$OMP END PARALLEL




contains

  subroutine get_new_random_numbers()
    !------------------------------------------------------------
    ! Get new random number every time you call this routine
    !------------------------------------------------------------
    real :: randomreal
    integer :: i

    do i=1, 10
      call random_number(randomreal)
      write(*,'(F10.6,x)', advance='no') randomreal
    enddo
    write(*,*)

    return

  end subroutine get_new_random_numbers


  subroutine get_same_random_numbers()
    !------------------------------------------------------------
    ! Get new random number every time you call this routine
    !------------------------------------------------------------
    real :: randomreal
    integer :: i

    do i=1, 10
      call random_seed()
      call random_number(randomreal)
      write(*,'(F10.6,x)', advance='no') randomreal
    enddo
    write(*,*)

    return
  end subroutine get_same_random_numbers



  subroutine test_multithreading()
    !------------------------------------------------------
    ! Test how multithreading behaves with randoms.
    ! compile with -fopenmp
    !------------------------------------------------------
    
    implicit none
    integer :: i, j, id, nthreads
    real :: randomreal 

    !$OMP PARALLEL DEFAULT(PRIVATE)
      nthreads=omp_get_num_threads()
      id=omp_get_thread_num()

      !$OMP CRITICAL
      do i = 0, nthreads-1
        if (id == i) then
          ! call random_seed()
          write(*,'(A7,I3,A5)', advance='no') "Thread", id, "got:"
          do j=1, 10
            call random_number(randomreal)
            write(*,'(F10.6)', advance='no') randomreal
          enddo
          write(*,*)
        endif
      enddo
      !$OMP END CRITICAL

    !$OMP END PARALLEL
  end subroutine test_multithreading



  real function independent_multithreading(id)
    !-----------------------------------------------
    ! Generate different seeds for each threads,
    ! but only once per run
    !-----------------------------------------------

    integer, intent(in) :: id   ! thread ID, number of threads
    integer :: n, clock, i
    integer, dimension(:), allocatable :: seed
    real :: temp

    if (.not.seed_set(id+1)) then
      call random_seed(size=n)
      allocate(seed(n))
      call system_clock(count=clock)
      seed = clock + id + 53 * [(n-i, i=1, n)]
      call random_seed(put=seed)
      deallocate(seed)
      seed_set(id+1) = .true.
    endif

    call random_number(temp)
    if (seed_set(id+1)) then
      independent_multithreading = temp
    else
      independent_multithreading = -temp
    endif

  end function independent_multithreading 

end program random

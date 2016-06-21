program timing

! How to time cpu-time with fortran


    implicit none

    real :: t_callcpu, t1, t2, mysum = 0.0
    integer :: someint, someotherint
    integer, parameter :: n = 10000

    call cpu_time(t_callcpu) !Fortran intrinsic
    t1 = t_callcpu
    
    do someint = 1, n
        do someotherint = 0, someint
            mysum = mysum + sqrt(real(someotherint))
            ! sqrt(x) takes only REALS as x. real(INT) transforms an
            ! integer to a real.
        end do
    end do

    call cpu_time(t_callcpu)
    t2 = t_callcpu
    
    write(*, *) "Sum: ", mysum
    write(*, *) "Time: ", t2 - t1, "s"


end program timing

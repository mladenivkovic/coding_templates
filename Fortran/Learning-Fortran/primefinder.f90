program primefinder
    implicit none
    integer :: divisor = 2   
    integer (selected_int_kind(18) ) :: dividend, quotient, rest 


    print*, "Enter integer to analyse"
    read*, dividend
!    print*, counter
!    print*, isprime
!    print*, MOD(isprime, counter)
!    print*, isprime / counter
    do
            quotient = dividend / divisor
            if (divisor > quotient) then !a/b = c; if b > c, stop iterating
                print*, "it's a prime"
                exit
            else !testing if it is a prime
                rest = mod(dividend, divisor)
                    if (rest == 0) then
                        print*, "not a prime number"
                        exit
                    else
                        divisor = divisor + 1
                    end if
            end if
    end do



end program primefinder

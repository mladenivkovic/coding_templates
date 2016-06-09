program simpleArithmetic
!this is a comment.
!hihihi.
!
implicit none
real :: n1, n2, n3, average = 0.0, total=0.0
integer :: n=3

    print *, 'type in 3 numbers'
    print *, 'separated by spaces or commas'
    read *, n1, n2, n3
    total = n1 + n2 + n3
    average = total/n
    print *, 'Total =', total
    print *, 'Average =', average

end program simpleArithmetic

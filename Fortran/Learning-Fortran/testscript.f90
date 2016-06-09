program testscript

    implicit none

!        real :: a
!        real :: b
!        a=1.4
!        b=2.5
!        answer = a**b
!        print *, answer
!	print *, tiny, digits, epsilon, huge
!	print *, maxexponent, minexponent, precision, radix, range
!	print *, range(1234531)
!    Print 1000, x,y,z
!    1000 format (1x,3(f5.2))

    integer, parameter :: ndim = 5
    integer, dimension(1:ndim, 1:ndim):: MYARRAY
    integer, dimension(2:ndim)::myarraynew
    integer :: spalte, zeile

    do spalte=1, 2
        do zeile = 1, ndim
            MYARRAY(zeile,spalte) = zeile*spalte
!            myarraynew(zeile,spalte) = (spalte*zeile) (spalte*zeile*4)
        end do
    end do

    do zeile=1, ndim
        print*, (MYARRAY(zeile, spalte), spalte=1,2)
    end do

    print*, MYARRAY(:, 2)
!    print*, MYARRAY
end program testscript

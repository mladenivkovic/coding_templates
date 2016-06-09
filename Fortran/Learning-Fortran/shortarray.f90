program shortarray

implicit none

integer, parameter :: i32 = selected_int_kind(9)

integer, parameter :: nelements = 10

integer (i32), dimension(1:nelements, 1:nelements, 1:nelements, 1:nelements) :: FourDimArray

integer, dimension(3, 4, 5) :: Testarray

integer :: zeile
integer :: spalte, whatever
integer (i32) :: ArrayElement

do zeile = 0, 3
    do spalte = 1, nelements
        !read *, ArrayElement
        !FourDimArray(spalte, zeile) = ArrayElement
    end do
end do

do zeile = 0, 2
    do spalte = 0, 3
        do whatever = 0, 4
        Testarray(zeile, spalte, whatever) = zeile + spalte + whatever 
        end do
    end do
end do

do zeile = 0, 2
    do spalte = 0, 3
        print*, (Testarray(zeile, spalte, whatever), whatever=0, 4)
    end do
end do
end program

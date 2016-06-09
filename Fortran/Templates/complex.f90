program usecomplex

! How to deal with complex numbers in fortran


    implicit none

    complex :: u, v
    integer :: i=1, j = 2
    
    ! Defining a double precision complex
    integer, parameter :: dp = selected_real_kind(15, 307)
    complex (kind=dp) :: w


    u = (1.0, 2.0)
    v = cmplx(i,j)
    w = (1.0, 2.0)




    write(*, '(A24, 2A9)') "", "real", "complex" 
    write(*, '(A24, 2F9.3)') "u = (1.0, 2.0) :", u
    write(*, '(A24, 2F9.3)') "v = cmplx(i, j) :", v
    write(*, *)
    write(*, '(A24, 2F9.3)') "real(u) :", real(u) ! real part
    write(*, '(A24, 2F9.3)') "aimag(u) :", aimag(u) ! imaginary part 
    write(*, '(A24, 2F9.3)') "abs(u) :", abs(u)
    write(*, '(A24, 2F9.3)') "conjg(u): ", conjg(u) 
    write(*, '(A24, 2F9.3)') "u + v : ", u + v 
    write(*, '(A24, 2F9.3)') "u - v :", u - v 
    write(*, '(A24, 2F9.3)') "u * v :", u * v
    write(*, '(A24, 2F9.3)') "u / v :", u / v
    write(*, *)
    write(*, *) "Comparing precision"
    write(*, *) "u =", u
    write(*, *) "w =", w



end program usecomplex

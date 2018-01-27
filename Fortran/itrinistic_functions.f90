program itrinistics

  !=============================================================
  !this program demonstrates some fortran itrinistric functions.
  !=============================================================

  implicit none
  integer:: someint
  real:: somereal

  write(*,'(A40)') "---- FORTRAN ITRINISTIC FUNCTIONS ----"
  write(*,*)
  write(*,'(A18,F10.5)') "abs(-1.0)", abs(-1.0)
  write(*,'(A18,F10.5)') "sqrt(25.0)", sqrt(25.0)
  write(*,'(A18,F10.5,A22)') "sin(3.141595256)", sin(3.141595256), "in radiants"
  write(*,'(A18,F10.5,A22)') "cos(3.141595256)", cos(3.141595256), "in radiants"
  write(*,'(A18,F10.5,A22)') "tan(3.141595256)", tan(3.141595256), "in radiants"
  write(*,'(A18,F10.5,A22)') "asin(1.0)", asin(1.0), "in radiants"
  write(*,'(A18,F10.5,A22)') "acos(0.0)", acos(0.0), "in radiants"
  write(*,'(A18,F10.5,A22)') "atan(1.0)", atan(1.0), "in radiants"
  write(*,'(A18,F10.3)') "exp(9.0)", exp(9.0)
  write(*,'(A18,F10.5,A28)') "log(8103.084)", log(8103.084), "natural logarithm"
  write(*,*)
  write(*,'(A18,I10)') "max(1,2)", max(1,2)
  write(*,'(A18,F10.5)') "max(1.0, 2.0)", max(1.0, 2.0)
  write(*,'(A18,I10)') "min(1,2)", min(1,2)
  write(*,'(A18,F10.5)') "min(1.0, 2.0)", min (1.0, 2.0)
  write(*,'(A18,I10)') "mod(5,2)", mod(5,2)
  write(*,'(A18,F10.5)') "mod(5.0, 2.0)", mod(5.0, 2.0)
  write(*,'(A18,F10.5)') "mod(5.5, 2.2)",mod(5.5, 2.2) 
  write(*,*)









end program itrinistics

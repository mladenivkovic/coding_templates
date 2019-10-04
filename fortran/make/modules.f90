!==============================================================
! This is a program to demonstrate how modules work.
! Modules have to be compiled before the main program
! is compiled, otherwise they won't be found.
! Gfortran is looking for modulename.mod.
!
! compile with gfortran precision_specification.f90 physical_constants.f90 simple_math_module.f90 modules.f90
! or use makefile
! 
! Implementations can be found:
! Variables: precision_specification
! subroutine: physical_constants
! function: simple_math_module
! type: simple_math_module
! private, public and protected variables: physical_constants 
!==============================================================
program main_program

    use precision_specification, only: qp ! Specify like this what you need if you don't want to use the whole module.
    use physical_constants
    use simple_math_module

    implicit none
    real (qp) :: mu_null = 4 * pi * 1E-7 ! from simple_math_module and precision_specification
    type (rectangle) :: myrect            ! from simple_math_module
    ! PARENTHESES NEEDED AROUND TYPE DEFINED IN A MODULE


    write(*, *)
    write(*, *) speedoflight            ! from physical_constants 
    call showconstants()                ! from physical_constants
    write(*, *) "area", circlearea(1.0) ! from simple_math_module


    ! defining the values of rectangle type.
    ! see simple_math_module for more.
    myrect%bottom_y = 0.0
    myrect%bottom_x = 0.0
    myrect%top_x = 4.0
    myrect%top_y = 2.0

    write(*, *) "Rectangle area: ", rectanglearea(myrect)
end program main_program 


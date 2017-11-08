module simple_math_module

    use precision_specification
    use physical_constants

    implicit none
    real (dp), parameter :: pi = 3.14159265359
    real (dp), parameter :: euler = 2.71828182845

    type rectangle 
        ! define a rectangle via top-right corner and
        ! bottom-left corner.
        real :: bottom_x = 0.0
        real :: bottom_y = 0.0
        real :: top_x = 0.0
        real :: top_y = 0.0
    end type rectangle


contains



    real function circlearea(radius)
        implicit none
    ! returns the area of a circle.
    ! The argument is radius in whatever unit.
        real, intent(in) :: radius
        
        circlearea = pi * radius * radius

    end function circlearea




    real function rectanglearea(rect)
        ! a function to compute the area of a rectangle type.
        implicit none
        type (rectangle), intent(in) :: rect 
        ! WATCH OUT! BRACES AROUND TYPE NAME NEEDED!
        real :: xborder, yborder

        xborder = abs(rect%top_x - rect%bottom_x)
        yborder = abs(rect%top_y - rect%bottom_y)
        rectanglearea = xborder * yborder
        end function rectanglearea






end module simple_math_module

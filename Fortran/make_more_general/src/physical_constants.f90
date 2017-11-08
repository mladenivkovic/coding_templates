module physical_constants
    ! use another module:
    use precision_specification

    implicit none
    
    real (dp), parameter :: speedoflight = 299792458
    real (dp), parameter :: epsilonnull = 8.854187871E-12
    real (qp), public, protected :: planck_h = 6.626070040E-34
    ! public: every program/module that uses this module can access this variable.
    ! protected: this variable cannot be changed anywhere else but here.
    real (qp), private :: planck_eV = 4.135667662E-15
    !private: only this module has access to this var.


contains
    subroutine showconstants()

    write(*, *)
    write(*, *) "---------------------------------"
    write(*, *)
    write(*, *) "Printing constants saved in the physical_constant module"
    write(*, *)
    write(*, '(A20, ES20.5E3)') "speedoflight", speedoflight
    write(*, '(A20, ES20.5E3)') "epsilonnull", epsilonnull 




    write(*, *)
    write(*, *) "---------------------------------"
    write(*, *)

    end subroutine showconstants


end module physical_constants


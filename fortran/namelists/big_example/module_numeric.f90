module numeric_
  implicit none
  ! define several KIND parameters for use everywhere

  ! 32bit integer
  integer, parameter :: INT_   = selected_int_kind(R=9)

  ! 64bit real
  integer, parameter :: REAL_  = selected_real_kind(P=13,R=300)

  ! 64+64bit complex
  integer, parameter :: COMPLEX_  = selected_real_kind(P=13,R=300)

!
! _X(create an overloaded 'generic' interface which depends)
! _X(on the subroutine 'signature' based on data type,)
! _X(kind number, and rank (TKR))
  interface show_kind
    module procedure show_int_, show_real_, show_complex_
  end interface
!
! _X(force the use of the generic interface by excluding)
! _X(any public access to specific routines)
  private show_int_, show_real_, show_complex_

contains

subroutine show_int_(x)
  implicit none
  integer(kind=INT_), intent(in) :: x ! kind must not be variable

  write(6,'(/1x,"***INT_      KIND = ", i6, 10x,"requested = ", i6)') &
    kind(x), selected_int_kind(R=9)
  write(6,'(1x,"RADIX             = ", i6)') radix(x)
  write(6,'(1x,"DIGITS            = ", i6)') digits(x)
  write(6,'(1x,"RANGE             = ", i6)') range(x)
  write(6,'(1x,"HUGE              = ")',advance='NO')
  write(6, *) huge(x)
  write(6,'(1x,"value             = ")',advance='NO')
  write(6, *) x
end subroutine show_int_

subroutine show_real_(x)
  implicit none
  real(kind=REAL_), intent(in) :: x ! kind must not be variable

  write(6,'(/1x,"***REAL_     KIND = ", i6, 10x,"requested = ", i6)') &
    kind(x), selected_real_kind(P=13,R=300)
  write(6,'(1x,"PRECISION         = ", i6)') precision(x)
  write(6,'(1x,"MAXEXPONENT       = ", i6)') maxexponent(x)
  write(6,'(1x,"MINEXPONENT       = ", i6)') minexponent(x)
  write(6,'(1x,"RADIX             = ", i6)') radix(x)
  write(6,'(1x,"DIGITS            = ", i6)') digits(x)
  write(6,'(1x,"EPSILON           = ")',advance='NO')
  write(6, *) epsilon(x)
  write(6,'(1x,"value             = ")',advance='NO')
  write(6, *) x
end subroutine show_real_

subroutine show_complex_(x)
  implicit none
  complex(kind=COMPLEX_), intent(in) :: x ! kind must not be variable

  write(6,'(/1x,"***COMPLEX_  KIND = ", i6, 10x,"requested = ", i6)') &
    kind(x), selected_real_kind(P=13,R=300)
  write(6,'(1x,"PRECISION         = ", i6)') precision(real(x))
  write(6,'(1x,"MAXEXPONENT       = ", i6)') maxexponent(real(x))
  write(6,'(1x,"MINEXPONENT       = ", i6)') minexponent(real(x))
  write(6,'(1x,"RADIX             = ", i6)') radix(real(x))
  write(6,'(1x,"DIGITS            = ", i6)') digits(real(x))
  write(6,'(1x,"EPSILON           = ")',advance='NO')
  write(6, *) epsilon(real(x))
  write(6,'(1x,"value             = ")',advance='NO')
  write(6, *) x
end subroutine show_complex_

end module numeric_



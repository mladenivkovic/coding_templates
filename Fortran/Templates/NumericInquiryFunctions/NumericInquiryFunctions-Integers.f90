!http://www.fortranplus.co.uk/book3_examples.html

program ch0508a

  implicit none
  
  ! example of the use of the kind function
  ! and the numeric inquiry functions
  ! for integer kind types
  
  ! 8 bit            -128  to
  ! 127      10**2
  
  ! 16 bit          -32768 to
  ! 32767     10**4
  
  ! 32 bit     -2147483648 to
  ! 2147483647     10**9
  
  ! 64 bit
  ! -9223372036854775808 to
  ! 9223372036854775807    10**18
  
    integer :: i
    integer, parameter :: i8 = selected_int_kind(2)
    integer, parameter :: i16 = selected_int_kind(4)
    integer, parameter :: i32 = selected_int_kind(9)
    integer, parameter :: i64 = selected_int_kind(18)
    integer (i8) :: i1
    integer (i16) :: i2
    integer (i32) :: i3
    integer (i64) :: i4
        

    print *, ' '
    print *, ' integer kind support'
    print *, 'bit         kind    huge'
    print *, ' '
    print *, '32', kind(i), ' ', huge(i), '(not specified integer)'
    print *, ' '
    print *, '8 ', kind(i1), ' ', huge(i1)
    print *, '16',kind(i2), ' ', huge(i2)
    print *, '32', kind(i3), ' ', huge(i3)
    print *, '64', kind(i4), ' ', huge(i4)
    print *, ' '
end program ch0508a


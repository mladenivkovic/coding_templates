program pointers
    !------------------------------------------------------------------------
    ! A pointer is a variable that has the pointer attribute.
    ! A pointer is associated with a target by allocation or 
    ! pointer assignment. 
    ! Association:
    ! - allocate statement referencing the pointer
    ! - pointer-assigned to a target that is associated or is 
    !   specified with the target attribute and, if allocatable, 
    !   is currently allocated.
    !
    ! A pointer may have a pointer association status
    ! - associated
    ! - disassociated
    ! - undefined
    ! The association status may change during execution of a program.
    ! A pointer can't be referenced nor defined until it is associated.
    ! A pointer is disassociated following execution of a deallocate or
    ! nullify statement.
    !------------------------------------------------------------------------


    implicit none
    integer, pointer :: a => null(), b => null() ! a and b can point to integer values. => null() : set the status of a and b to disassociated.
    ! This way, no space is set aside for variables a and b.
    integer, target :: c ! c can be targeted by pointers.
    integer :: d

    write(*, *) "associated(a): ", associated(a)
    c = 1
    a => c
    c = 2
    b => c
    d = a + b
    write(*, *)  "associated(a): ", associated(a)
    write(*, *) "result: ", a, b, c, d
    a => null()
    b => null()  
    write(*, *) "associated(a): ", associated(a)

    ! another way to associate pointers. They still remain pointers though!

    write(*, *)
    write(*, *) "Allocate"
    write(*, *) "associated(a): ", associated(a)
    
    allocate(a)
    a = 1
    write(*, *) "a =", a
    write(*, *) "loc(a)", loc(a)
    deallocate(a)
    write(*, *) "associated(a): ", associated(a)
    write(*, *) "loc(a)", loc(a)
end program pointers

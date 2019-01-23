program types

    !====================================
    ! Creating and using your own types.
    !====================================

    !------------------------------------
    ! Syntax:
    !
    !--- Defining types:
    !
    ! type typename
    !   data type :: component_name
    !   etc
    ! end type typename
    !
    ! ---initiating types:
    !
    ! type (typename) :: variablename
    !
    ! ---referring to components:
    ! variablename%component_name
    !
    !------------------------------------

    implicit none

    type date

        integer :: day = 1
        integer :: month = 1
        integer :: year = 2000
    ! initial values can be defined

    end type date

    type (date) :: d

    write(*, '(I3, I3, I5)') d%day, d%month, d%year

    d%day = 19
    d%month = 04
    d%year = 2016
    write(*, '(I3, I3, I5)') d%day, d%month, d%year

end program types

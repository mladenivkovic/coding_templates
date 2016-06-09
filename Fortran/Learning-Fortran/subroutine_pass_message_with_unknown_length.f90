program mp

    implicit none

    write(*, *) "call message()"
    call message()
    write(*, *)
    write(*, *) "call message('Hello)"
    call message('Hello')


contains
    subroutine message(somemessage)
        implicit none
        character(len=*), intent(in), optional :: somemessage
        
        if (present(somemessage)) then
            write(*, *) "got the message"
            write(*, *) "Your message was:"
            write(*, *) somemessage
        else
            write(*, *) "Nothing to tell you."
        end if
    end subroutine message
end program mp

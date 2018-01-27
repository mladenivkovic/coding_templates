program linkedlist

    !==============================================================
    ! Singly linked list
    ! Concept: Consist of a sequence of boxes with compartments
    ! Simplest case: First compartment holds a data item, second
    ! Compartment contains directions to the next box.
    !==============================================================

    implicit none

    type link
        character(len=1) :: c
        type (link), pointer :: next => null()
    end type link

    type (link), pointer :: root, current
    character (len=:), allocatable :: word
    character (len=80) :: fname="inputfiles/loremipsum.txt"
    integer :: io_stat_number = 0
    integer :: n, i = 0
    

    ! Singly linked list:
    ! Reads an arbitrary amount of text from a file
    ! -eof terminates it

    ! open file
    open(unit=1, file=fname, status='old')
    


    !reads one char at a time until EOF

    !reading in the first char, saved in the root:
    allocate(root)
    read(1, fmt='(A)', advance='no', iostat=io_stat_number) root%c
    if (io_stat_number/=-1) then
        allocate (root%next)
        i = i + 1
    end if
    
    current => root ! point the second char to the first
    ! The link type current itself points to the previous element.
    ! current%next points to the next element.
    ! The current%next is allocated at whatever place in memory.
    ! Then the following current will point to the place where
    ! the previous current%next was allocated to.    

    do while (associated(current%next))
        current => current%next
        read(1, fmt='(A)', advance='no', iostat=io_stat_number) current%c ! read in next char to current
        if (io_stat_number/=-1) then
            allocate(current%next)
            !write(*, *) "i =", i, "loc =", loc(current%next), "loc cur = ", loc(current)
            i = i + 1
        end if
    end do

    write(*, *) i, "characters read"
   

    ! Now connect all of them again.
    n = i
    allocate(character(len=n) :: word)
    i = 0
    current => root
   
    do while (associated(current%next))
        i = i + 1
        word(i:i) = current%c
        current => current%next
    end do

    print *, 'text read in was:'
    print *, word

    close(1)
end program linkedlist

module link_module

  type link
    character (len=1) :: c
    type (link), pointer :: next => null()
  end type link

end module link_module

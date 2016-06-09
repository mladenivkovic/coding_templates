	program simpleIO
!
!reads in name, prints it
!
		implicit none 
		character *20 : : first_name
		print *, 'type in your first name'
		print *,'up to 20 characters'
		read *, first_name
		print *, first_name
	end program simpleIO

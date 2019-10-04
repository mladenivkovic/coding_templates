! Compiling notice:
! First compile module_numeric.f90 with -c flag
! Then compile module_datastuff.f90 together with namelist.f90

program tnmlist
	use datastuff
	implicit none
  
  ! demonstrate the f90 standard namelist
  !
  ! Syntax: namelist /blockinthenamelist/ variables, that, will, be, imported.
  ! Here: Variables declared in module datastuff.

	namelist /tdata/ treal, tinteger, tcomplex, tchar, tbool
	namelist /adata/ areal, ainteger, acomplex, achar, abool
	namelist /xdata/ xreal, xinteger, xcomplex

	namelist /ttdata/ ttreal, ttinteger, ttcomplex, ttchar, ttbool
	namelist /aadata/ aareal, aainteger, aacomplex, aachar, aabool
	namelist /xxdata/ xxreal, xxinteger, xxcomplex

  ! _X(the OPEN statement defines many of the)
  ! _X(NAMELIST characteristics)
	! need the delim, else some implementations will not surround
	! character strings with delimiters
	! recl limits the I/O to 80 character lines
	! open(6, recl=80, delim='APOSTROPHE')
  open(8,file="input.nml", status='OLD', recl=80, delim='APOSTROPHE')

  ! _X(NAMELIST output varies with compilers)
	! how NAMELIST data displays on your system
    write(6, *)
    write(6, *) "WRITING NAMELIST"
    write(6, *) "----------------"
    write(6, *)
	write(6,nml=tdata)
	write(6,nml=adata)
	write(6,nml=xdata)
    write(6, *)
	write(6,*) "-----------------"
    write(6, *)
	write(6,*) 'Read first batch'
	call clearstuff()
	read(8,nml=ttdata)
	read(8,nml=aadata)
	read(8,nml=xxdata)
  ! _X(using the KEYWORD feature of routine)
  ! _X(calls adds clarity)
	call diffstuff(fullout=.TRUE.)

	write(6,*) 'Read second batch'
	call clearstuff()
	read(8,nml=ttdata)
	read(8,nml=aadata)
	read(8,nml=xxdata)
	call diffstuff(fullout=.TRUE.)

end program tnmlist

module datastuff

! This module contains:
! - variable declarations. For each type, one variable is assigned 
!   and one is not. (Will be assigned via the namelist.)
!
! - subroutine clearstuff():
!   set all variables read in from namefile to 0/empty
!
! - subroutine diffstuff(fullout):
!   compare read in values with assigned values.
!   if fullout = true, print difference as well.


	use numeric_ ! used for precision specification
	implicit none

! _X(for comparison purposes)
	real ::				ttreal, 	treal = 1.0
	integer ::			ttinteger,	tinteger = 2
	complex ::			ttcomplex,	tcomplex = (3.0, 4.0)
	character *10 ::	ttchar,		tchar = 'namelist'
	logical ::			ttbool,		tbool = .TRUE.

	real, dimension(4) ::		aareal,	areal = (/ 1.0, 1.0, 2.0, 3.0 /)
	integer, dimension(4) ::	aainteger, ainteger = (/ 2, 2, 3, 4 /)
	complex, dimension(4) ::	aacomplex, 	acomplex = (/ (3.0, 4.0), (3.0, 4.0), (5.0, 6.0), (7.0, 7.0) /)
	character*10, dimension(4) ::	aachar, achar = (/ 'namelist  ', 'namelist  ',	'array     ', ' the lot  ' /)
	logical, dimension(4) ::	aabool, abool = (/ .TRUE., .TRUE., .FALSE., .FALSE. /)

!   precisions come from module numeric_
	real(kind=REAL_) ::		    xxreal,		xreal = 1.0_REAL_
	integer(kind=INT_) ::		xxinteger,	xinteger = 2_INT_
	complex(kind=COMPLEX_) ::	xxcomplex,  xcomplex = (3.0_COMPLEX_, 4.0_COMPLEX_)


contains
	subroutine clearstuff()
		ttreal =	0.0
		ttinteger =	0
		ttcomplex =	(0.0,0.0)
		ttchar =	''
		ttbool =	.FALSE.
		aareal(1:4) =	0.0
		aainteger(1:4) =0
		aacomplex(1:4) =(0.0,0.0)
		aachar(1:4) =	''
		aabool(1:4) =	.FALSE.
		xxreal =	0.0_REAL_
		xxinteger =	0_INT_
		xxcomplex =	(0.0_COMPLEX_,0.0_COMPLEX_)
	end subroutine clearstuff

	subroutine diffstuff(fullout)
	implicit none
	logical :: fullout
	integer :: numbad,i
! _X(Compare the input data to the expected value)
! if fullout = true, print full output, including the difference.

	numbad = 0
	if (ttreal .ne. treal) then
		numbad = numbad + 1
		if (fullout) write(6,*) 'treal diff = ', ttreal - treal
	endif
	if (ttinteger .ne. tinteger) then
		numbad = numbad + 1
		if (fullout) write(6,*) 'tinteger diff = ', ttinteger - tinteger
	endif
	if (ttcomplex .ne. tcomplex) then
		numbad = numbad + 1
		if (fullout) write(6,*) 'tcomplex diff = ', ttcomplex - tcomplex
	endif
	if (ttchar .ne. tchar) then
		numbad = numbad + 1
		if (fullout) write(6,*) 'tchar diff = ', ttchar, tchar
	endif
	if (ttbool .neqv. tbool) then
		numbad = numbad + 1
		if (fullout) write(6,*) 'tbool diff = ', ttbool, tbool
	endif

	do i = 1,4
		if (aareal(i) .ne. areal(i)) then
			numbad = numbad + 1
	if (fullout) write(6,*) 'areal diff = ', aareal(i) - areal(i)
		endif
		if (aainteger(i) .ne. ainteger(i)) then
			numbad = numbad + 1
	if (fullout) write(6,*) 'ainteger diff = ', aainteger(i) - ainteger(i)
		endif
		if (aacomplex(i) .ne. acomplex(i)) then
			numbad = numbad + 1
	if (fullout) write(6,*) 'acomplex diff = ', aacomplex(i) - acomplex(i)
		endif
		if (aachar(i) .ne. achar(i)) then
			numbad = numbad + 1
	if (fullout) write(6,*) 'achar diff = ', aachar(i), achar(i)
		endif
		if (aabool(i) .neqv. abool(i)) then
			numbad = numbad + 1
	if (fullout) write(6,*) 'abool diff = ', aabool(i), abool(i)
		endif
	enddo

	if (xxreal .ne. xreal) then
		numbad = numbad + 1
		if (fullout) write(6,*) 'xreal diff = ', xxreal - xreal
	endif
	if (xxinteger .ne. xinteger) then
		numbad = numbad + 1
		if (fullout) write(6,*) 'xinteger diff = ', xxinteger - xinteger
	endif
	if (xxcomplex .ne. xcomplex) then
		numbad = numbad + 1
		if (fullout) write(6,*) 'xcomplex diff = ', xxcomplex - xcomplex
	endif

	if (numbad .ne. 0) then
		write(6,'(1x,"found ",i3," differences")') numbad
	else
		write(6,'(1x,"found "," no"," differences")')
	endif

	end subroutine diffstuff

end module datastuff




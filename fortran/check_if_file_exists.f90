program check_if_file_exists

  !===============================
  ! Check whether a file exists.
  !===============================


  implicit none


  logical:: file_exists = .false.
  logical:: dir_exists = .false.

  character(len=10) :: filename = 'loops.f90'
  character(len=10) :: dirname = 'MPI/'

  inquire(file=TRIM(filename), exist=file_exists)

  inquire(file=TRIM(dirname), exist=dir_exists)  


  write(*,*) "File ", filename, "exists?", file_exists
  write(*,*) "Directory ", dirname, "exists?", dir_exists
 
 

end program check_if_file_exists

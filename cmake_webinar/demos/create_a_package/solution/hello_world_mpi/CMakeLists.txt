cmake_minimum_required( VERSION 3.2 )

project( hello_world LANGUAGES Fortran )

find_package( MPI REQUIRED )
add_executable( hello )
target_sources( hello PRIVATE hello.f90 )
target_link_libraries( hello PUBLIC MPI::MPI_Fortran )


install( TARGETS hello )

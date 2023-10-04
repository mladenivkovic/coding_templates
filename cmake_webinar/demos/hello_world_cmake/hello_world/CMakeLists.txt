cmake_minimum_required( VERSION 3.2 )

project( hello_world LANGUAGES Fortran )

add_executable( hello )
target_sources( hello PRIVATE hello.f90 )

install( TARGETS hello )

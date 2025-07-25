# initialise autoconf: package name, version, contact for bug reports
AC_INIT([amhello], [1.0], [bug-automake@gnu.org])
# initialise automake. Give it a list of options. NOTE: THESE ARE AUTOMAKE ERRORS, NOT COMPILER ERRORS.
# 'foreign': means this package will not follow GNU standards.
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
# search for a c compiler and define CC variable
AC_PROG_CC 
# make config.h and store #defines in there
AC_CONFIG_HEADERS([config.h])
# list of files `configure` should create from their *.in templates.
AC_CONFIG_FILES([
 Makefile
 src/Makefile
])
AC_OUTPUT

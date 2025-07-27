# initialise autoconf: package name, version, contact for bug reports
AC_INIT([c_with_cuda_autotools], [1.0], [mivkov@protonmail.com])

# initialise automake. Give it a list of options. 
# NOTE: THESE ARE AUTOMAKE ERRORS, NOT COMPILER ERRORS.
# 'foreign': means this package will not follow GNU standards.
AM_INIT_AUTOMAKE([-Wall -Werror foreign subdir-objects])

# search for a c compiler and define CC variable
AC_PROG_CC 

# list of files `configure` should create from their *.in templates.
AC_CONFIG_FILES([
 Makefile
 src/Makefile
 src/cuda/Makefile
])


# Check for CUDA
have_cuda="no"
AC_ARG_WITH([cuda],
    [AS_HELP_STRING([--with-cuda=PATH],
       [root directory where CUDA is installed @<:@yes/no@:>@]
    )],
    [],
    [with_cuda="no"]
)

if test "x$with_cuda" != "xno"; then
  # grab the flags we want from provided path
  CUDA_CFLAGS="$CUDA_CFLAGS -I$with_cuda/include"
  CUDA_CXXFLAGS="$CUDA_CXXFLAGS -I$with_cuda/include"
  CUDA_LIBS="-L$with_cuda/lib -L$with_cuda/lib64 -lcudart"
  NVCC="$with_cuda/bin/nvcc"
  NVCC_FLAGS="$NVCC_FLAGS"
  NVCC_LDFLAGS="$NVCC_LDFLAGS"
  NVCC_LIBS="$NVCC_LIBS"
  have_cuda="yes"
else
  # try finding nvcc and go from there.
  AC_PATH_PROG([NVCC],[nvcc])
  echo "Found nvcc = ${NVCC}"
  if test -n "$NVCC"; then
    CUDA_HOME="`dirname $NVCC`/.."
    CUDA_CFLAGS="$CUDA_CFLAGS -I${CUDA_HOME}/include"
    CUDA_CXXFLAGS="$CUDA_CXXFLAGS -I${CUDA_HOME}/include"
    CUDA_LIBS="-L${CUDA_HOME}/lib -L${CUDA_HOME}/lib64 -lcudart"
    NVCC_FLAGS="$NVCC_FLAGS"
    NVCC_LDFLAGS="$NVCC_LDFLAGS"
    NVCC_LIBS="$NVCC_LIBS"
    have_cuda="yes"
  fi
fi

if test "x$have_cuda" != "xyes"; then
 AC_MSG_ERROR("Couldn't find a cuda installation and/or nvcc. Specify with --with-cuda=/path/")
fi


AC_SUBST(CUDA_CFLAGS)
AC_SUBST(CUDA_CXXFLAGS)
AC_SUBST(CUDA_LIBS)
AC_SUBST(NVCC)
AC_SUBST(NVCC_FLAGS)
AC_SUBST(NVCC_LDFLAGS)
AC_SUBST(NVCC_LIBS)

AM_PROG_AR # needed when using non-POSIX archiver. Call before LT_INIT
LT_INIT([enable-static, disable-shared])
AC_CONFIG_MACRO_DIRS([m4]) # recommended when libtool is in use

AC_OUTPUT


# List required headers
noinst_HEADERS = cfunc1.h cfunc2.h clib2func.h

# Specify extra flags in every Makefile.am anew. They aren't inherited.
AM_CFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src -I$(top_srcdir)/src/cuda 

# we want to build libraries, but not install them.
# so use "noinst" prefix.
# noinst_LIBRARIES = libmylib1.a libmylib2.a
# libmylib1_a_SOURCES = cfunc1.c cfunc2.c
# libmylib1_a_CFLAGS = $(AM_CFLAGS)
# libmylib1_a_LIBADD = cuda/libmycudalib.a

# libmylib2_a_SOURCES = clib2func.c
# libmylib2_a_CFLAGS = $(AM_CFLAGS)
# $(CUDA_CFLAGS) $(CUDA_INCLUDES)

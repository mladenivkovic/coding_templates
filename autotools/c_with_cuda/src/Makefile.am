# List required headers
noinst_HEADERS = cfunc1.h cfunc2.h clib2func.h

SUBDIRS = cuda

# Specify extra flags in every Makefile.am anew. They aren't inherited.
AM_CFLAGS = -I$(top_srcdir) -I$(top_srcdir)/src -I$(top_srcdir)/src/cuda 

# we want to build libraries, but not install them.
# so use "noinst" prefix.
noinst_LTLIBRARIES = libmylib1.la libmylib2.la

# specifications for libmylib1.la
libmylib1_la_SOURCES = cfunc1.c cfunc2.c
libmylib1_la_CFLAGS = $(AM_CFLAGS)
libmylib1_la_LIBADD = cuda/libmycudalib.a 

# specifications for libmylib2.la
libmylib2_la_SOURCES = clib2func.c
libmylib2_la_CFLAGS = $(AM_CFLAGS)
# $(CUDA_CFLAGS) $(CUDA_INCLUDES)

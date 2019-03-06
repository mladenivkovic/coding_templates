F90 = gfortran
FFLAGS = -Wall -fbacktrace -g
LIBS = -I$(HDF5_ROOT)/include -L$(HDF5_ROOT)/lib -lhdf5_fortran



all: simple_hdf5 append_datasets concatenate_arrays groups attributes

%: %.f90
	$(F90) $(FFLAGS) $^ -o $@ $(LIBS)
clean:
	rm simple_hdf5 append_datasets *.h5

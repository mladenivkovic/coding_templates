CXX=icpx


OPTFLAGS = 	-O2 -g 
WFLAGS= 	-Wall -Wextra -pedantic -Warray-bounds
STDFLAGS= 	-std=c++20


# CXXFLAGS=
CXXFLAGS=   $(OPTFLAGS) $(WFLAGS) $(STDFLAGS)

LIBS=-ltbb
LDFLAGS=-L/$(INTEL_ONEAPI_TBB_ROOT)/lib
INCLUDE=-I/$(INTEL_ONEAPI_TBB_ROOT)/include


simple_loop.o: simple_loop.cpp ../applications/timer/timer.h
	$(CXX) $(CXXFLAGS) $< -o $@ $(INCLUDE) $(LDFLAGS) $(LIBS)



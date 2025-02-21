# Select compiler.
# ------------------------------------------------------------
CXX=icpx

OPTFLAGS = 	-O3
WFLAGS= 	-Wall -Wextra -pedantic -Warray-bounds # -fsanitize=address
STDFLAGS= 	-std=c++20

# OPTFLAGS = 	-O0 -g
# OPTFLAGS = 	-O3
# WFLAGS= 	-Wall -Wextra -pedantic -Warray-bounds
# STDFLAGS= 	-std=c++20

CXXFLAGS=   $(OPTFLAGS) $(WFLAGS) $(STDFLAGS)
# ------------------------------------------------------------
# CXX=g++
#
# OPTFLAGS = 	-O3
# WFLAGS= 	-Wall -Wextra -pedantic -Warray-bounds
# STDFLAGS= 	-std=c++20
#
# CXXFLAGS=   $(OPTFLAGS) $(WFLAGS) $(STDFLAGS)
# ------------------------------------------------------------


array_speed.o: array_speed.cpp timer.h
	$(CXX) $(CXXFLAGS) $< -o $@


.PHONY: clean # keeps makefile from doing something with a file named clean

clean:
	rm -f *.o

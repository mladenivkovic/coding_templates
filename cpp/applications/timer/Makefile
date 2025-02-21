CXX=icpx


OPTFLAGS = 	-O2 -g 
WFLAGS= 	-Wall -Wextra -pedantic -Warray-bounds
STDFLAGS= 	-stdlib=libc++ -std=c++20 

CXXFLAGS=   $(OPTFLAGS) $(WFLAGS) $(STDFLAGS)



timer_example.o: timer_example.cpp timer.h
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f *.o

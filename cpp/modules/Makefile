default:main.o


my_module.o: my_module.cxx
	g++ my_module.cxx -c my_module.o -std=c++20 -Wall -fmodules-ts


main.o: main.cpp my_module.o
	g++ main.cpp -o main.o my_module.o -std=c++20 -Wall  -fmodules-ts

clean:
	rm -f *.o

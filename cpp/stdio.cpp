#include <iostream> // IO library
#include <string>   // string type

// using namespace std; // skip this for now to explicitly trace where you get things from

int main(){


  // Using the iostream
  int i = -1;
  std::cout << "\nEnter some integer: ";
  std::cin >> i;
  // NOTE: `std::cin >> int i` works (compiles) too, but `i` will not be in scope in the following lines.
  std::cout << "\n you gave me " << i << std::endl;

  std::cerr << "\aThis is an error written to stderr." << std::endl;
  // check e.g. by redirecting stderr to /dev/null : `./stdio.o 2> /dev/null `



  /* TODO: formatting strings etc */

  return 0;
}



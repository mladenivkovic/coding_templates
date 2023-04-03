#include <iostream> // IO library

#include "namespaces_header.h"


// using namespace std; // skip this for now to explicitly trace where you get things from

// NOTE: the "::" in e.g. "std::cout" is the 'scope resolution operator'


// TODO: all of this


// Define a global variable to provoke name clashes.
int global_nameclash_var = 15;

// Define some name space to provoke name clashes.
namespace some_namespace_with_nameclash {
  int global_nameclash_var = -23;
}


namespace some_namespace_split_among_multiple_files {
  // define part of the namespace here in this file,
  // and part of it in the namespace_header.h file
  int var1 = 10;
}


// Use/ "import" name spaces.
using namespace some_namespace_with_nameclash;

int main(){

  // This fails due to the name clash in the namespaces. gcc won't compile it and throws an error.
  // std::cout << "Global nameclash var: " << global_nameclash_var;

  std::cout << "Global nameclash var: " << some_namespace_with_nameclash::global_nameclash_var << std::endl;
  std::cout << "Global nameclash var: " << ::global_nameclash_var << std::endl; // this uses the global namespace.


  std::cout << "Split namespace var1: " << some_namespace_split_among_multiple_files::var1 << std::endl;
  std::cout << "Split namespace var2: " << some_namespace_split_among_multiple_files::var2 << std::endl;

  return 0;
}



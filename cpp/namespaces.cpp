#include <iostream> // IO library
#include <string>   // string type

// using namespace std; // skip this for now to explicitly trace where you get things from

// NOTE: the "::" in e.g. "std::cout" is the 'scope resolution operator'


// TODO: all of this


// Define a global variable to provoke name clashes.
int global_nameclash_var = 15;

// Define some name space to provoke name clashes.
namespace some_namespace_with_nameclash {
  int global_nameclash_var = -23;
}


// Use/ "import" name spaces.
using namespace some_namespace_with_nameclash;

int main(){


  // This fails due to the name clash in the namespaces. gcc won't compile it and throws an error.
  // std::cout << "Global nameclash var: " << global_nameclash_var;

  std::cout << "Global nameclash var: " << some_namespace_with_nameclash::global_nameclash_var << std::endl;
  std::cout << "Global nameclash var: " << ::global_nameclash_var << std::endl; // this uses the global namespace.
  return 0;
}



#include <iostream>

#include "header_for_initializations.h"
#include "print_var_and_name.h" // DUMP() macro

// Initialize static vars here now.
double myClass::static_var = 4.2345;

int main() {

  std::cout << "Hello World!" << std::endl;

  myClass obj = myClass();

  DUMP(obj.static_var);
  DUMP(obj.s);

  obj.print_sarr();

  return 0;
}

// using functions as arguments for other functions

// #include <functional> // needed for std::function
#include <iostream>

#include "PeanoPart.h"
#include "templated_functions_as_function_arguments.h"

int main(void) {

  hydroPartSet myPartSet = generateDummyPartSet(10);

  myPartSet.sanityCheck("Initialization");

  call_function_with_one_templated_parameter(myPartSet, &workOnPart);

  // Sanity check
  // for (auto p : myPartSet){
  //   std::cout << p->getPartID() << " " << *(p->getX().begin()) << "\n";
  // }

  return 0;
}

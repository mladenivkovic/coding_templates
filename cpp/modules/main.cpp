#include <iostream>

import my_module;

int main(void) {

  int result = add(3, 6);
  result = multiply(2, 4);

  std::cout << "Hello world! Result=" << result << "\n";

  return 0;
}

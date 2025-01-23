#include <iostream>
#include <limits>

// Prints out some numerical limits.

int main(void) {

  std::cout << "Bool      min: " << std::numeric_limits<bool>::min()
            << std::endl;
  std::cout << "Bool      max: " << std::numeric_limits<bool>::max()
            << std::endl;
  std::cout << std::endl;
  std::cout << "Int       min: " << std::numeric_limits<int>::min()
            << std::endl;
  std::cout << "Int       max: " << std::numeric_limits<int>::max()
            << std::endl;
  std::cout << std::endl;
  std::cout << "Uint      min: " << std::numeric_limits<unsigned int>::min()
            << std::endl;
  std::cout << "Uint      max: " << std::numeric_limits<unsigned int>::max()
            << std::endl;
  std::cout << std::endl;
  std::cout << "Long      min: " << std::numeric_limits<long>::min()
            << std::endl;
  std::cout << "Long      max: " << std::numeric_limits<long>::max()
            << std::endl;
  std::cout << std::endl;
  std::cout << "ULong     min: " << std::numeric_limits<unsigned long>::min()
            << std::endl;
  std::cout << "ULong     max: " << std::numeric_limits<unsigned long>::max()
            << std::endl;
  std::cout << std::endl;
  std::cout << "LongLong  min: " << std::numeric_limits<long>::min()
            << std::endl;
  std::cout << "LongLong  max: " << std::numeric_limits<long>::max()
            << std::endl;
  std::cout << std::endl;
  std::cout << "ULongLong min: "
            << std::numeric_limits<unsigned long long>::min() << std::endl;
  std::cout << "ULongLong max: "
            << std::numeric_limits<unsigned long long>::max() << std::endl;
  std::cout << std::endl;
  std::cout << "Float     min: " << std::numeric_limits<float>::min()
            << std::endl;
  std::cout << "Float     max: " << std::numeric_limits<float>::max()
            << std::endl;
  std::cout << std::endl;
  std::cout << "Double    min: " << std::numeric_limits<double>::min()
            << std::endl;
  std::cout << "Double    max: " << std::numeric_limits<double>::max()
            << std::endl;
  std::cout << std::endl;

  return 0;
}

#include <iostream> // IO library
#include <climits>  // char, int, long limits
#include <limits>   // std::numeric_limits
#include <float.h>  // float mins, maxs, epsilons

int main(){

  // these need #include <climits>
  std::cout << "bits per char:          " << CHAR_BIT << std::endl;
  std::cout << "signed char minimum:    " << SCHAR_MIN << std::endl;
  std::cout << "signed char maximum:    " << SCHAR_MAX << std::endl;
  // std::cout << "signed char maximum:    " << UCHAR_MIN << std::endl;
  std::cout << "unsigned char maximum:  " << UCHAR_MAX << std::endl;
  std::cout << "int minimum:            " << INT_MIN << std::endl;
  std::cout << "int maximum:            " << INT_MAX << std::endl;
  std::cout << "unsigned int maximum:   " << UINT_MAX << std::endl;
  std::cout << "long minimum:           " << LONG_MIN << std::endl;
  std::cout << "long maximum:           " << LONG_MAX << std::endl;
  std::cout << "unsigned long maximum:  " << ULONG_MAX << std::endl;
  std::cout << std::endl;

  // this needs #include <float>
  std::cout << "float epsilon:          " << FLT_EPSILON << std::endl;
  std::cout << "float min:              " << FLT_MIN << std::endl;
  std::cout << "float max:              " << FLT_MAX << std::endl;
  std::cout << "double epsilon:         " << DBL_EPSILON << std::endl;
  std::cout << "double min:             " << DBL_MIN << std::endl;
  std::cout << "double max:             " << DBL_MAX << std::endl;
  std::cout << std::endl;

  // this needs #include <limits>
  std::cout << "Also an option: std::numeric_limits<type>::max()" << std::endl;
  std::cout << "   long minimum:           " << std::numeric_limits<long>::min() << std::endl;
  std::cout << "   long maximum:           " << std::numeric_limits<long>::max() << std::endl;
  std::cout << "   double max:             " << std::numeric_limits<double>::max() << std::endl;


  return 0;
}



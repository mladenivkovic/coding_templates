#include <iostream> // IO library

#include "storage_classes_header.h"

// Every variable and function in C++ kernel language has two attributes, type
// and storage class.
// The storage classes are `automatic`, `external`, `register`, and `static`.

// functions are always assumed `external`, i.e. compilers look for it either
// in this file, or in some other file.
// This particular function is declared here, but defined in
// storage_classes_second_file.cpp
int some_int_function(int a);

/**
 * Function to demonstrate behaviour of static variables inside it
 */
void func_testing_static(void) {
  // Static variables retain their value inside a block
  // even when the block is left.
  // Call this function multiple times to see that the
  // value of `called` keeps increasing.
  static int called = 0;
  called++;
  std::cout << "func with static variable called count=" << called << std::endl;
}

/**
 * A static function is only available in the scope of the file
 * where it is defined.
 */
static void some_static_function(void) {
  std::cout << "called some static function" << std::endl;
}

int main() {

  // AUTO
  // ===============
  // NOTE:  In C++11 this keyword was repurposed to be used for implicit type
  // deduction. So the two lines below would throw an error "error: two or more
  // data types in declaration of ‘i’"
  //
  // "auto" is the default storage class. The variables are available
  // withing the scope of the enclosing compunt statement (i.e. within {})
  // auto int i = 10;        // might as well have written `int i = 10;`
  // auto float f = 23.231;  // might as well have written `float f = 23.231;`

  // EXTERN
  // ===============
  // Variable defined in some other object file (in this case, in
  // storage_classes_second_file.cpp) See instructions therein.
  extern int extern_int;
  std::cout << "Extern variable: " << extern_int << std::endl;

  // Directly included headers do not need to be declared extern
  // extern int extern_header_int; // this works
  std::cout << "Extern header variable: " << extern_header_int
            << std::endl; // this works too
  // functions are always assumed extern
  int sif = some_int_function(3);
  std::cout << "Result is = " << sif << std::endl;

  // REGISTER
  // ===============
  // warning: ISO C++17 does not allow ‘register’ storage class specifier

  // store variable in high-speed memory registers
  // for (register int i = 0; i < 10; i++){
  //   std::cout << " " << i;
  // }
  // std::cout << std::endl;

  // STATIC
  // ==============

  // `static` has two main applications.
  // First, static variables retain values between calls of same block:
  func_testing_static();
  func_testing_static();
  func_testing_static();

  // Second: static functions are only available in the
  // file where they were defined in.
  some_static_function(); // this works

  some_static_function_from_header(); // this also works

  // This static function is defined in storage_classes_second_file.cpp
  // Code won't compile like this.
  // some_second_static_function();

  return 0;
}

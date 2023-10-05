#pragma once

#include <iostream>
#include <string>

class myClass {

public:
  // non-const static data member must be initialized out of line
  // can't do this: static double a_static = 1.0;
  // You gotta initialize this static var somewhere else.
  static double static_var;

  // But this works
  inline const static double inline_const_static_var = 2.0;

  std::string s = "Hello";

  std::string sarr[3] = {"Foo", "Bar", "Baz"};

  // Print out sarr
  void print_sarr(void) {
    std::cout << "sarr = " << std::endl;
    int test = sizeof(sarr) / sizeof(std::string);
    std::cout << "test = " << test << std::endl;
    for (size_t i = 0; i < sizeof(sarr) / sizeof(std::string); i++) {
      std::cout << "       " << sarr[i] << std::endl;
    }
  }
};

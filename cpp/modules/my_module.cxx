module; // optional, only if you need to use #include


// all include statements must appear before first export keyword

// module declaration
#include <iostream>


export module my_module;

// function to add two integers
export int add(int a, int b) {
  std::cout << "called add\n";
  return a + b;
}

// function to multiply two integers
export int multiply(int a, int b) {
  std::cout << "called multiply\n";
  return a * b;
}

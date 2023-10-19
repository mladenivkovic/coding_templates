// using functions as arguments for other functions

#include <functional> // needed for std::function
#include <iostream>



int add(int a, int b){
  return a + b;
}

int multiply(int a, int b){
  return a * b;
}


int invoke_via_pointer(int a, int b, int (*func)(int, int)){
  return func(a, b);
}

int invoke_via_std_function(int a, int b, std::function<int(int, int)>func){
  return func(a, b);
}



int main(void) {

  int a = 5;
  int b = 3;

  std::cout << "Calling add: " << invoke_via_pointer(a, b, &add) << "\n";
  std::cout << "Calling multiply: " << invoke_via_pointer(a, b, &multiply) << "\n";

  std::cout << "Calling add: " << invoke_via_std_function(a, b, &add) << "\n";
  std::cout << "Calling multiply: " << invoke_via_std_function(a, b, &multiply) << "\n";

  return 0;
}

#include <iostream> // IO library
#include <string>   // string type


/**
 * void function with no arguments
 */
void void_func_with_no_args(void){
  std::cout << "void_func_with_no_args:                ";
  std::cout << "Hello world!\n";
}

/**
 * void function with single argument
 */
void void_func_with_single_arg(std::string s){
  std::cout << "void_func_with_single_arg:             ";
  std::cout << s << "\n";
}

/**
 * void function with single argument and default value
 */
void void_func_with_single_arg_and_default(std::string s = "Default argument taken"){
  std::cout << "void_func_with_single_arg_and_default: ";
  std::cout << s << "\n";
}


int main(){

  void_func_with_no_args();
  void_func_with_single_arg("single argument passed");
  void_func_with_single_arg_and_default();
  void_func_with_single_arg_and_default("Passed argument");

  return 0;
}



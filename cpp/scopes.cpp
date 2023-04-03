#include <iostream> // IO library

// About variable scopes

int main(){

  int a = 2;                      //outer block a
  std::cout << a << std::endl;    //prints 2
  {                               //enter inner block
    std::cout << a << std::endl;  //prints 2: "a" is valid until inner block re-defines it
    int a = 7;                    //inner block a
    std::cout << a << std::endl;  //prints 7
    std::cout << ++a << std::endl;//prints 8
  }                               //exit inner block
  std::cout << ++a << std::endl;  //3 is printed
  return 0;
}



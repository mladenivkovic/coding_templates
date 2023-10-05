// Demo how to deal with class memeber qualifiers

#include <iostream>
#include "print_var_and_name.h"


class MyClass {
  // NOTE: All values initialized inside the class definition
  // will be negative integers.
  // All positive integers are defined outside, except for the
  // necessary definitions of static variables.
public:
  int myInt;
  int myIntWithInitval = -1;
  const int myConstInt = -2;
  // const int myConstIntWithoutInitval; // ERROR: constructor for 'MyClass' must explicitly initialize the const member 'myConstIntWithInitval'
  static int myStaticInt; // NEEDS TO BE DEFINED SOMEWHERE BEFORE FIRST USE/INTANTIATION
  // static int myStaticIntWithInitval = -3; // error: non-const static data member must be initialized out of line
  inline static int myStaticIntWithInitval = -3; // error: non-const static data member must be initialized out of line
  const static int myConstStaticInt;
  const static int myConstStaticIntWithInitval = -4;
  // constexpr int myConstExprInt = -5; // ERROR: non-static data member cannot be constexpr
  static constexpr int myStaticConstExprInt = -5;

  // Constructor 1
  MyClass(){};

  // Constructor 2
  MyClass(int _myInt,
          int _myIntWithInitval,
          // int _myConstInt
          int _myStaticInt,
          int _myStaticIntWithInitval
          // int _myConstStaticInt,
          // int _myConstStaticIntInitval,
          // int _myConstExprInt
          ){
    myInt = _myInt;
    myIntWithInitval = _myIntWithInitval;
    // myConstInt = _myConstInt; ERROR: cannot assign to non-static data member 'myConstInt' with const-qualified type 'const int'
    myStaticInt = _myStaticInt;
    myStaticIntWithInitval = _myStaticIntWithInitval;
    // myConstStaticInt = _myConstStaticInt; ERROR: error: cannot assign to variable 'myConstStaticInt' with const-qualified type 'const int'
    // myConstStaticIntWithInitval = _myConstStaticIntInitval; // same error as above
    // myStaticConstExprInt = _myConstExprInt; // same error as above
  }
};

// Define statics
int MyClass::myStaticInt = -123; // Otherwise: undefined reference to `MyClass::myStaticInt'
const int MyClass::myConstStaticInt = -1234; // Otherwise: undefined reference to `MyClass::myConstStaticInt'


// Main business
int main(void){

  // CONSTRUCTOR 1: NO ARGUMENTS
  MyClass obj = MyClass();

  DUMP(obj.myInt);
  DUMP(obj.myIntWithInitval);
  DUMP(obj.myConstInt);
  DUMP(obj.myStaticInt);
  DUMP(obj.myStaticIntWithInitval);
  DUMP(obj.myConstStaticInt);
  DUMP(obj.myStaticConstExprInt);

  std::cout << std::endl;

  // CONSTRUCTOR 2: ALL ARGUMENTS
  MyClass obj2 = MyClass(12, 13, 14, 15);
  DUMP(obj2.myInt);
  DUMP(obj2.myIntWithInitval);
  DUMP(obj2.myConstInt);
  DUMP(obj2.myStaticInt);
  DUMP(obj2.myStaticIntWithInitval);
  DUMP(obj2.myConstStaticInt);
  DUMP(obj2.myStaticConstExprInt);

  std::cout << std::endl;

  std::cout << "NOTE! " << std::endl;
  DUMP(obj.myStaticInt);
  DUMP(obj.myConstStaticInt);
  DUMP(obj.myStaticConstExprInt);


  return 0;
}

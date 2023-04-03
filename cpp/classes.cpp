#include <iostream> // IO library
// #include <string>   // string type
// #include <iomanip>   // string type



// Declaration of a simple class
// --------------------------------

class point {
// contrary to structs, the default setting 
// for classes is that members are private.
  double value;
public:
  double x;
  double y;
  void sayHello(void){
    std::cout << "Hello from particle with value " << value << std::endl;
  }
}; // Note the semicolon here

// Check scope rules.
int scopeCheckVal = 10;

class check_scope {
// Check scope rules for classes.
public:
  int scopeCheckVal = 20;
  void sayHello(void){
    // this uses the external scope
    std::cout << "::scopeCheckVal = " << ::scopeCheckVal << std::endl;
    // this uses the local scope
    std::cout << "scopeCheckVal = " << scopeCheckVal << std::endl;
    // this also uses the local scope
    std::cout << "check_scope::scopeCheckVal = " << check_scope::scopeCheckVal << std::endl;
  }
};


// Check static and constant variables for classes
class static_const_vars {
  // TODO: left off here
// public:
  // whatevs
};


int main(){

  // Initialization options
  point p1;
  p1.x = 1.;
  p1.y = -1.;
  // p1.value = 123.; // doesn't work, since value is implicitly private

  std::cout << "p1.x: " << p1.x << std::endl;
  std::cout << "p1.y: " << p1.y << std::endl;
  // std::cout << "p1.value: " << p1.value << std::endl; // doesn't work, since value is implicitly private
  p1.sayHello();

  std::cout << "Scope Checks" << std::endl;
  check_scope cs;
  cs.sayHello();


  return 0;
}



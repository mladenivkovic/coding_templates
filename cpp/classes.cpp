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
}; // Note the semicolon here





int main(){

  // Initialization options
  point p1;
  p1.x = 1.;
  p1.y = -1.;
  // p1.value = 123.; // doesn't work, since value is implicitly private

  std::cout << "p1.x: " << p1.x << std::endl;
  std::cout << "p1.y: " << p1.y << std::endl;
  // std::cout << "p1.value: " << p1.value << std::endl; // doesn't work, since value is implicitly private


  return 0;
}



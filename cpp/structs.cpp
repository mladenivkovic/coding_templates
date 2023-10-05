#include <iostream> // IO library

// Declaration of a simple struct
// --------------------------------
struct point {
  double x;
  double y;
  double value;
}; // Note the semicolon here

// structs are allowed to have member functions
// ------------------------------------------------
struct struct_with_func {
  int some_internal_val;
  void print(void) {
    std::cout << "\t Internal print:" << some_internal_val << std::endl;
  }
};

// You can define functions of structs from outside the struct declaration
// ------------------------------------------------------------------------
struct struct_with_later_defined_function {
  int some_internal_val;
  void function_defined_later(void);
};

void struct_with_later_defined_function::function_defined_later(void) {
  std::cout << "--- this is a later defined function." << std::endl;
}

// Declaring members public and private
// --------------------------------------
struct struct_with_public_private_vars {
public:
  int i, j;
  void public_function(void);
  void set_position(double x, double y) {
    _x = x;
    _y = y;
  }

private:
  double _x, _y;
  void private_function(void);
};

void struct_with_public_private_vars::public_function() {
  // access public members of struct
  std::cout << "--- public function called: i=" << i << " j=" << j << std::endl;
  // access private members of struct
  std::cout << "--- public function called: x=" << _x << " y=" << _y
            << std::endl;
  // access private function
  private_function();
}

// You can still define private functions outside of the declaration
void struct_with_public_private_vars::private_function() {
  // access public members of struct
  std::cout << "------ private function called: i=" << i << " j=" << j
            << std::endl;
  // access private members of struct
  std::cout << "------ private function called: x=" << _x << " y=" << _y
            << std::endl;
}

int main() {

  // Initialization options
  struct point p1 = {1., -1., 123.};

  std::cout << "p1.x: " << p1.x << std::endl;
  std::cout << "p1.y: " << p1.y << std::endl;
  std::cout << "p1.value: " << p1.value << std::endl;

  // element-wise initialization
  struct point p2;
  p2.x = 2.;
  p2.y = -2.;
  p2.value = 345.;
  struct point *p2_p = &p2;

  // Dealing with pointers to structs: Dereferencing with ->
  std::cout << "p2_p->x: " << p2_p->x;
  std::cout << "\t(*p2_p).x: " << (*p2_p).x << std::endl;
  std::cout << "p2_p->y: " << p2_p->y;
  std::cout << "\t(*p2_p).y: " << (*p2_p).y << std::endl;
  std::cout << "p2_p->value: " << p2_p->value;
  std::cout << "\t(*p2_p).value: " << (*p2_p).value << std::endl;

  // Calling member functions
  struct struct_with_func swf = {42};
  swf.print();
  struct struct_with_later_defined_function swldf = {3};
  swldf.function_defined_later();

  // Playing with private and public members
  struct struct_with_public_private_vars swppv;
  swppv.i = 13;
  swppv.j = 14;
  swppv.set_position(3.2, 4.3);
  // swppv.x = 3.141; // This throws an error and doesn't compile because x is
  // private.
  swppv.public_function();
  // swppv.private_function(); // This throws an error and doesn't compile
  // because x is private.

  return 0;
}

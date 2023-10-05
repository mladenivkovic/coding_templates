#include <iostream> // IO library

// using namespace std; // skip this for now to explicitly trace where you get
// things from

/**
 * increase an integer by one and return the result.
 *
 * Intended to demonstrate how to get rid of const qualifyer
 * later.
 **/
int &inc_by_one(int &val) {
  val++;
  return val;
}

int main() {

  // std::fixed: print fixed number of digits after comma.
  std::cout << std::fixed;

  // Note that this works
  int i = 3.2;
  float f = 5.6;

  int itest;
  float ftest;

  // Implicit casting/promoting
  itest = f;
  ftest = i;
  std::cout << "Integer " << i << " to float: " << ftest << std::endl;
  std::cout << "Float " << f << " to integer: " << itest << std::endl;
  std::cout << std::endl;

  // Explicit casting. A conversion that is well defined, portable, and
  // invertible.
  itest = static_cast<int>(f);
  ftest = static_cast<int>(i);
  std::cout << "Integer " << i << " to float: " << ftest << std::endl;
  std::cout << "Float " << f << " to integer: " << itest << std::endl;
  std::cout << std::endl;

  // Explicit casting - this works too, but considered obsolete
  itest = (int)f;
  ftest = (float)i;
  std::cout << "Integer " << i << " to float: " << ftest << std::endl;
  std::cout << "Float " << f << " to integer: " << itest << std::endl;
  std::cout << std::endl;

  // Explicit casting - this works too
  itest = int(f);
  ftest = float(i);
  std::cout << "Integer " << i << " to float: " << ftest << std::endl;
  std::cout << "Float " << f << " to integer: " << itest << std::endl;
  std::cout << std::endl;

  // TODO: figure out what and how const_cast works
  // const int ic[] = {1, 2, 3, 4};
  // int* test = inc_by_one(const_cast<int &>(ic)) ;

  return 0;
}

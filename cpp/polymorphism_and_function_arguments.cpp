#include <iostream>

class A {
protected:
  int val;

public:
  void print() { std::cout << "Hello from A! My val is=" << val << std::endl; };
  void virtual vprint() {
    std::cout << "Hello from virtual A! My val is=" << val << std::endl;
  };
  A(int _val) { val = _val; };
};

class B : public A {
public:
  void print() { std::cout << "Hello from B! My val is=" << val << std::endl; };
  void virtual vprint() {
    std::cout << "Hello from virtual B! My val is=" << val << std::endl;
  };
  B(int _val) : A(_val) {};
};

void f_by_value(A a) {
  a.print();
  a.vprint();
}

void f_by_ref(A &a) {
  a.print();
  a.vprint();
}

// --------------------------------------------------

int main() {

  A myA = A(1);
  B myB = B(2);

  std::cout << "Calling by val" << std::endl;
  f_by_value(myA);
  f_by_value(myB);
  std::cout << "Calling by ref" << std::endl;
  f_by_ref(myA);
  f_by_ref(myB);

  return 0;
}

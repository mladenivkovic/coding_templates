#include <cstddef>
#include <iostream>

class SimpleArray {

  size_t _n;
  int *_myArr;

public:
  SimpleArray(size_t n) : _n(n), _myArr(nullptr) { _myArr = new int[n]; }

  ~SimpleArray() { delete[] _myArr; }

  int &operator[](size_t index) {
    if (index >= _n) {
      std::cout << "Invalid index:" << index << "\n";
      std::abort();
    }
    return _myArr[index];
  }

  void print() {
    for (size_t i = 0; i < _n; i++) {
      std::cout << _myArr[i] << " ";
    }
    std::cout << "\n";
  }
};

class PointerArray {

  size_t _n;
  int **_myArr;

public:
  PointerArray(size_t n) : _n(n), _myArr(nullptr) { _myArr = new int *[n]; }

  ~PointerArray() { delete[] _myArr; }

  int *&operator[](size_t index) {
    if (index >= _n) {
      std::cout << "Invalid index:" << index << "\n";
      std::abort();
    }
    return _myArr[index];
  }

  void print() {
    for (size_t i = 0; i < _n; i++) {
      std::cout << *_myArr[i] << " ";
    }
    std::cout << "\n";
  }
};

int main() {

  SimpleArray myS(4);
  // myS.print();

  myS[0] = 1;
  myS[1] = 2;
  myS[2] = 3;
  myS[3] = 4;
  myS.print();

  PointerArray myPointer(3);
  // segfauls: pointers in array of pointers not allocated yet
  // myPointer.print();
  int a = 1;
  int b = 2;
  int c = 3;
  myPointer[0] = &a;
  myPointer[1] = &b;
  myPointer[2] = &c;
  myPointer.print();

  return 0;
}

#include "singleton.h"

#include <iostream>

void singleton::S::setVar(int x) {
  std::cout << "Setting someVar       at " << &_someVar << " with value=" << x
            << std::endl;
  _someVar = x;
}

void singleton::S::setOtherVar(int x) {
  std::cout << "Setting someVar       at " << &_someOtherVar
            << " with value=" << x << std::endl;
  _someOtherVar = x;
}

int singleton::S::getVar() const {
  std::cout << "Fetching someVar      at " << &_someVar
            << " with value=" << _someVar << std::endl;
  return _someVar;
}

int singleton::S::getOtherVar() const {
  std::cout << "Fetching someOtherVar at " << &_someOtherVar
            << " with value=" << _someOtherVar << std::endl;
  return _someOtherVar;
}

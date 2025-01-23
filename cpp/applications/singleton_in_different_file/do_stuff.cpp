#include "do_stuff.h"
#include "singleton.h"

#include <iostream>

void doStuff::foo() {

  singleton::S &s = singleton::S::getInstance();
  std::cout << "In foo: Got singleton ref at         " << &s << std::endl;

  s.getVar();
  s.getOtherVar();
  s.setVar(17);
  s.setOtherVar(18);
  s.getVar();
  s.getOtherVar();
}

namespace doStuff {
singleton::S &s_outer = singleton::S::getInstance();
}

void doStuff::bar() {

  std::cout << "In foo: Got singleton ref at         " << &s_outer << std::endl;

  s_outer.getVar();
  s_outer.getOtherVar();
  s_outer.setVar(170);
  s_outer.setOtherVar(180);
  s_outer.getVar();
  s_outer.getOtherVar();
}

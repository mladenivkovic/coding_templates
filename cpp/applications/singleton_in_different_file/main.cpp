#include "singleton.h"
#include "do_stuff.h"

#include <iostream>



void fooMain(){

  singleton::S& s = singleton::S::getInstance();
  std::cout << "In foo main: Got singleton ref at   " << &s << std::endl;

  s.getVar();
  s.getOtherVar();
  s.setVar(27);
  s.setOtherVar(28);
  s.getVar();
  s.getOtherVar();

}

singleton::S& s_outer = singleton::S::getInstance();

void barMain(){

  std::cout << "In bar main: Got singleton ref at   " << &s_outer << std::endl;

  s_outer.getVar();
  s_outer.getOtherVar();
  s_outer.setVar(270);
  s_outer.setOtherVar(280);
  s_outer.getVar();
  s_outer.getOtherVar();

}


int main(void){

  singleton::S& s = singleton::S::getInstance();
  std::cout << "In main/inner: Got singleton ref at " << &s << std::endl;

  s.getVar();
  s.getOtherVar();
  s.setVar(27);
  s.setOtherVar(28);
  s.getVar();
  s.getOtherVar();


  std::cout << "In main/outer: Got singleton ref at " << &s_outer << std::endl;

  s_outer.getVar();
  s_outer.getOtherVar();
  s_outer.setVar(270);
  s_outer.setOtherVar(280);
  s_outer.getVar();
  s_outer.getOtherVar();



  fooMain();
  barMain();
  doStuff::foo();
  doStuff::bar();


  return 0;
}

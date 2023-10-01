#pragma once

#include <iostream>



class myClass{

public:

  // non-const static data member must be initialized out of line
  // can't do this: static double a_static = 1.0;
  static double static_var;

  // But this works
  inline const static double inline_const_static_var = 2.0;


};




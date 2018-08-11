#include "mylib.h"

void Rectangle::set_values (int x, int y) {
    width = x;
    height = y;
};


//constructors
Rectangle::Rectangle (int a, int b) {
  width = a;
  height = b;
};

Rectangle::Rectangle () {
    width=1;
    height=1;
};




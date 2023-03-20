//  g++ mylib.cc library.cc 
#include <iostream>
#include "mylib.h"

using namespace std;

int main () {
    Rectangle rect (3,4);
    Rectangle rectb;
    cout << "rect area: " << rect.area() << endl;
    cout << "rectb area: " << rectb.area() << endl;
    rectb.set_values(4,5);
    cout << "rectb area: " << rectb.area() << endl;
    return 0;;
}

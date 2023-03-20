// classes example
#include <iostream>
using namespace std;

class Rectangle {
        int width, height;
    public:
        //constructurs:
        Rectangle(int,int);
        Rectangle ();

        //methods:
        void set_values (int,int);
        int area() {return width*height;}
};

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




int main () {
    Rectangle rect (3,4);
    Rectangle rectb;
    cout << "rect area: " << rect.area() << endl;
    cout << "rectb area: " << rectb.area() << endl;
    rectb.set_values(4,5);
    cout << "rectb area: " << rectb.area() << endl;
    return 0;;
}

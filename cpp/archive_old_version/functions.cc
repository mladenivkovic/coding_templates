#include <iostream>
using namespace std;


// Functions must be declared before the main!

int addition (int a, int b){
    int r;
    r=a+b;
    return r;
}



void printmessage (){
    cout << "I'm a function!";
    cout << "\n";
}


void duplicate (int& a, int& b, int& c){
    a*=2;
    b*=2;
    c*=2;
}


void printarray (int somearray[], int len){
    cout << "Printing array: ";
    for (int i=0; i<len; i++){
        cout << somearray[i] << " ";
    }

    cout << endl;
}


int movedefaftermain(int someint);



//-----------------------------------------------




int main ()
{
    // calling a function
    int a;
    a = addition (5,3);
    cout << "The result is " << a;
    cout << "\n";

    // void function
    printmessage ();


    //arguments can be passed by reference, instead of by value.
    int x=1, y=3, z=7;
    duplicate (x, y, z);
    cout << "Duplicate: x=" << x << ", y=" << y << ", z=" << z;
    cout << "\n";


    int myarray [5] = {1,2,3,4,5};
    printarray(myarray,5);

    cout << "Fct defined after main: " << movedefaftermain(27) << endl;
    
    return 0;
}

//-----------------------------------------------


int movedefaftermain(int someint){
    // Define function here now.
    return someint - 17;
}

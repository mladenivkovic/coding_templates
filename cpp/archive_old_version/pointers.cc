#include <iostream>
using namespace std;


int main () {

    //a variable which stores the address of another variable is called a pointer.
    //Pointers are said to "point to" the variable whose address they store.

    int a = 25;
    int * b = &a; //declare a pointer
    cout << "a = " << a << " \n";
    cout << "pointer b: int * b = &a \n";
    cout << "adress of a = &a = b = " << b <<"\n";
   
    //dereference operator (*). The operator itself can be read as 
    //"value pointed to by"
    cout << "Dereference pointer b: *b = " << *b << "\n\n\n";




    int firstvalue = 5, secondvalue = 15;
    int * p1, * p2;

    p1 = &firstvalue;  // p1 = address of firstvalue
    p2 = &secondvalue; // p2 = address of secondvalue
    *p1 = 10;          // value pointed to by p1 = 10
    *p2 = *p1;         // value pointed to by p2 = value pointed to by p1
    p1 = p2;           // p1 = p2 (value of pointer is copied)
    *p1 = 20;          // value pointed to by p1 = 20
  
    cout << "firstvalue is " << firstvalue << '\n';
    cout << "secondvalue is " << secondvalue << '\n';
   



    // an array can always be implicitly converted to the 
    // pointer of the proper type.
    // The main difference being that mypointer can be assigned 
    // a different address, whereas myarray can never be assigned anything
   

    cout << endl; 
    cout << endl; 
    int numbers[5];
    int * p;
    p = numbers;  *p = 10;  // pointer = array works!!!
    p++;  *p = 20;
    p = &numbers[2];  *p = 30;
    p = numbers + 3;  *p = 40; //change adress, not value
    p = numbers;  *(p+4) = 50;
    for (int n=0; n<5; n++)
        cout << numbers[n] << ", ";
    cout << endl; 
    
    
    
    return 0;
}




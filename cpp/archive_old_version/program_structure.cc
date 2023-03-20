// A short program to demonstrate the program structure of a c++ program.

//preprocessing directives start with a '#'
#include <iostream>
using namespace std; //use std library

// declare function main: main is always run, when a c++ program is run.
int main()
{
    // declare variables
    int a, b, c, result;
    
    

    //write to screen:
    std::cout << "Hello World!" << endl;
    //statement: STanDard::CharacterOUTput << insert the following: "Some characters" ;
    // \n creates a new line.
    cout << "Equivalent to std::cout, if 'using namespace std' was used\n\n" << endl;

    // some calculations: 
    a = 5;
    b = 10;
    c = 3;
    result =(a+b)/c;

    //print result:
    cout << "Result is: " << result << "\n" ;

    // terminate program
    return 0; //return 0 = everything ended well.
}

#include <iostream>
using namespace std;

int main ()
{
    int a,b,c;

    a=2;
    b=7;
    // conditional operator
    //if a>b, c=a; else c=b
    c = (a>b) ? a : b; 

    cout << c << '\n';



    //The comma operator (,) 
    //is used to separate two or more expressions that are included where only
    //one expression is expected. When the set of expressions has to be 
    //evaluated for a value, only the right-most expression is considered.
    a = (b=3, b+2);
    cout << "a = " << a << " b = " << b << '\n';


    //  LOGICAL OPERATORS
    //  !   not
    //  &&  and
    //  ||  or



    // BITWISE OPERATORS
    //  operator	asm equivalent	description
    //  &	        AND	            Bitwise AND
    //  |	        OR	            Bitwise inclusive OR
    //  ^	        XOR	            Bitwise exclusive OR
    //  ~	        NOT	            Unary complement (bit inversion)
    //  <<	      SHL	            Shift bits left
    //  >>	      SHR	            Shift bits right



    return 0;
	
}

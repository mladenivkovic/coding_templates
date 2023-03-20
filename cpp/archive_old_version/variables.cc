// initialization of variables

#include <iostream>
#include <string>
using namespace std;


//define a constant.
const double pi = 3.1415926;\

//Preprocessor definitions:
//#define identifier replacement
#define PI 3.14159

int main ()
{
    // INITIALIZING VARIABLES 
    int a=5;               // initial value: 5 ("c-like initialization")
    int b(3);              // initial value: 3 ("constructor initialization")
    int c{2};              // initial value: 2 ("uniform initialization")
    int result;            // initial value undetermined



    // TYPES
    //  TYPE 	              TYPICAL BIT WIDTH 	TYPICAL RANGE
    //  char 	              1 byte 	            -128 to 127 or 0 to 255
    //  unsigned char 	    1 byte            	0 to 255
    //  signed char 	      1 byte            	-128 to 127
    //  int 	              4 bytes             -2147483648 to 2147483647
    //  unsigned int 	      4 bytes           	0 to 4294967295
    //  signed int 	        4 bytes           	-2147483648 to 2147483647
    //  short int 	        2 bytes           	-32768 to 32767
    //  unsigned short int 	2 bytes           	0 to 65,535
    //  signed short int 	  2 bytes           	-32768 to 32767
    //  long int 	          4 bytes           	-2,147,483,648 to 2,147,483,647
    //  signed long int 	  4 bytes           	-2,147,483,648 to 2,147,483,647
    //  unsigned long int 	4 bytes           	0 to 4,294,967,295
    //  float 	            4 bytes           	+/- 3.4e +/- 38 (~7 digits)
    //  double 	            8 bytes           	+/- 1.7e +/- 308 (~15 digits)
    //  long double 	      8 bytes           	+/- 1.7e +/- 308 (~15 digits)
    //  wchar_t 	          2 or 4 bytes 	      1 wide chara
    //	bool                1 byte              true or false	
    //  void	              no storage

   cout << "Size of char : " << sizeof(char) << endl;
   cout << "Size of int : " << sizeof(int) << endl;
   cout << "Size of short int : " << sizeof(short int) << endl;
   cout << "Size of long int : " << sizeof(long int) << endl;
   cout << "Size of float : " << sizeof(float) << endl;
   cout << "Size of double : " << sizeof(double) << endl;
   cout << "Size of wchar_t : " << sizeof(wchar_t) << endl;
   cout << "Size of bool : " << sizeof(bool) << endl;



    // ESCAPE CHARACTERS
    //  Escape code	  Description
    //  \n	          newline
    //  \r	          carriage return
    //  \t	          tab
    //  \v	          vertical tab
    //  \b	          backspace
    //  \f	          form feed (page feed)
    //  \a	          alert (beep)
    //  \'	          single quote (')
    //  \"	          double quote (")
    //  \?	          question mark (?)
    //  \\	          backslash (\)



    // Some random operations because reasons.
    a = a + b;
    result = a - c;
    cout << "\nResult is: " << result << endl;

    int declareafteroperations=3;
    //You don't have to declare all variables in the beginning.


    // TYPE DEDUCTIONS
    auto d = a; //automatically set the correct type
    decltype (a) e;  //take the same type for e that a has

    // STRINGS
    // needs #include <string>
    string mystring1 = "This is a string";
    string mystring2 ("This is a string");
    string mystring3 {"This is a string"};


    cout << "\n" << endl;
    cout << pi;
    cout << PI;
	

    return 0;
}

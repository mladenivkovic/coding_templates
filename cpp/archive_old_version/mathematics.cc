#include <iostream>
#include <math.h>
#include <cmath>  // abs that can handle doubles
#include <stdio.h>      /* printf */
using namespace std;

#define PI 3.14159265

// http://www.cplusplus.com/reference/cmath/
int main () 
{

    double param, result;
    param = 60;


    // trigonometric functions
   
    cout << "TRIGONOMETRIC FUNCTIONS \n";
    cout << "sin(60*PI/180) \t ";
    cout << sin(60*PI/180);
    cout << "\n";

    cout << "cos(60*PI/180) \t ";
    cout << cos(60*PI/180);
    cout << "\n";

    cout << "tan(60*PI/180) \t ";
    cout << tan(60*PI/180);
    cout << "\n";


    cout << "asin(0.1) \t ";
    cout << asin(0.1);
    cout << "\n";

    cout << "acos(0.1) \t ";
    cout << acos(0.1);
    cout << "\n";

    cout << "atan(0.1) \t ";
    cout << atan(0.1);
    cout << "\n";



    // Hyperbolic functions

    cout << "\n";
    cout << "HYPERBOLIC FUNCTIONS \n";
    cout << "sinh(60*PI/180) \t ";
    cout << sinh(60*PI/180);
    cout << "\n";

    cout << "cosh(60*PI/180) \t ";
    cout << cosh(60*PI/180);
    cout << "\n";

    cout << "tanh(60*PI/180) \t ";
    cout << tanh(60*PI/180);
    cout << "\n";


    cout << "asinh(0.1) \t ";
    cout << asinh(0.1);
    cout << "\n";

    cout << "acosh(1.1) \t ";
    cout << acosh(1.1);
    cout << "\n";

    cout << "atanh(0.1) \t ";
    cout << atanh(0.1);
    cout << "\n";





    // Exponential and logarithmic functions
    cout << "\n";
    cout << "EXPONENTIAL AND LOGARITHMIC FUNCTIONS\n";

    cout << "exp(1.0) \t ";
    cout << exp(1.0);
    cout << "\n";

    cout << "log(1.0) \t ";
    cout << log(1.0);
    cout << "\n";

    cout << "log10(10.0) \t ";
    cout << log10(10.0);
    cout << "\n";




    // various other usefull stuff

    cout << "\n";
    cout << "MISC\n";

    cout << "sqrt(4.0) \t ";
    cout << sqrt(4.0);
    cout << "\n";

    cout << "pow(2.0, 5.0) \t ";
    cout << pow(2.0, 5.0);
    cout << "\n";

    cout << "abs(-2.0) \t ";
    cout << abs(-2.0);
    cout << "\n";


    cout << "\n";




    // Rounding
    cout << "\n";
    cout << "ROUNDING\n";

    const char * format = "%.1f \t%.1f \t%.1f \t%.1f \t%.1f\n";
    printf ("value\tround\tfloor\tceil\ttrunc\n");
    printf ("-----\t-----\t-----\t----\t-----\n");
    printf (format, 2.3,round( 2.3),floor( 2.3),ceil( 2.3),trunc( 2.3));
    printf (format, 3.8,round( 3.8),floor( 3.8),ceil( 3.8),trunc( 3.8));
    printf (format, 5.5,round( 5.5),floor( 5.5),ceil( 5.5),trunc( 5.5));
    printf (format,-2.3,round(-2.3),floor(-2.3),ceil(-2.3),trunc(-2.3));
    printf (format,-3.8,round(-3.8),floor(-3.8),ceil(-3.8),trunc(-3.8));
    printf (format,-5.5,round(-5.5),floor(-5.5),ceil(-5.5),trunc(-5.5));
    
    cout << "\n";
    
    
    
    return 0;
}

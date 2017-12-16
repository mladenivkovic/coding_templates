//====================
// How math is done.
//====================

#include <stdio.h>
#include <math.h>
#include <stdlib.h> // abs()

#define PI 3.14159


int main(void){

    int negative = -3;
    double somefloat = 16.342;


    printf("Abs of int  : %d\n", abs(negative));
    printf("Abs of float: %.5f\n",fabs((float)(negative) ));

    printf("round up    : %.5lf\n", ceil(somefloat));
    printf("round down  : %.5f\n", floor(somefloat));

    printf("exp         : %.5f\n", exp(somefloat) );
    printf("ln          : %.5f\n",log(somefloat) );
    printf("log_10      : %.5f\n",log10(somefloat) );

    printf("power       : %.5f\n",pow(somefloat,somefloat/2.43) );
    printf("sqrt        : %.5f\n",sqrt(somefloat) );
 
    printf("sin         : %.5f\n",sin(PI/4) );
    printf("cos         : %.5f\n",cos(PI/4) );
    printf("tan         : %.5f\n",tan(PI/4) );


    /*printf("arccos %.5f\n", );*/
    /*printf("arcsin %.5f\n", );*/
    /*printf("arctan %.5f\n", );*/
    /**/
    /*printf("arctan %.5f\n", );*/





}

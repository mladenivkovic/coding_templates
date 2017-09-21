/*
 * Define my own "print()" function that always prints a new line.
 * Used the same internal functions as printf(), but added a newline.
 * http://www.firmcodes.com/write-printf-function-c/
 */

#include<stdio.h> 
#include<stdarg.h>					// standard argument library	
 
void print(const char *format,...) 
{ 
	  va_list arg;                //from stdarg.h; va_list type defined in stdarg.h
    int done;

	  va_start(arg, format);      //initialises arg variable, starting with "format" 
                                //variable given as an argument to print()
	
    done = vfprintf(stdout, format, arg);   // call the formatting and printing 
                                            // function that printf also uses

    va_end(arg);                // do whatever cleanup is necessary


    printf("\n");                //always end with a newline! :)

    /*if (done !=0)*/
    /*{*/
    /*    printf("ERROR: your print() function exited with error code %d\n", done);*/
    /*}*/

} 
 

int main() 
{ 
	print(" Here I go using my own print! \n %d %.3f", 999, 89234.24243); 
	
	return 0;
} 


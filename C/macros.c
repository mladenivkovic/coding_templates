//===================================
// Defining macros
//===================================



#include <stdio.h>      /* input, output    */


// PARENTHESES AROUND FORMAL PARAMETERS ARE VITAL!
#define PRINT_INT(label, num) printf("%s %d\n", (label), (num))
#define PRINT_DOUBLE(label, num) printf("%s %lf\n", (label), (num))
#define PI 3.14152926



int
main(void)    
{

  PRINT_INT("The answer is", 42);
  PRINT_DOUBLE("Pi is", PI);
  
  return(0);
}


/* 
 * Write some comments in here.
 */



#include <stdio.h>      /* input, output    */


// PARENTHESES AROUND FORMAL PARAMETERS ARE VITAL!
#define PRINT_INT(label, num) printf("%s %d\n", (label), (num))



int
main(void)    
{

  PRINT_INT("The answer is", 42);
  return(0);
}


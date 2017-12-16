//====================================
// Handling command line arguments
//====================================




#include <stdio.h>      /* input, output    */
#include <string.h>



int
main( int argc,     // input argument count (including program name)
      char *argv[]  // input argument vector
    )
{
  
  printf("These were your cmd line args:\n");
  char str[80];

  for (int i = 0; i < argc; i++){
    printf("Arg %d: %s\n", i, argv[i]);
  }


  return(0);
}


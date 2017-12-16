//====================== 
// Loopityloops
//======================



#include <stdio.h>      /* input, output    */



int
main(void)    
{

  int counter, i;

  //=======================
  // WHILE Loop
  //=======================

  counter = 0;
  while (counter < 7)
  {
      counter += 1;
  }
  printf("1.1 - Counter after loop: %d\n", counter);
  
  while (counter < 7)
  {
      counter += 1;
  }
  printf("1.2 - Counter after loop: %d\n", counter);
  printf("\n");





  //=======================
  // FOR Loop
  //=======================


  counter = 0;
  
  for (i = 0; i<7; i++)
  {
      counter += 1;
  }

  printf("2.1 - Counter after loop: %d\n", counter);



  counter = 0;
 

  for (i = 0; i<7; ++i)
  {
      counter += 1;
  }

  printf("2.2 - Counter after loop: %d\n", counter);
  printf("\n");
  ///////////////////////////////////////////////////////////////////
  // You're not using the return value of i, only its actual value -
  // therefore it makes no difference whether you use i++ or ++i !
  ///////////////////////////////////////////////////////////////////






  //=======================
  // DO WHILE Loop
  //=======================

  counter = 0;

  do {
      counter += 1;
  }
  while (counter < 3);

  printf("3.1 - Counter after loop: %d\n", counter);


  counter = 0;
  do {
      counter += 1;
  }
  while (counter < 0);
  printf("3.2 - Counter after loop: %d\n", counter);

  ////////////////////////////////////////////////////
  // DO WHILE executes command block at least once!
  ////////////////////////////////////////////////////

  return(0);

}


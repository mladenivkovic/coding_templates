//=======================================================
// On bitwise comparisons.
//=======================================================


#include <stdio.h>      /* input, output    */
#include <math.h>
#include <stdlib.h>



//===================================
char * binary(int a){
//===================================
  // returns binary representation of integer a as a string,
  // ready to be printed
  

  char * binary_string;

  int bits = sizeof(int)*8-1;

  binary_string = malloc(sizeof(char) * bits);

  for (int i = bits-1; i >= 0; i--){
    int pow2 = pow(2, i);
    int div = a / pow2;
    if ( div > 0){
      binary_string[bits-i-1] = '1';
      a -= div * pow2;
    }
    else{
      binary_string[bits-i-1] = '0';
    }
  }

  return(binary_string);
}




//===================================
int main()    
//===================================
{


  int a, b;

  /* a = 1; */
  /* b = 0; */
  /*  */
  /* printf("a = %d, b = %d, a & b = %d\n",  a, b, a & b); */
  /* printf("a = %d, b = %d, a && b = %d\n", a, b, a && b); */
  /* printf("a = %d, b = %d, a | b = %d\n",  a, b, a | b); */
  /* printf("a = %d, b = %d, a || b = %d\n",  a, b, a || b); */
  /* printf("\n"); */
  /*  */
  /* a = 3; */
  /* b = 10; */
  /* printf("a = %d, b = %d, a & b = %d\n",  a, b, a & b); */
  /* printf("a = %d, b = %d, a && b = %d\n", a, b, a && b); */
  /* printf("a = %d, b = %d, a | b = %d\n",  a, b, a | b); */
  /* printf("a = %d, b = %d, a || b = %d\n",  a, b, a || b); */




  a = 3;
  b = 10;

  printf("\n\n");
  printf("    In binary:                        In decimal:\n\n");
  printf(" Binary operators:\n\n");

  printf("    %s   %d\n", binary(a), a);
  printf(" &  %s   %d\n", binary(b), b);
  printf("  -------------------------------------------------\n");
  printf("    %s   %d\n", binary(a & b), a&b);
  printf("\n");

  printf("    %s   %d\n", binary(a), a);
  printf(" |  %s   %d\n", binary(b), b);
  printf("  -------------------------------------------------\n");
  printf("    %s   %d\n", binary(a | b), a|b);
  printf("\n");

  printf("    %s   %d\n", binary(a), a);
  printf(" ^  %s   %d\n", binary(b), b);
  printf("  -------------------------------------------------\n");
  printf("    %s   %d\n", binary(a ^ b), a^b);
  printf("\n");





  printf(" Boolean operators:\n\n");

  printf("    %s   %d\n", binary(a), a);
  printf(" && %s   %d\n", binary(b), b);
  printf("  -------------------------------------------------\n");
  printf("    %s   %d\n", binary(a && b), a&&b);
  printf("\n");


  printf("    %s   %d\n", binary(a), a);
  printf(" || %s   %d\n", binary(b), b);
  printf("  -------------------------------------------------\n");
  printf("    %s   %d\n", binary(a || b), a||b);
  printf("\n");





  printf("\n");
  printf("Using bitwise shifts, you can use this to store multiple booleans into one single int:\n");


  int storage = 0;
  int true = 1;
  int false = 0;

  int bool0 = true;
  int bool1 = false;
  int bool2 = true;
  int bool3 = true;

  int val0 = 0;
  int val1 = 0;
  int val2 = 0;
  int val3 = 1;

  if (bool0) {
    val0 = 1<<0;
    storage = (storage | val0);
  }
  if (bool1) {
    val1 = 1<<1;
    storage = (storage | val1);
  }
  if (bool2) {
    val2 = 1<<2;
    storage = (storage | val2);
  }
  if (bool3) {
    val3 = 1<<3;
    printf("val3 %d\n", val3);
    storage = (storage | val3);
  }


  printf("stored booleans: %s\n", binary(storage));
  if (val0 & storage) printf("bool0 = true\n");
  else printf("bool0 = false\n");
  if (val1 & storage) printf("bool1 = true\n");
  else printf("bool1 = false\n");
  if (val2 & storage) printf("bool2 = true\n");
  else printf("bool2 = false\n");
  if (val3 & storage) printf("bool3 = true\n");
  else printf("bool3 = false\n");



  return(0);
}





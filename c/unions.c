/* UNIONS
 * A union is a special data type available in C that allows to store different 
 * data types in the same memory location. You can define a union with many  
 * members, but only one member can contain a value at any given time. Unions  
 * provide an efficient way of using the same memory location for multiple-purpose. */


#include <stdio.h>


union myunion {
  int i;
  float f;
  char str[20];
} ;



int main(int argc, char* argv[])
{

  int i;
  float f;
  char c[20];
  union myunion union_instance;

  printf("Memory size occupied by int i : \t%lu\n", sizeof(i));
  printf("Memory size occupied by float f : \t%lu\n", sizeof(f));
  printf("Memory size occupied by char[20] : \t%lu\n", sizeof(c));
  printf("Memory size occupied by myunion : \t%lu\n", sizeof(union_instance));
  printf("\n");


  union_instance.i = 20;
  printf("union_instance.i : \t%d\n", union_instance.i);

  union_instance.f = 123.423;
  printf("union_instance.f : \t%f\n", union_instance.f);

  printf("WARNING: -Wall will not warn you if you use the 'wrong' data type.\n");
  printf("WARNING: If you're not careful, you'll get gibberish.\n");
  printf("union_instance.i : \t%d\n", union_instance.i);

  printf("union_instance.str : \t%s\n", union_instance.str);



  return 0;

}

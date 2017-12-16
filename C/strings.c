//============================ 
// Dealing with strings.
//============================ 



#include <stdio.h>      /* input, output    */
#include <string.h>     /* string stuff     */






//====================
int main(void)    
//====================
{

  // Strings are arrays of chars
  char initv[20] = "Initial value";

  // Array of strings is multidim char array
  // Don't forget to include space for the "\0" char, which marks the end of the string!
  char months_wrong[12][3] = { "Jan", "Feb", "Mar", "May", "Aug", "Jun", "Jul", "Sep", "Oct", "Nov", "Dec" };
  char months[12][4] = { "Jan", "Feb", "Mar", "May", "Aug", "Jun", "Jul", "Sep", "Oct", "Nov", "Dec" };

  printf("Initv: %s \n", initv);
  printf("Months wrong: %s \n", months_wrong[4]);
  printf("Months right: %s \n", months[4]);
 

  // only works with the library string.h
  char mysting[10];
  
  //---------------------------------
  // string assignment
  //---------------------------------
  strcpy(mysting, "Hi there!");  //copy string to string
  printf("%s \n", mysting);

  // DONT FORGET ABOUT OVERFLOWS!!!!!!
  // strcpy(mysting, "Hello there! This is an overflow.");  //overdo it
  // printf("%s \n", mysting);

  //---------------------------------
  //concatenate
  //---------------------------------
  char concatenated[40] = "Concatenate: ";
  strcat(concatenated, initv);
  printf("%s \n", concatenated);


  
  //---------------------------------
  //comparisons
  //---------------------------------
  char str1[16] = "This is string 1";
  char str2[16] = "This is string 2";

  //String comparisons don't work this way.
  if (str1 < str2)
    printf("str1 has adress value smaller than str2\n");
  else
    printf("str2 has adress value smaller than str1\n");



  //Compare first 10 characters of strings
  if (str1[10] < str2[10])
    printf("str1 is alphabetically earlier than str2\n");
  else
    printf("str1 is not alphabetially earlier than str2, but it should be\n");



  //Compare first 20 characters of strings
  if (str1[20] < str2[20])
    printf("str1 is alphabetically earlier than str2. Now we got it right.\n");
  else
    printf("str1 is not alphabetially earlier than str2, but it should be\n");


  //---------------------------------
  // better way of comparing strings:
  //---------------------------------

  if (strcmp(str1,str2) < 0 )
    printf("str1 is alphabetically earlier than str2. \n");
  else if (strcmp(str1,str2) == 0 )
    printf("str1 and str2 are the same. \n");
  else
    printf("str1 is not alphabetially earlier than str2\n");

  return(0);
}


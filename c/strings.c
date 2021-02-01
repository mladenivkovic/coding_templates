/* ============================ */
/*  Dealing with strings.       */
/* ============================ */

#include <stdio.h>  /* input, output    */
#include <stdlib.h> /* atoi, atof       */
#include <string.h> /* string stuff     */

int main(void) {

  /* Strings are arrays of chars */
  char initv[20] = "Initial value";

  /* Array of strings is multidim char array
   * Don't forget to include space for the "\0" char,
   * which marks the end of the
   * string! */
  char months_wrong[12][3] = {"Jan", "Feb", "Mar", "May", "Aug", "Jun",
                              "Jul", "Sep", "Oct", "Nov", "Dec"};
  char months[12][4] = {"Jan", "Feb", "Mar", "May", "Aug", "Jun",
                        "Jul", "Sep", "Oct", "Nov", "Dec"};

  printf("Initv: %s \n", initv);
  printf("Months wrong: %s \n", months_wrong[4]);
  printf("Months right: %s \n", months[4]);

  /* only works with the library string.h */
  char mysting[10];

  /* --------------------------------- */
  /*  string assignment */
  /* --------------------------------- */
  strcpy(mysting, "Hi there!"); /* copy string to string */
  printf("%s \n", mysting);

  /* DONT FORGET ABOUT OVERFLOWS!!!!!! */
  /* strcpy(mysting, "Hello there! This is an overflow.");  [> overdo it <] */
  /* printf("%s \n", mysting); */

  /* --------------------------------- */
  /*  concatenate */
  /* --------------------------------- */
  char concatenated[40] = "Concatenate: ";
  strcat(concatenated, initv);
  printf("%s \n", concatenated);

  /* --------------------------------- */
  /*  comparisons */
  /* --------------------------------- */
  char str1[16] = "This is string 1";
  char str2[16] = "This is string 2";

  /* String comparisons don't work this way. */
  if (str1 < str2)
    printf("str1 has adress value smaller than str2\n");
  else
    printf("str2 has adress value smaller than str1\n");

  /* Compare first 10 characters of strings */
  if (str1[10] < str2[10])
    printf("str1 is alphabetically earlier than str2\n");
  else
    printf("str1 is not alphabetially earlier than str2, but it should be\n");

  /* Compare first 20 characters of strings */
  if (str1[20] < str2[20])
    printf("str1 is alphabetically earlier than str2. Now we got it right.\n");
  else
    printf("str1 is not alphabetially earlier than str2, but it should be\n");

  /* --------------------------------- */
  /*  better way of comparing strings: */
  /* --------------------------------- */

  if (strcmp(str1, str2) < 0)
    printf("str1 is alphabetically earlier than str2. \n");
  else if (strcmp(str1, str2) == 0)
    printf("str1 and str2 are the same. \n");
  else
    printf("str1 is not alphabetially earlier than str2\n");

  char shortstring[20] = "string short";
  char longstring[30] = "string short is shorter";
  printf("Short string is: %s\n", shortstring);
  printf("Long string is: %s\n", longstring);
  if (strcmp(shortstring, longstring) < 0)
    printf("shortstring is alphabetically earlier than longstring. \n");
  else if (strcmp(shortstring, longstring) == 0)
    printf("shortstring and longstring are the same. \n");
  else
    printf("shortstring is not alphabetially earlier than longstring\n");

  /* --------------------------- */
  /*  String conversions */
  /* --------------------------- */

  int someint = 12312;
  float somefloat = 8234.13920;
  double somedouble = 1124081274102874124.4817091724;

  char stringint[30];
  char stringfloat[30];
  char stringdouble[30];

  sprintf(stringint, "%d", someint);
  sprintf(stringfloat, "%f", somefloat);
  sprintf(stringdouble, "%g", somedouble);

  printf("\n String conversions \n");
  printf("Nonstring to string \n");
  printf("Int:     %10d\t ; String: %s\n", someint, stringint);
  printf("Float:  %10f\t ; String: %s\n", somefloat, stringfloat);
  printf("Double: %10g\t ; String: %s\n", somedouble, stringdouble);

  printf("string to nonstring\n");

  int secondint = atoi(stringint);
  float secondfloat = atof(stringfloat);

  printf("Int:     %10d\t ; String: %s\n", secondint, stringint);
  printf("Float:  %10f\t ; String: %s\n", secondfloat, stringfloat);

  return (0);
}

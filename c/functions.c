/* =============================== */
/* Functions baby, functions! */
/* =============================== */

#include <stdio.h>
#define PI 3.14159

/* =================================================== */
/* Declare functions that will be defined after main: */
/* =================================================== */
void print_useless_shit(void);
void compute_and_print_area(double radius);
double circumference(double radius);
double cylinder_volume(double radius, double height);
void output_values_with_pointers(double num, char *signp, int *wholep,
                                 int *fracp);
void passing_output_parameters_as_arguments(double num, char *signp,
                                            int *wholep, int *fracp);
int fibonacci(int n);

/* ======================== */
int main(void) {
  /* ======================== */

  double radius = 13.2;

  print_useless_shit();

  compute_and_print_area(radius);

  double circ = circumference(radius);
  printf("The circumference is %.3f\n", circ);

  double cyl_vol = cylinder_volume(radius, 28.4982);
  printf("The cylinder volume is %.3g\n", cyl_vol);

  double somenumber = 23408.283904;
  char sign;
  int wholepart;
  int fracpart;
  output_values_with_pointers(somenumber, &sign, &wholepart, &fracpart);
  printf("%c%6d.%6d\n", sign, wholepart, fracpart);

  passing_output_parameters_as_arguments(somenumber, &sign, &wholepart,
                                         &fracpart);
  printf("%c%6d.%6d\n", sign, wholepart, fracpart);

  /*int *counter;*/

  int n = 12;
  int fib = fibonacci(n);
  printf("Fibonacci nr %d : %d\n", n, fib);

  return (0);
}

/* ============================== */
void print_useless_shit(void)
/* ============================== */
{
  /* This is a function that doesn't need or return anything. */

  printf("Hey there! Here's some more useless text for you.\n");
}

/* ======================================== */
void compute_and_print_area(double radius)
/* ======================================== */
{
  /* takes input, but creates no output. */

  double area = PI * radius * radius;
  printf("The area is %.3f\n", area);
}

/* ======================================== */
double circumference(double radius)
/* ======================================== */
{
  /* takes input, creates only one output */

  double circumference = 2 * PI * radius;

  return (circumference);
}

/* ==================================================== */
double cylinder_volume(double radius, double height)
/* ==================================================== */
{
  /* takes multiple arguments, gives one out */
  double volume = PI * radius * radius * height;
  return (volume);
}

/* ====================================================================================
 */
void output_values_with_pointers(double num, char *signp, int *wholep,
                                 int *fracp)
/* ====================================================================================
 */
{
  /* Separates float "num" into sign, whole value and fractional value. */
  /* Sign, whole value and fractional value are "returned" to the main. */

  int magnitude;
  double remainder;

  if (num < 0) {
    *signp = '-';
    num = -num;
  } else if (num == 0)
    *signp = ' ';
  else
    *signp = '+';

  magnitude = (int)num;
  *wholep = magnitude;

  remainder = num - magnitude;
  /* assuming up to 6 decimal digits */
  *fracp = remainder * 1000000;
}

/* ============================================================================================
 */
void passing_output_parameters_as_arguments(double num, char *signp,
                                            int *wholep, int *fracp)
/* ============================================================================================
 */
{
  /*This function passes the arguments to the function
   * output_values_with_pointers()*/
  output_values_with_pointers(num, signp, wholep, fracp);
  printf("Accessed function passing_output_parameters_as_arguments :)\n");
}

/* ================================ */
int fibonacci(int n)
/* ================================ */
{
  /* Determine the n-th fibonacci number recursively. */
  /* This is just an auxilliary function to keep the code clean. */
  /* The actual computation happens with compute_fibonacci(n) */

  int result;
  int previous = 0;

  /* define function here */
  int compute_fibonacci(int n, int *previous);

  if (n > 1) {
    result = compute_fibonacci(n, &previous);
    return result;
  } else if (n < 1) {
    printf("Something went wrong.\n");
    return 0;
  } else {
    return 1;
  }
}

/* =========================================== */
int compute_fibonacci(int n, int *previous)
/* =========================================== */
{

  int step, result;

  if (n > 1) {
    step = compute_fibonacci(n - 1, previous);
    result = step + *previous;
    *previous = step;
    return result;
  } else if (n < 1) {
    printf("Something went wrong.\n");
    return 0;
  } else {
    /* *previous = 0; [> not necessary...<] */
    return 1;
  }
}

/* Writing your own header file.  */

// allows new types
typedef struct { /* rectangle */
  double a;
  double b;
} rectangle_t;

// allows functions to be defined elsewhere
// and then linked to the compiler.
extern void print_rectangle_surface(rectangle_t r);

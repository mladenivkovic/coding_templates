/* Writing your own header file.  */

#define PI 3.1415926

// allows new types
typedef struct { /* cylinder */
  double radius;
  double height;
} cylinder_t;

// allows functions to be defined elsewhere
// and then linked to the compiler.
extern void print_cylinder_volume(cylinder_t c);

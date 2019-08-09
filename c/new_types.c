//=========================
// New types and structs.
//=========================

#include <stdio.h> /* input, output    */
#include <stdlib.h>
#include <string.h>  // strcpy()

#define STRINGSIZE 30

//====================
// enumerated type
//====================

// enumeration constants must be identifiers, they cannot be numeric,character,
// or string literals
// enum types are mainly used to make the code more readable.

typedef enum { student_id, grade, income } student;

typedef enum { Mon, Tue, Wed, Thur, Fri, Sat, Sun } week;

//====================
// struct type
//====================
typedef struct {
  char name[STRINGSIZE];  // name of planet
  double diameter;        // diameter in km
  int moons;              // # of moons
  double orbit_time;      // orbit around sun in yrs
  double rotation_time;   // time for one revolution in hrs
} planet;

//===========================
// functions using new types
//===========================
void printplanet(planet p) {
  printf("Name:          %s\n", p.name);
  printf("Diameter:      %f\n", p.diameter);
  printf("Moons:         %d\n", p.moons);
  printf("Orbit time:    %f\n", p.rotation_time);
  printf("Rotation time: %f\n", p.orbit_time);
}

//====================
int main(void)
//====================
{

  // type enum

  printf("\n ENUM TYPES\n\n");

  // note how you're not integers to assign the value,
  // but the keywords you defined!
  // 'Wed' and 'grade' are not variables!

  week day;
  day = Wed;
  printf("The day is %d (Wed)\n", day);

  int result;
  student teststudent = grade;

  switch (teststudent) {
    case student_id:
      result = 234872;
      break;

    case grade:
      result = 4;
      break;

    case income:
      result = 20000;
      break;

    default:
      result = 0;
  }

  printf("enum result: %d\n", result);

  printf("\n\n\n STRUCT TYPES\n\n");

  // struct type
  planet jupiter = {"Jupiter", 142800, 16, 11.9, 9.925};
  planet earth;
  strcpy(earth.name, "Earth");
  earth.diameter = 6371;
  earth.moons = 1;
  earth.orbit_time = 1;
  earth.rotation_time = 24;

  printf("%s\n", jupiter.name);
  printf("%f\n", jupiter.diameter);
  printf("%s\n", earth.name);
  printf("%f\n", earth.diameter);

  // Assigning whole structs to others works
  earth = jupiter;
  printf("%s\n", earth.name);
  printf("%f\n", earth.diameter);

  // using functions
  printplanet(jupiter);

  //=================================
  // WORKING WITH ARRAYS OF STRUCTS
  //=================================

  // define new struct
  typedef struct {
    int someint;
    double somedouble;
  } TEST;

  // initialize new array of pointers to structs
  int ntest = 3;
  TEST** testarr = malloc(ntest * sizeof(TEST*));

  // fill array
  for (int i = 0; i < ntest; i++) {
    TEST* newtest = malloc(sizeof(newtest));
    newtest->someint = i;
    // same as (*newtest).someint = i;
    newtest->somedouble = i * 0.33;
    testarr[i] = newtest;
  }

  // print
  printf("\n\n\n");
  for (int i = 0; i < ntest; i++) {
    printf("ARRAYTEST %d : %d %g\n", i, testarr[i]->someint,
           testarr[i]->somedouble);
  }
  printf("\n\n\n");

  return (0);
}

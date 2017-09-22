/* 
 * Write some comments in here.
 */



#include <stdio.h>      /* input, output    */


// enumerated type
typedef enum 
  // enumeration constants must be indetifiers, they cannot be numeric,character,
  // or string literals
  { student_id, grade, income}
  student;


int
main(void)    
{

    int result;
    student teststudent;

    switch (teststudent) {
      case student_id:
        result=234872;
        break;
      
      case grade:
        result=4;
        break;

      case income:
        result=20000;
        break;
    
      default:
        result=0;
    }  

    printf("result: %d\n", result);

  return(0);
}


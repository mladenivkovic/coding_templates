//================= 
// Using enum.
//================= 
 



#include <stdio.h>      /* input, output    */


enum week { monday, tuesday, wednesday, thursday, friday, saturday, sunday };


//assign your own values
enum cards {
  ace = 15,
  jack = 12,
  queen = 13,
  king = 14
  };



//define as a new type to use in arrays
//can't define same names twice!
/*typedef enum {monday, tuesday, wednesday, thursday, friday } workweekday;*/
typedef enum {mon, tue, wed, thu, fri } workweekday;





int
main(void)    
{

  // simple usage
  enum week today;
    today = wednesday;
    printf("Day %d\n",today+1);


  enum cards mycard = jack;
  enum cards yourcard = king;
  if (mycard <= yourcard)
    printf("I lost :(\n");
  else
    printf("I won :)\n");
 


  // use in arrays

  int tasks_pendent[5];
  workweekday tomorrow = thu;
  tasks_pendent[tomorrow] = 7;



  return(0);
}


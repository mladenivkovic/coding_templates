/*
 * Branches: if and switch
 */


#include <stdio.h>


int main(void)
{
    char somechar;
    int true, false;

    false = 0;
    true = 1;



    printf("Logical expressions with ints:\n");

    printf("!0 =     %d\n", !false);

    printf("!1 =     %d\n", !true);

    printf("!(!1) =  %d\n", !(!true));

    printf("1 || 0 = %d\n", true || false);

    printf("1 && 0 = %d\n", true && false);




/*=============================*/
/*=============================*/
/*=============================*/



    int i = 12, j = 7;

    printf("\nIF\n");

    if (i > j)
    {
        printf("i >= j \n");
        printf("Statement was true.\n");

        if (i >= 2*j)
        {
            printf("i > 2j\n");
            printf("Nested ifs work.\n");
        }
        else
        {
            printf("i <= 2j\n");
            printf("Nested ifs work.\n");
        }
    }
    else if (i == j)
    {
        printf("i == j\n");
    }
    else
    {
        printf("j > i \n");
        printf("Statement was false.\n");
    }



/*=============================*/
/*=============================*/
/*=============================*/



    printf("\nSWITCH\n");
    
    /*int watts_of_bulb =240;*/
    int watts_of_bulb =75;
    /*int watts_of_bulb =40;*/
    int life;

    switch (watts_of_bulb)
    {
    case 25:
        life = 2500;
        printf("Now checking 25\n");
        break;      // doesn't check other cases
    
    case 40:        // if true, it will go on until the first break.
    case 50:

    case 60:
        life = 1000;
        break;
    
    case 75:
        // if true, it will go on until the first break.
        printf("Now checking 75\n");
    
    case 100:
        life = 750;
        printf("Now checking 100\n");
   
    case 150:
        life = 500; 
        printf("Now checking 150\n");
        break;

    case 200:
        life = 250; 
        printf("Now checking 200\n");
        break;

    default:
        life = 0; // if watts > 200
    }

    printf("expected bulb life: %d\n", life);




    return(0);


}

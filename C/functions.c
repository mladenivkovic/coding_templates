/*
* Functions baby, functions!
*/

#include <stdio.h>
#define PI 3.14159



//================================
//================================
//================================


void print_useless_shit(void)
{
    //This is a function that doesn't need or return anything.
    
    printf("Hey there! Here's some more useless text for you.\n");
}

//================================
//================================
//================================


void compute_and_print_area(double radius)
{   
    // takes input, but creates no output.

    double area = PI * radius * radius;
    printf("The area is %.3f\n", area);

}


//================================
//================================
//================================


double circumference(double radius)
{
    // takes input, creates only one output

    double circumference = 2 * PI * radius;

    return(circumference);
}



//================================
//================================
//================================


double cylinder_volume(double radius, double height)
{
    // takes multiple arguments, gives one out
    double volume = PI * radius * radius * height;
    return(volume);

}





//================================
//================================
//================================




int main(void){

    double radius = 13.2;
    
    print_useless_shit();
    
    compute_and_print_area(radius);

    double circ = circumference(radius);
    printf("The circumference is %.3f\n", circ);

    double cyl_vol = cylinder_volume(radius, 28.4982);
    printf("The cylinder volume is %.3g\n", cyl_vol);


    return(0);
}

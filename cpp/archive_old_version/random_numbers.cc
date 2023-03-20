#include <stdlib.h>
#include <iostream>
using namespace std;


int main ()
{
    int randint, seededrandint;
    int intseed;
    
    double randdouble, seededranddouble; 
    long seededrandlong;
    long longseed;

    // remain the same. Good for debugging.
    cout << "Not seeded random numbers: \n";
    randint = rand();
    randdouble = drand48();
    cout << randint << "\t" << randdouble ;
    cout << endl;



    cout << "Seeded random numbers: \n";

    intseed=23;
    longseed=61;
    
    // get "new" random numbers. To get the first ones you created, use seed 1.
    srand(intseed);
    srand48(longseed);

    seededrandint = rand ();
    seededranddouble = drand48();
    cout << seededrandint << "\t" << seededranddouble ;
    cout << endl;


    return 0;
}

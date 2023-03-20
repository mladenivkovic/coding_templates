#include <iostream>
#include <string>
using namespace std;

int main()
{
    int n = 10;

    while (n>0) 
    {
        cout << n << ", ";
        --n;
    }

    cout << "liftoff!\n";



    // do while: executes first, then checks condition.
    n = -1;
    do
    {
        cout << n << ", ";
        --n;
    } while (n>0); 
    cout << endl;
    cout << "\n";



    for (int n=10; n>0; n--) 
    {
        cout << n << ", ";
    }
    cout << "liftoff!\n";
    cout << "\n";


    // for loops can handle two variables:
    int i;
    for ( n=0, i=10 ; n!=i ; ++n, --i )
    {
        cout << "n = " << n;
        cout << ", i = " << i;
        cout << endl;
    }
    cout << "\n";


    // for loop over range
    string str {"mystring"};

    cout << "Letters of string str: ";
    for ( char c : str) // also possible: auto c
    {
        cout << "'" << c << "' ";
    }
    cout << "\n";
    cout << "\n";




    // break: leaves a loop, even if the condition for its end is not fulfilled.
    for (int n=10; n>0; n--)
    {
        cout << n << ", ";
        if (n==3)
        {
            cout << "countdown aborted!";
            break;
        }
    }
    cout << "\n";
    cout << "\n";




    // continue: skip the rest of the loop in the current iteration
    // here: skip nr 5
    for (int n=10; n>0; n--) 
    {
        if (n==5) continue;
        cout << n << ", ";
    }
    cout << "liftoff!\n";
    cout << "\n";




    // switch
    int x = 2;
    switch (x) 
    {
        case 1:
            cout << "x is 1";
            break;
        case 2:
            cout << "x is 2";
            break;
        default:
            cout << "value of x unknown";
    }
    cout << "\n";


    // multiple cases with same result:
    switch (x) 
    {
        case 1:
        case 2:
        case 3:
            cout << "x is 1, 2 or 3";
            break;
        default:
            cout << "x is not 1, 2 nor 3";
    }
    cout << "\n";

return 0;
}

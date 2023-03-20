#include <iostream>
using namespace std;

int main()
{
    int x;

    x=20;

    if (x > 0)
    {
        cout << "x is positive";
        x-=1;
        cout << endl; 
    }
    else if (x < 0)
    {
        cout << "x is negative";
        x+=1;
        cout << endl; 
    }
    else
    {
        cout << "x is 0";
        cout << endl; 
    }


    return 0;
}

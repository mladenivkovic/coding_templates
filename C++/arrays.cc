#include <iostream>

using namespace std;


int main ()
{

    int intarray [5] = {7, 8, 9, 10, 11};
    int otherintarray [] = {4, 5, 6};
    int thirdintarray [5] = {14, 15, 16};


    cout << "Printing arrays.\n";

    for (int n = 0; n<5; n++){
        cout << intarray[n] << " "; 
    }
    cout << endl;

    for (int n = 0; n<5; n++){
        cout << otherintarray[n] << " "; 
    }
    cout << endl;

    for (int n = 0; n<5; n++){
        cout << thirdintarray[n] << " "; }
    cout << endl;
    
    
    
    cout << "\n";
    cout << "Accessing elements.\n";
    
    int twodimarray [5][3] = {1};    
    for (int i=0; i<3; i++){
        for (int j=0; j<5;j++) {
            cout << twodimarray[j][i] << " ";
        }
        cout << "\n";
    }
    
    
    
    
    
    return 0;

}

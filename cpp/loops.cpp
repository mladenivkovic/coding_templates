#include <iostream> // IO library
#include <string>   // string type
#include <iomanip>  // IO formatting




int main(){

  // For loops
  // =======================
  
  for (int i = 0; i < 5; i++){
    // set width of printed i to 2
    std::cout << std::setw(2) << i;
  }
  std::cout << std::endl;

  // i is not defined outside of FOR LOOP SCOPE; This causes an error at compile time
  // std::cout << "i after loop = " << i << std::endl;

  // Note that ++i doesn't change anything compared to i++
  for (int i = 0; i < 5; ++i){
    // set width of printed i to 2
    std::cout << std::setw(2) << i;
  }
  std::cout << std::endl;



  // MULTIPLE CONDITIONS can be done, separated by a comma
  for (int i = 0, j = 1; i < 5 && j < 8; i++, j+=2){
    // set width of printed i to 2
    std::cout << i << "/" << j << " ";
  }
  std::cout << std::endl;


  return 0;
}



#include <iostream> // IO library
#include <string>   // string type
#include <iomanip>  // manipulations, e.g. formatting

// using namespace std; // skip this for now to explicitly trace where you get things from

/**
 * Make a template to write a printing function for any variable type.
 */
template<typename T>
void my_formatted_print(const T & val_to_print) {
    std::cout << "Left fill:\n" << std::left << std::setfill('*')
              << std::setw(12) << val_to_print  << '\n';
              // << std::setw(12) << std::hex << std::showbase << 42 << '\n';
}



int main(){

  // Using the iostream
  // =================================
  // int a = -1;
  // std::cout << "\nEnter some integer: ";
  // std::cin >> a;
  // // NOTE: `std::cin >> int i` works (compiles) too, but `i` will not be in scope
  // // in the following lines.
  // std::cout << "\nyou gave me " << a << std::endl;
  //
  //
  // // \a should be an "alert", but apparently it doesn't do much nowadays.
  // std::cerr << "\aThis is an error written to stderr." << std::endl;
  // // check that this works when running the program e.g. by redirecting stderr
  // // to /dev/null : `./stdio.o 2> /dev/null `
  

  // Some Formatting
  // =================================

  int i = 7;
  float f = 3.14159;

  std::cout << "\n\nINTEGERS\n\n" << std::endl;
  std::cout << "Setting width to 24:";
  std::cout << std::setw(24) << i << std::endl;
  std::cout << "Setting width to 24, and filling up with '-':" << std::setfill('-') << std::setw(24) << i << std::endl;

  std::cout << "Aligning left, no width specified:" << std::left << i << std::endl;
  std::cout << "Aligning right, no width specified:" << std::right << i << std::endl;
  std::cout << "Aligning left with width=12:" << std::left << std::setw(12) <<  i << std::endl;
  std::cout << "Aligning right with width=12:" << std::right << std::setw(12) <<  i << std::endl;
  std::cout << "Setting 'fill' to ' '" << std::setfill(' ') << std::endl;
  std::cout << "Aligning left with width=12:" << std::left << std::setw(12) <<  i << std::endl;
  std::cout << "Aligning right with width=12:" << std::right << std::setw(12) <<  i << std::endl;

  std::cout << "Setting 'fill' to '-'" << std::setfill('-') << std::endl;
  std::cout << "Using setw() only once for two variables:" << std::setw(12) <<  i << i << std::endl;
  std::cout << "Changing setw() mid-line" << std::setw(12) <<  i << std::setw(3) << i << std::endl;



  std::cout << "\n\nFLOATS\n\n" << std::endl;
  std::cout << "Setting width to 24:";
  std::cout << std::setw(24) << f << std::endl;
  std::cout << "Setting width to 24, and filling up with '-':" << std::setfill('-') << std::setw(24) << f << std::endl;

  std::cout << "Aligning left, no width specified:" << std::left << f << std::endl;
  std::cout << "Aligning right, no width specified:" << std::right << f << std::endl;
  std::cout << "Aligning left with width=12:" << std::left << std::setw(12) << f << std::endl;
  std::cout << "Aligning right with width=12:" << std::right << std::setw(12) << f << std::endl;
  std::cout << "Setting 'fill' to ' '" << std::setfill(' ') << std::endl;
  std::cout << "Aligning left with width=12:" << std::left << std::setw(12) << f << std::endl;
  std::cout << "Aligning right with width=12:" << std::right << std::setw(12) << f << std::endl;

  std::cout << "Setting 'fill' to '-'" << std::setfill('-') << std::endl;
  std::cout << "Using setw() only once for two variables:" << std::setw(12) << f << f << std::endl;
  std::cout << "Changing setw() mid-line" << std::setw(12) << f << std::setw(8) << f << std::endl;

  std::cout << "Setting precision to 4: " << std::setprecision(4) << f << std::endl;
  std::cout << "Setting precision to 12: " << std::setw(15) << std::setprecision(12) << f <<  std::setw(15) << f << std::endl;



  return 0;
}



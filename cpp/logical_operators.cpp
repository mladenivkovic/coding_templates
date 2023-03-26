#include <cstdlib>
#include <iostream> // IO library
#include <string>   // string type


bool print_warning_return_true(std::string s){
  std::cerr << "\t" << s <<" print_warning_return_true executed!" << std::endl;;
  return true;
}

bool print_warning_return_false(std::string s){
  std::cerr << "\t" << s <<" print_warning_return_true executed!" << std::endl;;
  return false;
}



int main(){

  bool t = true;
  bool f = false;

  std::cout << "t = true  = " << t << std::endl;
  std::cout << "f = false = " << f << std::endl;

  std::cout << std::endl;
  std::cout << "t && t = " << (t && t) << std::endl;
  std::cout << "t && f = " << (t && f) << std::endl;
  std::cout << "f && t = " << (f && t) << std::endl;
  std::cout << "f && l = " << (f && f) << std::endl;

  std::cout << std::endl;
  std::cout << "t || t = " << (t || t) << std::endl;
  std::cout << "t || f = " << (t || f) << std::endl;
  std::cout << "f || t = " << (f || t) << std::endl;
  std::cout << "f || f = " << (f || f) << std::endl;

  // Some peculiarities
  if (t && print_warning_return_false("Check 1.1")) std::cout << "Check 1.1: Second expression executed!" << std::endl;
  if (t && print_warning_return_true("Check 1.2"))  std::cout << "Check 1.2: Second expression executed!" << std::endl;
  if (f && print_warning_return_false("Check 1.3")) std::cout << "Check 1.3: Second expression executed!" << std::endl;
  if (f && print_warning_return_true("Check 1.4"))   std::cout << "Check 1.4: Second expression executed!" << std::endl;

  if (t || print_warning_return_false("Check 2.1")) std::cout << "Check 2.1: Second expression executed!" << std::endl;
  if (t || print_warning_return_true("Check 2.2"))  std::cout << "Check 2.2: Second expression executed!" << std::endl;
  if (f || print_warning_return_false("Check 2.3")) std::cout << "Check 2.3: Second expression executed!" << std::endl;
  if (f || print_warning_return_true("Check 2.4"))  std::cout << "Check 2.4: Second expression executed!" << std::endl;

  return 0;
}



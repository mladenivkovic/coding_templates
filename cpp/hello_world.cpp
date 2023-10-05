#include <iostream> // IO library
#include <string>   // string type

// using namespace std; // skip this for now to explicitly trace where you get
// things from

void pr_hello_world() { std::cout << "Hello world!" << std::endl; }

void pr_message(std::string s = "Default string") {
  std::cout << s << std::endl;
}

void pr_messge_with_printf(std::string s = "Default printf string") {

  printf("%s\n", s.c_str());
}

int main() {

  pr_hello_world();
  pr_message();
  pr_message("Hello World without default argument!");
  pr_messge_with_printf("Hello world with printf!");

  return 0;
}

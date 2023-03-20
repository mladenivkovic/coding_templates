#include <iostream> // IO library
#include <string>   // string type

// using namespace std; // skip this for now to explicitly trace where you get things from

int main(){

  std::cout << "alert:                " << "\\a   " << " -- " << "'\a'" << std::endl; // should make a sound?
  std::cout << "backslash:            " << "\\\\  " << " -- " << "'\\'" << std::endl;
  std::cout << "backspace:            " << "\\b   " << " -- " << "'\b'" << std::endl;
  std::cout << "carriage return:      " << "\\r   " << " -- " << "'\r' additional text" << std::endl;
  std::cout << "double quote:         " << "\\\"  " << " -- " << "'\"'" << std::endl;
  std::cout << "formfeed:             " << "\\f   " << " -- " << "'\f'" << std::endl; // cause a printer to eject paper to the top of the next page, or a video terminal to clear the screen.
  std::cout << "newline:              " << "\\n   " << " -- " << "'\n'" << std::endl;
  std::cout << "Null character:       " << "\\0   " << " -- " << "'\0'" << std::endl;
  std::cout << "single quote:         " << "\\\'  " << " -- " << "'\''" << std::endl;
  std::cout << "tab:                  " << "\\t   " << " -- " << "'\t'" << std::endl;
  std::cout << "vertical tab:         " << "\\v   " << " -- " << "'\v'" << std::endl;
  std::cout << "octal ASCII A:        " << "\\101 " << " -- " << "'\101'" << std::endl;
  std::cout << "hexadecimal ASCII A:  " << "\\x041" << " -- " << "'\x041'" << std::endl;

  return 0;
}



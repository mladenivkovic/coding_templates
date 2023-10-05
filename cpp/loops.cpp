#include <iomanip>  // IO formatting
#include <iostream> // IO library

int main() {

  // For loops
  // =======================

  std::cout << "for loop" << std::endl;
  std::cout << "===============================" << std::endl;

  for (int i = 0; i < 5; i++) {
    // set width of printed i to 2
    std::cout << std::setw(2) << i;
  }
  std::cout << std::endl;

  // i is not defined outside of FOR LOOP SCOPE; This causes an error at compile
  // time std::cout << "i after loop = " << i << std::endl;

  // Note that ++i doesn't change anything compared to i++
  for (int i = 0; i < 5; ++i) {
    // set width of printed i to 2
    std::cout << std::setw(2) << i;
  }
  std::cout << std::endl;

  // MULTIPLE CONDITIONS can be done, separated by a comma
  for (int i = 0, j = 1; i < 5 && j < 8; i++, j += 2) {
    // set width of printed i to 2
    std::cout << i << "/" << j << " ";
  }
  std::cout << std::endl;

  std::cout << std::endl;
  std::cout << "while loop" << std::endl;
  std::cout << "===============================" << std::endl;

  int check = 0;
  while (check < 5) {
    std::cout << std::setw(2) << check;
    check++;
  }
  std::cout << std::endl;

  std::cout << std::endl;
  std::cout << "do while loop" << std::endl;
  std::cout << "===============================" << std::endl;

  check = 0;
  do {
    std::cout << std::setw(2) << check;
    check++;
  } while (check < 5);
  std::cout << std::endl;

  // do .. while will check for condition *after* statement is executed,
  // such as here:
  check = 5;
  do {
    std::cout << std::setw(2) << check;
    check++;
  } while (check < 5);
  std::cout << std::endl;

  std::cout << std::endl;
  std::cout << "continue & break" << std::endl;
  std::cout << "===============================" << std::endl;

  int i = 0;
  while (i < 21) {
    i++;
    // Print even numbers only
    if (i % 2 == 0) {
      std::cout << std::setw(3) << i;
    } else {
      // skip rest of the statements in loop iteration
      continue;
    }

    if (i > 13)
      break; // stop early
  }
  std::cout << std::endl;

  return 0;
}

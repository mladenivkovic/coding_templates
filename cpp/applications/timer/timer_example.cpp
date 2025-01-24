#include "timer.h"
#include <iostream>

void do_something(int iter = 10000) {

  // Should print out timing after function finishes
  timer::Timer tic = timer::Timer<timer::unit::ns>("internal");

  int print_interval = iter / 10;
  int result = 0;
  for (int i = 0; i < iter; i++) {
    result += 1;
    if (i % print_interval == 0) {
      std::cout << "Done " << (((float)i) / ((float)iter) * 100.) << "%\n";
    }
  }
  std::cout << "Result: " << result << "\n";
}

int main(void) {

  timer::Timer tic = timer::Timer<timer::unit::mus>("external");
  do_something(10000000);
  tic.end(); // Manually print out timing
}

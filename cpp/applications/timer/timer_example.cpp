#include "timer.h"
#include <iostream>



void do_something(int iter = 10000) {

  // Should print out timing after function finishes
  timer::timer tic = timer::timer("internal");

  int print_interval = iter / 10;
  int result = 0;
  for (int i = 0; i < iter; i++){
    result += 1;
    if (i % print_interval == 0) {
      std::cout << "Done " << (((float) i) / ((float) iter) * 100.) << "%\n";
    }
  }
  std::cout << "Result: " << result << "\n";
}


int main(void) {

  timer::timer tic = timer::timer("external");
  do_something();
  tic.end(); // Manually print out timing

}

// How to time execution time with cpp

#include <iostream>
#include <chrono>


void do_something(void){

  const int maxiter = 1000;

  int result = 0;
  for (int i = 0; i < maxiter; i++){
    result += i;
    if ((i % 100) == 0) {
      std::cout << "Completed " << (float)i / (float)maxiter * 100 << "%; result=" << result <<"\n";
    }
  }
}


int main(void) {

  auto start = std::chrono::high_resolution_clock::now();

  do_something();

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = duration_cast<std::chrono::microseconds>(stop - start);

  std::cout << "This took " << duration.count() << "ms" << std::endl;


  return 0;
}

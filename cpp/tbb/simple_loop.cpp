// https://www.intel.com/content/www/us/en/docs/onetbb/developer-guide-api-reference/2021-6/parallel-for.html


#include "../applications/timer/timer.h"
#include <iostream>

#include "oneapi/tbb.h"

using namespace oneapi::tbb;

size_t do_something(size_t iter = 10000) {
  size_t result = 0;
  for (size_t i = 0; i < iter; i++){
    result += i;
  }
  return result;
}


void SerialDoSomething( size_t n ) {
  timer::Timer t = timer::Timer("serial");
  size_t result;
  for( size_t i=0; i < n; ++i ){
    result=do_something(i);
  }
  std::cout << "Result:" << result << "\n";
}


class DoSomething {
public:
  void operator()( const blocked_range<size_t>& r ) const {
    for( size_t i=r.begin(); i!=r.end(); ++i ){
      do_something(i);
    }
    // std::cout << "parallel for " << r.begin() << "->" << r.end() << "\n";
  }
  // Constructor not necessarily needed if I don't use arrays as intel does in its example?
  // DoSomething(){}
};

void ParallelDoSomething(size_t n ) {
  timer::Timer t = timer::Timer("parallel");
  parallel_for(blocked_range<size_t>(0,n), DoSomething());
}


int main(void) {

  const long niter = 10000000000;
  SerialDoSomething(niter);
  ParallelDoSomething(niter);

  return 0;
}

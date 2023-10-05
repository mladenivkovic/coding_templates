#include <iostream> // IO library
// #include <string>   // string type

// using namespace std; // skip this for now to explicitly trace where you get
// things from

int main() {

  constexpr long double pi{std::numbers::pi_v<long double>};

  return 0;
}

#include <iostream> // IO library

// using namespace std; // skip this for now to explicitly trace where you get things from

// Note no assignment: no '=' between card_suit and '{}'
enum card_suit { clubs, diamonds, spades, hearts };

enum ages {aaron = 7, beth, charlie=42, dora};

// enumerators can be defined without a name
enum {anonymous_enum1, anonymous_enum2, anonymous_enum3=42};

int main(){

  std::cout << "Card Suits" << std::endl;
  std::cout << "clubs:    " << clubs << std::endl;
  std::cout << "diamonds: " << diamonds << std::endl;
  std::cout << "spades:   " << spades << std::endl;
  std::cout << "hearts:   " << hearts << std::endl;


  std::cout << std::endl;
  std::cout << "Ages" << std::endl;
  std::cout << "Aaron:     " << aaron << std::endl;
  std::cout << "Beth:      " << beth << std::endl;
  std::cout << "Charlie:   " << charlie << std::endl;
  std::cout << "Dora:      " << dora << std::endl;

  // Operations with enums
  // ============================

  // int from enum
  int i1 = aaron;

  // enum to enum
  card_suit new_card = clubs;

  int i2 = anonymous_enum2;

  return 0;
}



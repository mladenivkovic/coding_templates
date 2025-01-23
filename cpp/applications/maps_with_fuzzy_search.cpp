#include <cassert>
#include <iostream>
#include <map>

// Make a map, and allow for a fuzzy search of its nodes.
// I.e. we use a class containing a double as an identifier.
// We later want to recover that element, but with a tolerance
// in that identifier.
// To do so, we need 2 objects: 1 with strict comparisons, and another
// where the tolerance is enabled. We use the one with the strict comparisons
// to add elements to the map, while we use the one with a fuzzy
// comparison to fish them out later.

class myIDKey {
public:
  myIDKey(double x);
  myIDKey() {};
  double getX() const;

private:
  double _x;
};

myIDKey::myIDKey(double x) { _x = x; }

double myIDKey::getX() const { return _x; }

bool operator<(const myIDKey &lhs, const myIDKey &rhs) {
  // std::cout << "Called ID/ID <" << std::endl;
  return lhs.getX() < rhs.getX();
}

bool operator==(const myIDKey &lhs, const myIDKey &rhs) {
  // std::cout << "Called ID/ID ==" << std::endl;
  return lhs.getX() == rhs.getX();
}

class mySearchKey : public myIDKey {
public:
  mySearchKey(double x, double tolerance = 0.5);
  mySearchKey() {};
  double getTolerance() const;

private:
  double _tolerance;
};

mySearchKey::mySearchKey(double x, double tolerance) : myIDKey::myIDKey(x) {
  _tolerance = tolerance;
}

double mySearchKey::getTolerance() const { return _tolerance; }

bool operator<(const mySearchKey &lhs, const mySearchKey &rhs) {
  // std::cout << "Called Search/Search <" << std::endl;
  return lhs.getX() < rhs.getX();
}

bool operator==(const mySearchKey &lhs, const mySearchKey &rhs) {
  // std::cout << "Called Search/Search ==" << std::endl;
  return lhs.getX() == rhs.getX();
}

bool operator<(const mySearchKey &lhs, const myIDKey &rhs) {
  // std::cout << "Called Search/ID <" << std::endl;
  return (lhs.getX() - rhs.getX()) < -lhs.getTolerance();
}

bool operator==(const mySearchKey &lhs, const myIDKey &rhs) {
  // std::cout << "Called Search/ID ==" << std::endl;
  return std::abs(rhs.getX() - lhs.getX()) < lhs.getTolerance();
}

bool operator<(const myIDKey &lhs, const mySearchKey &rhs) {
  // std::cout << "Called Search/ID <" << std::endl;
  return (lhs.getX() - rhs.getX()) < -rhs.getTolerance();
}

bool operator==(const myIDKey &lhs, const mySearchKey &rhs) {
  // std::cout << "Called Search/ID ==" << std::endl;
  return std::abs(rhs.getX() - lhs.getX()) < rhs.getTolerance();
}

void testTruthTableSearchAndIDKeys() {

  // create a key and assign it a value in the map
  myIDKey a = myIDKey(1.);
  myIDKey b = myIDKey(2.);

  // std::cout << "a < b:" << (a < b) << "(1) a==b:" << (a == b) <<
  // "(0) a==a:" <<  (a == a) << "(1) b==b:" << (b == b) << "(1)" << std::endl;

  assert(a < b);
  assert(not(a == b));
  assert(not(b == a));
  assert(not(b < a));

  const double tol = 0.5;
  mySearchKey A = mySearchKey(1., tol);
  mySearchKey B = mySearchKey(2., tol);

  // std::cout << "A < B:" << (A < B) << "(1) A==B:" << (A == B) <<
  // "(0) A==A:" <<  (A == A) << "(1) B==B:" << (B == B) << "(1)" << std::endl;

  assert(A < B);
  assert(not(A == B));
  assert(not(B == A));
  assert(not(B < A));

  // cross-comparisons
  // std::cout << "a < A:" << (a < A) << "(0) A < a:" << (A < a) << "(0) a==A:"
  //   << (a == A) << "(1) A == a:" <<  (A == a) << "(1)" << std::endl;
  // std::cout << "a < B:" << (a < B) << "(1) B < a:" << (B < a) << "(0) a==B:"
  //   << (a == B) << "(0) B == a:" <<  (B == a) << "(0)" << std::endl;
  // std::cout << "b < A:" << (b < A) << "(0) A < b:" << (A < b) << "(1) b==A:"
  //   << (b == A) << "(0) A == b:" <<  (A == b) << "(0)" << std::endl;
  // std::cout << "b < B:" << (b < B) << "(0) B < b:" << (B < b) << "(0) b==B:"
  //   << (b == B) << "(1) B == b:" <<  (B == b) << "(1)" << std::endl;

  assert(not(a < A));
  assert(not(A < a));
  assert(A == a);
  assert(a == A);

  assert(a < B);
  assert(not(B < a));
  assert(not(B == a));
  assert(not(a == B));

  assert(not(b < A));
  assert(A < b);
  assert(not(A == b));
  assert(not(b == A));

  assert(not(b < B));
  assert(not(B < b));
  assert(b == B);
  assert(B == b);
}

// --------------------------------------------------

int main() {

  // tell the map to use std::less<> as comparator instead
  // of std::less<key>. std::less<> invokes the `<` operator.
  // We want that to be able to use our bespoke `<` operator
  // for different search and ID keys.
  std::map<myIDKey, double, std::less<>> myMap;

  // unit test for operator implementation
  // testTruthTableSearchAndIDKeys();

  myIDKey keyA(1.);
  myMap[keyA] = 1.;

  myIDKey keyB(2.);
  myMap[keyB] = 2.;

  mySearchKey searchKeyA = mySearchKey(1., 0.5);
  mySearchKey searchKeyA2 = mySearchKey(1.2, 0.5);

  auto search1 = myMap.find(searchKeyA);
  if (search1 != myMap.end()) {
    std::cout << "Search 1 successful" << std::endl;
    assert(search1->first.getX() == keyA.getX());
  } else {
    std::cout << "Search 1 failed" << std::endl;
  }

  auto search2 = myMap.find(searchKeyA2);
  if (search2 != myMap.end()) {
    std::cout << "Search 2 successful" << std::endl;
    assert(search2->first.getX() == keyA.getX());
  } else {
    std::cout << "Search 2 failed" << std::endl;
  }

  // Check that we don't find B
  assert(myMap.count(keyA) == 1);
  assert(myMap.count(searchKeyA) == 1);
  assert(myMap.count(searchKeyA2) == 1);

  std::cout << "Done." << std::endl;
  return 0;
}

#pragma once

namespace singleton {

class S {
public:
  static S &getInstance() {
    static S instance; // Guaranteed to be destroyed.
                       // Instantiated on first use.
    return instance;
  }

private:
  int _someVar;
  int _someOtherVar;

  // Constructor? (the {} brackets) are needed here, even if you leave it empty.
  S() {
    _someVar = -123;
    _someOtherVar = -456;
  }

public:
  // C++ 11 way of doing things. Explicitly delete the methods
  // we don't want.
  S(S const &) = delete;
  void operator=(S const &) = delete;

  // Note: Scott Meyers mentions in his Effective Modern
  //       C++ book, that deleted functions should generally
  //       be public as it results in better error messages
  //       due to the compilers behavior to check accessibility
  //       before deleted status

  void setVar(int x);
  void setOtherVar(int x);
  int getVar() const;
  int getOtherVar() const;
};

} // namespace singleton

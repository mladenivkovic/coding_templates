#include <iostream>


// These three macros dump the variable value and its name.
#define DUMPSTR_WNAME(os, name, a) \
    do { (os) << (name) << " has value " << (a) << std::endl; } while(false)

#define DUMPSTR(os, a) DUMPSTR_WNAME((os), #a, (a))
#define DUMP(a)        DUMPSTR_WNAME(std::cout, #a, (a))



class myClass {

public:
  int a;

};


int main(void){

  std::cout << "Hello world!" << std::endl;

  double a = 3.14156;

  DUMP(a);

  myClass obj = myClass();
  obj.a = 20;
  DUMP(obj.a);

  return 0;
}

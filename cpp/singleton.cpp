#include <iostream>


class S
{
    public:
        static S& getInstance()
        {
            static S    instance; // Guaranteed to be destroyed.
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
        S(S const&)               = delete;
        void operator=(S const&)  = delete;

        // Note: Scott Meyers mentions in his Effective Modern
        //       C++ book, that deleted functions should generally
        //       be public as it results in better error messages
        //       due to the compilers behavior to check accessibility
        //       before deleted status

        void setVar(int x){
          std::cout << "Setting someVar       at " << &_someVar << " with value=" << x  << std::endl;
          _someVar = x;
        }

        void setOtherVar(int x){
          std::cout << "Setting someVar       at " << &_someOtherVar  << " with value=" <<  x << std::endl;
          _someOtherVar = x;
        }

        int getVar() const{
          std::cout << "Fetching someVar      at " << &_someVar << " with value=" << _someVar <<  std::endl;
          return _someVar;
        }

        int getOtherVar() const{
          std::cout << "Fetching someOtherVar at " << &_someOtherVar << " with value=" << _someOtherVar <<  std::endl;
          return _someOtherVar;
        }
};

// This being a reference is crucial!
// static S& singleton = S::getInstance();
// This being a reference is crucial!
S& singleton = S::getInstance();


void foo(){

  singleton.getVar();
  singleton.getOtherVar();
  singleton.setVar(17);
  singleton.setOtherVar(18);
  singleton.getVar();
  singleton.getOtherVar();

}

int main(void){

  singleton.getVar();
  singleton.getOtherVar();
  singleton.setVar(3);
  singleton.setOtherVar(4);
  singleton.getVar();
  singleton.getOtherVar();

  foo();

  return 0;
}

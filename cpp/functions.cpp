#include <iostream> // IO library
#include <string>   // string type

/**
 * void function with no arguments
 */
void void_func_with_no_args(void) {
  std::cout << "void_func_with_no_args:                ";
  std::cout << "Hello world!\n";
}

/**
 * void function with single argument
 */
void void_func_with_single_arg(std::string s) {
  std::cout << "void_func_with_single_arg:             ";
  std::cout << s << "\n";
}

/**
 * void function with single argument and default value
 */
void void_func_with_single_arg_and_default(
    std::string s = "Default argument taken") {
  std::cout << "void_func_with_single_arg_and_default: ";
  std::cout << s << "\n";
}

/**
 * function returning an int intended to be passed as an
 * argument to a different function
 */
int int_func_to_be_passed_as_argument(int a) {
  std::cout << "Called function to be passed as argument with a = " << a
            << std::endl;
  return a + 1;
}

/**
 * function taking another function as the argument
 */
void void_func_calling_other_func(int f(int a), int a) {
  int new_a = f(a);
  std::cout << "After call: new_a = " << new_a << std::endl;
}

/**
 * void function intended to be passed as an
 * argument to a different function
 */
void void_func_to_be_passed_as_argument(void) {
  std::cout << "Called void function passed as argument" << std::endl;
}

/**
 * function taking another function as the argument
 */
void void_func_calling_other_void_func(void f(void)) {
  // You need to specify `void f(void)`; just `void f` is invalid
  f();
  std::cout << "Finished void_func_calling_other_void_func" << std::endl;
}

/**
 * Compute the average of an array.
 * Function is intended to be overloaded to work for doubles as well.
 */
int avg_arr_overloaded(const int arr[], int size) {
  int avg = 0;
  for (int i = 0; i < size; i++) {
    avg += arr[i];
  }
  avg = avg / size;
  return avg;
}

/**
 * Compute the average of an array.
 * Function is intended to be overloaded to work for integers as well.
 */
double avg_arr_overloaded(const double arr[], int size) {
  double avg = 0;
  for (int i = 0; i < size; i++) {
    avg += arr[i];
  }
  avg = avg / size;
  return avg;
}

int main() {

  void_func_with_no_args();
  void_func_with_single_arg("single argument passed");
  void_func_with_single_arg_and_default();
  void_func_with_single_arg_and_default("Passed argument");

  void_func_calling_other_func(int_func_to_be_passed_as_argument, 3);
  void_func_calling_other_void_func(void_func_to_be_passed_as_argument);

  const int size = 7;
  int intarr[size] = {2, 3, 4, 5, 6, 7, 8};
  double doublearr[size] = {2.1, 3.2, 4.3, 5.4, 6.5, 7.6, 8.7};

  // Note how the function names that take the int and the double arrays are the
  // same
  int int_avg = avg_arr_overloaded(intarr, size);
  std::cout << "Average of int array:    " << int_avg << std::endl;
  // Note how the function names that take the int and the double arrays are the
  // same
  double double_avg = avg_arr_overloaded(doublearr, size);
  std::cout << "Average of double array: " << double_avg << std::endl;

  return 0;
}

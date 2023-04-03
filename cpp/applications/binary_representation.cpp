#include <iostream> // IO library
#include <iomanip>

/**
 * Get bit representation of some variable as a string.
 * Optionally, add "prefix" string to add in front of the operation
 **/
template <typename T>
std::string bitwise_representation(T val){

  std::string result = "";

  // Go backwards: write most significant bits first
  for (size_t i = sizeof(T) * 8; i > 0; i--) {
    // compare whether the passed value has a 1 at the digit by
    // shifting a 1 to the corresponding digit and using a binary and comparison
    bool digit = val & (1 << (i - 1)); // use bool to get 1 or 0 only. Otherwise, you get 2**i as the result.
    result.append(std::to_string(digit));
  } 

  return result;
}


/**
 * Get bit representation of some variable as a string.
 * Optionally, add "prefix" string to add in front of the operation
 **/
template <typename T>
std::string get_bitwise_representation(T val, std::string prefix = ""){

  std::string result;

  // add prefix?
  if (prefix != ""){
    result = prefix;
    result.append(" ");
  }

  std::string binRep = bitwise_representation(val);
  result.append(binRep);

  return result;
}




/**
 * Write a binary operation `a X b = c` (and the result c thereof) represented
 * as binary numbers.
 * @param a: a
 * @param b: b
 * @param result: c
 * @param operation_symbol: symbol for X
 **/
template <typename T>
void write_operation(T a, T b, T result, std::string operation_symbol) {

  int width = 8 * sizeof(T) + 2 + operation_symbol.length();
  int width_decimal = 6;

  std::string a_str = get_bitwise_representation(a);
  std::string b_str = get_bitwise_representation(b, operation_symbol);
  std::string c_str = get_bitwise_representation(result);

  std::cout << std::setw(width) << a_str << " | " << std::setw(width_decimal) << std::to_string(a) << std::endl;
  std::cout << std::setw(width) << b_str << " | " << std::setw(width_decimal) << std::to_string(b) << std::endl;
  for (int i = 0; i < width+1; i++) std::cout << "-";
  std::cout << "|";
  for (int i = 0; i < width_decimal + 1; i++) std::cout << "-";
  std::cout << std::endl;
  std::cout << std::setw(width) << c_str << " | " << std::setw(width_decimal) << std::to_string(result) << std::endl;
  std::cout << std::endl;
}






int main(){

  
  // Use `char` here to limit number of bits to a sensible length for human eyes.
  char a, b;

  a = 10;
  b = 3;
  write_operation(a, b, static_cast<char>(a & b), "&");

  a = 10;
  std::cout << get_bitwise_representation(a,                     " a = ") << std::endl;
  std::cout << get_bitwise_representation(static_cast<char>(~a), "~a = ") << std::endl;

  return 0;
}



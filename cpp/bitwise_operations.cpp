#include <iomanip>
#include <iostream> // IO library

bool verbose = false;

/**
 * Get bit representation of an integer.
 * Optionally, add "prefix" string to add in front of the operation
 **/
template <typename T>
std::string bitwise_representation(T val, std::string prefix = "") {

  std::string result = prefix;
  result.append(" ");

  // Go backwards: write most significant bits first
  for (size_t i = sizeof(T) * 8; i > 0; i--) {
    // compare whether the passed value has a 1 at the digit by
    // shifting a 1 to the corresponding digit and using a binary and comparison
    bool digit = val & (1 << (i - 1)); // use bool to get 1 or 0 only.
                                       // Otherwise, you get 2**i as the result.
    result.append(std::to_string(digit));
    if (verbose)
      std::cout << "i=" << i - 1 << " digit=" << digit << " res=" << result
                << std::endl;
  }

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

  std::string a_str = bitwise_representation(a);
  std::string b_str = bitwise_representation(b, operation_symbol);
  std::string c_str = bitwise_representation(result);

  std::cout << std::setw(width) << a_str << " | " << std::setw(width_decimal)
            << std::to_string(a) << std::endl;
  std::cout << std::setw(width) << b_str << " | " << std::setw(width_decimal)
            << std::to_string(b) << std::endl;
  for (int i = 0; i < width + 1; i++)
    std::cout << "-";
  std::cout << "|";
  for (int i = 0; i < width_decimal + 1; i++)
    std::cout << "-";
  std::cout << std::endl;
  std::cout << std::setw(width) << c_str << " | " << std::setw(width_decimal)
            << std::to_string(result) << std::endl;
  std::cout << std::endl;
}

int main() {

  // Use `char` here to limit number of bits to a sensible length for human
  // eyes.
  char a, b;

  a = 10;
  b = 3;
  write_operation(a, b, static_cast<char>(a & b), "&");

  a = 10;
  b = 3;
  write_operation(a, b, static_cast<char>(a | b), "|");

  a = 10;
  b = 3;
  write_operation(a, b, static_cast<char>(a ^ b), "^");

  a = 10;
  std::cout << bitwise_representation(a, " a = ") << std::endl;
  std::cout << bitwise_representation(static_cast<char>(!a), "!a = ");
  std::cout << " Note that this is not a bitwise operation. This is a logical "
               "operation."
            << std::endl;
  std::cout << bitwise_representation(static_cast<char>(~a), "~a = ")
            << std::endl;

  std::cout << std::endl;
  std::cout << bitwise_representation(static_cast<char>(1), "1 =       ")
            << std::endl;
  std::cout << bitwise_representation(static_cast<char>(1 << 3), "1 << 3 =  ")
            << std::endl;
  std::cout << std::endl;
  std::cout << bitwise_representation(static_cast<char>(23), "23 =      ")
            << std::endl;
  std::cout << bitwise_representation(static_cast<char>(23 << 0), "23 << 0 = ")
            << std::endl;
  std::cout << bitwise_representation(static_cast<char>(23 << 1), "23 << 1 = ")
            << std::endl;
  std::cout << bitwise_representation(static_cast<char>(23 << 2), "23 << 2 = ")
            << std::endl;
  std::cout << bitwise_representation(static_cast<char>(23 << 5), "23 << 5 = ")
            << std::endl;
  std::cout << std::endl;
  std::cout << bitwise_representation(static_cast<char>(23), "23 =      ")
            << std::endl;
  std::cout << bitwise_representation(static_cast<char>(23 >> 0), "23 >> 0 = ")
            << std::endl;
  std::cout << bitwise_representation(static_cast<char>(23 >> 1), "23 >> 1 = ")
            << std::endl;
  std::cout << bitwise_representation(static_cast<char>(23 >> 2), "23 >> 2 = ")
            << std::endl;

  return 0;
}

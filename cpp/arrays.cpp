#include <iostream> // IO library
#include <string>

/**
 * Print an array element by element.
 * Optionally pass a message as a string as well.
 */
template <typename T>
void print_array(T array[], int size, std::string s = ""){

  std::cout << s << " ";
  for (int i = 0; i < size; i++){
    std::cout << array[i] << " ";
  }
  std::cout << std::endl;
}


/**
 * "Allocate" an array in this function, and access it
 * from outside the scope of this function.
 */
int* get_array_allocd_in_function(void){
  const int size = 6;
  int * array = new int[size];
  for (int i = 0; i < size; i++) array[i] = i+6;
  return array;
}


int main(){


  // Initialization
  // ===========================

  int array1[3] = {3, 4, 5};
  print_array(array1, 3, "array1:");

  int array2[] = {7, 8, 9, 10};
  print_array(array2, 4, "array2:");

  int array3[2][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  print_array(array3[0], 4, "array3 line 0:");
  print_array(array3[1], 4, "array3 line 1:");

  // "allocate" memory manually
  const int size = 5;
  int * array4 = new int[size];
  for (int i = 0; i < size; i++) array4[i] = i+3;
  print_array(array4, size, "array4: created with `new`:");
  delete[] array4;

  // you can access the memory outside the scope of the function now.
  int* array5 = get_array_allocd_in_function();
  print_array(array5, 6, "array alloc'd in function:");
  delete[] array5;

  return 0;
}



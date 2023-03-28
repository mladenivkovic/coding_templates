#include <iostream> // IO library
#include <vector>


/**
 * print all elements of a vector in a line.
 * Optionally, add a message.
 **/
template <typename T>
void print_vector(std::vector<T> v, std::string s = ""){
  std::cout << s;
  for (T element: v) std::cout << " " << element ;
  std::cout << std::endl;
}

void empty_line(){
  std::cout << std::endl;
}


int main(){

  // Initialization
  // =====================

  // Definition with number of elements
  std::vector <int> intvector1 (5);
  print_vector(intvector1, "Definition with number of elements:");

  // Definition with number of elements
  std::vector <int> intvector2;
  print_vector(intvector2, "Definition without elements:");

  // Initializer list
  std::vector <int> intvector3 = {1, 2, 3, 4, 5};
  print_vector(intvector3, "Initializer list:");

  // Initializer list
  std::vector <int> intvector4 {6, 7, 8, 9, 10};
  print_vector(intvector4, "Initializer list v2:");

  // Uniform initialization
  std::vector <int> intvector5 (5, 11);
  print_vector(intvector5, "Uniform initialization:");

  empty_line();




  // Loops
  // =====================
  std::cout << "Loop over all elements: ";
  for (int i : intvector4) {
    std::cout << i << " ";
  }
  std::cout << std::endl;

  // Loop using iterators.
  // For more on iterators, see below.
  std::cout << "Loop over all elements using iterators: ";
  for (
    std::vector <int>:: iterator it4 = intvector4.begin();
    it4 != intvector4.end();
    it4++
      ){
    std::cout << *it4 << " ";
  }
  std::cout << std::endl;


  empty_line();




  // Adding Elements
  // =====================

  print_vector(intvector5, "Before push_back:");
  // adds element to the end
  intvector5.push_back(13);
  intvector5.push_back(14);
  print_vector(intvector5, "After  push_back:");

  empty_line();




  // Accessing Elements
  // =====================
  
  // use vec.at(index) or vec[index]

  std::cout << "Accessing elements: " << std::endl;
  for (int i = 0; i < 5; i++) 
    std::cout << "\t Index" << i << ": " << intvector3.at(i)  << " or " << intvector3[i] << std::endl;

  // differences when accessing outside of range/bounds
  try {
    int a1 = intvector3[5];
    std::cout << "out of bounds/range for access vector[index] not detected; val="<< a1 << std::endl;
  } catch (std::out_of_range &) {
    std::cout << "out of bounds/range for access vector[index] detected"<< std::endl;
  }
  try {
    int a2 = intvector3.at(5);
    std::cout << "out of bounds/range for access vector.at(index) not detected; val="<< a2 << std::endl;
  } catch (std::out_of_range &) {
    std::cout << "out of bounds/range for access vector.at(index) detected" << std::endl;
  }

  empty_line();




  // Deleting Elements
  // =====================

  print_vector(intvector3, "Before vector.pop_back (deleting elements):");
  // Note that vec.pop_back() doesn't return the value.
  intvector3.pop_back();
  print_vector(intvector3, "After  vector.pop_back (deleting elements):");

  // Delete all elements with vector.clear()
  print_vector(intvector4, "Before vector.clear():");
  intvector4.clear();
  print_vector(intvector4, "After  vector.clear():");

  empty_line();




  // Other Functions
  // =====================

  print_vector(intvector3, "Vector used for functions below:");

  std::cout << "\tvec.front()    " << intvector3.front()    << " // first element of the vector" << std::endl;
  std::cout << "\tvec.back()     " << intvector3.back()     << " // last element of the vector" << std::endl;
  std::cout << "\tvec.empty()    " << intvector3.empty()    << " // is vector empty?" << std::endl;
  std::cout << "\tvec.size()     " << intvector3.size()     << " // number of elements present in vector" << std::endl;
  std::cout << "\tvec.capacity() " << intvector3.capacity() << " // overall size of vector" << std::endl;

  std::cout << "Using these functions on an empty vector: " << std::endl;
  std::vector <int> emptyVec;
  // These two cause segfaults
  // std::cout << "vec.front()    " << emptyVec.front()    << " // first element of the vector" << std::endl;
  // std::cout << "vec.back()     " << emptyVec.back()     << " // last element of the vector" << std::endl;
  std::cout << "\tvec.empty()    " << emptyVec.empty()    << " // is vector empty?" << std::endl;
  std::cout << "\tvec.size()     " << emptyVec.size()     << " // number of elements present in vector" << std::endl;
  std::cout << "\tvec.capacity() " << emptyVec.capacity() << " // overall size of vector" << std::endl;


  empty_line();




  // Iterators
  // =====================

  // Basically pointers to vector elements.

  std::vector<int> myVec = {1, 2, 3, 4, 5};
  std::vector<int>::iterator iter;

  // Initializing an iterator.
  // iter points to myVec[0]
  iter = myVec.begin();

  std::cout << "Iterators" << std::endl;
  int *myPointer = &myVec[0];
  std::cout << "*iter=" << *iter << "; iter[0]=" << iter[0] << "; iter[3]=" << iter[3] << std::endl;
  std::cout << "int *p = &myVec works too: *p=" << *myPointer << "; *(p+3)=" << *(myPointer + 3) << std::endl;

  // Play with addresses
  int i = 0;
  std::cout << "Loop over all elements using iterators:\n";
  for (std::vector <int>:: iterator myIter = myVec.begin(); myIter != myVec.end(); myIter++){
    std::cout << "\ti=" << i;
    std::cout << "\t*myIter=" << *myIter;
    std::cout << "\t&(*myIter)=" << &(*myIter);
    // std::cout << "\tmyIter=" << myIter; // This doesn't work; iterators don't have a printable value!
    std::cout << std::endl;
    i++;
  }

  empty_line();






  return 0;
}



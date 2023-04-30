#include <iostream> // IO library
#include <string>   // string type
#include <list>


/**
 * print all elements of a list in a line.
 * Optionally, add a message.
 **/
template <typename T>
void print_list(std::list<T> list, std::string s=""){
  std::cout << s;
  for (T element: list) std::cout << " " << element ;
  std::cout << std::endl;
}

void empty_line(){
  std::cout << std::endl;
}


int main(){


  // Declaration
  std::list<int> intlist1;
  print_list(intlist1, "Uninitialized list:");

  // Initialization
  std::list<int> intlist2 = {1, 2, 3, 4, 5};
  print_list(intlist2, "Initialized list:");

  // Add element to front
  intlist2.push_front(-1);
  print_list(intlist2, "Added element in front:");

  // Add element to back
  intlist2.push_back(-2);
  print_list(intlist2, "Added element in back:");


  // Accessing elements
  std::cout << "First element:" << intlist2.front() << std::endl;
  std::cout << "Last element:" << intlist2.back() << std::endl;

  // Deleting elements
  intlist2.pop_front();
  print_list(intlist2, "Removed element in front:");

  // Add element to back
  intlist2.pop_back();
  print_list(intlist2, "Removed element in back:");

  // Deleting elements via iterators
  // WARNING: Don't try to erase stuff while you're iterating over the list.
  std::list<int>::iterator index_to_erase;
  for (std::list<int>::iterator i=intlist2.begin(); i!=intlist2.end(); i++){
    if (*i == 3) index_to_erase = i;
  }
  intlist2.erase(index_to_erase);
  print_list(intlist2, "Removed element with value 3:");


  // Deleting elements from starting iterator to end iterator
  // WARNING: Don't try to erase stuff while you're iterating over the list.
  std::list<int>::iterator start_erase;
  std::list<int>::iterator stop_erase;
  for (std::list<int>::iterator i=intlist2.begin(); i!=intlist2.end(); i++){
    if (*i == 2) start_erase = i;
    if (*i == 5) stop_erase = i;
  }

  intlist2.erase(start_erase, stop_erase);
  print_list(intlist2, "Removed all elements starting with element of value 2, stoppinng with element with value 4:");

  intlist2 = {1, 2, 3, 4, 5};
  print_list(intlist2, "Restored intlist2 to:");


  // Reverse order of elements
  intlist2.reverse();
  print_list(intlist2, "Reversed:");


  // Reverse order of elements
  intlist2.sort();
  print_list(intlist2, "Sorted:");


  std::list <int> intlist3 = { 3, 4, 4, 5, 4, 5, 5, 7};
  print_list(intlist3, "New list to work with:        ");

  // Remove consecutive duplicates
  intlist3.unique();
  print_list(intlist3, "Removed consecutive duplicates");

  // Get element count
  std::cout << "list.size(): " << intlist3.size() << std::endl;
  std::cout << "Is empty? " << intlist3.empty() << std::endl;

  // remove all values from list
  intlist3.clear();
  print_list(intlist3, "List after clear:");
  std::cout << "Is empty? " << intlist3.empty() << std::endl;

  // Merging
  std::cout << "Using two unsorted lists:" << std::endl;
  std::list<int> l1 = {1, 5, 3, 4};
  std::list<int> l2 = {4, 3, 2, 1};
  print_list(l1, "\tBefore merge L1 = ");
  print_list(l2, "\tBefore merge L2 = ");

  l1.merge(l2);

  print_list(l1, "\tAfter merge L1 = ");
  print_list(l2, "\tAfter merge L2 = ");

  std::cout << "Using two sorted lists:" << std::endl;
  std::list<int> l3 = {1, 2, 3, 4};
  std::list<int> l4 = {3, 4, 5, 6};
  print_list(l3, "\tBefore merge L3 = ");
  print_list(l4, "\tBefore merge L4 = ");

  l3.merge(l4);

  print_list(l3, "\tAfter merge L3 = ");
  print_list(l4, "\tAfter merge L4 = ");


  // TODO: Iterators


  return 0;
}

#include <iostream> // IO library
#include <list>
#include <string> // string type
#include <vector>

// ===========================================
// Remove elements from a list while looping
// over it. Simultaneously copy elements from
// a second list to a third resulting list.
// ===========================================

/**
 * print all elements of a list in a line.
 * Optionally, add a message.
 **/
template <typename T> void print_list(std::list<T> list, std::string s = "") {
  // void print_list(std::list<T> list, std::string s=""){
  std::cout << s;
  for (T element : list)
    std::cout << " " << element;
  std::cout << std::endl;
}

template <typename T> void print_list(std::vector<T> list, std::string s = "") {
  // void print_list(std::list<T> list, std::string s=""){
  std::cout << s;
  for (T element : list)
    std::cout << " " << element;
  std::cout << std::endl;
}

/**
 * Remove elements from a list while looping over it.
 * Simultaneously copy elements from a second vector to
 * a third resulting vector. Also store which elements have
 * been removed.
 *
 * Remove elements if their modulo with provided integer
 * parameter `divisor` is zero.
 *
 * This version leads to wrong results if called more than
 * once, since the `extra_data` doesn't get updated alongside
 * with the `list_to_purge`.
 */
template <typename T>
void purge_list_wrong(std::list<T> *list_to_purge,
                      std::list<T> *list_of_purged_elements,
                      std::vector<T> *extra_data,
                      std::vector<T> *extra_data_of_purged_elements,
                      T divisor) {

  auto it = list_to_purge->begin();
  auto end = list_to_purge->end();

  int index = 0;
  // Removal and copying loop
  while (it != end) {
    T val = *it;
    if (val % divisor == 0) {
      // purge.
      list_of_purged_elements->push_back(val);
      it = list_to_purge->erase(it);

      // copy extra data.
      T extraval = (*extra_data)[index];
      extra_data_of_purged_elements->push_back(extraval);
    } else {
      it++;
    }

    index++;
  }
}

/**
 * Remove elements from a list while looping over it.
 * Simultaneously copy elements from a second vector to
 * a third resulting vector. Also store which elements have
 * been removed.
 *
 * Remove elements if their modulo with provided integer
 * parameter `divisor` is zero.
 *
 * NOTE: The vector removal stuff doesn't work. And I don't remember
 * what I was trying to do there any more.
 */
template <typename T>
void purge_list(std::list<T> *list_to_purge,
                std::list<T> *list_of_purged_elements,
                std::vector<T> *extra_data,
                std::vector<T> *extra_data_of_purged_elements, T divisor) {

  std::cout << "Purging list with divisor = " << divisor << std::endl;

  auto it = list_to_purge->begin();
  auto end = list_to_purge->end();

  int index = 0;
  // Removal and copying loop
  while (it != end) {
    T val = *it;
    if (val % divisor == 0) {
      // purge.
      list_of_purged_elements->push_back(val);
      it = list_to_purge->erase(it);

      // copy extra data.
      T extraval = (*extra_data)[index];
      extra_data_of_purged_elements->push_back(extraval);
      // extra_data->erase(extraval);
    } else {
      index++;
      it++;
    }
  }
}

int main() {

  // list where we remove elements from
  std::list<int> list_to_purge;

  // vector whose elements get copied alongside `list_to_clean`
  std::vector<int> extra_data;

  // Initialize the lists with identical elements
  for (int i = 1; i < 51; i++) {
    list_to_purge.push_back(i);
    extra_data.push_back(i);
  }
  print_list(list_to_purge, "list to purge:\n\t");
  // print_list(extra_data, "extra data:\n\t");

  // Store purged elements here
  std::list<int> list_of_purged_elements;
  // Store extra data of purged elements here
  std::vector<int> extra_data_of_purged_elements;
  std::vector<bool> skip;

  // purge_list_wrong(&list_to_purge, &list_of_purged_elements,
  //     &extra_data, &extra_data_of_purged_elements, 2);
  // // doing it once is not a problem. But doing it twice
  // // messes with the 'index'. So you also need to remove
  // // elements from 'extra data', otherwise you get wrong results.
  // // purge_list_wrong(&list_to_purge, &list_of_purged_elements,
  //     &extra_data, &extra_data_of_purged_elements, 3);

  purge_list(&list_to_purge, &list_of_purged_elements, &extra_data,
             &extra_data_of_purged_elements, 2);

  print_list(list_of_purged_elements, "purged elements:\n\t");
  print_list(list_to_purge, "list is now:\n\t");
  // print_list(extra_data_of_purged_elements,
  //            "extra data of purged elements:\n\t");

  std::cout << "\n";
  list_of_purged_elements.clear();

  purge_list(&list_to_purge, &list_of_purged_elements, &extra_data,
             &extra_data_of_purged_elements, 3);

  print_list(list_of_purged_elements, "purged elements:\n\t");
  print_list(list_to_purge, "list is now:\n\t");
  // print_list(extra_data_of_purged_elements,
  //            "extra data of purged elements:\n\t");

  return 0;
}

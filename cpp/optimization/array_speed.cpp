/**
 * Test array access speeds between C-style arrays, std::array, and std::vector.
 */


#include <array>
#include <iomanip>
#include <iostream>
#include <vector>

#include "timer.h"

//! Number of elements in small array
constexpr size_t Nsmall = 10;
//! Number of elements in midsized array
constexpr size_t Nmid = 10000;
//! Number of elements in large array
constexpr size_t Nlarge = 1000000000;

//! stride for strided access in small array
constexpr size_t stride_small = 3;
//! stride for strided access in mid array
constexpr size_t stride_mid = 300;
//! stride for strided access in large array
constexpr size_t stride_large = 30000;

//! How many times to repeat the experiment
constexpr size_t Nrepeat = 10;



using datatype = double;


void allocate_vector( std::vector<datatype>& vec, const size_t size){
  vec.resize(size);
}


datatype* allocate_array( const size_t size){
  datatype* arr = new datatype[size];
  return arr;
}


template <typename T>
void print_array(T& arr, const size_t size){
  for (size_t i = 0; i < size; i++){
    std::cout << arr[i] << ",";
  }
  std::cout << "\n";
}


template <typename T>
void fill_array_sequential(T& arr, const size_t size){
  for (size_t i = 0; i < size; i++){
    // datatype x = static_cast<datatype>(i);
    // arr[i] = x * x + 7.;
    arr[i] = static_cast<datatype>(i);
  }
}


template <typename T>
void fill_array_strided(T& arr, const size_t size, const size_t stride){
  for (size_t i = 0; i < stride; i++){
    for (size_t j = 0; j < size; j+= stride){
      // datatype x = static_cast<datatype>(i);
      // arr[i] = x * x + 7.;
      arr[i] = static_cast<datatype>(i);
    }
  }
}







int main() {

  // Vector
  std::vector<datatype> vec_small;

  timer::Timer t_vec_alloc;
  allocate_vector(vec_small, Nsmall);
  timer::dt_type dt_vec_alloc = t_vec_alloc.end();

  timer::Timer t_vec_seq;
  fill_array_sequential(vec_small, Nsmall);
  timer::dt_type dt_vec_seq = t_vec_seq.end();

  timer::Timer t_vec_stride;
  fill_array_strided(vec_small, Nsmall, stride_small);
  timer::dt_type dt_vec_stride = t_vec_stride.end();
  // print_array(vec_small, Nsmall);


  // C Array
  timer::Timer t_carr_alloc;
  datatype* carr_small =  allocate_array(Nsmall);
  timer::dt_type dt_carr_alloc = t_carr_alloc.end();

  timer::Timer t_carr_seq;
  fill_array_sequential(carr_small, Nsmall);
  timer::dt_type dt_carr_seq = t_carr_seq.end();

  timer::Timer t_carr_stride;
  fill_array_strided(carr_small, Nsmall, stride_small);
  timer::dt_type dt_carr_stride = t_carr_stride.end();
  // print_array(carr_small, Nsmall);


  // std::array
  timer::Timer t_stdarr_alloc;
  std::array<datatype, Nsmall> stdarr_small;
  timer::dt_type dt_stdarr_alloc = t_stdarr_alloc.end();

  timer::Timer t_stdarr_seq;
  fill_array_sequential(stdarr_small, Nsmall);
  timer::dt_type dt_stdarr_seq = t_stdarr_seq.end();

  timer::Timer t_stdarr_stride;
  fill_array_strided(stdarr_small, Nsmall, stride_small);
  timer::dt_type dt_stdarr_stride = t_stdarr_stride.end();

  // print_array(stdarr_small, Nsmall);



  std::cout << std::setw(20) << "stage" <<
    std::setw(20) << "c-array" <<
    std::setw(20) << "std::vector" <<
    std::setw(20) << "std::array"
    << "\n";

  std::cout << std::setw(20) << "alloc" <<
    std::setw(20) << dt_carr_alloc <<
    std::setw(20) << dt_vec_alloc <<
    std::setw(20) << dt_stdarr_alloc
    << "\n";

  std::cout << std::setw(20) << "sequential" <<
    std::setw(20) << dt_carr_seq <<
    std::setw(20) << dt_vec_seq <<
    std::setw(20) << dt_stdarr_seq
    << "\n";

  std::cout << std::setw(20) << "strided" <<
    std::setw(20) << dt_carr_stride <<
    std::setw(20) << dt_vec_stride <<
    std::setw(20) << dt_stdarr_stride
    << "\n";




  return 0;
}

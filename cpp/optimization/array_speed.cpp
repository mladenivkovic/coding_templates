/**
 * Test array access speeds between C-style arrays, std::array, and std::vector.
 *
 * NOTE: This program requires a big stack. Allow it to have it with `ulimit -s
 * unlimited`.
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
// constexpr size_t Nmid = 100;
//! Number of elements in large array
constexpr size_t Nlarge = 50000000;
// constexpr size_t Nlarge = 1000;

//! stride for strided access in small array
constexpr size_t stride_small = 3;
//! stride for strided access in mid array
constexpr size_t stride_mid = 300;
//! stride for strided access in large array
constexpr size_t stride_large = 300000;

//! How many times to repeat the experiment
constexpr size_t Nrepeat = 10;

//! Talk to me?
constexpr bool verbose = false;

using datatype = double;
using time_units = timer::unit::ns;

/**
 * Print out the array.
 */
template <typename T> void print_array(T &arr, const size_t size) {
  for (size_t i = 0; i < size; i++) {
    std::cout << arr[i] << ",";
  }
  std::cout << "\n";
}

/**
 * Fill the array with dummy values using sequential access.
 */
template <typename T>
inline void fill_array_sequential(T &arr, const size_t size) {
  for (size_t i = 0; i < size; i++) {
    datatype x = static_cast<datatype>(i);
    arr[i] = x * x + 7.;
    // arr[i] = static_cast<datatype>(i);
  }
}

/**
 * Fill the array with dummy values using sequential access.
 */
template <typename T>
inline void fill_array_sequential2(T &arr, const size_t size) {
  for (size_t i = 0; i < size; i++) {
    size_t j = i < size - 1 ? i + 1 : i - 1;
    arr[i] = arr[i] + 2 * arr[j];
  }
}

/**
 * Fill the array with dummy values using strided access.
 */
template <typename T>
inline void fill_array_strided(T &arr, const size_t size, const size_t stride) {
  // start index within stride
  for (size_t i = 0; i < stride; i++) {
    for (size_t j = i; j < size; j += stride) {
      datatype x = static_cast<datatype>(j);
      arr[j] = x * x + 7.;
    }
  }
}

/**
 * Fill the array with dummy values using strided access.
 */
template <typename T>
inline void fill_array_strided2(T &arr, const size_t size,
                                const size_t stride) {
  // start index within stride
  for (size_t i = 0; i < stride; i++) {
    for (size_t j = i; j < size; j += stride) {
      size_t k = j + stride < size ? j + stride : j - stride;
      arr[j] = arr[j] + arr[k] * 2 + 7;
    }
  }
}

/**
 * Compute average time interval
 */
double average(const timer::dt_type dt) {
  return static_cast<double>(dt) / static_cast<double>(Nrepeat);
}

int main() {

  const size_t sizes[3] = {Nsmall, Nmid, Nlarge};
  const size_t strides[3] = {stride_small, stride_mid, stride_large};
  const char *runnames[3] = {"small", "mid", "large"};

  for (int i = 0; i < 3; i++) {

    const size_t N = sizes[i];
    const size_t stride = strides[i];
    const char *name = runnames[i];

    // Reset timers
    timer::dt_type dt_vec_alloc = 0;
    timer::dt_type dt_vec_seq1 = 0;
    timer::dt_type dt_vec_seq2 = 0;
    timer::dt_type dt_vec_stride1 = 0;
    timer::dt_type dt_vec_stride2 = 0;
    timer::dt_type dt_carr_alloc = 0;
    timer::dt_type dt_carr_seq1 = 0;
    timer::dt_type dt_carr_seq2 = 0;
    timer::dt_type dt_carr_stride1 = 0;
    timer::dt_type dt_carr_stride2 = 0;
    timer::dt_type dt_stdarr_alloc = 0;
    timer::dt_type dt_stdarr_seq1 = 0;
    timer::dt_type dt_stdarr_seq2 = 0;
    timer::dt_type dt_stdarr_stride1 = 0;
    timer::dt_type dt_stdarr_stride2 = 0;

    timer::Timer<time_units> runTimer;
    if (verbose) {
      std::cout << "Running " << name << " (N=" << N << ")\n";
    }

    for (size_t r = 0; r < Nrepeat; r++) {

      timer::Timer<time_units> iterTimer;
      if (verbose) {
        std::cout << ">>> r = " << r;
      }

      {
        // Vector
        // Put it into its own block so object deconstructs automatically
        std::vector<datatype> vec;

        timer::Timer<time_units> t_vec_alloc;
        vec.resize(N);
        dt_vec_alloc += t_vec_alloc.end();

        timer::Timer<time_units> t_vec_seq1;
        fill_array_sequential(vec, N);
        dt_vec_seq1 += t_vec_seq1.end();

        timer::Timer<time_units> t_vec_seq2;
        fill_array_sequential2(vec, N);
        dt_vec_seq2 += t_vec_seq2.end();

        timer::Timer<time_units> t_vec_stride1;
        fill_array_strided(vec, N, stride);
        dt_vec_stride1 += t_vec_stride1.end();

        timer::Timer<time_units> t_vec_stride2;
        fill_array_strided2(vec, N, stride);
        dt_vec_stride2 += t_vec_stride2.end();

        // destruct so we don't run out of memory
        // vec.~vector();
      }

      // C Array
      timer::Timer<time_units> t_carr_alloc;
      datatype *carr = new datatype[N];
      dt_carr_alloc += t_carr_alloc.end();

      timer::Timer<time_units> t_carr_seq1;
      fill_array_sequential(carr, N);
      dt_carr_seq1 += t_carr_seq1.end();

      timer::Timer<time_units> t_carr_seq2;
      fill_array_sequential2(carr, N);
      dt_carr_seq2 += t_carr_seq1.end();

      timer::Timer<time_units> t_carr_stride1;
      fill_array_strided(carr, N, stride);
      dt_carr_stride1 += t_carr_stride1.end();

      timer::Timer<time_units> t_carr_stride2;
      fill_array_strided2(carr, N, stride);
      dt_carr_stride2 += t_carr_stride2.end();

      // dealloc so we don't run out of memory
      delete[] carr;

      // std::array
      // needs size to be known at compile time, so we do this
      // in three branches
      if (N == Nsmall) {
        timer::Timer<time_units> t_stdarr_alloc;
        std::array<datatype, Nsmall> stdarr;
        dt_stdarr_alloc += t_stdarr_alloc.end();

        timer::Timer<time_units> t_stdarr_seq1;
        fill_array_sequential(stdarr, Nsmall);
        dt_stdarr_seq1 += t_stdarr_seq1.end();

        timer::Timer<time_units> t_stdarr_seq2;
        fill_array_sequential2(stdarr, Nsmall);
        dt_stdarr_seq2 += t_stdarr_seq2.end();

        timer::Timer<time_units> t_stdarr_stride1;
        fill_array_strided(stdarr, Nsmall, stride_small);
        dt_stdarr_stride1 += t_stdarr_stride1.end();

        timer::Timer<time_units> t_stdarr_stride2;
        fill_array_strided2(stdarr, Nsmall, stride_small);
        dt_stdarr_stride2 += t_stdarr_stride2.end();

        // destruct so we don't run out of memory
        // not needed since scope ends automatically
        // stdarr.~array();

      } else if (N == Nmid) {

        timer::Timer<time_units> t_stdarr_alloc;
        std::array<datatype, Nmid> stdarr;
        dt_stdarr_alloc += t_stdarr_alloc.end();

        timer::Timer<time_units> t_stdarr_seq1;
        fill_array_sequential(stdarr, Nmid);
        dt_stdarr_seq1 += t_stdarr_seq1.end();

        timer::Timer<time_units> t_stdarr_seq2;
        fill_array_sequential2(stdarr, Nmid);
        dt_stdarr_seq2 += t_stdarr_seq2.end();

        timer::Timer<time_units> t_stdarr_stride1;
        fill_array_strided(stdarr, Nmid, stride_small);
        dt_stdarr_stride1 += t_stdarr_stride1.end();

        timer::Timer<time_units> t_stdarr_stride2;
        fill_array_strided2(stdarr, Nmid, stride_small);
        dt_stdarr_stride2 += t_stdarr_stride2.end();

        // destruct so we don't run out of memory
        // not needed since scope ends automatically
        // stdarr.~array();

      } else if (N == Nlarge) {

        timer::Timer<time_units> t_stdarr_alloc;
        std::array<datatype, Nlarge> stdarr;
        dt_stdarr_alloc += t_stdarr_alloc.end();

        timer::Timer<time_units> t_stdarr_seq1;
        fill_array_sequential(stdarr, Nlarge);
        dt_stdarr_seq1 += t_stdarr_seq1.end();

        timer::Timer<time_units> t_stdarr_seq2;
        fill_array_sequential2(stdarr, Nlarge);
        dt_stdarr_seq2 += t_stdarr_seq2.end();

        timer::Timer<time_units> t_stdarr_stride1;
        fill_array_strided(stdarr, Nlarge, stride_small);
        dt_stdarr_stride1 += t_stdarr_stride1.end();

        timer::Timer<time_units> t_stdarr_stride2;
        fill_array_strided2(stdarr, Nlarge, stride_small);
        dt_stdarr_stride2 += t_stdarr_stride2.end();

        // destruct so we don't run out of memory
        // not needed since scope ends automatically
        // stdarr.~array();
      }

      if (verbose) {
        std::cout << " took " << iterTimer.end() << " "
                  << timer::Timer<time_units>::units_str() << "\n";
      }
    }

    if (verbose) {
      std::cout << "Run " << name << " (N=" << N << ") took " << runTimer.end()
                << " " << timer::Timer<time_units>::units_str() << "\n";
    }

    std::cout << "Run " << name << " (N=" << N << ", iters=" << Nrepeat
              << ", units=" << timer::Timer<time_units>::units_str()
              << "):\n\n";

    std::cout << std::setw(20) << "stage" << std::setw(20) << "c-array"
              << std::setw(20) << "std::vector" << std::setw(20) << "std::array"
              << "\n";

    std::cout << std::setw(20) << "alloc" << std::setw(20)
              << average(dt_carr_alloc) << std::setw(20)
              << average(dt_vec_alloc) << std::setw(20)
              << average(dt_stdarr_alloc) << "\n";

    std::cout << std::setw(20) << "sequential 1" << std::setw(20)
              << average(dt_carr_seq1) << std::setw(20) << average(dt_vec_seq1)
              << std::setw(20) << average(dt_stdarr_seq1) << "\n";

    std::cout << std::setw(20) << "sequential 1" << std::setw(20)
              << average(dt_carr_seq2) << std::setw(20) << average(dt_vec_seq2)
              << std::setw(20) << average(dt_stdarr_seq2) << "\n";

    std::cout << std::setw(20) << "strided 1" << std::setw(20)
              << average(dt_carr_stride1) << std::setw(20)
              << average(dt_vec_stride1) << std::setw(20)
              << average(dt_stdarr_stride1) << "\n";

    std::cout << std::setw(20) << "strided 2" << std::setw(20)
              << average(dt_carr_stride2) << std::setw(20)
              << average(dt_vec_stride2) << std::setw(20)
              << average(dt_stdarr_stride2) << "\n";
  }

  return 0;
}

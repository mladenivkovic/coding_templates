// A small class and tools to help with timing.

#include <chrono>
#include <iostream>
#include <string>
#include <typeinfo>

namespace timer {

// Alias the namespace for convenience.
namespace chr = std::chrono;

// Rename time units for convenience.
namespace unit {
using ms = chr::milliseconds;
using ns = chr::nanoseconds;
using mus = chr::microseconds;
using s = chr::seconds;
} // namespace unit

using default_time_units = unit::ms;

/**
 * @brief a small class to time code execution.
 * Usage: Create a Timer object. Time count starts at instantiation.
 * The timing is printed out when the object destructs, or do it manually
 * using timer::Timer.end()
 */
template <typename time_units = default_time_units> class Timer {

private:
  chr::time_point<chr::high_resolution_clock> _start;
  std::string _msg;
  bool _printed;

  void _start_timing(void) { _start = chr::high_resolution_clock::now(); }

  /**
   * @brief get duration since object was created.
   */
  long _get_duration(void) {
    chr::time_point<chr::high_resolution_clock> _stop =
        chr::high_resolution_clock::now();
    auto duration = duration_cast<time_units>(_stop - _start);
    return duration.count();
  }

  /**
   * Get the used units as a string.
   */
  std::string _units_str() {

    if (typeid(time_units) == typeid(chr::nanoseconds)) {
      return "[ns]";
    } else if (typeid(time_units) == typeid(chr::microseconds)) {
      return "[mus]";
    } else if (typeid(time_units) == typeid(chr::milliseconds)) {
      return "[ms]";
    } else if (typeid(time_units) == typeid(chr::seconds)) {
      return "[s]";
    } else if (typeid(time_units) == typeid(chr::minutes)) {
      return "[min]";
    } else if (typeid(time_units) == typeid(chr::hours)) {
      return "[h]";
    } else {
      return "[unknown units]";
    }
  }

  /**
   * @brief compute and print out the duration since the creation
   * of this object.
   * Marks message as "printed" so it doesn't autoprint at
   * desctruction too.
   */
  void _print_timing(std::string msg = "") {
    long duration = _get_duration();
    // Allow users to pass a second message, if they want.
    // Use case can be e.g. to add function call automatically at
    // creation time, and additional msg at end of measurement.
    std::cout << "[Timing] " << _msg << " " << msg << ": " << duration << " "
              << _units_str() << "\n";
    _printed = true;
  }

public:
  // Constructor
  Timer() {
    _start_timing();
    _printed = false;
  }

  Timer(std::string msg) : _msg(msg) {
    _start_timing();
    _printed = false;
  }

  ~Timer() {
    if (not _printed) {
      _print_timing();
    }
  }

  void end(std::string msg = "") { _print_timing(msg); }
};
} // namespace timer

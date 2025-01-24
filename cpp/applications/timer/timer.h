// A small class and tools to help with timing.

#include <chrono>
#include <iostream>
#include <string>
#include <typeinfo>

namespace timer {

  namespace unit {
    using ms = std::chrono::milliseconds;
    using ns = std::chrono::nanoseconds;
    using mus = std::chrono::microseconds;
    using s = std::chrono::seconds;
  }

  using default_time_units = unit::ms;

  /**
   * @brief a small class to help with timing.
   * Either print out timing when the object destructs, or
   * do it manually using timer::timer.end()
   */
template <typename time_units = default_time_units>
  class timer {

private:

  std::chrono::time_point<std::chrono::high_resolution_clock> _start;
  std::string _msg;
  bool _printed;

  void _start_timing(void) {
    _start = std::chrono::high_resolution_clock::now();
  }

  /**
   * @brief get duration since object was created.
   */
  long _get_duration(void) {
    std::chrono::time_point<std::chrono::high_resolution_clock> _stop = std::chrono::high_resolution_clock::now();
    auto duration = duration_cast<time_units>(_stop - _start);
    return duration.count();
  }

  /**
   * Get the used units as a string.
   */
  std::string _units_str() {

    if (typeid(time_units) == typeid(std::chrono::nanoseconds)) {
      return "[ns]";
    } else if (typeid(time_units) == typeid(std::chrono::microseconds)) {
      return "[mus]";
    } else if (typeid(time_units) == typeid(std::chrono::milliseconds)) {
      return "[ms]";
    } else if (typeid(time_units) == typeid(std::chrono::seconds)) {
      return "[s]";
    } else if (typeid(time_units) == typeid(std::chrono::minutes)) {
      return "[min]";
    } else if (typeid(time_units) == typeid(std::chrono::hours)) {
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
    std::cout << "[Timing] " << _msg << " " << msg << ": " << duration << " " << _units_str() << "\n";
    _printed = true;
  }

public:

    // Constructor
    timer() {
      _start_timing();
      _printed = false;
    }

    timer(std::string msg) :
      _msg(msg)
    {
      _start_timing();
      _printed = false;
    }

    ~timer() {
      if (not _printed) {
        _print_timing();
      }
    }

    void end(std::string msg = "") {
      _print_timing(msg);
    }
  };
}

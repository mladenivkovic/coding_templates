#!/bin/bash

# generate build directory
mkdir build
cd build

# call cmake
cmake -S ../ -B .
make
make install
# This leads to:
# CMake Error at cmake_install.cmake:52 (file):
#   file INSTALL cannot copy file
#   "/home/XXX/coding/cmake_webinar/demos/hello_world_cmake/hello_world/build/hello"
#   to "/usr/local/bin/hello": Permission denied.


# pass directory where to install
cmake ../ -DCMAKE_INSTALL_PREFIX=../hello_install
make
make install

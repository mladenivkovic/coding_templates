#!/bin/bash

mkdir build
cd build
cmake -S ../ -B .
# this leads to
#     -- The Fortran compiler identification is IntelLLVM 2023.1.0
#     CMake Error at /usr/share/cmake-3.22/Modules/CMakeDetermineCUDACompiler.cmake:179 (message):
#       Failed to find nvcc.
#
#       Compiler requires the CUDA toolkit.  Please set the CUDAToolkit_ROOT
#       variable.
#     Call Stack (most recent call first):
#       CMakeLists.txt:23 (project)

module restore swift-intel-openmpi
# look for definable/used variables in CMakeCache.txt
cmake -S .. -B . -DCMAKE_INSTALL_PREFIX=wind_tunnel_install -DUSE_CUDA=OFF

# for cmake with GUI - allows you to see variables you can tweak
cmake-gui ..

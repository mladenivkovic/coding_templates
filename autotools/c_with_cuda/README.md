# About

A minimal working example to reproduce the structure of (GPU-)SWIFT's
build system with autotools:

- There is a main file, `main.c`
- We want to build three libraries using autoconf, automake, and libtool:
  - a main library, FIRSTLIBRARY, storing the contents of most C files
    (`cfunc1.c`, `cfunc2.c`). (This mirrors `libswift.a`)
  - a second library, SECONDLIBRARY, storing the contents of some other C
    files (`clib2func.c`). (This mirrors `libgrav.a`)
  - a third library, CUDALIBRARY, storing the contents of cuda objects
- We compile (but don't link) `main.c`
- We link `main.o` with all three libraries.
- We have a running executable :)


# Usage





# Notes

## Compiling and linking cuda & C code

You can compile cuda stuff and C stuff separately, but the linking then needs
special attention:
- The compiled cuda object files need to be run through the "device linker"
  first. This should create device-linked objects
- At link time, you need to provide both the cuda objects and the device-linked
  objects to the linker.

Minimal example:

```
nvcc –arch=sm_35 –dc a.cu b.cu
nvcc –arch=sm_35 –dlink a.o b.o –o dlink.o
g++ a.o b.o dlink.o x.cpp –lcudart
```

See https://stackoverflow.com/a/16310324/6168231 and
https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#using-separate-compilation-in-cuda



## Manual Makefile

For dev purposes, there is also a `Makefile_manual` which I wrote by hand
and which compiles the executable correctly (but doesn't create libraries
first, as we are trying to do here.) Used that to verify code actually
compiles and runs as intended. Use it with

```
make -f Makefile_manual
```


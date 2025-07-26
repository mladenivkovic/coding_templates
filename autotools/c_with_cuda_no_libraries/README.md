# About

A minimal example to compile some C code and some CUDA code separately
and link them together using the host compiler using autotools.

In this version, we're not building any libraries; Just passing all objects to
the linker.


# Usage

- Run `autogen.sh` (or `autoreconf --install --symlinks`).
- Run `./configure`.
  - You may need to pass `./configure` the `--with-cuda=/path/to/cuda` flag if
    it can't detect your installation automatically.
- Run `make`.
- Run your executable, `./my_exec`



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


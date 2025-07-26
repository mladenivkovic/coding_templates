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



For dev purposes, there is also a `Makefile_manual` which I wrote by hand
which compiles the executable correctly (but doesn't create libraries
first, as we are trying to do here.) Used that to verify code actually
compiles and runs as intended. Use it with

```
make -f Makefile_manual
```


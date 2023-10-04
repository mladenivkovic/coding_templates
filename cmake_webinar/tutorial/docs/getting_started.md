# Getting started

In order to build a CMake package you first need to install CMake itself.

## Getting CMake

You can install CMake from several package managers. For instance, using `apt` on Ubuntu:

```bash
sudo apt install cmake
```

On computing clusters CMake is usually already centrally installed. Like on the ARCHER2 machine, loading the centrally installed module is done with:

```bash
module load cmake
```

## Compiling a CMake package

Compiling a CMake package requires three steps:

- A Configuration step
- A Build step
- An install step

### Configuration

First you need to create a build directory. It will be populated with temporary build files, such as object files and internal CMake configuration files.
You then need to go inside that newly created directory as the following commands will need to be run from within it:

```bash
mkdir $BUILD_DIRECTORY 
cd $BUILD_DIRECTORY
```

You also need to tell CMake to look for compilers, check for dependencies and find where libraries are stored on your system. Let us assume you wish to compile the source code in `$CMAKE_PACKAGE_DIRECTORY` and that the resulting binaries should be installed in `$INSTALL_DIR_PACKAGE`.
From the build directory run:

```bash
cmake $CMAKE_PACKAGE_DIRECTORY -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR_PACKAGE .
```

This will generate a build system in the build directory. On UNIX based systems, the default build system is `Makefile`.
Alternatively, you can run CMake from any directory, by specifying the the source and build directory respectively with the `-S` and `-B` flag.

```bash
cmake -S $CMAKE_PACKAGE_DIRECTORY -B $BUILD_DIRECTORY
```

### Build

Upon successful completion of the configuration step CMake will output a `Makefile` in your build directory.
In order to compile the package you can type

```bash
make
```

If you wish to see some more verbose output, including the compilation commands being executed type

```bash
make VERBOSE=1
```

After a successful build, the binary files will be saved somewhere in the build directory.
Alternatively, you can use

```bash
cmake --build .
```

### Install

Once all the executable and libraries have been generated, these need to be copied to the installation directory you specified during the configuration phase. You can copy the binaries in your specified install directory by running:

```bash
make install
```

Alternatively, you can run

```bash
cmake --build . --target install
```

### Tests

CMake contains a mechanism to run tests. This can be useful to test that the package was successfully installed. If tests have been defined by the CMake package developers, from the build directory you can type

```bash
ctest 
```

This command will run all the tests. If a test fails it will print out an error message.

!!! Exercise
    Try to build the hello world CMake package contained in `demos/hello_world_cmake`.

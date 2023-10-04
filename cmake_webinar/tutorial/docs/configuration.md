# Configuration

When building a package you often need to configure the build process.
You may need to specify which compiler to use or turn on/off some functionalities in the package, specify which libraries to link with and what compiler to use, etc..

## Setting variables

Packages often define some variables that can be set by the user. Some of these variables are builtin in the CMake framework, while others are created by package developers.

### Listing variables

You can find a list of all variables, including variables defined by the CMake package and built-in variables by typing

```bash
cmake ${CMAKE_PACKAGE_DIRECTORY} -LHA
```

Here you invoke cmake with the source directory as a first argument and the additional flags:

- `-L`: print a list of CMake variables
- `-A`: print a list of all advanced variables. Variables marked as `advanced` are hidden by default and usually do not need to be changed by the package user.
- `-H`: print helper strings for each variable. A helper string is meant to provide a description of the variable.

Alternatively, you can use the interactive tool

```bash
ccmake ${CMAKE_PACKAGE_DIRECTORY}
```

This will open an interactive program from which the user is able to view and modify user-modifiable variables.

### Specifying variables

Once you know the name of a variable and which value you want to set it to, you can specify a variable at configure time by adding the argument `-DVAR_NAME=${VAR_VALUE}`

For instance, you can configure a project with the variable named `VAR_NAME` set to `${VAR_VALUE}` by typing

```
cmake ${CMAKE_PACKAGE_DIRECTORY} -DVAR_NAME=${VAR_VALUE}
```

### Builtin variables

Some variables are defined by CMake itself and not by the package developers.
Some commonly used CMake variables are

- **CMAKE_BUILD_TYPE**: Can be set to one of `Debug`, `Release`, `RelWithDebInfo` and `MinSizeRel`. You can use it to turn on optimisation flags (`Release`) or debugging flags (`Debug`).
- **CMAKE_INSTALL_PREFIX**: The directory where to copy installation files after you built the package in the build folder. After running the install step, the installation directory will contain all executables, libraries and modules built by the CMake package.

You can obtain a list of all pre-defined CMake variables withe flag `--help-variable-list`

```bash
cmake --help-variable-list
```

## Specifying the compilers

In order to build a CMake package, you will need a compiler.
If not specified, CMake will look for a compiler on your system and use the first it finds.

!!! Warning
    If multiple compilers are present on the system CMake may find the wrong compiler, without returning any error. This results in errors at building or run time.

You can define the compiler by setting an environment variable before configuring the project. 
The name of the environment variable will depend on the choice of the compiler. Common variables are `CXX`,`CC` and `FC`, respectively for the `C++`,`C` and `Fortran` languages.
For instance for a GNU compiler, one could run

```bash
export CXX=g++
export CC=gcc
export FC=gfortran
```

before cmake.

Alternatively, you can specify the compilers by setting the appropriate CMake variables:

```bash
 cmake ${CMAKE_PACKAGE_DIRECTORY} -DCMAKE_Fortran_COMPILER=gfortran -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
```

!!! Exercise
    Try to build the **windtunnel** CMake package contained in `demos/wind_tunnel`.You will need to define some variables in order to successfully build the package.

# Writing a CMake package

Occasionally, in order to build a code, you need to modify a CMake package.
Learning to recognise the different components of a CMake package is useful even if you are not a CMake package developer. This page gives a brief overview of how to write a CMake package.

# Structure of a CMake project

A CMake package needs to contain at least one `CMakeLists.txt` file.
This file describes the CMake package using the CMake programming language. The CMake programming language is a complete programming language in which one can define variables, conditionals, loops and all the features you would expect from a modern programming language.

### Project definition

You start writing a `CMakeLists.txt` file by defining the minimum supported version of cmake and which compiler languages are supported.
Then you need to define the **project** with the `project( ... )` function. This function takes as arguments the name of the project and accepts optionally a list of programming languages included in the project.

```cmake
cmake_minimum_required( VERSION 3.2 )
project( wind_tunnel LANGUAGES FORTRAN CUDA )
```

## Targets

A **target** defines an object to be built by CMake. A target may be either an *executable* or a *library*. You can define a property on a target to control how the target is built. For instance, you can compile and link flags, external libraries and the location of source files by defining the appropriate properties for the target.

<figure markdown>
![Compilation process](images/targets.png)
<figcaption> The output executable my_exec is defined as a target. In order to build the target, you can set the target properties `source_files` and `link_libraries`. A property can contain a list of elements and those elements can be other targets.</figcaption>
</figure>

To create an executable target you can use the `add_executable` function, which takes as argument the name of the new target to be created:
```cmake
add_executable( my_exec )
```
You can also specify the location of the source files needed to build the executable by setting the `sources` property with the `target_sources` function:

```cmake
target_sources( my_exec PRIVATE source1.f90 source2.f90 )
```

The function accepts as the first argument the name of the target. The following argument is a declaration of how the property should be propagated to other targets that depend on the target `my_exec`.
Valid accessor values are:

- **PUBLIC**: the property gets forwarded to targets which depend on the current target.
- **PRIVATE**: the property is used for the current target and is forwarded to other targets which depend on the current target
- **INTERFACE**: the property is not used to build the current target, but is forwarded to other targets which depend on the current target

In this case, you only need the source files to build the current target, so you may declare the property as private. The declaration is followed by a list of filenames.
In order to create an executable, you often need to link with external libraries in the linking phase. You may do so using the `target_link_libraries` function. This function sets the `link_libraries` property for a target. If you wish `my_exec` to use `MPI` in `Fortran`, you can use:

```cmake
target_link_libraries( my_exec  MPI::MPI_Fortran )
```

Where the first argument is the name of the target and the second one is the name of the external library.

You might also need to include external files, such as `C/C++` header files or `Fortran` modules. You may do so with the `target_include_directories` function. This function takes as argument the name of the target, followed by the access specifier and a list of directories. For instance you can include `MPI` Fortran modules with:

```cmake
target_include_directories( my_exec PUBLIC ${MPI_FORTRAN_DIRS} )
```


## find_package

This command looks for a specific package. The command will look for instructions on how to find a package by looking for a `Find<Packagename>.cmake` file.
You can specify whether the package is optional or required for a successful build. The command `find_package` will look for the package in a [pre-defined set of directories](https://cmake.org/cmake/help/latest/command/find_package.html?highlight=find_package#search-procedure) and will return the first one it finds. If no package is found and the package is marked as required a fatal error will be thrown.
If a package is found it will define targets and/or variables which you can use later in your project.

!!! Exercise
    The folder `demos/create_a_package` contains the source code of a MPI hello world fortran program. It also contains a `CMakeLists.txt` but it does not define any target. Fill in the CMake code required to build an executable.

## Variables

You can define custom variables in a CMake package using the function `set`. The function accepts as first argument the name of the variable and as second argument the value of the variable. 
For instance, in order to set a variable with name `USE_CUDA` to the value `false`:

```cmake
set( USE_CUDA false )
```

You can access the value of a variable by prefixing with a `$` sign. For instance, for the variable defined above the command

```cmake
${USE_CUDA}
```

will evaluate to false.
CMake variables are usually local in scope to the function that they are defined in.
If defined outside of a function, they are local to the subdirectory.
Hence, a variable defined in a subfolder will not be accessible from the parent directory. 
One exception are variables defined as `CACHE`. These variables are global variables which can be accessed from anywhere in the project, no matter where they are defined. Additionally, once a cache variable has been set, it will not be changed by successive cmake calls.
Cached variables are used for variables that need to be set from the user. A variable can be set as cached by setting a third optional argument to `CACHE`. 
For instance, to set the `CMAKE_BUILD_TYPE` variable to `Release` you can use

```cmake
set( CMAKE_BUILD_TYPE Release CACHE )
```

## Options

These are logical variables which you can turn on or off at configure time. You can use options to enable compilation of additional components or to provide support for additional parallel paradigms.
You can define an option with the `option( ...)` function, which can accept as arguments the name of the variable to be set, an helper string to provide a human-readable description of the option and the value ON or OFF.
For instance, in order to turn on CUDA support, you may use

```cmake
option( USE_CUDA "Enable using CUDA for optimization" ON )
```

## Include subdirectories

It is good practice to organise your package in different modules. You can place each module in a separate directory. Then you can write a `CMakeLists.txt` file in each subdirectory describing how to build the corresponding module.
You can have an additional `CMakeLists.txt` in the root folder of your project and add the subdirectories to your main project. You can add sub directories with the `add_subdirectory( ... ) `function. This function takes as argument the location of the sub folder. For instance, if you want to include a `src` folder in your project you may use 

```cmake
add_subdirectory( src )
```

!!! Exercise
    Look at the CMake code for the **windtunnel** program.

    - Can you recognise the different sections ?
    - Where is the executable target defined ?
    - Where is the target linked to MPI ?
    - Where is the default build type defined ?

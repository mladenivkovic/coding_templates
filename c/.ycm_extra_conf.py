import os
import subprocess

DIR_OF_THIS_SCRIPT = os.path.abspath( os.path.dirname( __file__ ) )


def _add_include(path, include="include"):
    """
    Add the "include" subdir to a path. E.g.
    /some/path -> /some/path/include
    """
    return os.path.join(os.path.abspath(path), include)


c_flags = [
    '-Wall',
    '-Wextra',
    # THIS IS IMPORTANT! Without the '-x' flag, Clang won't know which language to
    # use when compiling headers. So it will guess. Badly. So C++ headers will be
    # compiled as C headers. You don't want that so ALWAYS specify the '-x' flag.
    # For a C project, you would set this to 'c' instead of 'c++'.
    '-x', 'c',
    #  '-Wno-unused-includes',
    #  '-Wno-unused-parameter',
    '-Wno-unused-function',
    '-std=c11'
]


defines = [
    #  "-DWITH_MPI",
    #  "-DWITH_CUDA",
    ]

include = [
    "-I", DIR_OF_THIS_SCRIPT,                     # add config.h
    #  "-I", DIR_OF_THIS_SCRIPT+"/argparse",
    #  "-I", os.path.join(DIR_OF_THIS_SCRIPT, "src"),
        ]

libs = []


def Settings( **kwargs ):

    if kwargs[ 'language' ] == 'cfamily':

        all_defines = defines
        all_includes = include

        # add current directory of the file
        fname = kwargs[ "filename" ]
        absfname = os.path.abspath(fname)
        fdir = os.path.dirname(absfname)
        all_includes.append("-I")
        all_includes.append(fdir)

        for lib in libs:
            all_includes.append("-I")
            all_includes.append(_add_include(lib))

        file_flags= c_flags

        final_flags = file_flags + all_defines + all_includes

        return {
            'flags': final_flags,
            'do_cache': True
          }

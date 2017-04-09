#!/bin/bash




# Write output and errors to null:

input/somescript.sh  > /dev/null 2>&1 





# Also works with functions:
# Define function f
# First command: say hi
# Second command: Bullshit, throws an error
# Write Error to file stderrlog, write stdout to screen AND file stdoutlog

f() {
    echo "hi"
    asd
}


f 2>stderrlog | tee stdoutlog





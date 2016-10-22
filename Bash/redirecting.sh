#!/bin/bash

# Define function f
# First command: say hy
# Second command: Bullshit, throws an error
# Write Error to file stderrlog, write stdout to screen AND file stdoutlog

f() {
    echo "hi"
    asd
}


f 2>stderrlog | tee stdoutlog



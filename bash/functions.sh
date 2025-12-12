#!/bin/bash

#----------------------
# Defining function
#----------------------

myfunction() {
  nargs=$# #store initial number of args; shift reduces $#
  echo "I was given" $nargs "arguments"
  for ((i=0; i<$nargs; i++)); do
    echo "argument "$i " is" $1
    shift
  done;
}




#----------------------
# Calling function
#----------------------

myfunction 2 asd 5 3kj hihi








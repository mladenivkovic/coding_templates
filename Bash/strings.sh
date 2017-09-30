#!/bin/bash


mystring="This is a string of mine."
filename="somefile.txt"



echo "mystring = " $mystring
echo Sting length: ${#mystring}


# ${somestring%extract_substring_from_back}
echo extract filename: ${filename%.*txt}


#!/bin/bash



# This script shows some useful stuff with sed.
# It takes input from the file "input/sedinput.txt"
# and writes its output to the "output/sedoutput.txt" file.


inputfile="input/sedinput.txt"
outputfile="output/sedoutput.txt"

cp "$inputfile" "$outputfile" # start with a clean file.

# We will work with -i: do changes "in place", meaning in the file directly.


#1) Work only on lines starting with keyword "print"
sed -i '/^print/ s/print/print\(/' "$outputfile"


#2) Add a parenthesis at the end of each line that starts with keyword print
sed -i '/^print/ s/$/ \)/' "$outputfile"


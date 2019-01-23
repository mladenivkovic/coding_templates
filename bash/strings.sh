#!/bin/bash


mystring="This is a string of mine."
filename="somefile.txt"



echo "mystring = " $mystring
echo Sting length: ${#mystring}


# ${somestring%extract_substring_from_back}
echo extract filename: ${filename%.*txt}

echo extract suffix: ${filename#somefile}


echo everything after keyword string: ${mystring#*string}


number=5
paddednumber=`printf "%05d" $number`
echo padded number: $paddednumber



# Substing extraction
echo "Substring starting from 5th character:" ${mystring:5}
echo "Substring starting from 5th character, 11 chars long:" ${mystring:5:11}

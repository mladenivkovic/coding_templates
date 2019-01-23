#!/bin/bash

echo "format specifiers"
echo ""
printf "%s\t%s\n" "1" "2 3" "4" "5"

##############################################
##############################################
##############################################

echo ""
echo ""


echo ""
echo "allow to interpret escape sequences"
printf "%b\n" "1" "2" "\n3"



##############################################
##############################################
##############################################


echo ""
echo ""

echo "integers"
echo ""
for i in $( seq 1 10 ); do 
    printf "%05d\t" "$i"; 
done
printf "\n"


##############################################
##############################################
##############################################


echo ""
echo ""


echo "floats"
printf "%f\n" 255 0xff 0377 3.5

echo ""

printf "%.3f\n" 255 0xff 0377 3.5




##############################################
##############################################
##############################################

echo ""
echo ""


echo "read in file line for line"
echo ""
printf "%s\t%s\t%s\t%s\t%s\n" "name" "value1" "value2" "value3" "value4"

file='input/table.dat'
while read line; do
    printf "%s\t%s\t%s\t%s\t%s\n" $line
done <$file




##############################################
##############################################
##############################################

echo ""
echo ""


echo "whole table"


divider===============================
divider=$divider$divider

header="\n %-10s %8s %10s %11s\n"
format=" %-10s %08d %10s %11.2f\n"

width=43

printf "$header" "ITEM NAME" "ITEM ID" "COLOR" "PRICE"

printf "%$width.${width}s\n" "$divider"

printf "$format" \
Triangle 13  red 20 \
Oval 204449 "dark blue" 65.656 \
Square 3145 orange .7




echo ""
echo ""
echo ""
echo ""
echo "OTHER PRINTF POSSIBILITIES"
echo "Save printf output into variable"
echo "printf -v newvar #format \$oldvar"
echo "example: i=50; printf -v j %05d \$i; echo \$j"
i=50; printf -v j %05d $i; echo $j

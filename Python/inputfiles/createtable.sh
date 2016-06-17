#!/bin/bash

printf "%s\t%s\t%s\n" "# name" "int" "float" > "table.txt"
for i in {1..50}; do 
    int=`echo "scale=0; $i*3 " | bc` ; 
    float=`echo "scale=4; $i/7 " | bc` ; 
    printf "%s%02d\t%s\t%.4f\n" "stud" "$i" "$int" "$float" >> "table.txt"
done 

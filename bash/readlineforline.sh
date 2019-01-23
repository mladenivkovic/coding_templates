#!/bin/bash

echo "read in file line for line"

file='input/loremipsum.txt'
while read line; do
    echo $line
    echo "NEXT LINE"
done <$file


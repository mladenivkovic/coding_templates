#!/bin/bash


input='input/table.dat'

echo "read in selected parts/columns of file"

echo ""
echo "unformatted"
cat $input | awk '{print $1, $3, $4}'

echo ""
echo "formatted"
printf "%s\t%s\t%s\n" `cat $input | awk '{print $1, $3, $4}'`

echo ""
echo "only line 2"
printf "%s\t%s\t%s\n" `cat $input | awk 'NR==2{print $1, $3, $4}'`

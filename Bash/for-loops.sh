#!/bin/bash


echo "  accessing every cmd line arg"
num_variables=$# #Number of positional arguments given
for ((i=1 ; i <= num_variables ; i++));do
		echo $1
		shift
	done

echo ""
echo "----------"
echo ""


echo "  looping through a sequence of integers"
for i in {1..10}; do
    echo $i
done

echo ""
echo "----------"
echo ""


echo "  looping through a list"
for i in `ls`; do
    echo $i
done

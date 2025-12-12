#!/bin/bash

echo "doing simple math. Executing 2 + 3^(0.5)"

echo ""

echo "python"
echo "print 2 + 3**(0.5)" | python

echo ""
echo "bc"
echo "scale=11; 2 + sqrt(3)" | bc


echo ""
echo "awk"
echo 2 3 0.5 | awk '{result = $1 + $2 ** $3; print result}'
#see https://www.gnu.org/software/gawk/manual/html_node/Arithmetic-Ops.html for more


echo ""
echo ""
echo "bash internal stuff: arithmetics in loops"

for i in {1..10}; do
  result=$(( $i -20 ))
  echo "result is: " $result
  calc=$((i % 2))
  if [[ $calc == 0 ]]; then
    echo "   i= " $i " is even."
  fi
done

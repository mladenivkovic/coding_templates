#!/bin/bash

#==============================
# Indexed arrays
#==============================

declare -a ind_array

ind_array=(foo bar)
echo [1]: $ind_array
echo [2]: ${ind_array[@]}

ind_array[2]=linx
echo [3.1]: ${ind_array[@]}
echo [3.2]: ${ind_array[*]}


# array[@] gives word-wise output;
# array[*] gives one string

echo [4.1]:
for i in "${ind_array[@]}"; do echo $i; done
echo [4.2]:
for i in "${ind_array[*]}"; do echo $i; done




echo
echo




#==============================
# Associative arrays
#==============================

# think of python dictionaries.

declare -A ass_array
ass_array=([foo]=bar [baz]=foobar)

echo [5.1]: ${ass_array[@]}
echo [5.2]: ${ass_array[*]}


# array[@] gives word-wise output;
# array[*] gives one string

echo [6.1]:
for i in "${ass_array[@]}"; do echo $i; done
echo [6.2]:
for i in "${ass_array[*]}"; do echo $i; done


# printing keys
echo [7]:
for key in "${!ass_array[@]}"; do echo $key; done





echo
echo
echo [8.1]: Adding elements - like this

# parentheses are vital!
ind_array+=(baz hihi)
echo ${ind_array[*]}


echo [8.2]: Adding elements - not like this

# parentheses are vital!
ind_array+=crop
echo ${ind_array[*]}


echo [9]: array len ${#ind_array[@]}


echo [10.1]: before deleting element : ${ind_array[*]}
unset ind_array[1]
echo [10.2]: after deleting element  : ${ind_array[*]}

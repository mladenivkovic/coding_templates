#!/bin/bash

#--------------------------------------------
# Short demonstration for ifs and elses
#--------------------------------------------



if [ "foo" == "foo" ]; then
    echo "true"
else
    echo "false"
fi


# one-liners for if [var=true] do stuff:
# var needs to be either defined or undefined, what value you set doesn't matter


do_check=asdjhasidh
[ $do_check ] && echo doing check first time
do_check=
[ $do_check ] && echo doing check second time

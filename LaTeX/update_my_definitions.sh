#!/bin/bash


#===================================================
# Update all the my_defines.tex files present
# in this directory to the last modified version.
#===================================================



skip_checks=false

if [ $# -gt 0 ]; then
    case $1 in
        -h | --help )
            echo "usage: update_my_definitions.sh   [-h | --help ] [ -f | --force ]"
            echo "       -h | --help:  print this message"
            echo "       -f | --force: ignore checks, update all versions to the last modified one."
            exit
            ;;
        -f | --force )
            skip_checks=true
            ;;
        * )
            echo "didn't recognize argument" $1.
            echo "use -h for help."
            exit
            ;;
    esac
fi




filelist=`find . -name 'my_defines.tex'`

# get all my_defines files in array sorted by newest modification time
declare -a filelist_sorted
temp=`ls -t $filelist`
for file in $temp; do
    filelist_sorted+=($file)
done

# get last three modified files
first=${filelist_sorted[0]}     # last modified
second=${filelist_sorted[1]}    # second last modified
third=${filelist_sorted[2]}     # third last modified

# get their modification times
firsttime=`stat -c %Y $first`
secondtime=`stat -c %Y $second`
thirdtime=`stat -c %Y $third`



# do checks, unless specified not to
if [ "$skip_checks" == 'false' ]; then

    # check that the last modified time is not the same as the second to last
    if [ "$firsttime" == "$secondtime" ]; then
        echo "last two modification times are identical. Aborting."
        exit
    fi

    # check that the second and third to last modified are not modified at different times;
    # if they aren't, there is probably more than one change going on
    if [ "$secondtime" != "$thirdtime" ]; then
        echo "second and thrid to last modification times are not the same."
        echo "this means that there may be more than one unsynchronized change going on."
        echo "If you still want to overwrite all the files with the newest one, then use -f."
        exit
    fi
fi




# now copy newest file to all others
for file in ${filelist_sorted[@]}; do
    if [ "$file" == "$first" ]; then
        touch $file
        # create a backup for unforseen cases
        cp $first my_defines_backup.tex
    else
        echo "$first" "-->" "$file"
        cp -u "$first" "$file"
    fi
done


#!/bin/bash

clang=${CLANG_FORMAT_CMD:="clang-format-18"}

# Formatting command
cmd="$clang -style=file $(git ls-files | grep '\.[ch]$')"

# Test if `clang-format-5.0` works
command -v $clang > /dev/null
if [[ $? -ne 0 ]]
then
    echo "ERROR: cannot find $clang"
    exit 1
fi

# Print the help
function show_help {
    echo -e "This script formats according to Google style"
    echo -e "  -h, --help \t Show this help"
    echo -e "  -t, --test \t Test if .c and .h files are well formatted"
}

# Parse arguments (based on https://stackoverflow.com/questions/192249)
TEST=0
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
	# print the help and exit
	-h|--help)
	    show_help
	    exit
	    ;;
	# check if the code is well formatted
	-t|--test)
	    TEST=1
	    shift
	    ;;
	# unknown option
	*)
	    echo "Argument '$1' not implemented"
	    show_help
	    exit
	    ;;
    esac
done

# Run the required commands
if [[ $TEST -eq 1 ]]
then
    # Note trapping the exit status from both commands in the pipe. Also note
    # do not use -q in grep as that closes the pipe on first match and we get
    # a SIGPIPE error.
    echo "Testing if files are correctly formatted"
    $cmd -output-replacements-xml | grep "<replacement " > /dev/null
    status=("${PIPESTATUS[@]}")

    #  Trap if first command failed. Note 141 is SIGPIPE, that happens when no
    #  output
    if [[ ${status[0]} -ne 0 ]]
    then
       echo "ERROR: $clang command failed"
       exit 1
    fi

    # Check formatting
    if [[ ${status[1]} -eq 0 ]]
    then
 	echo "ERROR: needs formatting"
 	exit 1
    else
        echo "...is correctly formatted"
    fi
else
    echo "Formatting"
    $cmd -i
fi

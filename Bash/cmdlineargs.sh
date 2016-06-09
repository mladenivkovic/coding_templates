#!/bin/bash


if [[ $# == 0 ]]; then
    echo "no arguments given. Can't handle that."
    exit
else

    while [[ $# > 0 ]]; do
    arg="$1"

    case $arg in 
        -s | --something)
        echo "detected something."
        echo "it is:"
        echo $2
        stuff=$2
        shift   #another shift here, so the argument after -s will not be taken as the next argument.
        ;;

        -h | --help)
        echo ""
        echo "A small template script to demonstrate command line passing."
        echo "-s <arg> or --something <arg> will get your argument recognised."
        echo "Otherwise, the script will complain."
        echo ""
        
        exit
        ;;

        *)
        echo "unknown argument:" $arg
        echo "use -h or --help for help."
        echo ""
        exit
        ;;
    esac

    shift
    done
fi


echo "here you can do stuff with your args."
echo "This stuff got passed: " $stuff

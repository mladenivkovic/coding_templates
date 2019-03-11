#!/usr/bin/env python3
#--------------------------------------
# Parsing arguments using argparse
#--------------------------------------

import argparse

parser = argparse.ArgumentParser(description='A program to demonstrate parsing arguments.')
parser.add_argument('some_argument')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true') # defaults to false if not given
parser.add_argument('--noutput')

args = parser.parse_args()
some_argument = args.some_argument
verbose = args.verbose
noutput = args.noutput
print(args, some_argument, verbose, noutput)




#=====================================================================
# give any amount of cmd line arguments. 
# examlpe: python cmdline_args_and_workdir.py 2 34 53 f.235 hihihi
#=====================================================================

from os import getcwd
import sys

print( "Workdir:", getcwd() )

print( "" )
print( "Number of arguments: ", len(sys.argv) )
print( "Number of arguments given: ", len(sys.argv)-1 )

for i in range(len(sys.argv)):
    print(('{0:6}{1:2d}{2:10}{3:20}'.format(" Arg nr. ",i,"    arg:", str(sys.argv[i]) )) )




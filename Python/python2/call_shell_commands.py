#!/usr/bin/python2

# How to call shell commands and get output into the script
import subprocess

#cmd='ls ../inputfiles/mpi_multiple_files/output_00004.0*'
cmd=['ls *.py']
p1=subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout_val, stderr_val=p1.communicate()
p1.stdout.close()
files=list(stdout_val.split())
print files

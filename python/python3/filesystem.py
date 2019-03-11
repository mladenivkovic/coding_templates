#!/usr/bin/env python3

#==========================================================
#This script shows how to deal with files on the filesystem.
#==========================================================


import os
import fnmatch
import shutil

print("=================")
print("Get information")
print("=================")
print()

print("My current path:")
print(os.getcwd())

print()
print("Files in directory:")
print(os.listdir())

print()
print("Files in some other dir: somedir='..'")
print(os.listdir(".."))


print()
print("Get only files matching pattern from a dir:")

inputfiles=[]
fileloc='inputfiles/mpi_multiple_files/'
for filename in os.listdir(fileloc):
    if fnmatch.fnmatch(filename, 'output_00008*'):
        inputfiles.append(filename)
        # inputfiles.append(fileloc+filename)

inputfiles.sort() #sort alphabetically!

print(inputfiles)












print()
print()
print("=================")
print("Interact")
print("=================")

print()
print("Create directory if it doesn't exist already:")
dirname='python_created_this_directory'
if not os.path.exists(dirname):
    os.makedirs(dirname)
    print("Created directory ", dirname, "", sep="'")
else:
    print("Directory ", dirname, " already exists.", sep="'")


print("This directory's contents are now:", os.listdir())


print("")
print("Create file if it doesn't exist")
filename='python_created_this_file.txt'
fileloc=dirname+'/'+filename

try:
    file = open(fileloc, 'r')
    print("File exists already.")
except IOError:
    print("File didn't exist. Writing a new one.")
    file = open(fileloc, 'w')
    file.write('some text to put in the file\n')
    # If opened this way while the file exists, it will be overwritten completely!
    file.close()
    print("New file with some useless content written.", os.listdir(dirname))
    file = open(fileloc, 'r')

print()
print("Reading from created file:")
print(file.read())
print()


input("The script has stopped so you can check out that the directory './"+dirname+"/' and the file were created. Press any key to continue to delete them.")

print()
print("Now deleting file.")
os.remove(fileloc)
print("File removed. See for yourself:", os.listdir(dirname))

print()
print("Now deleting directory.")
os.rmdir(dirname)
print("Directory removed. See for yourself:", os.listdir())


print()
print()

print("Königsdisziplin: Remove directory and contents recursively.")
print("First creating directory and file again.")

os.makedirs(dirname)
file = open(fileloc, 'w')
file.write('some text to put in the file\n')
# If opened this way while the file exists, it will be overwritten completely!
file.close()

print("directory and file created: ", os.listdir(dirname))

shutil.rmtree(dirname)

print("directory and file removed: ", os.listdir())


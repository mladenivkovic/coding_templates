#/bin/bash

# compile and run the program
# then create coverage html and open it with firefox

make clean
make
./my_program
gcovr . --root ../src --html --html-details -o coverage.html
# firefox coverage.html

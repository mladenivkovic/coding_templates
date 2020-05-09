#/bin/bash

# run each test case first, create json output
rm *html *json

make clean
make
./my_program
gcovr . --root ../src/ --json -o run1.json


make clean
make -f Makefile-is-defined
./my_program
gcovr . --root ../src/ --json -o run2.json


# now combine tracefiles and create coverage html
gcovr . --root ../src/ --add-tracefile run1.json --add-tracefile run2.json  --html-details -o coverage.html



firefox coverage.html

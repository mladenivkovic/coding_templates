#!/bin/bash

# sequence of commands to upload the package to PyPI.org
# twine can be installed with pip

rm -r dist/
python setup.py sdist bdist_wheel
twine upload  dist/*

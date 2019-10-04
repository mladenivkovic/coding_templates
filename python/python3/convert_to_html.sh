#!/bin/bash
for f in *.ipynb; do
    jupyter-nbconvert --execute --allow-errors --output-dir=html $f
done

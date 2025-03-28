#!/bin/bash


# remove all auxiliary files, rename main document
# and finally remove this scripts for a clean, fresh start
# also create figures/ dir if it doesn't exist

mkdir -p figures

for file in *.aux *.run.xml *.snm *.log *-blx.bib *.bbl *.nav *.out *.toc *.md *.synctex.gz *.blg *.pdf; do
    rm $file
done

master=minimalist.tex
read -p "Enter a name to rename the master document into (don't include .tex suffix): " newname
mv $master $newname.tex
sed -i "s/main_file = minimalist/main_file = ${newname}/" Makefile


read -p "Finished. Hit any key to remove this script. ctrl-c to quit" junk
rm cleanup_and_rename.sh

#!/bin/bash


# remove all auxiliary files, rename main document
# and finally remove this scripts for a clean, fresh start
# also create figures/ dir if it doesn't exist

mkdir -p figures

for file in *.aux *.run.xml *.snm *.log *-blx.bib *.bbl *.nav *.out *.toc *.md *.synctex.gz *.blg *.pdf; do
    rm $file
done

master=LaTeXdoc-formal-template.tex
read -p "Enter a name to rename the master document into (don't include .tex suffix): " newname
mv $master $newname.tex
sed -i "s/main_file = LaTeXdoc-formal-template/main_file = ${newname}/" Makefile


read -p "Finished. Hit any key to remove this script. ctrl-c to quit" junk
rm cleanup_and_rename.sh

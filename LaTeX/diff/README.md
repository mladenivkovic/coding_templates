Create a pdf containing a diff of a previous (main_previous.tex) version of a document and a current version (main_updated.tex) version of the document.

Needs latexdiff.

We first create a file containing the diffs using latexdiff. See the recipe in the Makefile for `diff.tex` on which command to call.

Then we simply compile the new diff.tex file to obtain our diff'd pdf.

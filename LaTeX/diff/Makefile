TEXFILES = $(wildcard *.tex)

default: diff.pdf

main_updated.pdf: main_updated.tex include_file_updated.tex
	pdflatex main_updated.tex
	pdflatex main_updated.tex

main_previous.pdf: main_previous.tex include_file_previous.tex
	pdflatex main_previous.tex
	pdflatex main_previous.tex

# First create a diff file
diff.tex: $(TEXFILES) # main_updated.tex main_previous.tex
	latexdiff main_previous.tex main_updated.tex --flatten > diff.tex

diff.pdf: diff.tex
	pdflatex diff.tex
	pdflatex diff.tex

clean:
	rm -f main_updated.pdf main_previous.pdf diff.tex diff.pdf *.aux *.log *.out *.dvi *.bbl *.blg *.toc *.gz *.lof *.lot


.PHONY: clean

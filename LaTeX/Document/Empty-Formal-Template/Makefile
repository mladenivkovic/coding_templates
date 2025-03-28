# make the pdf.

SUBDIRS = 
TEXFILES = #$(wildcard *.tex $(foreach fd, $(SUBDIRS), $(fd)/*.tex))

main_file = LaTeXdoc-formal-template

default: all

all: ${main_file}.pdf 

${main_file}.pdf: ${main_file}.tex ${TEXFILES}
	pdflatex ${main_file}.tex
	bibtex ${main_file}
	pdflatex ${main_file}.tex
	pdflatex ${main_file}.tex


clean:
	rm -f ${main_file}.pdf *.aux *.log *.out *.dvi *.bbl *.blg *.toc *.gz *.lof *.lot

.PHONY: clean


LATEX = @pdflatex -output-directory build main.tex
BUILD = @mkdir -p build

.PHONY: all bibtex latex clean

all: latex

bibtex:
	$(BUILD)
	$(LATEX)
	cp main.bib build/main.bib
	cd build && bibtex main
	$(LATEX)
	$(LATEX)

latex:
	$(BUILD)
	$(LATEX)

clean:
	rm -rf build


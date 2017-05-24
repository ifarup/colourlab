all: class.pdf package.pdf

%.pdf: %.gv
	dot -Tpdf $< -o $@

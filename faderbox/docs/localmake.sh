
set -e

if [ wires.tex -nt wires.pdf ]; then
	rm -rf wires_outdir
	mkdir wires_outdir
	pdflatex -output-directory wires_outdir wires.tex
	mv wires_outdir/wires.pdf .
	rm -rf wires_outdir
fi


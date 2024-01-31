
world: disk_add.pdf disk_sub.pdf disk_speed.pdf disk_resistor.pdf
	rm -f texjob.*

disk_%.pdf: disk_%.spl csrgen.spl
	splrun -e csrgen.spl $< > texjob.tex
	pdflatex -halt-on-error texjob.tex
	cp texjob.pdf $@


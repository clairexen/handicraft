
all:
	@echo "Run 'sudo make install' to install."

example:
	bash mojo-synth.sh example.ucf example.v
	python mojo-prog.py -s example.bit

install:
	install mojo-prog.py  /usr/local/bin/mojo-prog
	install mojo-synth.sh /usr/local/bin/mojo-synth

uninstall:
	rm -f /usr/local/bin/mojo-prog
	rm -f /usr/local/bin/mojo-synth


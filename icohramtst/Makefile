
SSH_RASPI ?= ssh pi@raspi

run: hramtst.bin
	$(SSH_RASPI) 'killall -9 icoprog || true'
	$(SSH_RASPI) 'icoprog -p' < hramtst.bin
	sedexpr="$$( grep '// debug_.*->' hramtst.v | sed 's,.*// \(debug_\),s/\1,; s, *-> *, /,; s, *$$, /;,;'; )"; \
		$(SSH_RASPI) "icoprog -V $$( grep '// debug_.*->' hramtst.v | wc -l; )" | sed -e "$$sedexpr" > trace.vcd

hramtst.bin: hramtst.asc libretto_data.hex
	icebram -v libretto_seed.hex libretto_data.hex < hramtst.asc | icepack > hramtst.bin

hramtst.asc: hramtst.blif hramtst.pcf
	arachne-pnr -o hramtst.asc -p hramtst.pcf -d 8k hramtst.blif

hramtst.blif: hramtst.v icosoc_raspif.v icosoc_crossclkfifo.v icosoc_debugger.v libretto_seed.hex
	yosys -v2 -l yosys.log -p 'synth_ice40 -top top -blif hramtst.blif' \
		hramtst.v icosoc_raspif.v icosoc_crossclkfifo.v icosoc_debugger.v

libretto_seed.hex:
	icebram -g 16 1024 > libretto_seed.hex

libretto_data.hex: libretto.py
	python3 libretto.py

clean:
	rm -f yosys.log libretto_seed.hex libretto_data.hex trace.vcd
	rm -f hramtst.blif hramtst.asc hramtst.bin


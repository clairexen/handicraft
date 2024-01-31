yosys -p 'synth_ice40 -blif test.blif -top test' test.v 
arachne-pnr -d 8k -o test.asc -p test.pcf test.blif 
icepack test.asc test.bin
iceprog -S test.bin

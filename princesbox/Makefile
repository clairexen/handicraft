run: princesbox
	/opt/riscv64b/bin/spike --isa=RV64GCB pk $<

princesbox: princesbox.c
	/opt/riscv64b/bin/riscv64-unknown-elf-gcc -march=rv64gcb -o $@ -Wall -O3 $^

clean:
	rm -f princesbox

.PHONY: run clean

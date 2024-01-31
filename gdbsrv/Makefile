CXXFLAGS = -MD -Wextra -O0 -ggdb -std=c++11
LDLIBS = -lstdc++ -lm

all: gdbsrv test.elf

gdbsrv: picorv32.v gdbsrv.cc
	verilator -Wno-lint --exe --top-module picorv32 -CFLAGS "$(CXXFLAGS)" --cc picorv32.v gdbsrv.cc
	$(MAKE) -C obj_dir -f Vpicorv32.mk
	cp obj_dir/Vpicorv32 gdbsrv

test.elf: test.s test.lds
	riscv32-unknown-elf-gcc -Os -m32 -ffreestanding -nostdlib -o test.elf -Wl,-Bstatic,-T,test.lds test.s

clean:
	rm -rf gdbsrv obj_dir

.PHONY: clean

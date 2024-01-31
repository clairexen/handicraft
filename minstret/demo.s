# riscv32-unknown-elf-gcc -ffreestanding -nostdlib -o demo.elf demo.s
# ./spike.sh -m0x10000:0x10000 -l demo.elf 2>&1 | head -n20

.global _start

_start:
csrrwi zero, minstret, 0    # b0205073
nop                         # 00000013
csrrci zero, minstret, 1    # b020f073
csrrci ra, minstret, 0      # b02070f3

slli ra, ra, 8
ret # <-- the destination of this jump reveals the result

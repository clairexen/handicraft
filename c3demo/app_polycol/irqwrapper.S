#define q0 0
#define q1 1
#define q2 2
#define q3 3

.macro getq rd qs
custom0 \rd,\qs,0,0
.endm

.macro setq qd rs
custom0 \qd,\rs,0,1
.endm

.macro retirq
custom0 0,0,0,2
.endm

.macro maskirq rd rs
custom0 \rd,\rs,0,3
.endm

.macro waitirq rd
custom0 \rd,0,0,4
.endm

.macro timer rd rs
custom0 \rd,\rs,0,5
.endm

.section .text
.global reset_timer
.global maskirq
.global unmaskirq
.global irq_wrapper
.global irq

.set irq_regs, 0x20
.set irq_stack, 0x200

reset_timer:
	lui t0, %hi(20000)
	addi t0, t0, %lo(20000)
	timer zero, t0
	ret

maskirq:
	addi t0, zero, -1
	maskirq zero, t0
	ret

unmaskirq:
	maskirq zero, zero
	ret

irq_wrapper:
	/* save registers */

	setq q2, x1
	setq q3, x2

	lui x1, %hi(irq_regs)
	addi x1, x1, %lo(irq_regs)

	getq x2, q0
	sw x2,   0*4(x1)

	getq x2, q2
	sw x2,   1*4(x1)

	getq x2, q3
	sw x2,   2*4(x1)

	sw x3,   3*4(x1)
	sw x4,   4*4(x1)
	sw x5,   5*4(x1)
	sw x6,   6*4(x1)
	sw x7,   7*4(x1)
	sw x8,   8*4(x1)
	sw x9,   9*4(x1)
	sw x10, 10*4(x1)
	sw x11, 11*4(x1)
	sw x12, 12*4(x1)
	sw x13, 13*4(x1)
	sw x14, 14*4(x1)
	sw x15, 15*4(x1)
	sw x16, 16*4(x1)
	sw x17, 17*4(x1)
	sw x18, 18*4(x1)
	sw x19, 19*4(x1)
	sw x20, 20*4(x1)
	sw x21, 21*4(x1)
	sw x22, 22*4(x1)
	sw x23, 23*4(x1)
	sw x24, 24*4(x1)
	sw x25, 25*4(x1)
	sw x26, 26*4(x1)
	sw x27, 27*4(x1)
	sw x28, 28*4(x1)
	sw x29, 29*4(x1)
	sw x30, 30*4(x1)
	sw x31, 31*4(x1)

	/* call interrupt handler C function */

	lui sp, %hi(irq_stack)
	addi sp, sp, %lo(irq_stack)

	// arg0 = address of regs
	lui a0, %hi(irq_regs)
	addi a0, a0, %lo(irq_regs)

	// arg1 = interrupt type
	getq a1, q1

	// call to C function
	jal ra, irq

	/* restore registers */

	// new irq_regs address returned from C code in a0
	addi x1, a0, 0

	lw x2,   0*4(x1)
	setq q0, x2

	lw x2,   1*4(x1)
	setq q1, x2

	lw x2,   2*4(x1)
	setq q2, x2

	lw x3,   3*4(x1)
	lw x4,   4*4(x1)
	lw x5,   5*4(x1)
	lw x6,   6*4(x1)
	lw x7,   7*4(x1)
	lw x8,   8*4(x1)
	lw x9,   9*4(x1)
	lw x10, 10*4(x1)
	lw x11, 11*4(x1)
	lw x12, 12*4(x1)
	lw x13, 13*4(x1)
	lw x14, 14*4(x1)
	lw x15, 15*4(x1)
	lw x16, 16*4(x1)
	lw x17, 17*4(x1)
	lw x18, 18*4(x1)
	lw x19, 19*4(x1)
	lw x20, 20*4(x1)
	lw x21, 21*4(x1)
	lw x22, 22*4(x1)
	lw x23, 23*4(x1)
	lw x24, 24*4(x1)
	lw x25, 25*4(x1)
	lw x26, 26*4(x1)
	lw x27, 27*4(x1)
	lw x28, 28*4(x1)
	lw x29, 29*4(x1)
	lw x30, 30*4(x1)
	lw x31, 31*4(x1)

	getq x1, q1
	getq x2, q2

	retirq

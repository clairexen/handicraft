.section .init
.global main

/* set stack pointer */
lui sp, %hi(4096)
addi sp, sp, %lo(4096)

/* call main */
jal ra, main

/* reset stack pointer */
lui sp, %hi((64+128)*1024)
addi sp, sp, %lo((64+128)*1024)

/* jump to application */
j 0x10000

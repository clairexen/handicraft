# The RISC-V X-BitMagic ISA Extension

The X-BitMagic ISA Extension defines the following instructions:

```
|31         25|24     20|19     15|14 12|11      7|6           0|
+-------------+---------+---------+-----+---------+-------------+
|   funct7    |   rs2   |   rs1   |  f3 |   rd    |    opcode   | R-type
|        imm[11:0]      |   rs1   |  f3 |   rd    |    opcode   | I-type
+=+=======+===+=====+===+=========+=====+=========+=============+
|/|R| ZIP |SHF|ZIPXY|SHF|   rs1   | /// |   rd    | /////////// | RPERM.<perm>
+=+=+=====+===+=====+===+=========+=====+=========+=============+
|/|R| ZIP |0 0|   rs2   |   rs1   | /// |   rd    | /////////// | BMMOR.<perm>
|/|R| ZIP |0 1|   rs2   |   rs1   | /// |   rd    | /////////// | BMMXOR.<perm>
+-+-+---+-+---+-----+---+---------+-----+---------+-------------+
|/|R| ZIP |1 0|   rs2   |   rs1   | /// |   rd    | /////////// | BFLY64.<perm>
|/|R| ZIP |1 1|   rs2   |   rs1   | /// |   rd    | /////////// | BFLY16.<perm>
+-+-+---+-+---+-----+---+---------+-----+---------+-------------+
|/|0 1 1 0|0 0|   rs2   |   rs1   | /// |   rd    | /////////// | RSAG
|/|0 1 1 0|0 1|   rs2   |   rs1   | /// |   rd    | /////////// | NSAG
|/|0 1 1 0|1 0|   rs2   |   rs1   | /// |   rd    | /////////// | MSAG
|/|0 1 1 0|1 1|   rs2   |   rs1   | /// |   rd    | /////////// | ESAG
+-+-------+---+---------+---------+-----+---------+-------------+
|/|0 1 1 1|0 0|   rs2   |   rs1   | /// |   rd    | /////////// | IRSAG
|/|0 1 1 1|0 1|   rs2   |   rs1   | /// |   rd    | /////////// | INSAG
|/|0 1 1 1|1 0|   rs2   |   rs1   | /// |   rd    | /////////// | IMSAG
|/|0 1 1 1|1 1|   rs2   |   rs1   | /// |   rd    | /////////// | PACKU
+-+-------+---+---------+---------+-----+---------+-------------+
|/|1 1 1 0|0 0|   rs2   |   rs1   | /// |   rd    | /////////// | -reserved-
|/|1 1 1 0|0 1|   rs2   |   rs1   | /// |   rd    | /////////// | -reserved-
|/|1 1 1 0|1 0|   rs2   |   rs1   | /// |   rd    | /////////// | -reserved-
|/|1 1 1 0|1 1|   rs2   |   rs1   | /// |   rd    | /////////// | -reserved-
+-+-------+---+---------+---------+-----+---------+-------------+
|/|1 1 1 1|0 0|   rs2   |   rs1   | /// |   rd    | /////////// | -reserved-
|/|1 1 1 1|0 1|   rs2   |   rs1   | /// |   rd    | /////////// | -reserved-
|/|1 1 1 1|1 0|   rs2   |   rs1   | /// |   rd    | /////////// | -reserved-
|/|1 1 1 1|1 1|   rs2   |   rs1   | /// |   rd    | /////////// | -reserved-
+=+===========+=========+=========+=====+=========+=============+
|   funct7    |   rs2   |   rs1   |  f3 |   rd    |    opcode   | R-type
|        imm[11:0]      |   rs1   |  f3 |   rd    |    opcode   | I-type
+-------------+---------+---------+-----+---------+-------------+
|31         25|24     20|19     15|14 12|11      7|6           0|
```

## Regular Permutations (RPERM)

Each 64-bit permutation corresponds to an invertable 6-bit function that
maps the old bit index for a data bit to the index of the new bit position.

For example, rotate shift is equivalent to 6-bit addition/subtraction
with overflow/underflow.

Since every permutation is an invertible function we can define the class
of 64-bit permutations that correspond to bit-permutation functions over
the 6-bit indices for the bit positions.

Since there are `6! = 720` bit permutations for a 6-bit word, there are
also 720 *Regular Permutations* for a 64-bit word.

The RPERM instruction can perform all and any of the 720 regular permutations
in a single instruction. (The shuffle/unshuffle instructions in the old
xbitmanip draft spec needed up to 6 instructions to perform any regular
permutation.)

## Bit-Matrix-Multiply (BMMOR, BMMXOR)

## 64-Bit Butterfly Instruction (BFLY64)

## 16-Bit Butterfly Instruction (BFLY16)

## Sheep-And-Goats Instructions ([IRNME]SAG)

## Upper Pack Instruction (PACKU)

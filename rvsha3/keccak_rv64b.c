/*
 *  Copyright (C) 2018  Clifford Wolf <clifford@clifford.at>
 *
 *  Permission to use, copy, modify, and/or distribute this software for any
 *  purpose with or without fee is hereby granted, provided that the above
 *  copyright notice and this permission notice appear in all copies.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 *  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 *  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 *  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 *  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 *  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 */

#include <stdint.h>
#include <stdio.h>

#define PRINT_TRACE

static const uint64_t keccak_round_constants[24] = {
	0x0000000000000001LL,
	0x0000000000008082LL,
	0x800000000000808ALL,
	0x8000000080008000LL,
	0x000000000000808BLL,
	0x0000000080000001LL,
	0x8000000080008081LL,
	0x8000000000008009LL,
	0x000000000000008ALL,
	0x0000000000000088LL,
	0x0000000080008009LL,
	0x000000008000000ALL,
	0x000000008000808BLL,
	0x800000000000008BLL,
	0x8000000000008089LL,
	0x8000000000008003LL,
	0x8000000000008002LL,
	0x8000000000000080LL,
	0x000000000000800ALL,
	0x800000008000000ALL,
	0x8000000080008081LL,
	0x8000000000008080LL,
	0x0000000080000001LL,
	0x8000000080008008LL
};

// Mapping of Axy to registers

#define A00 reg[0]
#define A01 reg[1]
#define A02 reg[2]
#define A03 reg[3]
#define A04 reg[4]

#define A10 reg[5]
#define A11 reg[6]
#define A12 reg[7]
#define A13 reg[8]
#define A14 reg[9]

#define A20 reg[10]
#define A21 reg[11]
#define A22 reg[12]
#define A23 reg[13]
#define A24 reg[14]

#define A30 reg[15]
#define A31 reg[16]
#define A32 reg[17]
#define A33 reg[18]
#define A34 reg[19]

#define A40 reg[20]
#define A41 reg[21]
#define A42 reg[22]
#define A43 reg[23]
#define A44 reg[24]

// Mapping of Bxy to registers

#define B00 A00
#define B01 A13
#define B02 A21
#define B03 A34
#define B04 A42
#define B10 A02
#define B11 A10
#define B12 A23
#define B13 A31
#define B14 A44
#define B20 A04
#define B21 A12
#define B22 A20
#define B23 A33
#define B24 A41
#define B30 A01
#define B31 A14
#define B32 A22
#define B33 A30
#define B34 A43
#define B40 A03
#define B41 A11
#define B42 A24
#define B43 A32
#define B44 A40

// Temp registers

#define X reg[25]
#define Y reg[26]
#define Z reg[27]

#define ROT(_x, _shamt) (((_x) << (_shamt)) | ((_x) >> ((64-(_shamt)) & 63)))

void keccak_p_rv64b(uint64_t state[25], uint64_t round_constant)
{
	uint64_t reg[28];
	uint64_t stack[5];

	// Input

	for (int x = 0; x < 5; x++)
	for (int y = 0; y < 5; y++)
	{
		reg[5*x + y] = state[5*y + x];
	}

	// Theta Step

	Z = A00 ^ A01;         // INSN: XOR
	Z = Z ^ A02;           // INSN: XOR
	Z = Z ^ A03;           // INSN: XOR
	Z = Z ^ A04;           // INSN: XOR
	stack[0] = Z;          // INSN: SD

	Y = A10 ^ A11;         // INSN: XOR
	Y = Y ^ A12;           // INSN: XOR
	Y = Y ^ A13;           // INSN: XOR
	Y = Y ^ A14;           // INSN: XOR
	stack[1] = Y;          // INSN: SD

	X = A20 ^ A21;         // INSN: XOR
	X = X ^ A22;           // INSN: XOR
	X = X ^ A23;           // INSN: XOR
	X = X ^ A24;           // INSN: XOR
	stack[2] = X;          // INSN: SD

	X = A30 ^ A31;         // INSN: XOR
	X = X ^ A32;           // INSN: XOR
	X = X ^ A33;           // INSN: XOR
	X = X ^ A34;           // INSN: XOR
	stack[3] = X;          // INSN: SD

	X = A40 ^ A41;         // INSN: XOR
	X = X ^ A42;           // INSN: XOR
	X = X ^ A43;           // INSN: XOR
	X = X ^ A44;           // INSN: XOR
	stack[4] = X;          // INSN: SD

	/* X is already stack[4] */
	/* Y is already stack[1] */
	Y = ROT(Y, 1);         // INSN: ROLI
	X = X ^ Y;             // INSN: XOR
	A00 = A00 ^ X;         // INSN: XOR
	A01 = A01 ^ X;         // INSN: XOR
	A02 = A02 ^ X;         // INSN: XOR
	A03 = A03 ^ X;         // INSN: XOR
	A04 = A04 ^ X;         // INSN: XOR

	/* Z is already stack[0] */
	Y = stack[2];          // INSN: LD
	Y = ROT(Y, 1);         // INSN: ROLI
	Z = Z ^ Y;             // INSN: XOR
	A10 = A10 ^ Z;         // INSN: XOR
	A11 = A11 ^ Z;         // INSN: XOR
	A12 = A12 ^ Z;         // INSN: XOR
	A13 = A13 ^ Z;         // INSN: XOR
	A14 = A14 ^ Z;         // INSN: XOR

	X = stack[1];          // INSN: LD
	Y = stack[3];          // INSN: LD
	Y = ROT(Y, 1);         // INSN: ROLI
	X = X ^ Y;             // INSN: XOR
	A20 = A20 ^ X;         // INSN: XOR
	A21 = A21 ^ X;         // INSN: XOR
	A22 = A22 ^ X;         // INSN: XOR
	A23 = A23 ^ X;         // INSN: XOR
	A24 = A24 ^ X;         // INSN: XOR

	X = stack[2];          // INSN: LD
	Y = stack[4];          // INSN: LD
	Y = ROT(Y, 1);         // INSN: ROLI
	X = X ^ Y;             // INSN: XOR
	A30 = A30 ^ X;         // INSN: XOR
	A31 = A31 ^ X;         // INSN: XOR
	A32 = A32 ^ X;         // INSN: XOR
	A33 = A33 ^ X;         // INSN: XOR
	A34 = A34 ^ X;         // INSN: XOR

	X = stack[3];          // INSN: LD
	Y = stack[0];          // INSN: LD
	Y = ROT(Y, 1);         // INSN: ROLI
	X = X ^ Y;             // INSN: XOR
	A40 = A40 ^ X;         // INSN: XOR
	A41 = A41 ^ X;         // INSN: XOR
	A42 = A42 ^ X;         // INSN: XOR
	A43 = A43 ^ X;         // INSN: XOR
	A44 = A44 ^ X;         // INSN: XOR

#ifdef PRINT_TRACE
	printf("After theta:\n");
	printf("%016llX %016llX %016llX %016llX %016llX\n", (long long)A00, (long long)A10, (long long)A20, (long long)A30, (long long)A40);
	printf("%016llX %016llX %016llX %016llX %016llX\n", (long long)A01, (long long)A11, (long long)A21, (long long)A31, (long long)A41);
	printf("%016llX %016llX %016llX %016llX %016llX\n", (long long)A02, (long long)A12, (long long)A22, (long long)A32, (long long)A42);
	printf("%016llX %016llX %016llX %016llX %016llX\n", (long long)A03, (long long)A13, (long long)A23, (long long)A33, (long long)A43);
	printf("%016llX %016llX %016llX %016llX %016llX\n", (long long)A04, (long long)A14, (long long)A24, (long long)A34, (long long)A44);
#endif

	// Rho and Pi Step

	X = ROT(A01, 36);      // INSN: ROLI
	A01 = ROT(A30, 28);    // INSN: ROLI
	A30 = ROT(A33, 21);    // INSN: ROLI
	A33 = ROT(A23, 15);    // INSN: ROLI
	A23 = ROT(A12, 10);    // INSN: ROLI
	A12 = ROT(A21,  6);    // INSN: ROLI
	A21 = ROT(A02,  3);    // INSN: ROLI
	A02 = ROT(A10,  1);    // INSN: ROLI
	A10 = ROT(A11, 44);    // INSN: ROLI
	A11 = ROT(A41, 20);    // INSN: ROLI
	A41 = ROT(A24, 61);    // INSN: ROLI
	A24 = ROT(A42, 39);    // INSN: ROLI
	A42 = ROT(A04, 18);    // INSN: ROLI
	A04 = ROT(A20, 62);    // INSN: ROLI
	A20 = ROT(A22, 43);    // INSN: ROLI
	A22 = ROT(A32, 25);    // INSN: ROLI
	A32 = ROT(A43,  8);    // INSN: ROLI
	A43 = ROT(A34, 56);    // INSN: ROLI
	A34 = ROT(A03, 41);    // INSN: ROLI
	A03 = ROT(A40, 27);    // INSN: ROLI
	A40 = ROT(A44, 14);    // INSN: ROLI
	A44 = ROT(A14,  2);    // INSN: ROLI
	A14 = ROT(A31, 55);    // INSN: ROLI
	A31 = ROT(A13, 45);    // INSN: ROLI
	A13 = X;               // INSN: MV

#ifdef PRINT_TRACE
	printf("After rho:\n");
	printf("%016llX %016llX %016llX %016llX %016llX\n", (long long)B00, (long long)B10, (long long)B20, (long long)B30, (long long)B40);
	printf("%016llX %016llX %016llX %016llX %016llX\n", (long long)B01, (long long)B11, (long long)B21, (long long)B31, (long long)B41);
	printf("%016llX %016llX %016llX %016llX %016llX\n", (long long)B02, (long long)B12, (long long)B22, (long long)B32, (long long)B42);
	printf("%016llX %016llX %016llX %016llX %016llX\n", (long long)B03, (long long)B13, (long long)B23, (long long)B33, (long long)B43);
	printf("%016llX %016llX %016llX %016llX %016llX\n", (long long)B04, (long long)B14, (long long)B24, (long long)B34, (long long)B44);
	printf("After pi:\n");
	printf("%016llX %016llX %016llX %016llX %016llX\n", (long long)A00, (long long)A10, (long long)A20, (long long)A30, (long long)A40);
	printf("%016llX %016llX %016llX %016llX %016llX\n", (long long)A01, (long long)A11, (long long)A21, (long long)A31, (long long)A41);
	printf("%016llX %016llX %016llX %016llX %016llX\n", (long long)A02, (long long)A12, (long long)A22, (long long)A32, (long long)A42);
	printf("%016llX %016llX %016llX %016llX %016llX\n", (long long)A03, (long long)A13, (long long)A23, (long long)A33, (long long)A43);
	printf("%016llX %016llX %016llX %016llX %016llX\n", (long long)A04, (long long)A14, (long long)A24, (long long)A34, (long long)A44);
#endif

	// Chi Step

	X = A10 & (~A00);      // INSN: ANDC
	Y = A20 & (~A10);      // INSN: ANDC
	Z = A30 & (~A20);      // INSN: ANDC
	A10 = A10 ^ Z;         // INSN: XOR
	Z = A40 & (~A30);      // INSN: ANDC
	A20 = A20 ^ Z;         // INSN: XOR
	Z = A00 & (~A40);      // INSN: ANDC
	A30 = A30 ^ Z;         // INSN: XOR
	A40 = A40 ^ X;         // INSN: XOR
	A00 = A00 ^ Y;         // INSN: XOR

	X = A11 & (~A01);      // INSN: ANDC
	Y = A21 & (~A11);      // INSN: ANDC
	Z = A31 & (~A21);      // INSN: ANDC
	A11 = A11 ^ Z;         // INSN: XOR
	Z = A41 & (~A31);      // INSN: ANDC
	A21 = A21 ^ Z;         // INSN: XOR
	Z = A01 & (~A41);      // INSN: ANDC
	A31 = A31 ^ Z;         // INSN: XOR
	A41 = A41 ^ X;         // INSN: XOR
	A01 = A01 ^ Y;         // INSN: XOR

	X = A12 & (~A02);      // INSN: ANDC
	Y = A22 & (~A12);      // INSN: ANDC
	Z = A32 & (~A22);      // INSN: ANDC
	A12 = A12 ^ Z;         // INSN: XOR
	Z = A42 & (~A32);      // INSN: ANDC
	A22 = A22 ^ Z;         // INSN: XOR
	Z = A02 & (~A42);      // INSN: ANDC
	A32 = A32 ^ Z;         // INSN: XOR
	A42 = A42 ^ X;         // INSN: XOR
	A02 = A02 ^ Y;         // INSN: XOR

	X = A13 & (~A03);      // INSN: ANDC
	Y = A23 & (~A13);      // INSN: ANDC
	Z = A33 & (~A23);      // INSN: ANDC
	A13 = A13 ^ Z;         // INSN: XOR
	Z = A43 & (~A33);      // INSN: ANDC
	A23 = A23 ^ Z;         // INSN: XOR
	Z = A03 & (~A43);      // INSN: ANDC
	A33 = A33 ^ Z;         // INSN: XOR
	A43 = A43 ^ X;         // INSN: XOR
	A03 = A03 ^ Y;         // INSN: XOR

	X = A14 & (~A04);      // INSN: ANDC
	Y = A24 & (~A14);      // INSN: ANDC
	Z = A34 & (~A24);      // INSN: ANDC
	A14 = A14 ^ Z;         // INSN: XOR
	Z = A44 & (~A34);      // INSN: ANDC
	A24 = A24 ^ Z;         // INSN: XOR
	Z = A04 & (~A44);      // INSN: ANDC
	A34 = A34 ^ Z;         // INSN: XOR
	A44 = A44 ^ X;         // INSN: XOR
	A04 = A04 ^ Y;         // INSN: XOR

#ifdef PRINT_TRACE
	printf("After chi:\n");
	printf("%016llX %016llX %016llX %016llX %016llX\n", (long long)A00, (long long)A10, (long long)A20, (long long)A30, (long long)A40);
	printf("%016llX %016llX %016llX %016llX %016llX\n", (long long)A01, (long long)A11, (long long)A21, (long long)A31, (long long)A41);
	printf("%016llX %016llX %016llX %016llX %016llX\n", (long long)A02, (long long)A12, (long long)A22, (long long)A32, (long long)A42);
	printf("%016llX %016llX %016llX %016llX %016llX\n", (long long)A03, (long long)A13, (long long)A23, (long long)A33, (long long)A43);
	printf("%016llX %016llX %016llX %016llX %016llX\n", (long long)A04, (long long)A14, (long long)A24, (long long)A34, (long long)A44);
#endif

	// Iota Step

	X = 0;                 // INSN: AUIPC
	X = round_constant;    // INSN: LD
	A00 = A00 ^ X;         // INSN: XOR

#ifdef PRINT_TRACE
	printf("After iota:\n");
	printf("%016llX %016llX %016llX %016llX %016llX\n", (long long)A00, (long long)A10, (long long)A20, (long long)A30, (long long)A40);
	printf("%016llX %016llX %016llX %016llX %016llX\n", (long long)A01, (long long)A11, (long long)A21, (long long)A31, (long long)A41);
	printf("%016llX %016llX %016llX %016llX %016llX\n", (long long)A02, (long long)A12, (long long)A22, (long long)A32, (long long)A42);
	printf("%016llX %016llX %016llX %016llX %016llX\n", (long long)A03, (long long)A13, (long long)A23, (long long)A33, (long long)A43);
	printf("%016llX %016llX %016llX %016llX %016llX\n", (long long)A04, (long long)A14, (long long)A24, (long long)A34, (long long)A44);
#endif

	// Output

	for (int x = 0; x < 5; x++)
	for (int y = 0; y < 5; y++)
	{
		state[5*y + x] = reg[5*x + y];
	}
}

void print_state_1d(uint64_t data[25])
{
	for (int y = 0; y < 5; y++)
	{
		printf("%016llX %016llX %016llX %016llX %016llX\n",
			(long long)data[5*y + 0], (long long)data[5*y + 1], (long long)data[5*y + 2],
			(long long)data[5*y + 3], (long long)data[5*y + 4]);
	}
}

int main()
{
	uint64_t state[25] = { /* zeros */ };

#ifdef PRINT_TRACE
	printf("Input:\n");
	print_state_1d(state);
#endif

	for (int i = 0; i < 24; i++) {
#ifdef PRINT_TRACE
		printf("\n--- Round %d ---\n\n", i);
#endif
		keccak_p_rv64b(state, keccak_round_constants[i]);
	}

#ifdef PRINT_TRACE
	printf("\nOutput:\n");
	print_state_1d(state);
#endif

	return 0;
}

// clang++ -Wall -o aes32insn aes32insn.cc && ./aes32insn
//
//  -----------------------------------------------------------------
//
//  Demonstrator for 32-bit AES instructions
//
//  Copyright (C) 2020  Claire Wolf <claire@symbioticeda.com>
//
//  Permission to use, copy, modify, and/or distribute this software for any
//  purpose with or without fee is hereby granted, provided that the above
//  copyright notice and this permission notice appear in all copies.
//
//  THE SOFTWARE IS PROVIDED \"AS IS\" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
//  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
//  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
//  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
//  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
//  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
//
//  -----------------------------------------------------------------
//
//  This is a demonstrator for a set of four 32-bit RV32 instructions that can
//  implement a single round of AES in 8 instructions. The state is stored in
//  four 32-bit variables, each containing one QUADRANT of the AES state:
//
//          A00 A01 | A02 A03            Q1 = {A00 A01 A10 A11}
//          A10 A11 | A12 A13            Q2 = {A02 A03 A12 A13}
//          --------+--------     =>
//          A20 A21 | A22 A23            Q3 = {A20 A21 A30 A31}
//          A30 A31 | A32 A33            Q4 = {A22 A23 A32 A33}
//
//  The instructions rv32aes1() and rv32aes2() perform the AES SubBytes and
//  ShiftRows step:
//
//      Q1' = rv32aes1(Q1, Q2);
//      Q2' = rv32aes1(Q2, Q1);
//      Q3' = rv32aes2(Q3, Q4);
//      Q4' = rv32aes2(Q4, Q3);
//
//  The instruction rv32aes3() performs the AES MixColumns step:
//
//      Q1' = rv32aes3(Q1, Q3);
//      Q2' = rv32aes3(Q2, Q4);
//      Q3' = rv32aes3(Q3, Q1);
//      Q4' = rv32aes3(Q4, Q2);
//
//  -----------------------------------------------------------------
//
//  *** TEST VECTORS ***
//  http://www.herongyang.com/Cryptography/AES-Example-Vector-of-AES-Encryption.html
//
//  00102030405060708090a0b0c0d0e0f0
//  63cab7040953d051cd60e0e7ba70e18c <-- SubBytes()
//  6353e08c0960e104cd70b751bacad0e7 <-- ShiftRows()
//  5f72641557f5bc92f7be3b291db9f91a <-- MixColumns()
//
//  89d810e8855ace682d1843d8cb128fe4
//  a761ca9b97be8b45d8ad1a611fc97369 <-- SubBytes()
//  a7be1a6997ad739bd8c9ca451f618b61 <-- ShiftRows()
//  ff87968431d86a51645151fa773ad009 <-- MixColumns()

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

uint8_t AES_SubByte(uint8_t v);
uint32_t AES_MixColumn(uint32_t v);

// ======================== 32-Bit AES Instructions ========================

uint32_t rv32aes1(uint32_t a, uint32_t b)
{
	uint8_t a0 = a >> 24, a1 = a >> 16, a3 = a, b2 = b >> 8;

	uint32_t v = 0;
	v |= AES_SubByte(a0) << 24;
	v |= AES_SubByte(a1) << 16;
	v |= AES_SubByte(a3) <<  8;
	v |= AES_SubByte(b2);

	return v;
}

uint32_t rv32aes2(uint32_t a, uint32_t b)
{
	uint8_t a2 = a >> 8, b0 = b >> 24, b1 = b >> 16, b3 = b;

	uint32_t v = 0;
	v |= AES_SubByte(b0) << 24;
	v |= AES_SubByte(b1) << 16;
	v |= AES_SubByte(b3) <<  8;
	v |= AES_SubByte(a2);

	return v;
}

uint32_t rv32aes3(uint32_t a, uint32_t b)
{
	uint8_t a0 = a >> 24, a1 = a >> 16, a2 = a >> 8, a3 = a;
	uint8_t b0 = b >> 24, b1 = b >> 16, b2 = b >> 8, b3 = b;

	uint32_t p = (a0 << 24) | (a2 << 16) | (b0 << 8) | b2;
	uint32_t q = (a1 << 24) | (a3 << 16) | (b1 << 8) | b3;

	p = AES_MixColumn(p);
	q = AES_MixColumn(q);

	uint8_t p0 = p >> 24, p1 = p >> 16;
	uint8_t q0 = q >> 24, q1 = q >> 16;

	return (p0 << 24) | (q0 << 16) | (p1 << 8) | q1;
}

// =========================================================================

uint8_t AES_SubByte(uint8_t v)
{
	static const uint8_t rijndael_sbox[256] = {
		0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
		0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
		0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
		0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
		0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
		0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
		0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
		0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
		0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
		0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
		0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
		0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
		0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
		0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
		0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
		0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
	};
	return rijndael_sbox[v];
}

uint32_t AES_MixColumn(uint32_t v)
{
	uint8_t a0 = v >> 24, a1 = v >> 16, a2 = v >> 8, a3 = v;

	auto gfmul1 = [](int a){ return a; };
	auto gfmul2 = [](int a){ return a << 1; };
	auto gfmul3 = [](int a){ return (a << 1) ^ a; };
	auto gfreduce = [](int a){ return a > 255 ? a ^ 0x11b : a; };

	uint8_t b0 = gfreduce(gfmul2(a0) ^ gfmul3(a1) ^ gfmul1(a2) ^ gfmul1(a3));
	uint8_t b1 = gfreduce(gfmul1(a0) ^ gfmul2(a1) ^ gfmul3(a2) ^ gfmul1(a3));
	uint8_t b2 = gfreduce(gfmul1(a0) ^ gfmul1(a1) ^ gfmul2(a2) ^ gfmul3(a3));
	uint8_t b3 = gfreduce(gfmul3(a0) ^ gfmul1(a1) ^ gfmul1(a2) ^ gfmul2(a3));

	return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
}

struct aes_reference
{
	uint8_t state[4][4];

	void setcols(uint32_t a, uint32_t b, uint32_t c, uint32_t d)
	{
		for (int j = 0; j < 4; j++)
		{
			for (int i = 0; i < 4; i++)
				state[i][j] = a >> (24-8*i);
			a = b, b = c, c = d;
		}
	}

	void setq(uint32_t q, int i, int j)
	{
		state[i+0][j+0] = q >> 24;
		state[i+0][j+1] = q >> 16;
		state[i+1][j+0] = q >> 8;
		state[i+1][j+1] = q;
	}

	void setq(uint32_t q1, uint32_t q2, uint32_t q3, uint32_t q4)
	{
		setq(q1, 0, 0);
		setq(q2, 0, 2);
		setq(q3, 2, 0);
		setq(q4, 2, 2);
	}

	uint32_t getq1() { return (state[0][0] << 24) | (state[0][1] << 16) | (state[1][0] << 8) | state[1][1]; }
	uint32_t getq2() { return (state[0][2] << 24) | (state[0][3] << 16) | (state[1][2] << 8) | state[1][3]; }
	uint32_t getq3() { return (state[2][0] << 24) | (state[2][1] << 16) | (state[3][0] << 8) | state[3][1]; }
	uint32_t getq4() { return (state[2][2] << 24) | (state[2][3] << 16) | (state[3][2] << 8) | state[3][3]; }

	void print_matrix()
	{
		for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			printf("%02x%c", state[i][j], j == 3 ? '\n' : ' ');
	}

	void print_line()
	{
		for (int j = 0; j < 4; j++)
		for (int i = 0; i < 4; i++)
			printf("%02x", state[i][j]);
		printf("\n");
	}

	const char* get_line()
	{
		static char buffer[128];
		char *p = buffer;
		for (int j = 0; j < 4; j++)
		for (int i = 0; i < 4; i++)
			p += snprintf(p, sizeof(buffer)-(p-buffer), "%02x", state[i][j]);
		return buffer;
	}

	void check(const char *ref)
	{
		const char *val = get_line();
		printf("T: %s\nR: %s\n", ref, val);
		if (strcmp(ref, val)) {
			printf("** CHECK FAILED **\n");
			exit(1);
		}
	}

	void run_subbytes()
	{
		for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			state[i][j] = AES_SubByte(state[i][j]);
	}

	void run_shiftrows()
	{
		uint8_t old_state[4][4];
		memcpy(old_state, state, sizeof(state));
		for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			state[i][j] = old_state[i][(j+i)%4];
	}

	void run_mixcolumns()
	{
		for (int j = 0; j < 4; j++)
		{
			uint32_t v = 0;

			for (int i = 0; i < 4; i++)
				v = (v << 8) | state[i][j];

			v = AES_MixColumn(v);

			for (int i = 0; i < 4; i++)
				state[i][j] = v >> (24-8*i);
		}
	}
};

void rv32aes_print_line(uint32_t q1, uint32_t q2, uint32_t q3, uint32_t q4)
{
	aes_reference t;
	t.setq(q1, q2, q3, q4);
	t.print_line();
}

void rv32aes_print_matrix(uint32_t q1, uint32_t q2, uint32_t q3, uint32_t q4)
{
	aes_reference t;
	t.setq(q1, q2, q3, q4);
	t.print_matrix();
}

void rv32aes_check(const char *ref, uint32_t q1, uint32_t q2, uint32_t q3, uint32_t q4)
{
	aes_reference t;
	t.setq(q1, q2, q3, q4);
	t.check(ref);
}

int main()
{
	aes_reference r;

	printf("-- Init\n");
	r.setcols(0x00102030, 0x40506070, 0x8090a0b0, 0xc0d0e0f0);
	r.check("00102030405060708090a0b0c0d0e0f0");

	uint32_t t1 = r.getq1();
	uint32_t t2 = r.getq2();
	uint32_t t3 = r.getq3();
	uint32_t t4 = r.getq4();

	printf("-- SubBytes\n");
	r.run_subbytes();
	r.check("63cab7040953d051cd60e0e7ba70e18c");

	printf("-- ShiftRows\n");
	r.run_shiftrows();
	r.check("6353e08c0960e104cd70b751bacad0e7");

	printf("-- MixColumns\n");
	r.run_mixcolumns();
	r.check("5f72641557f5bc92f7be3b291db9f91a");

	printf("-- RV32AES\n");

	rv32aes_check("00102030405060708090a0b0c0d0e0f0", t1, t2, t3, t4);

	uint32_t u1 = rv32aes1(t1, t2);
	uint32_t u2 = rv32aes1(t2, t1);
	uint32_t u3 = rv32aes2(t3, t4);
	uint32_t u4 = rv32aes2(t4, t3);

	rv32aes_check("6353e08c0960e104cd70b751bacad0e7", u1, u2, u3, u4);

	t1 = rv32aes3(u1, u3);
	t2 = rv32aes3(u2, u4);
	t3 = rv32aes3(u3, u1);
	t4 = rv32aes3(u4, u2);

	rv32aes_check("5f72641557f5bc92f7be3b291db9f91a", t1, t2, t3, t4);

	// ----------------------------------------------------------

	printf("\n-- Init\n");
	r.setcols(0x89d810e8, 0x855ace68, 0x2d1843d8, 0xcb128fe4);
	r.check("89d810e8855ace682d1843d8cb128fe4");

	t1 = r.getq1();
	t2 = r.getq2();
	t3 = r.getq3();
	t4 = r.getq4();

	printf("-- SubBytes\n");
	r.run_subbytes();
	r.check("a761ca9b97be8b45d8ad1a611fc97369");

	printf("-- ShiftRows\n");
	r.run_shiftrows();
	r.check("a7be1a6997ad739bd8c9ca451f618b61");

	printf("-- MixColumns\n");
	r.run_mixcolumns();
	r.check("ff87968431d86a51645151fa773ad009");

	printf("-- RV32AES\n");

	rv32aes_check("89d810e8855ace682d1843d8cb128fe4", t1, t2, t3, t4);

	u1 = rv32aes1(t1, t2);
	u2 = rv32aes1(t2, t1);
	u3 = rv32aes2(t3, t4);
	u4 = rv32aes2(t4, t3);

	rv32aes_check("a7be1a6997ad739bd8c9ca451f618b61", u1, u2, u3, u4);

	t1 = rv32aes3(u1, u3);
	t2 = rv32aes3(u2, u4);
	t3 = rv32aes3(u3, u1);
	t4 = rv32aes3(u4, u2);

	rv32aes_check("ff87968431d86a51645151fa773ad009", t1, t2, t3, t4);

	return 0;
}

#include <assert.h>
#include <stdint.h>
#include <stdbool.h>

// -----------------------------------------------------------------
// All tests performed with CBMC version 5.3 and ESBMC version 2.1.0
//
// CBMC verifies that the hack works, with ESBMC the verification
// fails. My guess is that ESBMC is not using IEEE floats to represent
// floating point types.
// -----------------------------------------------------------------

// This version is by @rrika9
// https://twitter.com/rrika9/status/849619674690187264
//
// (minor modification to p and q initialization to make
// it work with the esbmc C front-end)
//
// Verification passes with cbmc:
// $ cbmc --function test1 fpintcmp_formal.c
//
// Verification fails with esbmc:
// $ esbmc --function test1 fpintcmp_formal.c
//
void test1(float a, float b)
{
	union { float f; unsigned int i; } p, q;
	p.f = a, q.f = b;

	if (a > 0 && b > 0) {
		assert ((a<b) == (p.i<q.i));
	}
}

// This is my original version using pointer casts for the conversion
// and checking of the assumptions using the integer representation.
// The version above is much much nicer. I don't know what I though
// when I wrote mine..
//
// Verification passes with cbmc:
// $ cbmc --function test2 fpintcmp_formal.c
//
// Verification fails with esbmc:
// $ esbmc --function test2 fpintcmp_formal.c
//
void test2(int32_t ai, int32_t bi)
{
	// clear sign bit
	ai &= 0x7fffffff;
	bi &= 0x7fffffff;

	// no nan/inf values
	if ((ai & 0x7f800000) == 0x7f800000) return;
	if ((bi & 0x7f800000) == 0x7f800000) return;

	float af = *(float*)&ai;
	float bf = *(float*)&bi;

	bool x = ai < bi;
	bool y = af < bf;

	assert(x == y);
}

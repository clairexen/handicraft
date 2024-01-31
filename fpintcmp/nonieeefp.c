#include <assert.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

// In IEEE floating point +0 and -0 are represented using
// distinct bit patterns, but checking them for equality
// using the == operator will return true. Thus the following
// two test will fail when IEEE floating point is used:
//
// cbmc --function test1 nonieeefp.c
// cbmc --function test2 nonieeefp.c
//
// However, esbmc seems to be using a non-ieee represenation
// that does not have this property. With esbmc verification
// of the two functions succeeds:
//
// esbmc --function test1 nonieeefp.c
// esbmc --function test2 nonieeefp.c

void test1(float a, float b)
{
	union { float f; unsigned int i; } p, q;
	p.f = a, q.f = b;
	if (a == b) assert (p.i == q.i);
}

void test2(float a, float b)
{
	int32_t ai = *(int32_t*)&a;
	int32_t bi = *(int32_t*)&b;
	if (a == b) assert (ai == bi);
}

int main()
{
	int32_t ai = 0x00000000;
	int32_t bi = 0x80000000;

	float af = *(float*)&ai;
	float bf = *(float*)&bi;

	printf("%f %f %d\n", af, bf, af == bf);
	return 0;
}

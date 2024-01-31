#include <stdint.h>
#include <stdio.h>

typedef union {
	float f;
	uint32_t i;
} fp32_t;

fp32_t mkfp32(int sign, int exponent, int mantissa)
{
	fp32_t x;
	x.i = mantissa;
	x.i |= exponent << 23;
	x.i |= sign << 31;
	return x;
}

int main()
{
	// A (fp #b1 #xdb #b00100000000000000000000)
	// B (fp #b0 #xd8 #b10010110100001100011111)
	// C (fp #b1 #xde #b11101000001001000000000)
	// D (fp #b0 #xd8 #b00001001000000000100001)

	fp32_t A = mkfp32(1, 0xdb, 0b00100000000000000000000);
	fp32_t B = mkfp32(0, 0xd8, 0b10010110100001100011111);
	fp32_t C = mkfp32(1, 0xde, 0b11101000001001000000000);
	fp32_t D = mkfp32(0, 0xd8, 0b00001001000000000100001);

	fp32_t X;
	float AB = A.f + B.f;
	float CD = C.f + D.f;
	X.f = AB + CD;

	printf("A    -> 0x%08x\n", A.i);
	printf("B    -> 0x%08x\n", B.i);
	printf("C    -> 0x%08x\n", C.i);
	printf("D    -> 0x%08x\n", D.i);
	printf("X    -> 0x%08x\n", X.i);

	fp32_t arr[4] = {A, B, C, D};

	for (int i0=0; i0 < 4; i0++)
	{
		float t0 = arr[i0].f;

		for (int i1=0; i1 < 4; i1++)
		{
			if (i1 == i0) continue;
			float t1 = t0 + arr[i1].f;

			for (int i2=0; i2 < 4; i2++)
			{
				if (i2 == i0) continue;
				if (i2 == i1) continue;
				float t2 = t1 + arr[i2].f;

				for (int i3=0; i3 < 4; i3++)
				{
					if (i3 == i0) continue;
					if (i3 == i1) continue;
					if (i3 == i2) continue;
					float t3 = t2 + arr[i3].f;

					fp32_t sum;
					sum.f = t3;
					printf("%c%c%c%c -> 0x%08x %s\n", "ABCD"[i0], "ABCD"[i1], "ABCD"[i2], "ABCD"[i3], sum.i, sum.i != X.i ? "Okay" : "ERR");
				}
			}
		}
	}

	return 0;
}

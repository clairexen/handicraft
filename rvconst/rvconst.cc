// g++ -Wall -Wextra -O2 -o rvconst rvconst.cc && ./rvconst
// cbmc --trace --function check_make_const rvconst.cc

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

int64_t xorshift64()
{
	static uint64_t x = 123456789;
	x ^= x << 13;
	x ^= x >> 7;
	x ^= x << 17;
	return x;
}

int64_t lui(int64_t v)
{
	return (int32_t)v & ~(int64_t)0xfff;
}

int64_t addi(int64_t a, int64_t b)
{
	return a + (b << 52 >> 52);
}

int64_t addiw(int64_t a, int64_t b)
{
	return (int32_t)(a + (b << 52 >> 52));
}

int64_t addiwu(int64_t a, int64_t b)
{
	return (uint32_t)a + (b << 52 >> 52);
}

int64_t zextw(int64_t a)
{
	return a & 0xffffffff;
}

int64_t make_const(int64_t v)
{
	int64_t lo12 = v << 52 >> 52;
	int64_t upper = v - lo12;
	int64_t result = 0;

	if ((v >> 11) == 0x1fffff) {
		result = zextw(addi(0, lo12));
		assert(result != addi(lui(upper), lo12));
		assert(result != addiwu(lui(upper), lo12));
		printf("addi rd, zero, %d\n", (int)lo12);
		printf("zext.w rd, rd\n");
		return result;
	}

	if (((v >> 31) & 3) == 1) {
		result = addiwu(lui(upper), lo12);
		assert(result != addiw(lui(upper), lo12));
		printf("lui rd, 0x%05x\n", ((unsigned int)upper) >> 12);
		printf("addiwu rd, rd, %d\n", (int)lo12);
		return result;
	}

	result = addiw(lui(upper), lo12);
	printf("lui rd, 0x%05x\n", ((unsigned int)upper) >> 12);
	printf("addiw rd, rd, %d\n", (int)lo12);
	return result;
}

extern "C" void check_make_const(int64_t v)
{
	int64_t up33 = v >> 31;

	if (up33 != -1 && up33 != 0 && up33 != 1)
		return;

	int64_t result = make_const(v);
	assert(result == v);
}

int main()
{
	int64_t constvals[] = {
		4294966272l,
		2697199616l,
		-38912l,
		2147481664l
	};

	for (int i = 0; i < 10000; i++)
	{
		int64_t v = xorshift64() >> 31;

		if (i < int(sizeof(constvals)/sizeof(*constvals)))
			v = constvals[i];
		else if ((v >> 31) == -2)
			continue;

		printf("-- 0x%016llx --\n", (long long)v);
		int64_t x = make_const(v);
		assert(v == x);
	}
	printf("OKAY.\n");
	return 0;
}


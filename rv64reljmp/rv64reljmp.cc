// gcc -o rv64reljmp rv64reljmp.cc && ./rv64reljmp
// cbmc --trace --function check_rv64reljmp rv64reljmp.cc

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#define RV64RELJMP_LOWER int64_t(0xFFFFFFFF7FFFF800LL)
#define RV64RELJMP_UPPER int64_t(0x000000007FFFF7FFLL)

int64_t lui(int64_t v)
{
	return (int32_t)v & ~(int64_t)0xfff;
}

int64_t addi(int64_t a, int64_t b)
{
	return a + (b << 52 >> 52);
}

int64_t rv64reljmp(int64_t v)
{
	int64_t lo12 = v << 52 >> 52;
	int64_t upper = (int32_t)(v - lo12);
	int64_t result = 0;

	result = addi(lui(upper), lo12);
	printf("auipc rt, %d\n", ((unsigned int)upper) >> 12);
	printf("jalr x0, %d(rt)\n", (int)lo12);
	return result;
}

extern "C" void check_rv64reljmp(int64_t v)
{
	if (v < RV64RELJMP_LOWER)
		return;

	if (v > RV64RELJMP_UPPER)
		return;

	int64_t result = rv64reljmp(v);
	assert(result == v);
}

int main()
{
	printf("%lld\n", (long long)(RV64RELJMP_UPPER - RV64RELJMP_LOWER + 1));
}

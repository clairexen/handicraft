#include <stdio.h>
#include <stdint.h>
#include <assert.h>

uint32_t rawbits32(uint32_t *data, int lo, int len)
{
	data += lo / 32;
	lo = lo % 32;

	uint32_t b = data[0];

	b = b >> lo;

	if (lo + len > 32)
		b |= (uint32_t)data[1] << (32-lo);
	
	return b;
}

uint32_t uintbits32(uint32_t *data, int lo, int len)
{
	uint32_t b = rawbits32(data, lo, len);

	return (b << (32-len)) >> (32-len);
}

int32_t intbits32(uint32_t *data, int lo, int len)
{
	uint32_t b = rawbits32(data, lo, len);

	b = b << (32-len);
	
	uint32_t int32_max = ((uint32_t)1 << 31) - 1;
	int32_t bs = b > int32_max ? -1-(int32_t)(~b) : (int32_t)b;

	return bs >> (32-len);
}

void checker(uint32_t *data, int lo, int len)
{
	if (data == NULL) return;
	if (lo < 0) return;
	if (lo > 32) return;
	if (len <= 0) return;
	if (len > 32) return;

	uintbits32(data, lo, len);
	intbits32(data, lo, len);
}

// -----------------------------------------------------------------------------

#ifdef SIM
uint64_t s[2] = { 123456789, 987654321 };

uint64_t xorshift128plus(uint64_t max_val)
{
	uint64_t x = s[0];
	uint64_t const y = s[1];
	s[0] = y;
	x ^= x << 23; // a
	s[1] = x ^ y ^ (x >> 17) ^ (y >> 26); // b, c
	return max_val ? (s[1] + y) % (max_val + 1) : (s[1] + y);
}

int main(int argc, char **argv)
{
	int i, j, k;
	uint32_t data[32];

	for (k = 0; k < 1000000; k++)
	{
		if ((k % 100000) == 99999)
			printf("%d%%\n", (k+1) / 10000);

		int lo = xorshift128plus(512);
		int len = 1+xorshift128plus(31);
		uint32_t refval = xorshift128plus(0);

		for (i = 0; i < 32; i++) {
			j = lo + i;
			if (refval & ((uint32_t)1 << i))
				data[j / 32] |= (uint32_t)1 << (j % 32);
			else
				data[j / 32] &= ~((uint32_t)1 << (j % 32));
		}

		uint32_t b = uintbits32(data, lo, len);
		int32_t bs = intbits32(data, lo, len);

#if 0
		printf("%3d %2d %016llx: %016llx %016llx\n", lo, len,
				(long long)refval, (long long)b, (long long)bs);
#endif

		assert(b == ((b << (32 - len)) >> (32 - len)));
		assert(bs == ((bs << (32 - len)) >> (32 - len)));

		b = b << (32 - len);
		bs = bs << (32 - len);
		refval = refval << (32 - len);

		assert(b == refval);
		assert(bs == refval);
	}

	printf("OK.\n");
	return 0;
}
#endif

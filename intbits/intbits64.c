#include <stdio.h>
#include <stdint.h>
#include <assert.h>

uint64_t rawbits64(uint32_t *data, int lo, int len)
{
	data += lo / 32;
	lo = lo % 32;

	uint64_t b = data[0];

	if (lo + len > 32)
		b |= (uint64_t)data[1] << 32;

	b = b >> lo;

	if (lo + len > 64)
		b |= (uint64_t)data[2] << (64-lo);
	
	return b;
}

uint64_t uintbits64(uint32_t *data, int lo, int len)
{
	uint64_t b = rawbits64(data, lo, len);

	return (b << (64-len)) >> (64-len);
}

int64_t intbits64(uint32_t *data, int lo, int len)
{
	uint64_t b = rawbits64(data, lo, len);

	b = b << (64-len);
	
	uint64_t int64_max = ((uint64_t)1 << 63) - 1;
	int64_t bs = b > int64_max ? -1-(int64_t)(~b) : (int64_t)b;

	return bs >> (64-len);
}

void checker(uint32_t *data, int lo, int len)
{
	if (data == NULL) return;
	if (lo < 0) return;
	if (lo > 64) return;
	if (len <= 0) return;
	if (len > 64) return;

	uintbits64(data, lo, len);
	intbits64(data, lo, len);
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
		int len = 1+xorshift128plus(63);
		uint64_t refval = xorshift128plus(0);

		for (i = 0; i < 64; i++) {
			j = lo + i;
			if (refval & ((uint64_t)1 << i))
				data[j / 32] |= (uint32_t)1 << (j % 32);
			else
				data[j / 32] &= ~((uint32_t)1 << (j % 32));
		}

		uint64_t b = uintbits64(data, lo, len);
		int64_t bs = intbits64(data, lo, len);

#if 0
		printf("%3d %2d %016llx: %016llx %016llx\n", lo, len,
				(long long)refval, (long long)b, (long long)bs);
#endif

		assert(b == ((b << (64 - len)) >> (64 - len)));
		assert(bs == ((bs << (64 - len)) >> (64 - len)));

		b = b << (64 - len);
		bs = bs << (64 - len);
		refval = refval << (64 - len);

		assert(b == refval);
		assert(bs == refval);
	}

	printf("OK.\n");
	return 0;
}
#endif

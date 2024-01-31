// cbmc --unwind 40 --trace --function check1a bsort.c
// cbmc --unwind 40 --trace --function check1b bsort.c
// cbmc --unwind 40 --trace --function check2a bsort.c
// cbmc --unwind 40 --trace --function check2b bsort.c

#define SZ 6
#define T uint8_t

#include <assert.h>
#include <stdint.h>

// original bungee sort
int bsort1(T *data, int n)
{
	for (int i = 1; i < n; i++) {
		T a = data[i-1];
		T b = data[i];
		if (a > b) {
			data[i] = a;
			data[i-1] = b;
			i = 0;
		}
	}
}

// improved bungee sort
int bsort2(T *data, int n)
{
	for (int i = 1; i < n; i++)
		if (i) {
			T a = data[i-1];
			T b = data[i];
			if (a > b) {
				data[i--] = a;
				data[i--] = b;
			}
		}
}

void check1a()
{
	T data[SZ];

	bsort1(data, SZ);

	for (int i = 1; i < SZ; i++)
		assert(data[i-1] <= data[i]);
}

void check1b()
{
	T data[SZ];

	T v1 = 0;
	for (int i = 0; i < SZ; i++)
		v1 |= data[i];

	bsort1(data, SZ);

	T v2 = 0;
	for (int i = 0; i < SZ; i++)
		v2 |= data[i];

	assert(v1 == v2);
}

void check2a()
{
	T data[SZ];

	bsort2(data, SZ);

	for (int i = 1; i < SZ; i++)
		assert(data[i-1] <= data[i]);
}

void check2b()
{
	T data[SZ];

	T v1 = 0;
	for (int i = 0; i < SZ; i++)
		v1 |= data[i];

	bsort2(data, SZ);

	T v2 = 0;
	for (int i = 0; i < SZ; i++)
		v2 |= data[i];

	assert(v1 == v2);
}

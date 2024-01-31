// nvcc -O3 -Wall -o findsqnum findsqnum.cu && ./findsqnum | head

#include <stdint.h>
#include <stdio.h>

static void HandleErrorCUDA(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(1);
	}
}

#define HANDLE_ERROR_CUDA(_err) HandleErrorCUDA(_err, __FILE__, __LINE__)

struct myint128_t
{
	uint64_t lo, hi;

	__device__ myint128_t(uint64_t lo_val = 0, uint64_t hi_val = 0) : lo(lo_val), hi(hi_val) { }

	__device__ bool operator!=(const myint128_t &b) const {
		const myint128_t &a = *this;
		return a.lo != b.lo || a.hi != b.hi;
	}

	__device__ myint128_t operator+(const myint128_t &b) const {
		const myint128_t &a = *this;
		myint128_t res;
		res.lo = a.lo + b.lo;
		res.hi = a.hi + b.hi + (res.lo < a.lo);
		return res;
	}

	__device__ myint128_t operator*(const myint128_t &b) const {
		const myint128_t &a = *this;
		myint128_t res;
		res.lo = a.lo * b.lo;
		res.hi = __umul64hi(a.lo, b.lo) + a.lo * b.hi + a.hi * b.lo;
		return res;
	}

	__device__ myint128_t operator<<(int i) const {
		const myint128_t &a = *this;
		myint128_t res;
		if (i < 64) {
			res.lo = (a.lo << i);
			res.hi = (a.hi << i) | (a.lo >> (64-i));
		} else {
			res.lo = 0;
			res.hi = a.lo << (i-64);
		}
		return res;
	}

	__device__ myint128_t operator>>(int i) const {
		const myint128_t &a = *this;
		myint128_t res;
		if (i < 64) {
			res.lo = (a.lo >> i) | (a.hi << (64-i));
			res.hi = (a.hi >> i);
		} else {
			res.lo = a.hi >> (i-64);
			res.hi = 0;
		}
		return res;
	}
};

__global__ void find_num(uint32_t offset)
{
	unsigned int n = offset + blockIdx.x*blockDim.x + threadIdx.x;
	myint128_t n0 = myint128_t(n) << 40;

	myint128_t n1 = n0 + myint128_t((uint64_t(1) << 40) * 0.0123456789 - 1);
	myint128_t n2 = n0 + myint128_t((uint64_t(1) << 40) * 0.0123456790 + 1);

	// myint128_t n1 = n0 + myint128_t((uint64_t(1) << 40) * 0.9876543210 - 1);
	// myint128_t n2 = n0 + myint128_t((uint64_t(1) << 40) * 0.9876543211 + 1);

	if (((n1*n1) >> 80) != ((n2*n2) >> 80)) {
		// printf("floor(%u0123456789^2 / (10^20)); sqrt(%);\n", n);
		// printf("floor(%u9876543210^2 / (10^20)); sqrt(%);\n", n);
		printf("%u\n", n);
	}
}

int main()
{
	for (uint32_t i = 0; i < 100; i++) {
		// printf("-- %d --\n", i);
		find_num<<<1000, 1000>>>(1000000 * i);
		HANDLE_ERROR_CUDA(cudaThreadSynchronize());
	}
	return 0;
}


#define PROFILER_SINGULAR_DEFS
#include "cudalu.h"
#include "profiler.h"
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#define SMALL_DIM 10
#define LARGE_DIM 1000

uint32_t xorshift128()
{
	static uint32_t x = 123456789;
	static uint32_t y = 362436069;
	static uint32_t z = 521288629;
	static uint32_t w = 88675123;
	uint32_t t = x ^ (x << 11);
	x = y; y = z; z = w;
	w ^= (w >> 19) ^ t ^ (t >> 8);
	return w;
}

#ifdef SMALL_DIM
void test_small()
{
	CudaLU d(SMALL_DIM);
	CudaLU::ScalarType v[SMALL_DIM];

	for (int i = 0; i < d.size(); i++)
	for (int j = 0; j < d.size(); j++)
		d(i, j) = (100 + xorshift128() % 900) / 100.0;
	for (int i = 0; i < d.size(); i++)
		v[i] = (100 + xorshift128() % 900) / 100.0;

	printf("A = [\n");
	d.printMatrix();
	printf("];\n");

	printf("b = [");
	for (int i = 0; i < d.size(); i++)
		printf(" %.2f", v[i]);
	printf(" ];\n");

	d.updateGPU();
	d.factorize_lu();
	d.solve(v);
	d.updateCPU();

	printf("L = [\n");
	d.printMatrix('l');
	printf("];\n");

	printf("U = [\n");
	d.printMatrix('u');
	printf("];\n");

	printf("x = [");
	for (int i = 0; i < d.size(); i++)
		printf(" %f", v[i]);
	printf(" ];\n");

	printf("A * x' - b'\n");
}
#endif

void test_large()
{
	CudaLU d(LARGE_DIM);
	CudaLU::ScalarType v[LARGE_DIM];

	for (int i = 0; i < d.size(); i++)
	for (int j = 0; j < d.size(); j++)
		d(i, j) = (100 + xorshift128() % 900) / 100.0;
	for (int i = 0; i < d.size(); i++)
		v[i] = (100 + xorshift128() % 900) / 100.0;

	Profiler("cudalu") {
		Profiler("cudalu_update")
			d.updateGPU();
		Profiler("cudalu_factorize")
			d.factorize_lu();
		Profiler("cudalu_solve")
			d.solve(v);
	}

#ifdef SMALL_DIM
	printf("x_large = [");
	for (int i = 0; i < 10; i++)
		printf(" %f", v[i]);
	printf(" ... ];\n");
#endif
}

int main()
{
#ifdef SMALL_DIM
	test_small();
	test_large();
#else
	printf("Running large test 10 times..\n");
	for (int i = 0; i < 10; i++)
		test_large();
#endif
	Profiler::show();
	return 0;
}


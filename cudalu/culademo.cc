#define PROFILER_SINGULAR_DEFS
#include "profiler.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cula_lapack.h>

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

void checkStatus(culaStatus status)
{
	char buf[256];

	if(!status)
		return;

	culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
	printf("%s\n", buf);

	culaShutdown();
	exit(1);
}

#ifdef SMALL_DIM
void test_small()
{
	culaFloat* A = (culaFloat*)malloc(SMALL_DIM*SMALL_DIM*sizeof(culaFloat));
	culaFloat* B = (culaFloat*)malloc(SMALL_DIM*sizeof(culaFloat));
	culaFloat* X = (culaFloat*)malloc(SMALL_DIM*sizeof(culaFloat));
	culaInt* IPIV = (culaInt*)malloc(SMALL_DIM*sizeof(culaInt));

	for (int i = 0; i < SMALL_DIM; i++)
	for (int j = 0; j < SMALL_DIM; j++)
		A[i + j*SMALL_DIM] = (100 + xorshift128() % 900) / 100.0;
	for (int i = 0; i < SMALL_DIM; i++)
		X[i] = B[i] = (100 + xorshift128() % 900) / 100.0;

	printf("A = [\n");
	for (int i = 0; i < SMALL_DIM; i++) {
		for (int j = 0; j < SMALL_DIM; j++)
			printf(" %7.2f", A[i + j*SMALL_DIM]);
		printf("\n");
	}
	printf("];\n");

	printf("b = [");
	for (int i = 0; i < SMALL_DIM; i++)
		printf(" %.2f", B[i]);
	printf(" ];\n");

	checkStatus(culaSgesv(SMALL_DIM, 1, A, SMALL_DIM, IPIV, X, SMALL_DIM));

	printf("x = [");
	for (int i = 0; i < SMALL_DIM; i++)
		printf(" %f", X[i]);
	printf(" ];\n");

	printf("A * x' - b'\n");

	free(A);
	free(B);
	free(X);
	free(IPIV);
}
#endif

void test_large()
{
	culaFloat* A = (culaFloat*)malloc(LARGE_DIM*LARGE_DIM*sizeof(culaFloat));
	culaFloat* X = (culaFloat*)malloc(LARGE_DIM*sizeof(culaFloat));
	culaInt* IPIV = (culaInt*)malloc(LARGE_DIM*sizeof(culaInt));

	for (int i = 0; i < LARGE_DIM; i++)
	for (int j = 0; j < LARGE_DIM; j++)
		A[i + j*LARGE_DIM] = (100 + xorshift128() % 900) / 100.0;
	for (int i = 0; i < LARGE_DIM; i++)
		X[i] = (100 + xorshift128() % 900) / 100.0;

	Profiler("large_cula_solve")
		checkStatus(culaSgesv(LARGE_DIM, 1, A, LARGE_DIM, IPIV, X, LARGE_DIM));

#ifdef SMALL_DIM
	printf("x_large = [");
	for (int i = 0; i < 10; i++)
		printf(" %f", X[i]);
	printf(" ... ];\n");
#endif

	free(A);
	free(X);
	free(IPIV);
}

int main()
{
	checkStatus(culaInitialize());
#ifdef SMALL_DIM
	test_small();
	test_large();
#else
	printf("Running large test 10 times..\n");
	for (int i = 0; i < 10; i++)
		test_large();
#endif
	culaShutdown();
	Profiler::show();
	return 0;
}


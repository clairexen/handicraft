#define PROFILER_SINGULAR_DEFS
#include "profiler.h"
#include <sys/time.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include "challenge.h"

challenge_s challenge;

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

int main()
{
	setvbuf(stdout, NULL, _IONBF, 0);
	printf("Testing GSL and ViennaCL by solving %d dense %dx%d problems..\n", NUM, DIM, DIM);

	Profiler("init")
	{
		for (int n = 0; n < NUM; n++)
		{
			for (int i = 0; i < DIM; i++)
			for (int j = 0; j < DIM; j++)
				challenge.A[n][i][j] = xorshift128() / double(UINT32_MAX);

			for (int i = 0; i < DIM; i++)
				challenge.b[n][i] = xorshift128() / double(UINT32_MAX);
		}
	}

	printf("Solving using GSL and OpenMP: ");
	Profiler("gsl")
		solve_gsl(challenge);
	printf("%.2f seconds\n", Profiler::get("gsl"));

	printf("Solving using ViennaCL and OpenCL: ");
	Profiler("viennacl") {
		// info_viennacl();
		solve_viennacl(challenge);
	}
	printf("%.2f seconds\n", Profiler::get("viennacl"));

	printf("GSL:");
	for (int i = 0; i < 5; i++)
		printf(" %10.5f", challenge.x_gsl[0][i]);
	printf("    ...\n");

	printf("VCL:");
	for (int i = 0; i < 5; i++)
		printf(" %10.5f", challenge.x_viennacl[0][i]);
	printf("    ...\n");

	Profiler("errnorm") {
		double total_errnorm = 0;
		for (int n = 0; n < NUM; n++) {
			double err_norm = 0;
			for (int i = 0; i < DIM; i++)
				err_norm += pow(challenge.x_gsl[n][i] - challenge.x_viennacl[n][i], 2);
			total_errnorm += sqrt(err_norm);
		}
		printf("Avg. norm of error vector: %.6g\n", total_errnorm / NUM);
	}

	Profiler("fileout")
	{
		FILE *f = fopen("matrix.txt", "wt");
		for (int i = 0; i < DIM; i++) {
			for (int j = 0; j < DIM; j++)
				fprintf(f, "%s%.6g", j ? " " : "", challenge.A[0][i][j]);
			fprintf(f, "\n");
		}
		fclose(f);

		f = fopen("vectors.txt", "wt");
		for (int i = 0; i < DIM; i++)
			fprintf(f, "%.6g %.6g %.6g\n", challenge.b[0][i], challenge.x_gsl[0][i], challenge.x_viennacl[0][i]);
		fclose(f);
	}

	Profiler::show();

	return 0;
}


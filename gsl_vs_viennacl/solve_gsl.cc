#include "profiler.h"
#include "challenge.h"
#include <gsl/gsl_linalg.h>

void solve_gsl(challenge_s &cl)
{
	#pragma omp parallel for
	for (int n = 0; n < NUM; n++)
	{
		gsl_matrix *m = gsl_matrix_alloc(DIM, DIM);
		gsl_vector *b = gsl_vector_alloc(DIM);
		gsl_vector *x = gsl_vector_alloc(DIM);
		gsl_permutation *p = gsl_permutation_alloc(DIM);

		Profiler("gsl_copy_matrix") {
			for (int i = 0; i < DIM; i++)
			for (int j = 0; j < DIM; j++)
				gsl_matrix_set(m, i, j, cl.A[n][i][j]);
		}

		Profiler("gsl_copy_vect1") {
			for (int i = 0; i < DIM; i++)
				gsl_vector_set(b, i, cl.b[n][i]);
		}

		Profiler("gsl_solve") {
			int s;
			gsl_linalg_LU_decomp(m, p, &s);
			gsl_linalg_LU_solve(m, p, b, x);
		}

		Profiler("gsl_copy_vect2") {
			for (int i = 0; i < DIM; i++)
				cl.x_gsl[n][i] = gsl_vector_get(x, i);
		}

		gsl_permutation_free(p);
		gsl_matrix_free(m);
		gsl_vector_free(b);
		gsl_vector_free(x);
	}
}


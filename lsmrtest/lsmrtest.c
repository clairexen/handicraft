#include <stdio.h>

// LSMR Documentation:
// See the large comment in lsmr/lsmrModule.f90 starting at line 67

extern void __lsmrmodule_MOD_lsmr(
	int *m, int *n,
	void (*Aprod1)(int *m, int *n, double *x, double *y),
	void (*Aprod2)(int *m, int *n, double *x, double *y),
	double *b, double *damp, double *atol, double *btol,
	double *conlim, int *itnlim, int *localSize, int *nout,
	double *x, int *istop, int *itn,
	double *normA, double *condA, double *normr, double *normAr, double *normx
);

// Solve A*x = b using the LSMR algorithm
struct LSMR_JOB
{
	// INPUT Arguments:
	int m; // # of rows in A (elements in b)
	int n; // # of columns in A (elements in x)
	void (*Aprod1)(int *m, int *n, double *x, double *y); // calculate y = y + A*x
	void (*Aprod2)(int *m, int *n, double *x, double *y); // calculate x = x + A'*y
	double *b;     // the rhs of the equation (m elements)
	double damp;   // damping parameter
	double atol;   // estimate for rel. error in A (pass zero for eps)
	double btol;   // estimate for rel. error in b (pass zero for eps)
	double conlim; // upper limit of cond(Abar) (pass zero for 1/eps)
	int itnlim;    // upper limit on the # of iterations
	int localSize; // # of vectors for local reorthogonalization
	int nout;      // set to > 0 for debug output

	// OUTPUT Arguments:
	double *x;      // the result vector (n elements)
	int istop;     // reason for termination
	int itn;       // number of iterations performed
	double normA;  // estimate of the Frobenius norm of Abar
	double condA;  // estimate of cond(Abar)
	double normr;  // estimate of the final value of norm(rbar) 
	double normAr; // estimate of the final value of norm(Abar'*rbar)
	double normx;  // estimate of norm(x) for the final solution x
};

void LSMR(struct LSMR_JOB *job)
{
	__lsmrmodule_MOD_lsmr(
		&job->m, &job->n,
		job->Aprod1, job->Aprod2,
		job->b, &job->damp, &job->atol, &job->btol,
		&job->conlim, &job->itnlim, &job->localSize, &job->nout,
		job->x, &job->istop, &job->itn,
		&job->normA, &job->condA, &job->normr, &job->normAr, &job->normx
	);
}

#define M 9
#define N 5

double A[M][N] = {
	{ 1, 2, 0, 0, 0 },
	{ 0, 1, 2, 0, 0 },
	{ 0, 0, 1, 2, 0 },
	{ 0, 0, 0, 1, 2 },
	{ 2, 0, 0, 0, 1 },
	{ 1, 2, 0, 0, 0 },
	{ 0, 1, 2, 0, 0 },
	{ 0, 0, 1, 2, 0 },
	{ 0, 0, 0, 1, 2 }
};
double b[M] = { 5, 8, 11, 14, 7, 5, 8, 11, 14 };
double x[N]; // should be: 1, 2, 3, 4, 5

void aprod1(int *m, int *n, double *x, double *y)
{
	// calculate y = y + A*x
	int i, j;
	for (i = 0; i < *m; i++)
	for (j = 0; j < *n; j++)
		y[i] += A[i][j] * x[j];
}

void aprod2(int *m, int *n, double *x, double *y)
{
	// calculate x = x + A'*y
	int i, j;
	for (i = 0; i < *m; i++)
	for (j = 0; j < *n; j++)
		x[j] += A[i][j] * y[i];
}

int main()
{
	int i;
	struct LSMR_JOB job;

	job.m = M;
	job.n = N;
	job.Aprod1 = &aprod1;
	job.Aprod2 = &aprod2;
	job.b = b;
	job.damp = 0;
	job.atol = 0;
	job.btol = 0;
	job.conlim = 0;
	job.itnlim = 10;
	job.localSize = 3;
	job.nout = 1;
	job.x = x;

	LSMR(&job);

	printf("x:");
	for (i = 0; i < N; i++)
		printf(" %f", job.x[i]);
	printf("\n");

	printf("istop=%d, itn=%d, normA=%g, condA=%g, normr=%g, normAr=%g, normx=%g\n",
		job.istop, job.itn, job.normA, job.condA, job.normr, job.normAr, job.normx);

	return 0;
}


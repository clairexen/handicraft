#ifndef CHALLENGE_H
#define CHALLENGE_H

#define DIM 1000
#define NUM 1

struct challenge_s
{
	double A[NUM][DIM][DIM];
	double b[NUM][DIM];
	double x_gsl[NUM][DIM];
	double x_viennacl[NUM][DIM];
};

void solve_gsl(challenge_s &cl);

void info_viennacl();
void solve_viennacl(challenge_s &cl);

#endif

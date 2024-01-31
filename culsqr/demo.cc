
#include "culsqr.h"
#include <stdio.h>
#include <stdlib.h>

int main()
{
	CuLsqr solver(15, 6, 3);
	double sol[10] = { 1, 2, 3, 4, 5, 6 };

	for (int i = 0; i < 15; i++)
	{
		double s = 0;
		for (int j = 0; j < 3; j++) {
			int idx = 2*j + lrand48() % 2;
			double koeff = drand48();
			solver.addEntry(i, idx, koeff);
			s += sol[idx] * koeff;
		}
		solver.rhs(0, i) = s; // + 0.01*xs128.real();
		solver.rhs(1, i) = s; // + 0.01*xs128.real();
		solver.rhs(2, i) = s; // + 0.01*xs128.real();
	}

	solver.solve(true);

	printf("\n");
	for (int i = 0; i < 6; i++)
		printf("%8.5f %8.5f %8.5f\n", solver.sol(0, i), solver.sol(1, i), solver.sol(2, i));

	return 0;
}


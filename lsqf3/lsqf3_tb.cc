#include <stdio.h>
#include <vector>
#include "lsqf3.h"

const double limit = 100;

struct eq_t {
	double lhs[3], rhs;
};

int main()
{
	double p[3] = { 1, 2, 3 };
	std::vector<eq_t> eqs;

	printf("--- inputs ---\n");
	for (int i = 0; i < 16; i++) {
		eq_t eq = {{
				drand48()*limit - limit/2,
				drand48()*limit - limit/2,
				drand48()*limit - limit/2},
				0
		};
		for (int j = 0; j < 3; j++)
			eq.rhs += p[j]*eq.lhs[j];
		printf("%10.3f %10.3f %10.3f %10.3f\n",
				eq.lhs[0], eq.lhs[1], eq.lhs[2], eq.rhs);
		eqs.push_back(eq);
	}

	din_t *input = new din_t[eqs.size()*4+4];
	dout_t output[3];
	for (size_t i = 0; i < eqs.size(); i++) {
		input[4*i+0] = eqs[i].lhs[0];
		input[4*i+1] = eqs[i].lhs[1];
		input[4*i+2] = eqs[i].lhs[2];
		input[4*i+3] = eqs[i].rhs;
	}
	for (size_t i = 0; i < 4; i++) {
		input[4*eqs.size()+i] = 0;
	}
	lsqf3(input, output);
	delete input;

	printf("--- results ---\n");
	for (int i = 0; i < 3; i++) {
		double v1 = p[i], v2 = output[i];
		printf("%10.3f %10.3f\n", v1, v2);
	}

	return 0;
}

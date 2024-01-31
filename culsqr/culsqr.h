#ifndef CULSQR_H
#define CULSQR_H

#include <sys/types.h>
#include <vector>

struct CuLsqr
{
	size_t m, n, r;
	std::vector< std::vector<size_t> > matrixIndex;
	std::vector< std::vector<double> > matrixValue;
	std::vector< std::vector<double> > rhsData;
	std::vector< std::vector<double> > solData;

	CuLsqr(size_t m, size_t n, size_t r);
	void addEntry(size_t i, size_t j, double value);
	void solve(bool debug = false);

	double &rhs(size_t k, size_t i) { return rhsData.at(k).at(i); }
	double &sol(size_t k, size_t i) { return solData.at(k).at(i); }

private:
	static void kernel_make_upper_triangle_Dcsr(int n, double *val, int *row, int *col, int *nnzPtr = NULL);
};

#endif

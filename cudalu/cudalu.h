#ifndef CUDALU_H
#define CUDALU_H

#include <cublas_v2.h>
#include <sys/types.h>

class CudaLU
{
public:
	typedef float ScalarType;

private:
	size_t dim;
	ScalarType *host_matrix, *dev_matrix, *dev_vect;
	cublasHandle_t cublasHandle;

public:
	CudaLU(size_t dim);
	~CudaLU();

	size_t size() const { return dim; }
	ScalarType &operator() (size_t i, size_t j);
	const ScalarType &operator() (size_t i, size_t j) const;
	void updateGPU();
	void updateCPU();
	void factorize_lu();
	void solve(ScalarType *vect) const;
	void printMatrix(char type = 0) const;
};

#endif

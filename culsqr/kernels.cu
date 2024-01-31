
#include "culsqr.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

#define CUDA_CHECK_RETURN(value) do {                                                           \
		cudaError_t _m_cudaStat = value;                                                \
		if (_m_cudaStat != cudaSuccess) {                                               \
			fprintf(stderr, "Error %s at line %d in file %s\n",                     \
					cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
			exit(1);                                                                \
	} } while (0)

static __global__ void kernel_make_upper_triangle_Dcsr_dev(int n, double *val, int *row, int *col)
{
	int k = row[0], p = k;
	for (int i = 0; i < n; i++) {
		while (row[i+1] != k) {
			if (col[k] >= i) {
				val[p] = val[k];
				col[p] = col[k];
				p++;
			}
			k++;
		}
		row[i+1] = p;
	}
}

void CuLsqr::kernel_make_upper_triangle_Dcsr(int n, double *val, int *row, int *col, int *nnzPtr)
{
	// just use a single thread. this does not need to be fast - we just want to
	// avoid transfering the matrix data back and forth between gpu and cpu
	kernel_make_upper_triangle_Dcsr_dev<<<1, 1>>>(n, val, row, col);

	if (nnzPtr)
		CUDA_CHECK_RETURN(cudaMemcpy(nnzPtr, row + n, sizeof(int), cudaMemcpyDeviceToHost));
}


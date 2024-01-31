#include "cudalu.h"
#include <stdio.h>

static void HandleErrorCUDA(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(1);
	}
}

static void HandleErrorCUBLAS(cublasStatus_t err, const char *file, int line)
{
	if (err != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLASS Error %d in %s at line %d\n", err, file, line);
		exit(1);
	}
}

#define HANDLE_ERROR_CUDA(err) HandleErrorCUDA(err, __FILE__, __LINE__)
#define HANDLE_ERROR_CUBLAS(err) HandleErrorCUBLAS(err, __FILE__, __LINE__)
#define HANDLE_ERROR_NULL(a) do { if ((a) == NULL) { printf( "Unexpected NULL in %s at line %d\n", __FILE__, __LINE__ ); exit( EXIT_FAILURE );}} while (0)

CudaLU::CudaLU(size_t dim)
{
	this->dim = dim;
	HANDLE_ERROR_NULL(host_matrix = (ScalarType*)malloc(sizeof(ScalarType) * dim * dim));
	HANDLE_ERROR_CUDA(cudaMalloc((void**)&dev_matrix, sizeof(ScalarType) * dim * dim));
	HANDLE_ERROR_CUDA(cudaMalloc((void**)&dev_vect, sizeof(ScalarType) * dim));
	HANDLE_ERROR_CUBLAS(cublasCreate(&cublasHandle));
}

CudaLU::~CudaLU()
{
	free(host_matrix);
	cudaFree(dev_matrix);
	cudaFree(dev_vect);
	cublasDestroy(cublasHandle);
}

CudaLU::ScalarType &CudaLU::operator()(size_t i, size_t j)
{
	return host_matrix[i + j*dim];
}

const CudaLU::ScalarType &CudaLU::operator()(size_t i, size_t j) const
{
	return host_matrix[i + j*dim];
}

void CudaLU::updateGPU()
{
	HANDLE_ERROR_CUDA(cudaMemcpy(dev_matrix, host_matrix, sizeof(ScalarType) * dim * dim, cudaMemcpyHostToDevice));
}

void CudaLU::updateCPU()
{
	HANDLE_ERROR_CUDA(cudaMemcpy(host_matrix, dev_matrix, sizeof(ScalarType) * dim * dim, cudaMemcpyDeviceToHost));
}

template<typename ScalarType>
__global__ void kernel_factors(int dim, int stage, ScalarType *matrix)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + stage;
	if (i < dim) {
		int idx_zeroer = (stage-1) + (stage-1)*dim;
		int idx_zeroee = i + (stage-1)*dim;
		matrix[idx_zeroee] = matrix[idx_zeroee] / matrix[idx_zeroer];
	}
}

template<typename ScalarType>
__global__ void kernel_lu(int dim, int stage, ScalarType *matrix)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + stage;
	int j = blockIdx.y * blockDim.y + stage;
	if (i < dim) {
		int idx_factor = i + (stage-1)*dim;
		int idx_adder = (stage-1) + j*dim;
		int idx_addee = i + j*dim;
		matrix[idx_addee] -= matrix[idx_adder] * matrix[idx_factor];
	}
}

void CudaLU::factorize_lu()
{
	int threads = 192;
	for (int stage = 1; stage < dim; stage++) {
		dim3 grid(ceil((dim-stage)/float(threads)), dim-stage);
		kernel_factors<ScalarType><<<grid.x,threads>>>(dim, stage, dev_matrix);
		kernel_lu<ScalarType><<<grid,threads>>>(dim, stage, dev_matrix);
	}
}

void CudaLU::solve(ScalarType *vect) const
{
	HANDLE_ERROR_CUDA(cudaMemcpy(dev_vect, vect, sizeof(ScalarType) * dim, cudaMemcpyHostToDevice));
	HANDLE_ERROR_CUBLAS(cublasStrsv(cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, dim, dev_matrix, dim, dev_vect, 1));
	HANDLE_ERROR_CUBLAS(cublasStrsv(cublasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, dim, dev_matrix, dim, dev_vect, 1));
	HANDLE_ERROR_CUDA(cudaMemcpy(vect, dev_vect, sizeof(ScalarType) * dim, cudaMemcpyDeviceToHost));
}

void CudaLU::printMatrix(char type) const
{
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++)
			if (type == 'l' && i <= j)
				printf(" %7s", i == j ? "1  " : "0  ");
			else if (type == 'u' && i > j)
				printf(" %7s", "0  ");
			else if (fabs((*this)(i, j)) < 0.001)
				printf(" %7s", "0  ");
			else
				printf(" %7.2f", (*this)(i, j));
		printf("\n");
	}
}



#include "culsqr.h"
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <stdlib.h>
#include <stdio.h>

#undef USE_LU_INSTEAD_OF_CHOLESKY
#undef USE_CSRSV_INSTEAD_OF_SCRSM

#define CUDA_CHECK_RETURN(value) do {                                                           \
		cudaError_t _m_cudaStat = value;                                                \
		if (_m_cudaStat != cudaSuccess) {                                               \
			fprintf(stderr, "Error %s at line %d in file %s\n",                     \
					cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
			exit(1);                                                                \
	} } while (0)

#define CUSPARSE_CHECK_RETURN(value) do {                                                       \
		cusparseStatus_t _m_cusparseStat = value;                                       \
		if (_m_cusparseStat != CUSPARSE_STATUS_SUCCESS) {                               \
			const char *errMessage = NULL;                                          \
			switch (_m_cusparseStat) {                                              \
			case CUSPARSE_STATUS_SUCCESS:                                           \
				errMessage = "SUCCESS"; break;                                  \
			case CUSPARSE_STATUS_NOT_INITIALIZED:                                   \
				errMessage = "NOT_INITIALIZED"; break;                          \
			case CUSPARSE_STATUS_ALLOC_FAILED:                                      \
				errMessage = "ALLOC_FAILED"; break;                             \
			case CUSPARSE_STATUS_INVALID_VALUE:                                     \
				errMessage = "INVALID_VALUE"; break;                            \
			case CUSPARSE_STATUS_ARCH_MISMATCH:                                     \
				errMessage = "ARCH_MISMATCH"; break;                            \
			case CUSPARSE_STATUS_MAPPING_ERROR:                                     \
				errMessage = "MAPPING_ERROR"; break;                            \
			case CUSPARSE_STATUS_EXECUTION_FAILED:                                  \
				errMessage = "EXECUTION_FAILED"; break;                         \
			case CUSPARSE_STATUS_INTERNAL_ERROR:                                    \
				errMessage = "INTERNAL_ERROR"; break;                           \
			case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:                         \
				errMessage = "MATRIX_TYPE_NOT_SUPPORTED"; break;                \
			default:                                                                \
				errMessage = "UNKOWN_ERROR_CODE";                               \
			}                                                                       \
			fprintf(stderr, "CUSPARSE Error code %d (%s) at line %d in file %s\n",  \
					int(_m_cusparseStat), errMessage, __LINE__, __FILE__);  \
			exit(1);                                                                \
	} } while (0)

CuLsqr::CuLsqr(size_t m, size_t n, size_t r)
{
	this->m = m;
	this->n = n;
	this->r = r;

	this->matrixIndex.resize(this->m);
	this->matrixValue.resize(this->m);

	this->rhsData.resize(this->r);
	this->solData.resize(this->r);

	for (size_t i = 0; i < this->r; i++) {
		this->rhsData[i].resize(this->m);
		this->solData[i].resize(this->n);
	}
}

void CuLsqr::addEntry(size_t i, size_t j, double value)
{
	this->matrixIndex.at(i).push_back(j);
	this->matrixValue.at(i).push_back(value);
}

static void printCSR(int m, int n, int nnzA, double *csrValA, int *csrRowPtrA, int *csrColIndA)
{
	int k = 0;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++) {
			if (k == csrRowPtrA[i+1] || j != csrColIndA[k])
				printf(" %8s", "0    ");
			else
				printf(" %8.3f", csrValA[k++]);
		}
		printf("\n");
	}
}

static void printCSRdev(int m, int n, int nnzA, double *csrValAdev, int *csrRowPtrAdev, int *csrColIndAdev)
{
	double *csrValA = NULL;
	int *csrRowPtrA = NULL;
	int *csrColIndA = NULL;

	CUDA_CHECK_RETURN(cudaMallocHost((void**)&csrValA, sizeof(double) * nnzA));
	CUDA_CHECK_RETURN(cudaMallocHost((void**)&csrRowPtrA, sizeof(int) * (m+1)));
	CUDA_CHECK_RETURN(cudaMallocHost((void**)&csrColIndA, sizeof(int) * nnzA));

	CUDA_CHECK_RETURN(cudaMemcpy(csrValA, csrValAdev, sizeof(double) * nnzA, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(csrRowPtrA, csrRowPtrAdev, sizeof(int) * (m+1), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(csrColIndA, csrColIndAdev, sizeof(int) * nnzA, cudaMemcpyDeviceToHost));

	printCSR(m, n, nnzA, csrValA, csrRowPtrA, csrColIndA);

	CUDA_CHECK_RETURN(cudaFreeHost(csrValA));
	CUDA_CHECK_RETURN(cudaFreeHost(csrRowPtrA));
	CUDA_CHECK_RETURN(cudaFreeHost(csrColIndA));
}

static void printDENSE(int m, int n, double *data)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
			printf(" %8.3f", data[j*m + i]);
		printf("\n");
	}
}

static void printDENSEdev(int m, int n, double *data_dev)
{
	double *data = NULL;

	CUDA_CHECK_RETURN(cudaMallocHost((void**)&data, sizeof(double) * m * n));
	CUDA_CHECK_RETURN(cudaMemcpy(data, data_dev, sizeof(double) * m * n, cudaMemcpyDeviceToHost));

	printDENSE(m, n, data);

	CUDA_CHECK_RETURN(cudaFreeHost(data));
}

void CuLsqr::solve(bool debug)
{
	cusparseHandle_t handle;
	CUSPARSE_CHECK_RETURN(cusparseCreate(&handle));

	cusparseSolveAnalysisInfo_t info;
	CUSPARSE_CHECK_RETURN(cusparseCreateSolveAnalysisInfo(&info));

	/*
	 * generate matrix A in host memory
	 */

	cusparseMatDescr_t descrA;
	CUSPARSE_CHECK_RETURN(cusparseCreateMatDescr(&descrA));

	int nnzA = 0;
	double *csrValA = NULL;
	int *csrRowPtrA = NULL;
	int *csrColIndA = NULL;

	for (auto &row : this->matrixIndex)
		nnzA += row.size();

	CUDA_CHECK_RETURN(cudaMallocHost((void**)&csrValA, sizeof(double) * nnzA));
	CUDA_CHECK_RETURN(cudaMallocHost((void**)&csrRowPtrA, sizeof(int) * (this->m+1)));
	CUDA_CHECK_RETURN(cudaMallocHost((void**)&csrColIndA, sizeof(int) * nnzA));

	int k = 0;
	for (size_t i = 0; i < this->m; i++) {
		csrRowPtrA[i] = k;
		for (size_t j = 0; j < this->matrixIndex.at(i).size(); j++) {
			csrColIndA[k] = this->matrixIndex.at(i).at(j);
			csrValA[k++] = this->matrixValue.at(i).at(j);
		}
	}
	csrRowPtrA[this->m] = k;

	/*
	 * copy matrix A to device memory
	 */

	double *csrValAdev = NULL;
	int *csrRowPtrAdev = NULL;
	int *csrColIndAdev = NULL;
	
	CUDA_CHECK_RETURN(cudaMalloc((void**)&csrValAdev, sizeof(double) * nnzA));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&csrRowPtrAdev, sizeof(int) * (this->m+1)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&csrColIndAdev, sizeof(int) * nnzA));

	CUDA_CHECK_RETURN(cudaMemcpy(csrValAdev, csrValA, sizeof(double) * nnzA, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(csrRowPtrAdev, csrRowPtrA, sizeof(int) * (this->m+1), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(csrColIndAdev, csrColIndA, sizeof(int) * nnzA, cudaMemcpyHostToDevice));

	if (debug) {
		printf("\nA = [\n");
		printCSRdev(this->m, this->n, nnzA, csrValAdev, csrRowPtrAdev, csrColIndAdev);
		printf("];\n");
	}

	/*
	 * create matrix AtA = A' * A on device
	 */

	cusparseMatDescr_t descrAtA;
	CUSPARSE_CHECK_RETURN(cusparseCreateMatDescr(&descrAtA));

	int nnzAtA = 0;
	double *csrValAtAdev = NULL;
	int *csrRowPtrAtAdev = NULL;
	int *csrColIndAtAdev = NULL;

	CUDA_CHECK_RETURN(cudaMalloc((void**)&csrRowPtrAtAdev, sizeof(int) * (this->n+1)));

	CUSPARSE_CHECK_RETURN(cusparseXcsrgemmNnz(handle,
			CUSPARSE_OPERATION_TRANSPOSE,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			this->n, this->n, this->m,
			descrA, nnzA, csrRowPtrAdev, csrColIndAdev,
			descrA, nnzA, csrRowPtrAdev, csrColIndAdev,
			descrAtA, csrRowPtrAtAdev, &nnzAtA));

	CUDA_CHECK_RETURN(cudaMalloc((void**)&csrValAtAdev, sizeof(double) * nnzAtA));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&csrColIndAtAdev, sizeof(int) * nnzAtA));

	CUSPARSE_CHECK_RETURN(cusparseDcsrgemm(handle,
			CUSPARSE_OPERATION_TRANSPOSE,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			this->n, this->n, this->m,
			descrA, nnzA, csrValAdev, csrRowPtrAdev, csrColIndAdev,
			descrA, nnzA, csrValAdev, csrRowPtrAdev, csrColIndAdev,
			descrAtA, csrValAtAdev, csrRowPtrAtAdev, csrColIndAtAdev));

	if (debug) {
		printf("\nAtA = [\n");
		printCSRdev(this->n, this->n, nnzAtA, csrValAtAdev, csrRowPtrAtAdev, csrColIndAtAdev);
		printf("];\n");
	}

	/*
	 * create dense matrix Y = rhs and copy to device
	 */

	double *Y = NULL;
	double *Ydev = NULL;

	CUDA_CHECK_RETURN(cudaMallocHost((void**)&Y, sizeof(double) * this->m * this->r));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&Ydev, sizeof(double) * this->m * this->r));

	for (size_t k = 0; k < this->r; k++)
	for (size_t i = 0; i < this->m; i++)
		Y[k*this->m + i] = this->rhs(k, i);

	CUDA_CHECK_RETURN(cudaMemcpy(Ydev, Y, sizeof(double) * this->m * this->r, cudaMemcpyHostToDevice));

	if (debug) {
		printf("\nY = [\n");
		printDENSEdev(this->m,  this->r, Ydev);
		printf("];\n");
	}

	/*
	 * create dense matrix X1 = A' * Y on device
	 */

	double *X1dev = NULL;
	double alpha = 1.0, beta = 0.0;

	CUDA_CHECK_RETURN(cudaMalloc((void**)&X1dev, sizeof(double) * this->n * this->r));

	CUSPARSE_CHECK_RETURN(cusparseDcsrmm(handle,
			CUSPARSE_OPERATION_TRANSPOSE,
			this->m, this->r, this->n,
			nnzA, &alpha, descrA, csrValAdev, csrRowPtrAdev, csrColIndA,
			Ydev, this->m, &beta, X1dev, this->n));

	if (debug) {
		printf("\nX1 = [\n");
		printDENSEdev(this->n,  this->r, X1dev);
		printf("];\n");
	}

#ifdef USE_LU_INSTEAD_OF_CHOLESKY

	/*
	 * create L * U = AtA on device (L and U are still stored in AtA memory)
	 */

	CUSPARSE_CHECK_RETURN(cusparseDcsrsv_analysis(handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE, this->n,
			nnzAtA, descrAtA, csrValAtAdev, csrRowPtrAtAdev, csrColIndAtAdev, info));

	CUSPARSE_CHECK_RETURN(cusparseDcsrilu0(handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE, this->n,
			descrAtA, csrValAtAdev, csrRowPtrAtAdev, csrColIndAtAdev, info));

	if (debug) {
		printf("\nLU = [\n");
		printCSRdev(this->n, this->n, nnzAtA, csrValAtAdev, csrRowPtrAtAdev, csrColIndAtAdev);
		printf("]; L = tril(LU, -1) + eye(size(LU,1)); U = triu(LU, 0);\n");
	}

	/*
	 * solve L * X2 = X1
	 */

	double *X2dev = NULL;

	CUDA_CHECK_RETURN(cudaMalloc((void**)&X2dev, sizeof(double) * this->n * this->r));

	CUSPARSE_CHECK_RETURN(cusparseSetMatType(descrAtA, CUSPARSE_MATRIX_TYPE_GENERAL));
	CUSPARSE_CHECK_RETURN(cusparseSetMatFillMode(descrAtA, CUSPARSE_FILL_MODE_LOWER));
	CUSPARSE_CHECK_RETURN(cusparseSetMatDiagType(descrAtA, CUSPARSE_DIAG_TYPE_UNIT));

	CUSPARSE_CHECK_RETURN(cusparseDcsrsm_analysis(handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE, this->n, nnzAtA,
			descrAtA, csrValAtAdev, csrRowPtrAtAdev, csrColIndAtAdev, info));

#  ifdef USE_CSRSV_INSTEAD_OF_SCRSM

	for (int i = 0; i < this->r; i++)
		CUSPARSE_CHECK_RETURN(cusparseDcsrsv_solve(handle,
				CUSPARSE_OPERATION_NON_TRANSPOSE, this->n, &alpha,
				descrAtA, csrValAtAdev, csrRowPtrAtAdev, csrColIndAtAdev, info,
				X1dev + i*this->n, X2dev + i*this->n));

#  else // USE_CSRSV_INSTEAD_OF_SCRSM

	CUSPARSE_CHECK_RETURN(cusparseDcsrsm_solve(handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE, this->n, this->r, &alpha,
			descrAtA, csrValAtAdev, csrRowPtrAtAdev, csrColIndAtAdev, info,
			X1dev, this->n, X2dev, this->n));

#  endif // USE_CSRSV_INSTEAD_OF_SCRSM

	if (debug) {
		printf("\nX2 = [\n");
		printDENSEdev(this->n,  this->r, X2dev);
		printf("];\n");
	}

	/*
	 * solve U * X3 = X2
	 */

	double *X3dev = NULL;

	CUDA_CHECK_RETURN(cudaMalloc((void**)&X3dev, sizeof(double) * this->n * this->r));

	CUSPARSE_CHECK_RETURN(cusparseSetMatType(descrAtA, CUSPARSE_MATRIX_TYPE_GENERAL));
	CUSPARSE_CHECK_RETURN(cusparseSetMatFillMode(descrAtA, CUSPARSE_FILL_MODE_UPPER));
	CUSPARSE_CHECK_RETURN(cusparseSetMatDiagType(descrAtA, CUSPARSE_DIAG_TYPE_NON_UNIT));

	CUSPARSE_CHECK_RETURN(cusparseDcsrsm_analysis(handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE, this->n, nnzAtA,
			descrAtA, csrValAtAdev, csrRowPtrAtAdev, csrColIndAtAdev, info));

#  ifdef USE_CSRSV_INSTEAD_OF_SCRSM

	for (int i = 0; i < this->r; i++)
		CUSPARSE_CHECK_RETURN(cusparseDcsrsv_solve(handle,
				CUSPARSE_OPERATION_NON_TRANSPOSE, this->n, &alpha,
				descrAtA, csrValAtAdev, csrRowPtrAtAdev, csrColIndAtAdev, info,
				X2dev + i*this->n, X3dev + i*this->n));

#  else // USE_CSRSV_INSTEAD_OF_SCRSM

	CUSPARSE_CHECK_RETURN(cusparseDcsrsm_solve(handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE, this->n, this->r, &alpha,
			descrAtA, csrValAtAdev, csrRowPtrAtAdev, csrColIndAtAdev, info,
			X2dev, this->n, X3dev, this->n));

#  endif // USE_CSRSV_INSTEAD_OF_SCRSM

	if (debug) {
		printf("\nX3 = [\n");
		printDENSEdev(this->n,  this->r, X3dev);
		printf("];\n");
	}

#else // USE_LU_INSTEAD_OF_CHOLESKY

	/*
	 * create R' * R = AtA on device (R is still stored in AtA memory)
	 */

	CUSPARSE_CHECK_RETURN(cusparseSetMatType(descrAtA, CUSPARSE_MATRIX_TYPE_SYMMETRIC));
	CUSPARSE_CHECK_RETURN(cusparseSetMatFillMode(descrAtA, CUSPARSE_FILL_MODE_UPPER));
	CUSPARSE_CHECK_RETURN(cusparseSetMatDiagType(descrAtA, CUSPARSE_DIAG_TYPE_NON_UNIT));

	kernel_make_upper_triangle_Dcsr(this->n, csrValAtAdev, csrRowPtrAtAdev, csrColIndAtAdev, &nnzAtA);

	CUSPARSE_CHECK_RETURN(cusparseDcsrsv_analysis(handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE, this->n,
			nnzAtA, descrAtA, csrValAtAdev, csrRowPtrAtAdev, csrColIndAtAdev, info));

	CUSPARSE_CHECK_RETURN(cusparseDcsric0(handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE, this->n,
			descrAtA, csrValAtAdev, csrRowPtrAtAdev, csrColIndAtAdev, info));

	if (debug) {
		printf("\nR = [\n");
		printCSRdev(this->n, this->n, nnzAtA, csrValAtAdev, csrRowPtrAtAdev, csrColIndAtAdev);
		printf("];\n");
	}

	/*
	 * solve R' * X2 = X1
	 */

	double *X2dev = NULL;

	CUDA_CHECK_RETURN(cudaMalloc((void**)&X2dev, sizeof(double) * this->n * this->r));

	CUSPARSE_CHECK_RETURN(cusparseSetMatType(descrAtA, CUSPARSE_MATRIX_TYPE_GENERAL));
	CUSPARSE_CHECK_RETURN(cusparseSetMatFillMode(descrAtA, CUSPARSE_FILL_MODE_UPPER));
	CUSPARSE_CHECK_RETURN(cusparseSetMatDiagType(descrAtA, CUSPARSE_DIAG_TYPE_NON_UNIT));

	CUSPARSE_CHECK_RETURN(cusparseDcsrsm_analysis(handle,
			CUSPARSE_OPERATION_TRANSPOSE, this->n, nnzAtA,
			descrAtA, csrValAtAdev, csrRowPtrAtAdev, csrColIndAtAdev, info));

#  ifdef USE_CSRSV_INSTEAD_OF_SCRSM

	for (int i = 0; i < this->r; i++)
		CUSPARSE_CHECK_RETURN(cusparseDcsrsv_solve(handle,
				CUSPARSE_OPERATION_TRANSPOSE, this->n, &alpha,
				descrAtA, csrValAtAdev, csrRowPtrAtAdev, csrColIndAtAdev, info,
				X1dev + i*this->n, X2dev + i*this->n));

#  else // USE_CSRSV_INSTEAD_OF_SCRSM

	CUSPARSE_CHECK_RETURN(cusparseDcsrsm_solve(handle,
			CUSPARSE_OPERATION_TRANSPOSE, this->n, this->r, &alpha,
			descrAtA, csrValAtAdev, csrRowPtrAtAdev, csrColIndAtAdev, info,
			X1dev, this->n, X2dev, this->n));

#  endif // USE_CSRSV_INSTEAD_OF_SCRSM

	if (debug) {
		printf("\nX2 = [\n");
		printDENSEdev(this->n,  this->r, X2dev);
		printf("];\n");
	}

	/*
	 * solve R * X3 = X2
	 */

	double *X3dev = NULL;

	CUDA_CHECK_RETURN(cudaMalloc((void**)&X3dev, sizeof(double) * this->n * this->r));

	CUSPARSE_CHECK_RETURN(cusparseSetMatType(descrAtA, CUSPARSE_MATRIX_TYPE_GENERAL));
	CUSPARSE_CHECK_RETURN(cusparseSetMatFillMode(descrAtA, CUSPARSE_FILL_MODE_UPPER));
	CUSPARSE_CHECK_RETURN(cusparseSetMatDiagType(descrAtA, CUSPARSE_DIAG_TYPE_NON_UNIT));

	CUSPARSE_CHECK_RETURN(cusparseDcsrsm_analysis(handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE, this->n, nnzAtA,
			descrAtA, csrValAtAdev, csrRowPtrAtAdev, csrColIndAtAdev, info));

#  ifdef USE_CSRSV_INSTEAD_OF_SCRSM

	for (int i = 0; i < this->r; i++)
		CUSPARSE_CHECK_RETURN(cusparseDcsrsv_solve(handle,
				CUSPARSE_OPERATION_NON_TRANSPOSE, this->n, &alpha,
				descrAtA, csrValAtAdev, csrRowPtrAtAdev, csrColIndAtAdev, info,
				X2dev + i*this->n, X3dev + i*this->n));

#  else // USE_CSRSV_INSTEAD_OF_SCRSM

	CUSPARSE_CHECK_RETURN(cusparseDcsrsm_solve(handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE, this->n, this->r, &alpha,
			descrAtA, csrValAtAdev, csrRowPtrAtAdev, csrColIndAtAdev, info,
			X2dev, this->n, X3dev, this->n));

#  endif // USE_CSRSV_INSTEAD_OF_SCRSM

	if (debug) {
		printf("\nX3 = [\n");
		printDENSEdev(this->n,  this->r, X3dev);
		printf("];\n");
	}

#endif // USE_LU_INSTEAD_OF_CHOLESKY

	/*
	 * copy X data back to host
	 */

	double *X = NULL;

	CUDA_CHECK_RETURN(cudaMallocHost((void**)&X, sizeof(double) * this->n * this->r));
	CUDA_CHECK_RETURN(cudaMemcpy(X, X3dev, sizeof(double) * this->n * this->r, cudaMemcpyDeviceToHost));

	for (size_t k = 0; k < this->r; k++)
	for (size_t i = 0; i < this->n; i++)
		this->sol(k, i) = X[k*this->n + i];

	/*
	 * cleanup
	 */

	CUDA_CHECK_RETURN(cudaFreeHost(csrValA));
	CUDA_CHECK_RETURN(cudaFreeHost(csrRowPtrA));
	CUDA_CHECK_RETURN(cudaFreeHost(csrColIndA));

	CUDA_CHECK_RETURN(cudaFree(csrValAdev));
	CUDA_CHECK_RETURN(cudaFree(csrRowPtrAdev));
	CUDA_CHECK_RETURN(cudaFree(csrColIndAdev));

	CUDA_CHECK_RETURN(cudaFree(csrValAtAdev));
	CUDA_CHECK_RETURN(cudaFree(csrRowPtrAtAdev));
	CUDA_CHECK_RETURN(cudaFree(csrColIndAtAdev));

	CUDA_CHECK_RETURN(cudaFreeHost(Y));
	CUDA_CHECK_RETURN(cudaFree(Ydev));

	CUDA_CHECK_RETURN(cudaFreeHost(X));
	CUDA_CHECK_RETURN(cudaFree(X1dev));
	CUDA_CHECK_RETURN(cudaFree(X2dev));
	CUDA_CHECK_RETURN(cudaFree(X3dev));

	CUSPARSE_CHECK_RETURN(cusparseDestroySolveAnalysisInfo(info));
	CUSPARSE_CHECK_RETURN(cusparseDestroy(handle));
}


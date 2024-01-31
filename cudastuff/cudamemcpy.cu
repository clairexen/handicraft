#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#include <cuda_profiler_api.h>

void cudaCheckError_(cudaError_t err, const char *filename, int linenr)
{
	if (err != cudaSuccess) {
		fprintf(stderr, "Cuda error '%s' in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		abort();
	}
}

#define cudaCheckError(expr_) cudaCheckError_(expr_, __FILE__, __LINE__)

__global__
void my_memcpy_test_1(const int *src, int *dst)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	dst[idx] = src[idx];
}

__global__
void my_memcpy_test_2(const int *src, int *dst)
{
	int idx = 32 * (threadIdx.x + blockIdx.x * blockDim.x);
	for (int i = 0; i < 32; i++)
		dst[idx + i] = src[idx + i];
}

__global__
void my_memcpy_test_3(const int *src, int *dst)
{
	int idx = 32 * (threadIdx.x + blockIdx.x * blockDim.x);
	for (int i = 0; i < 32; i++)
		dst[idx + i] = __ldg(&src[idx + i]);
}

int main()
{
	int *buffer_a, *buffer_b;

	/* setup */

	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	/* allocate */

	cudaCheckError(cudaMalloc((void**)&buffer_a, sizeof(int)*1024*1024));
	cudaCheckError(cudaMalloc((void**)&buffer_b, sizeof(int)*1024*1024));

	/* start collecting profiling data */

	cudaCheckError(cudaProfilerStart());

	/* run test #1 */

	my_memcpy_test_1<<<1024, 1024>>>(buffer_a, buffer_b);
	cudaCheckError(cudaDeviceSynchronize());

	/* run test #2 */

	my_memcpy_test_2<<<1024, 32>>>(buffer_a, buffer_b);
	cudaCheckError(cudaDeviceSynchronize());

	/* run test #3 */

	my_memcpy_test_3<<<1024, 32>>>(buffer_a, buffer_b);
	cudaCheckError(cudaDeviceSynchronize());

	/* done collecting profiling data */

	cudaCheckError(cudaProfilerStop());

	/* free */

	cudaCheckError(cudaFree(buffer_a));
	cudaCheckError(cudaFree(buffer_b));

	return 0;
}


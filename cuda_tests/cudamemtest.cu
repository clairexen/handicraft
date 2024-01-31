// Simple CUDA demo program. Demonstrating that it's ok to read/write access
// words in the same cache line from different GPU warps.

// nvcc -o cudamemtest -O3 -gencode arch=compute_20,code=sm_21 cudamemtest.cu
// ./cudamemtest

#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>

#define N 1000000
#define M 1000

#define BLOCK_A  128
#define BLOCK_B 1024

static void HandleErrorCUDA(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(1);
	}
}

#define HANDLE_ERROR_CUDA(_err) HandleErrorCUDA(_err, __FILE__, __LINE__)
#define HANDLE_ERROR_NULL(_ptr) do { if ((_ptr) == NULL) { printf( "Unexpected NULL in %s at line %d\n", __FILE__, __LINE__ ); exit(EXIT_FAILURE); }} while (0)

class XorShift128
{
	uint32_t x, y, z, w;
public:
	XorShift128() : x(123456789), y(362436069), z(521288629), w(88675123) { }
	uint32_t operator()(uint32_t max) {
		uint32_t t = x ^ (x << 11);
		x = y; y = z; z = w;
		w ^= (w >> 19) ^ t ^ (t >> 8);
		return w % max;
	}
};

__global__ void test_kernel(int *map, int *data, int offset)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < N)
		data[map[(idx + offset) % N]]++;
}

int main()
{
	cudaStream_t stream;
	HANDLE_ERROR_CUDA(cudaStreamCreate(&stream));

	int *host_map, *host_data;
	int *dev_map, *dev_data;
	bool found_error = false;

	HANDLE_ERROR_CUDA(cudaMallocHost((void**)&host_map, N * sizeof(int)));
	HANDLE_ERROR_CUDA(cudaMallocHost((void**)&host_data, N * sizeof(int)));

	HANDLE_ERROR_CUDA(cudaMalloc((void**)&dev_map, N * sizeof(int)));
	HANDLE_ERROR_CUDA(cudaMalloc((void**)&dev_data, N * sizeof(int)));

	XorShift128 xs128;

	for (int iter = 1; iter <= 100 && !found_error; iter++)
	{
		for (int i = 0; i < N; i++) {
			host_map[i] = i;
			host_data[i] = i;
		}
		for (int i = 0; i < N; i++) {
			int j = xs128(N);
			int k = host_map[i];
			host_map[i] = host_map[j];
			host_map[j] = k;
		}

		HANDLE_ERROR_CUDA(cudaMemcpyAsync(dev_map, host_map, N * sizeof(int), cudaMemcpyHostToDevice, stream));
		HANDLE_ERROR_CUDA(cudaMemcpyAsync(dev_data, host_data, N * sizeof(int), cudaMemcpyHostToDevice, stream));

		for (int i = 0; i < M/2; i++) {
			test_kernel<<<N/BLOCK_A + 1, BLOCK_A, 0, stream>>>(dev_map, dev_data, xs128(N));
			test_kernel<<<N/BLOCK_B + 1, BLOCK_B, 0, stream>>>(dev_map, dev_data, xs128(N));
		}

		HANDLE_ERROR_CUDA(cudaMemcpyAsync(host_data, dev_data, N * sizeof(int), cudaMemcpyDeviceToHost, stream));
		HANDLE_ERROR_CUDA(cudaStreamSynchronize(stream));

		for (int i = 0; i < N; i++)
			if (host_data[i] != M+i) {
				printf("host_data[%d] = %d\n", i, host_data[i]);
				found_error = true;
			}

		printf("pass %3d completed %s errors.\n", iter, found_error ? "with" : "without");
	}

	HANDLE_ERROR_CUDA(cudaFreeHost(host_map));
	HANDLE_ERROR_CUDA(cudaFreeHost(host_data));

	HANDLE_ERROR_CUDA(cudaFree(dev_map));
	HANDLE_ERROR_CUDA(cudaFree(dev_data));

	HANDLE_ERROR_CUDA(cudaStreamDestroy(stream));

	return found_error ? 1 : 0;
}


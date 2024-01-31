#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#define MAX_PRIMES 100000
#define MAX_INDEX (MAX_PRIMES/2)

extern "C" int primes_gpu(bool);
extern "C" void primes_gpu_init();
extern "C" void primes_gpu_done();

__global__
void primes_worker(int *data)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= MAX_INDEX)
		return;

	int p = 2*idx+1;
	for (int i = 3; i*i <= p; i += 2) {
		if (p % i == 0) {
			data[idx] = 0;
			return;
		}
	}

	data[idx] = idx ? p : 0;
}

__global__
void sum_worker(int *data, int *sum_ptr)
{
	__shared__ int block_sum;
	int idx = threadIdx.x;
	int thread_sum = 0;

	if (threadIdx.x == 0)
		block_sum = 2;

	for (int i = idx; i < MAX_INDEX; i += blockDim.x)
		thread_sum += data[i];
	
	__syncthreads();

	atomicAdd(&block_sum, thread_sum);

	__syncthreads();

	if (threadIdx.x == 0)
		*sum_ptr = block_sum;
}

int *primes_or_zeros;
int *sum_buffer;

void primes_gpu_init()
{
	cudaError_t err;

	err = cudaMalloc((void**)&primes_or_zeros, sizeof(int)*MAX_INDEX);

	if (err != cudaSuccess)
		printf("Cuda error '%s' in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);

	err = cudaMallocHost((void**)&sum_buffer, sizeof(int));

	if (err != cudaSuccess)
		printf("Cuda error '%s' in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
}

void primes_gpu_done()
{
	cudaError_t err;

	err = cudaFree(primes_or_zeros);

	if (err != cudaSuccess)
		printf("Cuda error '%s' in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);

	err = cudaFreeHost(sum_buffer);

	if (err != cudaSuccess)
		printf("Cuda error '%s' in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
}

int primes_gpu(bool big_chunks)
{
	int num_blocks = (MAX_INDEX + 31) / 32;
	int num_treads = 32;

	if (big_chunks) {
		for (int i = 0; i < 99; i++) {
			primes_worker<<<num_blocks, num_treads>>>(primes_or_zeros);
			sum_worker<<<1, 32>>>(primes_or_zeros, sum_buffer);
		}
	}

	primes_worker<<<num_blocks, num_treads>>>(primes_or_zeros);
	sum_worker<<<1, 32>>>(primes_or_zeros, sum_buffer);
	cudaError_t err = cudaDeviceSynchronize();

	if (err != cudaSuccess)
		printf("Cuda error '%s' in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);

	return *sum_buffer;
}


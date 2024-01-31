// nvcc -arch=sm_20 --ptxas-options=-v cuda_async.cu
// (and visualize runtime behavior using nvvp)

#include <cuda_runtime_api.h>
#include <assert.h>
#include <stdio.h>
#include <vector>

#define NUM_BLOCKS 512
#define NUM_THREADS 32

static void HandleErrorCUDA(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(1);
	}
}

#define HANDLE_ERROR_CUDA(_err) HandleErrorCUDA(_err, __FILE__, __LINE__)
#define HANDLE_ERROR_NULL(_ptr) do { if ((_ptr) == NULL) { printf( "Unexpected NULL in %s at line %d\n", __FILE__, __LINE__ ); exit(EXIT_FAILURE); }} while (0)

__global__ void kernel(int size, int count, unsigned int *data)
{
	int tid = blockDim.x*gridDim.x*blockIdx.y + blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < size) {
		unsigned int x = data[tid];
		// perform given # of cycles of xorshift32
		for (int i = 0; i < count; i++) {
			x ^= x << 13;
			x ^= x >> 17;
			x ^= x << 5;
		}
		data[tid] = x;
	}
}

struct job
{
	int size, index;
	cudaStream_t stream;
	unsigned int *buf_host, *buf_dev;
	job(int sz, int idx) {
		size = sz, index = idx;
		assert(size % (NUM_THREADS*NUM_BLOCKS) == 0);
		HANDLE_ERROR_CUDA(cudaMallocHost((void**)&buf_host, size * sizeof(unsigned int)));
		for (int i = 0; i < size; i++)
			buf_host[i] = i+1;
		HANDLE_ERROR_CUDA(cudaMalloc((void**)&buf_dev, size * sizeof(unsigned int)));
		HANDLE_ERROR_CUDA(cudaStreamCreate(&stream));
		printf("[INIT %d]\n", index);
	}
	~job() {
		HANDLE_ERROR_CUDA(cudaStreamSynchronize(stream));
		HANDLE_ERROR_CUDA(cudaFreeHost(buf_host));
		HANDLE_ERROR_CUDA(cudaFree(buf_dev));
		HANDLE_ERROR_CUDA(cudaStreamDestroy(stream));
		printf("[DONE %d]\n", index);
	}
	void copy_host_to_dev() {
		HANDLE_ERROR_CUDA(cudaMemcpyAsync(buf_dev, buf_host, size * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
		printf("[HtoD %d]\n", index);
	}
	void copy_dev_to_host() {
		HANDLE_ERROR_CUDA(cudaMemcpyAsync(buf_host, buf_dev, size * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
		printf("[DtoH %d]\n", index);
	}
	void run(int count) {
		dim3 grid(size/(NUM_THREADS*NUM_BLOCKS), NUM_BLOCKS);
		kernel<<<grid, NUM_THREADS, 0, stream>>>(size, count, buf_dev);
		HANDLE_ERROR_CUDA(cudaGetLastError());
		printf("[EXEC %d]\n", index);
	}
	void sync() {
		HANDLE_ERROR_CUDA(cudaStreamSynchronize(stream));
		printf("[SYNC %d]\n", index);
	}
};

int main()
{
	std::vector<job*> jobs;

	for (size_t i = 0; i < 4; i++)
		jobs.push_back(new job(1024*1024*64, i));

	for (size_t i = 0; i < jobs.size(); i++)
		jobs[i]->copy_host_to_dev(); 

	for (size_t i = 0; i < jobs.size(); i++) {
		jobs[i]->run(256); 
		jobs[i]->copy_dev_to_host(); 
	}

	for (size_t i = 0; i < jobs.size(); i++)
		jobs[i]->sync(); 

	for (size_t i = 0; i < jobs.size(); i++)
		delete jobs[i];
	jobs.clear();
	return 0;
}


// A really simple memory bandwidth and CPU benchmark.
//
// build and run:
// gcc -o membench -static -O3 -march=native -std=c99 membench.c -lrt
// ./membench

#ifdef WIN32
#include <Windows.h>
#else
#define _POSIX_C_SOURCE 200112L
#endif

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>

#define MEMORY_SIZE (1024*1024*128/sizeof(uint32_t))   // 128 MiB
#define INNER_CYCLES (1024*1024*1024/sizeof(uint32_t)) //   1 GiB
#define OUTER_CYCLES 16                                //  16 GiB
#define CACHELINE_WORDS (64/sizeof(uint32_t))          //  64 Bytes
#define MEAS_CLOCK CLOCK_PROCESS_CPUTIME_ID

uint32_t memblock[MEMORY_SIZE];
uint32_t myrngpad[256];
uint32_t myrng_state = 0;
uint32_t myrng_index = 0;
double total_seconds = 0;

#ifdef WIN32
DWORD start_time;
#else
struct timespec start_time;
#endif

static inline uint32_t myrng()
{
	myrng_state += myrngpad[myrng_index++ % 256];
	return myrng_state;
}

static inline void myrng_init()
{
	uint32_t state = 1;
	for (int i = 0; i < 256; i++) {
		state ^= state << 13;
		state ^= state >> 17;
		state ^= state << 5;
		myrngpad[i] = state;
	}
}

static inline uint32_t test_read_linear()
{
	uint32_t xor_sum = 0;
	for (int i = 0; i < INNER_CYCLES; i++)
		xor_sum ^= memblock[i % MEMORY_SIZE];
	return xor_sum;
}

static inline uint32_t test_write_linear()
{
	for (int i = 0; i < INNER_CYCLES; i++)
		memblock[i % MEMORY_SIZE] = myrng();
	return 123456789;
}

static inline uint32_t test_read_random()
{
	uint32_t xor_sum = 0;
	for (int i = 0; i < INNER_CYCLES/10; i++)
		xor_sum ^= memblock[myrng() % MEMORY_SIZE];
	return xor_sum;
}

static inline uint32_t test_write_random()
{
	for (int i = 0; i < INNER_CYCLES/10; i++)
		memblock[myrng() % MEMORY_SIZE] = myrng();
	return 123456789;
}

static inline uint32_t test_read_cacheline()
{
	uint32_t xor_sum = 0;
	assert(CACHELINE_WORDS == 16);
	for (int i = 0; i < INNER_CYCLES/CACHELINE_WORDS; i++) {
		int idx = (myrng() % MEMORY_SIZE) & ~0xf;
		for (int j = 0; j < CACHELINE_WORDS; j++)
			xor_sum ^= memblock[i+j];
	}
	return xor_sum;
}

static inline uint32_t test_write_cacheline()
{
	assert(CACHELINE_WORDS == 16);
	for (int i = 0; i < INNER_CYCLES/CACHELINE_WORDS; i++) {
		int idx = (myrng() % MEMORY_SIZE) & ~0xf;
		for (int j = 0; j < CACHELINE_WORDS; j++)
			memblock[i+j] = myrng();
	}
	return 123456789;
}

static inline uint32_t test_read_indirect()
{
	uint32_t xor_sum = 0;
	for (int i = 0; i < INNER_CYCLES/10; i++) {
		int idx = memblock[i % MEMORY_SIZE];
		xor_sum ^= memblock[idx % MEMORY_SIZE];
	}
	return xor_sum;
}

static inline uint32_t test_write_indirect()
{
	for (int i = 0; i < INNER_CYCLES/10; i++) {
		int idx = memblock[i % MEMORY_SIZE];
		memblock[idx % MEMORY_SIZE] = myrng();
	}
	return 123456789;
}

static inline uint32_t test_memcpy()
{
	for (int i = 0; i < INNER_CYCLES; i += MEMORY_SIZE/2)
		memcpy(memblock, memblock+MEMORY_SIZE/2, sizeof(uint32_t)*MEMORY_SIZE/2);
	return 123456789;
}

static inline uint32_t test_xorshift()
{
	uint32_t xor_sum = 0, state = 1;
	for (int i = 0; i < INNER_CYCLES/3; i++) {
		state ^= state << 13;
		state ^= state >> 17;
		state ^= state << 5;
		xor_sum ^= state;
	}
	return xor_sum;
}

static inline uint32_t test_bubblesort()
{
	int cycles = 0;
	uint32_t state = 1;
	uint32_t xor_sum = 0;
	int num = sqrt(MEMORY_SIZE);

	while (cycles < INNER_CYCLES/4)
	{
		for (int i = 0; i < num; i++) {
			state ^= state << 13;
			state ^= state >> 17;
			state ^= state << 5;
			memblock[i] = state;
		}

		bool keep_running = true;
		while (keep_running) {
			keep_running = false;
			for (int i = 0; i < num-1; i++) {
				if (memblock[i] > memblock[i+1]) {
					memblock[i+0] ^= memblock[i+1];
					memblock[i+1] ^= memblock[i+0];
					memblock[i+0] ^= memblock[i+1];
					xor_sum ^= memblock[i];
					keep_running = true;
				}
				cycles++;
			}
		}
	}

	return xor_sum;
}

static inline uint32_t test_linear_search()
{
	uint32_t xor_sum = 0;
	for (int i = 0; i < INNER_CYCLES; i++)
		if (memblock[i % MEMORY_SIZE] < 0x10000000)
			xor_sum ^= memblock[i % MEMORY_SIZE];
	return xor_sum;
}

static inline void start(const char *p)
{
	char buffer[1024];
#ifdef WIN32
	_snprintf_s(buffer, 1024, "Running %s test.........................", p);
#else
	snprintf(buffer, 1024, "Running %s test.........................", p);
#endif
	printf("%.40s", buffer);
	fflush(stdout);
#ifdef WIN32
	start_time = GetTickCount();
#else
	clock_gettime(MEAS_CLOCK, &start_time);
#endif
}

static inline void stop(uint32_t xor_sum)
{
#ifdef WIN32
	DWORD stop_time = GetTickCount();
	double secs = (stop_time - start_time) * 1e-3;
#else
	struct timespec stop_time;
	clock_gettime(MEAS_CLOCK, &stop_time);

	if (stop_time.tv_nsec < start_time.tv_nsec) {
		stop_time.tv_nsec += 1000000000;
		stop_time.tv_sec -= 1;
	}

	stop_time.tv_nsec -= start_time.tv_nsec;
	stop_time.tv_sec -= start_time.tv_sec;
	double secs = stop_time.tv_nsec * 1e-9 + stop_time.tv_sec;
#endif

	total_seconds += secs;
	printf(" 0x%08x %8.3f\n", xor_sum, secs);
	fflush(stdout);
}

int main()
{
	uint32_t xor_sum = 0;

	myrng_init();
	for (int i = 0; i < MEMORY_SIZE; i++)
		memblock[i] = myrng();

	start("linear read");
	for (int i = 0; i < OUTER_CYCLES; i++) {
		uint32_t result = test_read_linear();
		xor_sum ^= result + myrng();
	}
	stop(xor_sum);

	start("linear write");
	for (int i = 0; i < OUTER_CYCLES; i++) {
		uint32_t result = test_write_linear();
		xor_sum ^= result + myrng();
	}
	stop(xor_sum);

	start("random read (10x)");
	for (int i = 0; i < OUTER_CYCLES; i++) {
		uint32_t result = test_read_random();
		xor_sum ^= result + myrng();
	}
	stop(xor_sum);

	start("random write (10x)");
	for (int i = 0; i < OUTER_CYCLES; i++) {
		uint32_t result = test_write_random();
		xor_sum ^= result + myrng();
	}
	stop(xor_sum);

	start("cacheline read");
	for (int i = 0; i < OUTER_CYCLES; i++) {
		uint32_t result = test_read_cacheline();
		xor_sum ^= result + myrng();
	}
	stop(xor_sum);

	start("cacheline write");
	for (int i = 0; i < OUTER_CYCLES; i++) {
		uint32_t result = test_write_cacheline();
		xor_sum ^= result + myrng();
	}
	stop(xor_sum);

	start("indirect read (10x)");
	for (int i = 0; i < OUTER_CYCLES; i++) {
		uint32_t result = test_read_indirect();
		xor_sum ^= result + myrng();
	}
	stop(xor_sum);

	start("indirect write (10x)");
	for (int i = 0; i < OUTER_CYCLES; i++) {
		uint32_t result = test_write_indirect();
		xor_sum ^= result + myrng();
	}
	stop(xor_sum);

	start("memcpy");
	for (int i = 0; i < OUTER_CYCLES; i++) {
		uint32_t result = test_memcpy();
		xor_sum ^= result + myrng();
	}
	stop(xor_sum);

	start("xorshift");
	for (int i = 0; i < OUTER_CYCLES; i++) {
		uint32_t result = test_xorshift();
		xor_sum ^= result + myrng();
	}
	stop(xor_sum);

	start("bubblesort");
	for (int i = 0; i < OUTER_CYCLES; i++) {
		uint32_t result = test_bubblesort();
		xor_sum ^= result + myrng();
	}
	stop(xor_sum);

	start("linear search");
	for (int i = 0; i < OUTER_CYCLES; i++) {
		uint32_t result = test_linear_search();
		xor_sum ^= result + myrng();
	}
	stop(xor_sum);

	printf("Total: %.3f\n", total_seconds);
	return 0;
}


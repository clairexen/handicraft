#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

int primes_cpu(bool);
int primes_gpu(bool);
void primes_gpu_init();
void primes_gpu_done();

char timer_method = 'r';
bool meas_pieces = true;
bool use_gpu = false;
bool early_exit = false;
bool big_chunks = false;

int64_t get_the_timer()
{
	if (timer_method == 'r')
	{
		struct rusage ru;
		getrusage(RUSAGE_THREAD, &ru);

		int64_t utime_usec = (int64_t)ru.ru_utime.tv_sec * 1000000;
		utime_usec += ru.ru_utime.tv_usec;

		int64_t stime_usec = (int64_t)ru.ru_stime.tv_sec * 1000000;
		stime_usec += ru.ru_stime.tv_usec;

		return utime_usec + stime_usec;
	}

	if (timer_method == 'm' || timer_method == 'c' || timer_method == 't')
	{
		struct timespec ts;

		if (timer_method == 'm')
			clock_gettime(CLOCK_MONOTONIC, &ts);

		if (timer_method == 'c')
			clock_gettime(CLOCK_MONOTONIC_COARSE, &ts);

		if (timer_method == 't')
			clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);

		int64_t time_usec  = (int64_t)ts.tv_sec * 1000000;
		time_usec += ts.tv_nsec / 1000;

		return time_usec;
	}

	return 0;
}

int main(int argc, char **argv)
{
	int opt;

	while ((opt = getopt(argc, argv, "rmctngxb")) != -1)
		switch (opt)
		{
		case 'r':
		case 'm':
		case 'c':
		case 't':
			timer_method = opt;
			break;
		case 'n':
			meas_pieces = false;
			break;
		case 'g':
			use_gpu = true;
			break;
		case 'x':
			early_exit = true;
			break;
		case 'b':
			big_chunks = true;
			break;
		default:
			printf("Usage: %s [-r|-m|-c|-t] [-n] [-g] [-x] [-b]\n", argv[0]);
			return 1;
		}

	int64_t overall_start = get_the_timer();
	int64_t pieces_sum = 0;

	if (use_gpu)
		primes_gpu_init();

	for (int i = 0; i < (big_chunks ? 100 : 10000); i++)
	{
		int64_t piece_start = meas_pieces ? get_the_timer() : 0;

		int ret = use_gpu ? primes_gpu(big_chunks) : primes_cpu(big_chunks);
		int expected_ret = 454396537;

		if (i == 0)
			printf("%s primes sum: %d\n", use_gpu ? "GPU" : "CPU", ret);

		if (ret != expected_ret) {
			printf("Calc error!\n");
			return 1;
		}

		int64_t piece_stop = meas_pieces ? get_the_timer() : 0;
		pieces_sum += piece_stop - piece_start;

		if (early_exit)
			break;
	}

	if (use_gpu)
		primes_gpu_done();

	int64_t overall_stop = get_the_timer();
	int64_t overall_sum = overall_stop - overall_start;

	printf("Pieces:  %9.3f seconds\n", pieces_sum * 1e-6);
	printf("Overall: %9.3f seconds\n", overall_sum * 1e-6);

	return 0;
}


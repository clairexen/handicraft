// gcc -Wall -O3 -o judyprime judyprime.c -lJudy -lm

#include <Judy.h>
#include <math.h>
#include <stdio.h>
#include <signal.h>
#include <assert.h>
#include <unistd.h>

#define MAX_PRIME 1000000000UL
void *primeMap = NULL;

int phase = 0;
unsigned long state = 0;

void sig_report(int dummy)
{
	static int counter = 0;
	fprintf(stderr, "phase %d: %.3f%% %c    \r", phase,
			(state*100.0) / (phase == 1 ? sqrt(MAX_PRIME) :
			MAX_PRIME), "/-\\|"[counter++ % 4]);
	alarm(1);
}

int main()
{
	unsigned long i, j;
	int rc;

	signal(SIGALRM, &sig_report);
	sig_report(0);

	// initialize map with candidates
	phase = 0;
	J1S(rc, primeMap, 2)
	J1S(rc, primeMap, 3)
	J1S(rc, primeMap, 5)
	J1S(rc, primeMap, 7)
	unsigned long int i3 = 3, i5 = 5, i7 = 7;
	for (i = 9; i < MAX_PRIME; i += 2) {
		while (i3 < i) i3 += 3;
		while (i5 < i) i5 += 5;
		while (i7 < i) i7 += 7;
		if (i3 == i) continue;
		if (i5 == i) continue;
		if (i7 == i) continue;
		J1S(rc, primeMap, i)
		state = i;
	}

	// eliminate all non-primes
	phase = 1, state = i = 9;
	J1F(rc, primeMap, i)
	while (i*i < MAX_PRIME) {
		for (j = i*i; j < MAX_PRIME; j += i) {
			J1U(rc, primeMap, j)
		}
		J1N(rc, primeMap, i)
		state = i;
	}

#if 0
	// print the results
	phase = 2, state = i = 0;
	J1F(rc, primeMap, i)
	while (i < MAX_PRIME) {
		printf("%lu\n", i);
		J1N(rc, primeMap, i)
		if (rc == 0)
			break;
		state = i;
	}
#endif

	alarm(0);
	fprintf(stderr, "%*s\r", 20, "");
	return 0;
}


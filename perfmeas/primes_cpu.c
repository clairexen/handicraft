/* THE EVOLUTION OF PRIME - prime9.c
 *
 * Copyright (C) 2003 Clifford Wolf
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Limit range of prime numbers to process to sqrt(MAX_PRIMES) and start
 * with the square of a prime number (instead of it's double). Use even
 * deltas so we don't need the odd check in the main loop and can also
 * reduce the number of main-loop-cycles.
 *
 * Use the C version of the main loop because todays compiler generate
 * better code than this hand written assembler loop.
 */

#include <string.h>
#include <stdbool.h>
#include <math.h>

// max number of primes processed in parallel
#define GROUP_SIZE 2048
int group_c[GROUP_SIZE] __attribute__ ((aligned (4096))); // counters
int group_d[GROUP_SIZE] __attribute__ ((aligned (4096))); // deltas

// number of pages processed in one pass
#define PAGE_BLOCK 16

// 1 page = 4096 bytes = 4096*8 prime-flags
// 4096*8 = 32768 = 0x8000
#define PAGE_SIZE (0x8000*PAGE_BLOCK)
#define PAGE_MASK (~(PAGE_SIZE-1))

#define MAX_PRIMES 100000
/* zero-initialized, add an entire page padding */
unsigned char p[MAX_PRIMES/16+PAGE_SIZE+1] __attribute__ ((aligned (4096)));

int primes_cpu_worker()
{
	int a=1, b, g;
	int group_window, group_roof;
	int current_page, next_page;
	int maxp_roof = MAX_PRIMES/2;
	int maxp_sqrt = sqrt(MAX_PRIMES)/2;
	long long sum_primes = 2;

	memset(group_c, 0, sizeof(group_c));
	memset(group_d, 0, sizeof(group_d));
	memset(p, 0, sizeof(p));

	while (a<maxp_sqrt) {
		group_window = a+a > maxp_sqrt ? maxp_sqrt : a+a;
		for (group_roof=0; a < group_window && group_roof < GROUP_SIZE; a++) {
			if ( p[a>>3] & (1<<(a&7)) ) continue;
			group_d[group_roof] = 2*a+1;
			sum_primes += group_d[group_roof];
			group_c[group_roof++] = 2*a*a+2*a;
		}
		if ( ! group_roof ) continue;
		current_page = group_c[0] & PAGE_MASK;

		while ( current_page <= (maxp_roof & PAGE_MASK) ) {
			next_page = (maxp_roof & PAGE_MASK) + PAGE_SIZE;
			for (g=0; g<group_roof; g++) {
				for (b=group_c[g]; b < (current_page + PAGE_SIZE); b+=group_d[g])
					p[b>>3]=p[b>>3]|(1<<(b&7));
				if ( b < next_page ) next_page = b & PAGE_MASK;
				group_c[g]=b;
			}
			current_page = next_page;
		}
	}

	for (; a<maxp_roof; a++)
		if ( ! (p[a>>3] & (1<<(a&7))) ) sum_primes += 2*a+1;

	return sum_primes;
}

int primes_cpu(bool big_chunks)
{
	if (big_chunks) {
		for (int i = 0; i < 99; i++)
			primes_cpu_worker();
	}

	return primes_cpu_worker();
}


// Formal verification:
//    cbmc --function prove vlogrnd.c
//
// Simulation:
//    gcc -Wall -O3 -o vlogrnd vlogrnd.c && ./vlogrnd

# include  <assert.h>
# include  <stdlib.h>
# include  <math.h>
# include  <limits.h>
# include  <stdio.h>

#if ULONG_MAX > 4294967295UL
# define UNIFORM_MAX INT_MAX
# define UNIFORM_MIN INT_MIN
#else
# define UNIFORM_MAX LONG_MAX
# define UNIFORM_MIN LONG_MIN
#endif

static long rtl_dist_uniform(long *seed, long start, long end);
static double uniform(long *seed, long start, long end);

void check(long value)
{
	if ((value & 63) == 0)
		assert(((value >> 24) & 15) == 0);
}

void prove(long seed)
{
	long value = rtl_dist_uniform(&seed, UNIFORM_MIN, UNIFORM_MAX);
	check(value);
}

int main()
{
	long seed = 1234;
	int i;

	for (i = 1; i <= 1000000000; i++) {
		if ((i % 10000000) == 0) printf("Iteration %d.\n", i);
		long value = rtl_dist_uniform(&seed, UNIFORM_MIN, UNIFORM_MAX);
		check(value);
	}

	return 0;
}

/* copied from IEEE1364-2001, with slight modifications for 64bit machines. */
static long rtl_dist_uniform(long *seed, long start, long end)
{
      double r;
      long i;

      if (start >= end) return(start);

      /* NOTE: The cast of r to i can overflow and generate strange
         values, so cast to unsigned long first. This eliminates
         the underflow and gets the twos complement value. That in
         turn can be cast to the long value that is expected. */

      if (end != UNIFORM_MAX) {
            end++;
            r = uniform(seed, start, end);
            if (r >= 0) {
                  i = (unsigned long) r;
            } else {
	          i = - ( (unsigned long) (-(r - 1)) );
            }
            if (i < start) i = start;
            if (i >= end) i = end - 1;
      } else if (start != UNIFORM_MIN) {
            start--;
            r = uniform( seed, start, end) + 1.0;
            if (r >= 0) {
                  i = (unsigned long) r;
            } else {
	          i = - ( (unsigned long) (-(r - 1)) );
            }
            if (i <= start) i = start + 1;
            if (i > end) i = end;
      } else {
            r = (uniform(seed, start, end) + 2147483648.0) / 4294967295.0;
            r = r * 4294967296.0 - 2147483648.0;

            if (r >= 0) {
                  i = (unsigned long) r;
            } else {
	            /* At least some compilers will notice that (r-1)
		       is <0 when castling to unsigned long and
		       replace the result with a zero. This causes
		       much wrongness, so do the casting to the
		       positive version and invert it back. */
	          i = - ( (unsigned long) (-(r - 1)) );
            }
      }

      return i;
}

static double uniform(long *seed, long start, long end )
{
      double d = 0.00000011920928955078125;
      double a, b, c;
      unsigned long oldseed, newseed;

      oldseed = *seed;
      if (oldseed == 0)
            oldseed = 259341593;

      if (start >= end) {
            a = 0.0;
            b = 2147483647.0;
      } else {
            a = (double)start;
            b = (double)end;
      }

      /* Original routine used signed arithmetic, and the (frequent)
       * overflows trigger "Undefined Behavior" according to the
       * C standard (both c89 and c99).  Using unsigned arithmetic
       * forces a conforming C implementation to get the result
       * that the IEEE-1364-2001 committee wants.
       */
      newseed = 69069 * oldseed + 1;

      /* Emulate a 32-bit unsigned long, even if the native machine
       * uses wider words.
       */
#if ULONG_MAX > 4294967295UL
      newseed = newseed & 4294967295UL;
#endif
      *seed = newseed;


#if 0
      /* Cadence-donated conversion from unsigned int to double */
      {
            union { float s; unsigned stemp; } u;
            u.stemp = (newseed >> 9) | 0x3f800000;
            c = (double) u.s;
      }
#else
      /* Equivalent conversion without assuming IEEE 32-bit float */
      /* constant is 2^(-23) */
      c = 1.0 + (newseed >> 9) * 0.00000011920928955078125;
#endif


      c = c + (c*d);
      c = ((b - a) * (c - 1.0)) + a;

      return c;
}


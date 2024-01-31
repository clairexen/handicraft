// Source: http://graphics.stanford.edu/~seander/bithacks.html

#include <stdbool.h>
#include <stdint.h>

#define in(v) asm volatile ("" : "=r"(v))
#define out(v) asm volatile ("" : : "r"(v))

static inline uint64_t in64()
{
	uint32_t a, b;
	in(a);
	in(b);

	return ((uint64_t)a << 32) | b;
}

#define CHAR_BIT 8

// ------------------------------------

void aa()
{
	int v;      // we want to find the sign of v
	int sign;   // the result goes here 

	in(v);

	// CHAR_BIT is the number of bits per byte (normally 8).
	sign = -(v < 0);  // if v < 0 then -1, else 0. 
	out(sign);

	// or, to avoid branching on CPUs with flag registers (IA32):
	sign = -(int)((unsigned int)((int)v) >> (sizeof(int) * CHAR_BIT - 1));
	out(sign);

	// or, for one less instruction (but not portable):
	sign = v >> (sizeof(int) * CHAR_BIT - 1); 
	out(sign);

	sign = +1 | (v >> (sizeof(int) * CHAR_BIT - 1));  // if v < 0 then -1, else +1
	out(sign);

	sign = (v != 0) | -(int)((unsigned int)((int)v) >> (sizeof(int) * CHAR_BIT - 1));
	out(sign);

	// Or, for more speed but less portability:
	sign = (v != 0) | (v >> (sizeof(int) * CHAR_BIT - 1));  // -1, 0, or +1
	out(sign);

	// Or, for portability, brevity, and (perhaps) speed:
	sign = (v > 0) - (v < 0); // -1, 0, or +1
	out(sign);

	sign = 1 ^ ((unsigned int)v >> (sizeof(int) * CHAR_BIT - 1)); // if v < 0 then 0, else 1
	out(sign);
}


// ------------------------------------

void ab()
{
	int x, y;               // input values to compare signs
	in(x);
	in(y);

	bool f = ((x ^ y) < 0); // true iff x and y have opposite signs
	out(f);
}

// ------------------------------------

void ac()
{
	int v;           // we want to find the absolute value of v
	in(v);

	unsigned int r;  // the result goes here 
	int const mask = v >> (sizeof(int) * CHAR_BIT - 1);

	r = (v + mask) ^ mask;
	out(r);

	r = (v ^ mask) - mask;
	out(r);
}

// ------------------------------------

void ad()
{
	int x, y;  // we want to find the minimum of x and y
	in(x);
	in(y);

	int r;  // the result goes here 

	r = y ^ ((x ^ y) & -(x < y)); // min(x, y)
	out(r);

	r = x ^ ((x ^ y) & -(x < y)); // max(x, y)
	out(r);

	r = y + ((x - y) & ((x - y) >> (sizeof(int) * CHAR_BIT - 1))); // min(x, y)
	out(r);

	r = x - ((x - y) & ((x - y) >> (sizeof(int) * CHAR_BIT - 1))); // max(x, y)
	out(r);
}

// ------------------------------------

void ae()
{
	unsigned int v; // we want to see if v is a power of 2
	in(v);

	bool f;         // the result goes here 

	f = (v & (v - 1)) == 0;
	out(f);

	f = v && !(v & (v - 1));
	out(f);
}

// ------------------------------------

void af()
{
	int x; // convert this from using 5 bits to a full int
	in(x);

	int r; // resulting sign extended number goes here
	struct {signed int x:5;} s;
	r = s.x = x;
	out(r);
}

// ------------------------------------

void ag()
{
	unsigned b; // number of bits representing the number in x
	int x;      // sign extend this b-bit number to r
	in(b);
	in(x);

	int r;      // resulting sign-extended number
	int const m = 1U << (b - 1); // mask can be pre-computed if b is fixed

	x = x & ((1U << b) - 1);  // (Skip this if bits in x above position b are already zero.)
	r = (x ^ m) - m;
	out(r);
}

// ------------------------------------

void ah()
{
	unsigned b; // number of bits representing the number in x
	int x;      // sign extend this b-bit number to r
	in(b);
	in(x);

	int r;      // resulting sign-extended number
	int const m = CHAR_BIT * sizeof(x) - b;
	r = (x << m) >> m;
	out(r);
}

// ------------------------------------

void ai()
{
	unsigned b; // number of bits representing the number in x
	int x;      // sign extend this b-bit number to r

	in(b);
	in(x);

	int r;      // resulting sign-extended number
	#define M(B) (1U << ((sizeof(x) * CHAR_BIT) - B)) // CHAR_BIT=bits/byte
	static int const multipliers[] = 
	{
	  0,     M(1),  M(2),  M(3),  M(4),  M(5),  M(6),  M(7),
	  M(8),  M(9),  M(10), M(11), M(12), M(13), M(14), M(15),
	  M(16), M(17), M(18), M(19), M(20), M(21), M(22), M(23),
	  M(24), M(25), M(26), M(27), M(28), M(29), M(30), M(31),
	  M(32)
	}; // (add more if using more than 64 bits)
	static int const divisors[] = 
	{
	  1,    ~M(1),  M(2),  M(3),  M(4),  M(5),  M(6),  M(7),
	  M(8),  M(9),  M(10), M(11), M(12), M(13), M(14), M(15),
	  M(16), M(17), M(18), M(19), M(20), M(21), M(22), M(23),
	  M(24), M(25), M(26), M(27), M(28), M(29), M(30), M(31),
	  M(32)
	}; // (add more for 64 bits)
	#undef M
	r = (x * multipliers[b]) / divisors[b];

	out(r);
}

// ------------------------------------

void aj()
{
	unsigned b; // number of bits representing the number in x
	int x;      // sign extend this b-bit number to r

	in(b);
	in(x);

	const int s = -b; // OR:  sizeof(x) * CHAR_BIT - b;
	int r = (x << s) >> s;

	out(r);
}

// ------------------------------------

void ak()
{
	bool f;         // conditional flag
	unsigned int m; // the bit mask
	unsigned int w; // the word to modify:  if (f) w |= m; else w &= ~m; 

	in(f);
	in(m);
	in(w);

	w ^= (-f ^ w) & m;

	out(w);
}


// ------------------------------------

void al()
{
	bool f;         // conditional flag
	unsigned int m; // the bit mask
	unsigned int w; // the word to modify:  if (f) w |= m; else w &= ~m; 

	in(f);
	in(m);
	in(w);

	w = (w & ~m) | (-f & m);

	out(w);
}

// ------------------------------------

void am()
{
	bool fDontNegate;  // Flag indicating we should not negate v.
	int v;             // Input value to negate if fDontNegate is false.

	in(fDontNegate);
	in(v);

	int r;             // result = fDontNegate ? v : -v;

	r = (fDontNegate ^ (fDontNegate - 1)) * v;

	out(r);
}

// ------------------------------------

void an()
{
	bool fNegate;  // Flag indicating if we should negate v.
	int v;         // Input value to negate if fNegate is true.

	in(fNegate);
	in(v);

	int r;         // result = fNegate ? -v : v;

	r = (v ^ -fNegate) + fNegate;

	out(r);
}

// ------------------------------------

void ao()
{
	unsigned int a;    // value to merge in non-masked bits
	unsigned int b;    // value to merge in masked bits
	unsigned int mask; // 1 where bits from b should be selected; 0 where from a.

	in(a);
	in(b);
	in(mask);

	unsigned int r;    // result of (a & ~mask) | (b & mask) goes here

	r = a ^ ((a ^ b) & mask); 

	out(r);
}

// ------------------------------------

void ap()
{
	unsigned int v; // count the number of bits set in v

	in(v);

	unsigned int c; // c accumulates the total bits set in v

	for (c = 0; v; v >>= 1)
	{
	  c += v & 1;
	}

	out(c);
}

// ------------------------------------

void aq()
{
	static const unsigned char BitsSetTable256[256] = 
	{
	#   define B2(n) n,     n+1,     n+1,     n+2
	#   define B4(n) B2(n), B2(n+1), B2(n+1), B2(n+2)
	#   define B6(n) B4(n), B4(n+1), B4(n+1), B4(n+2)
	    B6(0), B6(1), B6(1), B6(2)
	};

	unsigned int v; // count the number of bits set in 32-bit value v

	in(v);

	unsigned int c; // c is the total bits set in v

	// Option 1:
	c = BitsSetTable256[v & 0xff] + 
	    BitsSetTable256[(v >> 8) & 0xff] + 
	    BitsSetTable256[(v >> 16) & 0xff] + 
	    BitsSetTable256[v >> 24]; 

	out(c);

	// Option 2:
	unsigned char * p = (unsigned char *) &v;
	c = BitsSetTable256[p[0]] + 
	    BitsSetTable256[p[1]] + 
	    BitsSetTable256[p[2]] +	
	    BitsSetTable256[p[3]];

	out(c);
}

// ------------------------------------

void ar()
{
	unsigned int v; // count the number of bits set in v

	in(v);

	unsigned int c; // c accumulates the total bits set in v

	for (c = 0; v; c++)
	{
	  v &= v - 1; // clear the least significant bit set
	}

	out(c);
}

// ------------------------------------

void as()
{
	unsigned int v; // count the number of bits set in v

	in(v);

	unsigned int c; // c accumulates the total bits set in v

	// option 1, for at most 14-bit values in v:
	c = (v * 0x200040008001ULL & 0x111111111111111ULL) % 0xf;

	out(c);

	// option 2, for at most 24-bit values in v:
	c =  ((v & 0xfff) * 0x1001001001001ULL & 0x84210842108421ULL) % 0x1f;
	c += (((v & 0xfff000) >> 12) * 0x1001001001001ULL & 0x84210842108421ULL) 
	     % 0x1f;

	out(c);

	// option 3, for at most 32-bit values in v:
	c =  ((v & 0xfff) * 0x1001001001001ULL & 0x84210842108421ULL) % 0x1f;
	c += (((v & 0xfff000) >> 12) * 0x1001001001001ULL & 0x84210842108421ULL) % 
	     0x1f;
	c += ((v >> 24) * 0x1001001001001ULL & 0x84210842108421ULL) % 0x1f;

	out(c);
}

// ------------------------------------

void at()
{
	unsigned int v; // count bits set in this (32-bit value)

	in(v);

	unsigned int c; // store the total here
	static const int S[] = {1, 2, 4, 8, 16}; // Magic Binary Numbers
	static const int B[] = {0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF};

	c = v - ((v >> 1) & B[0]);
	c = ((c >> S[1]) & B[1]) + (c & B[1]);
	c = ((c >> S[2]) + c) & B[2];
	c = ((c >> S[3]) + c) & B[3];
	c = ((c >> S[4]) + c) & B[4];

	out(c);

	v = v - ((v >> 1) & 0x55555555);                    // reuse input as temporary
	v = (v & 0x33333333) + ((v >> 2) & 0x33333333);     // temp
	c = (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24; // count

	out(c);
}

// ------------------------------------

void au()
{
	uint64_t v;       // Compute the rank (bits set) in v from the MSB to pos.
	unsigned int pos; // Bit position to count bits upto.

	v = in64();
	in(pos);

	uint64_t r;       // Resulting rank of bit at pos goes here.

	// Shift out bits after given position.
	r = v >> (sizeof(v) * CHAR_BIT - pos);
	// Count set bits in parallel.
	// r = (r & 0x5555...) + ((r >> 1) & 0x5555...);
	r = r - ((r >> 1) & ~0UL/3);
	// r = (r & 0x3333...) + ((r >> 2) & 0x3333...);
	r = (r & ~0UL/5) + ((r >> 2) & ~0UL/5);
	// r = (r & 0x0f0f...) + ((r >> 4) & 0x0f0f...);
	r = (r + (r >> 4)) & ~0UL/17;
	// r = r % 255;
	r = (r * (~0UL/255)) >> ((sizeof(v) - 1) * CHAR_BIT);

	out(r);
	out(r >> 32);
}

// ------------------------------------

void av()
{
	uint64_t v;          // Input value to find position with rank r.
	unsigned int r;      // Input: bit's desired rank [1-64].

	in(v);
	in(r);

	unsigned int s;      // Output: Resulting position of bit with rank r [1-64]
	uint64_t a, b, c, d; // Intermediate temporaries for bit count.
	unsigned int t;      // Bit count temporary.

	// Do a normal parallel bit count for a 64-bit integer,                     
	// but store all intermediate steps.                                        
	// a = (v & 0x5555...) + ((v >> 1) & 0x5555...);
	a =  v - ((v >> 1) & ~0UL/3);
	// b = (a & 0x3333...) + ((a >> 2) & 0x3333...);
	b = (a & ~0UL/5) + ((a >> 2) & ~0UL/5);
	// c = (b & 0x0f0f...) + ((b >> 4) & 0x0f0f...);
	c = (b + (b >> 4)) & ~0UL/0x11;
	// d = (c & 0x00ff...) + ((c >> 8) & 0x00ff...);
	d = (c + (c >> 8)) & ~0UL/0x101;
	t = (d >> 32) + (d >> 48);
	// Now do branchless select!                                                
	s  = 64;
	// if (r > t) {s -= 32; r -= t;}
	s -= ((t - r) & 256) >> 3; r -= (t & ((t - r) >> 8));
	t  = (d >> (s - 16)) & 0xff;
	// if (r > t) {s -= 16; r -= t;}
	s -= ((t - r) & 256) >> 4; r -= (t & ((t - r) >> 8));
	t  = (c >> (s - 8)) & 0xf;
	// if (r > t) {s -= 8; r -= t;}
	s -= ((t - r) & 256) >> 5; r -= (t & ((t - r) >> 8));
	t  = (b >> (s - 4)) & 0x7;
	// if (r > t) {s -= 4; r -= t;}
	s -= ((t - r) & 256) >> 6; r -= (t & ((t - r) >> 8));
	t  = (a >> (s - 2)) & 0x3;
	// if (r > t) {s -= 2; r -= t;}
	s -= ((t - r) & 256) >> 7; r -= (t & ((t - r) >> 8));
	t  = (v >> (s - 1)) & 0x1;
	// if (r > t) s--;
	s -= ((t - r) & 256) >> 8;
	s = 65 - s;

	out(s);
}

// ------------------------------------

void aw()
{
	unsigned int v;       // word value to compute the parity of

	in(v);

	bool parity = false;  // parity will be the parity of v

	while (v)
	{
	  parity = !parity;
	  v = v & (v - 1);
	}

	out(parity);
}

// ------------------------------------

void ax()
{
	static const bool ParityTable256[256] = 
	{
	#   define P2(n) n, n^1, n^1, n
	#   define P4(n) P2(n), P2(n^1), P2(n^1), P2(n)
	#   define P6(n) P4(n), P4(n^1), P4(n^1), P4(n)
	    P6(0), P6(1), P6(1), P6(0)
	};

	unsigned char b;  // byte value to compute the parity of

	in(b);

	bool parity = ParityTable256[b];

	out(parity);
}

// ------------------------------------

void ay()
{
	static const bool ParityTable256[256] = 
	{
	#   define P2(n) n, n^1, n^1, n
	#   define P4(n) P2(n), P2(n^1), P2(n^1), P2(n)
	#   define P6(n) P4(n), P4(n^1), P4(n^1), P4(n)
	    P6(0), P6(1), P6(1), P6(0)
	};

	// OR, for 32-bit words:
	unsigned int v;

	in(v);

	v ^= v >> 16;
	v ^= v >> 8;
	bool parity = ParityTable256[v & 0xff];

	out(parity);
}

// ------------------------------------

void az()
{
	static const bool ParityTable256[256] = 
	{
	#   define P2(n) n, n^1, n^1, n
	#   define P4(n) P2(n), P2(n^1), P2(n^1), P2(n)
	#   define P6(n) P4(n), P4(n^1), P4(n^1), P4(n)
	    P6(0), P6(1), P6(1), P6(0)
	};

	unsigned int v;

	in(v);

	// Variation:
	unsigned char * p = (unsigned char *) &v;
	bool parity = ParityTable256[p[0] ^ p[1] ^ p[2] ^ p[3]];

	out(parity);
}

// ------------------------------------

void ba()
{
	unsigned char b;  // byte value to compute the parity of

	in(b);

	bool parity = 
	  (((b * 0x0101010101010101ULL) & 0x8040201008040201ULL) % 0x1FF) & 1;

	out(parity);
}

// ------------------------------------

void bb()
{
	unsigned int v; // 32-bit word

	in(v);

	v ^= v >> 1;
	v ^= v >> 2;
	v = (v & 0x11111111U) * 0x11111111U;
	v = (v >> 28) & 1;

	out(v);
}

// ------------------------------------

void bc()
{
	unsigned long long v; // 64-bit word

	in(v);

	v ^= v >> 1;
	v ^= v >> 2;
	v = (v & 0x1111111111111111UL) * 0x1111111111111111UL;
	v = (v >> 60) & 1;

	out(v);
}

// ------------------------------------

void bd()
{
	unsigned int v;  // word value to compute the parity of

	in(v);

	v ^= v >> 16;
	v ^= v >> 8;
	v ^= v >> 4;
	v &= 0xf;
	v = (0x6996 >> v) & 1;

	out(v);
}

// ------------------------------------

void be()
{
	unsigned int i, j; // positions of bit sequences to swap
	unsigned int n;    // number of consecutive bits in each sequence
	unsigned int b;    // bits to swap reside in b

	in(i);
	in(j);
	in(n);
	in(b);

	unsigned int r;    // bit-swapped result goes here

	unsigned int x = ((b >> i) ^ (b >> j)) & ((1U << n) - 1); // XOR temporary
	r = b ^ ((x << i) | (x << j));

	out(r);
}

// ------------------------------------

void bf()
{
	unsigned int v;     // input bits to be reversed

	in(v);

	unsigned int r = v; // r will be reversed bits of v; first get LSB of v
	int s = sizeof(v) * CHAR_BIT - 1; // extra shift needed at end

	for (v >>= 1; v; v >>= 1)
	{   
	  r <<= 1;
	  r |= v & 1;
	  s--;
	}
	r <<= s; // shift when v's highest bits are zero

	out(r);
}

// ------------------------------------

void bg()
{
	static const unsigned char BitReverseTable256[256] = 
	{
	#   define R2(n)     n,     n + 2*64,     n + 1*64,     n + 3*64
	#   define R4(n) R2(n), R2(n + 2*16), R2(n + 1*16), R2(n + 3*16)
	#   define R6(n) R4(n), R4(n + 2*4 ), R4(n + 1*4 ), R4(n + 3*4 )
	    R6(0), R6(2), R6(1), R6(3)
	};

	unsigned int v; // reverse 32-bit value, 8 bits at time

	in(v);

	unsigned int c; // c will get v reversed

	// Option 1:
	c = (BitReverseTable256[v & 0xff] << 24) | 
	    (BitReverseTable256[(v >> 8) & 0xff] << 16) | 
	    (BitReverseTable256[(v >> 16) & 0xff] << 8) |
	    (BitReverseTable256[(v >> 24) & 0xff]);

	out(c);
}

// ------------------------------------

void bh()
{
	static const unsigned char BitReverseTable256[256] = 
	{
	#   define R2(n)     n,     n + 2*64,     n + 1*64,     n + 3*64
	#   define R4(n) R2(n), R2(n + 2*16), R2(n + 1*16), R2(n + 3*16)
	#   define R6(n) R4(n), R4(n + 2*4 ), R4(n + 1*4 ), R4(n + 3*4 )
	    R6(0), R6(2), R6(1), R6(3)
	};

	unsigned int v; // reverse 32-bit value, 8 bits at time

	in(v);

	unsigned int c; // c will get v reversed

	// Option 2:
	unsigned char * p = (unsigned char *) &v;
	unsigned char * q = (unsigned char *) &c;
	q[3] = BitReverseTable256[p[0]]; 
	q[2] = BitReverseTable256[p[1]]; 
	q[1] = BitReverseTable256[p[2]]; 
	q[0] = BitReverseTable256[p[3]];

	out(c);
}

// ------------------------------------

void bi()
{
	unsigned char b; // reverse this (8-bit) byte

	in(b);
	 
	b = (b * 0x0202020202ULL & 0x010884422010ULL) % 1023;

	out(b);
}

// ------------------------------------

void bj()
{
	unsigned char b; // reverse this byte
	 
	in(b);

	b = ((b * 0x80200802ULL) & 0x0884422110ULL) * 0x0101010101ULL >> 32;

	out(b);
}

// ------------------------------------

void bk()
{
	unsigned int v; // 32-bit word to reverse bit order

	in(v);

	// swap odd and even bits
	v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
	// swap consecutive pairs
	v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
	// swap nibbles ... 
	v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
	// swap bytes
	v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
	// swap 2-byte long pairs
	v = ( v >> 16             ) | ( v               << 16);

	out(v);
}

// ------------------------------------

void bl()
{
	unsigned int v; // 32-bit word to reverse bit order

	in(v);

	unsigned int s = sizeof(v) * CHAR_BIT; // bit size; must be power of 2 
	unsigned int mask = ~0;         

	while ((s >>= 1) > 0) 
	{
	  mask ^= (mask << s);
	  v = ((v >> s) & mask) | ((v << s) & ~mask);
	}

	out(v);
}

// ------------------------------------

void bm()
{
	unsigned int n;          // numerator
	unsigned int s;

	in(n);
	in(s);

	unsigned int d = 1U << s; // So d will be one of: 1, 2, 4, 8, 16, 32, ...
	unsigned int m;           // m will be n % d

	m = n & (d - 1); 

	out(m);
}

// ------------------------------------

void bn()
{
	unsigned int n;                      // numerator
	unsigned int s;                // s > 0

	in(n);
	in(s);

	unsigned int d = (1 << s) - 1; // so d is either 1, 3, 7, 15, 31, ...).
	unsigned int m;                      // n % d goes here.

	for (m = n; n > d; n = m)
	{
	  for (m = 0; n; n >>= s)
	  {
	    m += n & d;
	  }
	}

	// Now m is a value from 0 to d, but since with modulus division
	// we want m to be 0 when it is d.
	m = m == d ? 0 : m;

	out(m);
}

// ------------------------------------

void bo()
{
	// The following is for a word size of 32 bits!

	static const unsigned int M[] = 
	{
	  0x00000000, 0x55555555, 0x33333333, 0xc71c71c7,  
	  0x0f0f0f0f, 0xc1f07c1f, 0x3f03f03f, 0xf01fc07f, 
	  0x00ff00ff, 0x07fc01ff, 0x3ff003ff, 0xffc007ff,
	  0xff000fff, 0xfc001fff, 0xf0003fff, 0xc0007fff,
	  0x0000ffff, 0x0001ffff, 0x0003ffff, 0x0007ffff, 
	  0x000fffff, 0x001fffff, 0x003fffff, 0x007fffff,
	  0x00ffffff, 0x01ffffff, 0x03ffffff, 0x07ffffff,
	  0x0fffffff, 0x1fffffff, 0x3fffffff, 0x7fffffff
	};

	static const unsigned int Q[][6] = 
	{
	  { 0,  0,  0,  0,  0,  0}, {16,  8,  4,  2,  1,  1}, {16,  8,  4,  2,  2,  2},
	  {15,  6,  3,  3,  3,  3}, {16,  8,  4,  4,  4,  4}, {15,  5,  5,  5,  5,  5},
	  {12,  6,  6,  6 , 6,  6}, {14,  7,  7,  7,  7,  7}, {16,  8,  8,  8,  8,  8},
	  { 9,  9,  9,  9,  9,  9}, {10, 10, 10, 10, 10, 10}, {11, 11, 11, 11, 11, 11},
	  {12, 12, 12, 12, 12, 12}, {13, 13, 13, 13, 13, 13}, {14, 14, 14, 14, 14, 14},
	  {15, 15, 15, 15, 15, 15}, {16, 16, 16, 16, 16, 16}, {17, 17, 17, 17, 17, 17},
	  {18, 18, 18, 18, 18, 18}, {19, 19, 19, 19, 19, 19}, {20, 20, 20, 20, 20, 20},
	  {21, 21, 21, 21, 21, 21}, {22, 22, 22, 22, 22, 22}, {23, 23, 23, 23, 23, 23},
	  {24, 24, 24, 24, 24, 24}, {25, 25, 25, 25, 25, 25}, {26, 26, 26, 26, 26, 26},
	  {27, 27, 27, 27, 27, 27}, {28, 28, 28, 28, 28, 28}, {29, 29, 29, 29, 29, 29},
	  {30, 30, 30, 30, 30, 30}, {31, 31, 31, 31, 31, 31}
	};

	static const unsigned int R[][6] = 
	{
	  {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
	  {0x0000ffff, 0x000000ff, 0x0000000f, 0x00000003, 0x00000001, 0x00000001},
	  {0x0000ffff, 0x000000ff, 0x0000000f, 0x00000003, 0x00000003, 0x00000003},
	  {0x00007fff, 0x0000003f, 0x00000007, 0x00000007, 0x00000007, 0x00000007},
	  {0x0000ffff, 0x000000ff, 0x0000000f, 0x0000000f, 0x0000000f, 0x0000000f},
	  {0x00007fff, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f},
	  {0x00000fff, 0x0000003f, 0x0000003f, 0x0000003f, 0x0000003f, 0x0000003f},
	  {0x00003fff, 0x0000007f, 0x0000007f, 0x0000007f, 0x0000007f, 0x0000007f},
	  {0x0000ffff, 0x000000ff, 0x000000ff, 0x000000ff, 0x000000ff, 0x000000ff},
	  {0x000001ff, 0x000001ff, 0x000001ff, 0x000001ff, 0x000001ff, 0x000001ff}, 
	  {0x000003ff, 0x000003ff, 0x000003ff, 0x000003ff, 0x000003ff, 0x000003ff}, 
	  {0x000007ff, 0x000007ff, 0x000007ff, 0x000007ff, 0x000007ff, 0x000007ff}, 
	  {0x00000fff, 0x00000fff, 0x00000fff, 0x00000fff, 0x00000fff, 0x00000fff}, 
	  {0x00001fff, 0x00001fff, 0x00001fff, 0x00001fff, 0x00001fff, 0x00001fff}, 
	  {0x00003fff, 0x00003fff, 0x00003fff, 0x00003fff, 0x00003fff, 0x00003fff}, 
	  {0x00007fff, 0x00007fff, 0x00007fff, 0x00007fff, 0x00007fff, 0x00007fff}, 
	  {0x0000ffff, 0x0000ffff, 0x0000ffff, 0x0000ffff, 0x0000ffff, 0x0000ffff}, 
	  {0x0001ffff, 0x0001ffff, 0x0001ffff, 0x0001ffff, 0x0001ffff, 0x0001ffff}, 
	  {0x0003ffff, 0x0003ffff, 0x0003ffff, 0x0003ffff, 0x0003ffff, 0x0003ffff}, 
	  {0x0007ffff, 0x0007ffff, 0x0007ffff, 0x0007ffff, 0x0007ffff, 0x0007ffff},
	  {0x000fffff, 0x000fffff, 0x000fffff, 0x000fffff, 0x000fffff, 0x000fffff}, 
	  {0x001fffff, 0x001fffff, 0x001fffff, 0x001fffff, 0x001fffff, 0x001fffff}, 
	  {0x003fffff, 0x003fffff, 0x003fffff, 0x003fffff, 0x003fffff, 0x003fffff}, 
	  {0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff}, 
	  {0x00ffffff, 0x00ffffff, 0x00ffffff, 0x00ffffff, 0x00ffffff, 0x00ffffff},
	  {0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff}, 
	  {0x03ffffff, 0x03ffffff, 0x03ffffff, 0x03ffffff, 0x03ffffff, 0x03ffffff}, 
	  {0x07ffffff, 0x07ffffff, 0x07ffffff, 0x07ffffff, 0x07ffffff, 0x07ffffff},
	  {0x0fffffff, 0x0fffffff, 0x0fffffff, 0x0fffffff, 0x0fffffff, 0x0fffffff},
	  {0x1fffffff, 0x1fffffff, 0x1fffffff, 0x1fffffff, 0x1fffffff, 0x1fffffff}, 
	  {0x3fffffff, 0x3fffffff, 0x3fffffff, 0x3fffffff, 0x3fffffff, 0x3fffffff}, 
	  {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff}
	};

	unsigned int n;       // numerator
	unsigned int s; // s > 0

	in(n);
	in(s);

	unsigned int d = (1 << s) - 1; // so d is either 1, 3, 7, 15, 31, ...).
	unsigned int m;       // n % d goes here.

	m = (n & M[s]) + ((n >> s) & M[s]);

	for (const unsigned int * q = &Q[s][0], * r = &R[s][0]; m > d; q++, r++)
	{
	  m = (m >> *q) + (m & *r);
	}
	m = m == d ? 0 : m; // OR, less portably: m = m & -((signed)(m - d) >> s);

	out(m);
}

// ------------------------------------

void bp()
{
	unsigned int v; // 32-bit word to find the log base 2 of

	in(v);

	unsigned int r = 0; // r will be lg(v)

	while (v >>= 1) // unroll for more speed...
	{
	  r++;
	}

	out(r);
}

// ------------------------------------

void bq()
{
	int v; // 32-bit integer to find the log base 2 of

	in(v);

	int r; // result of log_2(v) goes here
	union { unsigned int u[2]; double d; } t; // temp

	t.u[1] = 0x43300000;
	t.u[0] = v;
	t.d -= 4503599627370496.0;
	r = (t.u[1] >> 20) - 0x3FF;

	out(r);
}

// ------------------------------------

void br()
{
	static const char LogTable256[256] = 
	{
	#define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
	    -1, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
	    LT(4), LT(5), LT(5), LT(6), LT(6), LT(6), LT(6),
	    LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7)
	};

	unsigned int v; // 32-bit word to find the log of

	in(v);

	unsigned r;     // r will be lg(v)
	register unsigned int t, tt; // temporaries

	if ((tt = (v >> 16)))
	{
	  r = (t = tt >> 8) ? 24 + LogTable256[t] : 16 + LogTable256[tt];
	}
	else 
	{
	  r = (t = v >> 8) ? 8 + LogTable256[t] : LogTable256[v];
	}

	out(r);
}

// ------------------------------------

void bs()
{
	static const char LogTable256[256] = 
	{
	#define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
	    -1, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
	    LT(4), LT(5), LT(5), LT(6), LT(6), LT(6), LT(6),
	    LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7)
	};

	unsigned int v; // 32-bit word to find the log of

	in(v);

	unsigned r;     // r will be lg(v)
	unsigned int tt; // temporaries

	if ((tt = (v >> 24)))
	{
	  r = 24 + LogTable256[tt];
	} 
	else if ((tt = (v >> 16)))
	{
	  r = 16 + LogTable256[tt];
	} 
	else if ((tt = (v >> 8)))
	{
	  r = 8 + LogTable256[tt];
	} 
	else 
	{
	  r = LogTable256[v];
	}

	out(r);
}

// ------------------------------------

void bt()
{
	unsigned int v;  // 32-bit value to find the log2 of 

	in(v);

	const unsigned int b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
	const unsigned int S[] = {1, 2, 4, 8, 16};
	int i;

	register unsigned int r = 0; // result of log2(v) will go here
	for (i = 4; i >= 0; i--) // unroll for speed...
	{
	  if (v & b[i])
	  {
	    v >>= S[i];
	    r |= S[i];
	  } 
	}

	out(r);
}

// ------------------------------------

void bu()
{
	// OR (IF YOUR CPU BRANCHES SLOWLY):

	unsigned int v;	         // 32-bit value to find the log2 of 

	in(v);

	register unsigned int r; // result of log2(v) will go here
	register unsigned int shift;

	r =     (v > 0xFFFF) << 4; v >>= r;
	shift = (v > 0xFF  ) << 3; v >>= shift; r |= shift;
	shift = (v > 0xF   ) << 2; v >>= shift; r |= shift;
	shift = (v > 0x3   ) << 1; v >>= shift; r |= shift;
						r |= (v >> 1);
	
	out(r);
}

// ------------------------------------

void bv()
{
	// OR (IF YOU KNOW v IS A POWER OF 2):

	unsigned int v;  // 32-bit value to find the log2 of 

	in(v);

	static const unsigned int b[] = {0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0, 
					 0xFF00FF00, 0xFFFF0000};
	register unsigned int r = (v & b[0]) != 0;
	int i;

	for (i = 4; i > 0; i--) // unroll for speed...
	{
	  r |= ((v & b[i]) != 0) << i;
	}

	out(r);
}

// ------------------------------------

void bw()
{
	uint32_t v; // find the log base 2 of 32-bit v

	in(v);

	int r;      // result goes here

	static const int MultiplyDeBruijnBitPosition[32] = 
	{
	  0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30,
	  8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31
	};

	v |= v >> 1; // first round down to one less than a power of 2 
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;

	r = MultiplyDeBruijnBitPosition[(uint32_t)(v * 0x07C4ACDDU) >> 27];

	out(r);
}

// ------------------------------------

void bx()
{
	uint32_t v; // find the log base 2 of 32-bit v

	in(v);

	int r;      // result goes here

	static const int MultiplyDeBruijnBitPosition2[32] = 
	{
	  0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8, 
	  31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
	};

	r = MultiplyDeBruijnBitPosition2[(uint32_t)(v * 0x077CB531U) >> 27];

	out(r);
}

// ------------------------------------

extern unsigned int IntegerLogBase2(unsigned int);

void by()
{
	unsigned int v; // non-zero 32-bit integer value to compute the log base 10 of 

	in(v);

	int r;          // result goes here
	int t;          // temporary

	static unsigned int const PowersOf10[] = 
	    {1, 10, 100, 1000, 10000, 100000,
	     1000000, 10000000, 100000000, 1000000000};

	t = (IntegerLogBase2(v) + 1) * 1233 >> 12; // (use a lg2 method from above)
	r = t - (v < PowersOf10[t]);

	out(r);
}

// ------------------------------------

void bz()
{
	unsigned int v; // non-zero 32-bit integer value to compute the log base 10 of 

	in(v);

	int r;          // result goes here

	r = (v >= 1000000000) ? 9 : (v >= 100000000) ? 8 : (v >= 10000000) ? 7 : 
	    (v >= 1000000) ? 6 : (v >= 100000) ? 5 : (v >= 10000) ? 4 : 
	    (v >= 1000) ? 3 : (v >= 100) ? 2 : (v >= 10) ? 1 : 0;

	out(r);
}

// ------------------------------------

void ca()
{
	// float v; // find int(log2(v)), where v > 0.0 && finite(v) && isnormal(v)
	int v;

	in(v);

	int c;         // 32-bit int c gets the result;

	c = *(const int *) &v;  // OR, for portability:  memcpy(&c, &v, sizeof c);
	c = (c >> 23) - 127;

	out(c);
}

// ------------------------------------

void cb()
{
	static const char LogTable256[256] = 
	{
	#define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
	    -1, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
	    LT(4), LT(5), LT(5), LT(6), LT(6), LT(6), LT(6),
	    LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7)
	};

	// float v;              // find int(log2(v)), where v > 0.0 && finite(v)
	int v;

	in(v);

	int c;                      // 32-bit int c gets the result;
	int x = *(const int *) &v;  // OR, for portability:  memcpy(&x, &v, sizeof x);

	c = x >> 23;          

	if (c)
	{
	  c -= 127;
	}
	else
	{ // subnormal, so recompute using mantissa: c = intlog2(x) - 149;
	  register unsigned int t; // temporary
	  // Note that LogTable256 was defined <a href="#IntegerLogLookup">earlier</a>
	  if ((t = (x >> 16)))
	  {
	    c = LogTable256[t] - 133;
	  }
	  else
	  {
	    c = (t = x >> 8) ? LogTable256[t] - 141 : LogTable256[x] - 149;
	  }
	}

	out(c);
}

// ------------------------------------

void cc()
{
	int r, v;
	// float v; // find int(log2(pow((double) v, 1. / pow(2, r)))), 
		       // where isnormal(v) and v > 0

	in(r);
	in(v);

	int c;         // 32-bit int c gets the result;

	c = *(const int *) &v;  // OR, for portability:  memcpy(&c, &v, sizeof c);
	c = ((((c - 0x3f800000) >> r) + 0x3f800000) >> 23) - 127;

	out(c);
}

// ------------------------------------

void cd()
{
	unsigned int v;  // input to count trailing zero bits

	in(v);

	int c;  // output: c will count v's trailing zero bits,
		// so if v is 1101000 (base 2), then c will be 3
	if (v)
	{
	  v = (v ^ (v - 1)) >> 1;  // Set v's trailing 0s to 1s and zero rest
	  for (c = 0; v; c++)
	  {
	    v >>= 1;
	  }
	}
	else
	{
	  c = CHAR_BIT * sizeof(v);
	}

	out(c);
}

// ------------------------------------

void ce()
{
	unsigned int v;      // 32-bit word input to count zero bits on right

	in(v);

	unsigned int c = 32; // c will be the number of zero bits on the right
	v &= -(signed)v;
	if (v) c--;
	if (v & 0x0000FFFF) c -= 16;
	if (v & 0x00FF00FF) c -= 8;
	if (v & 0x0F0F0F0F) c -= 4;
	if (v & 0x33333333) c -= 2;
	if (v & 0x55555555) c -= 1;

	out(c);
}

// ------------------------------------

void cf()
{
	unsigned int v;     // 32-bit word input to count zero bits on right

	in(v);

	unsigned int c;     // c will be the number of zero bits on the right,
			    // so if v is 1101000 (base 2), then c will be 3
	// NOTE: if 0 == v, then c = 31.
	if (v & 0x1) 
	{
	  // special case for odd v (assumed to happen half of the time)
	  c = 0;
	}
	else
	{
	  c = 1;
	  if ((v & 0xffff) == 0) 
	  {  
	    v >>= 16;  
	    c += 16;
	  }
	  if ((v & 0xff) == 0) 
	  {  
	    v >>= 8;  
	    c += 8;
	  }
	  if ((v & 0xf) == 0) 
	  {  
	    v >>= 4;
	    c += 4;
	  }
	  if ((v & 0x3) == 0) 
	  {  
	    v >>= 2;
	    c += 2;
	  }
	  c -= v & 0x1;
	}	

	out(c);
}

// ------------------------------------

void cg()
{
	unsigned int v;  // find the number of trailing zeros in v

	in(v);

	int r;           // put the result in r
	static const int Mod37BitPosition[] = // map a bit value mod 37 to its position
	{
	  32, 0, 1, 26, 2, 23, 27, 0, 3, 16, 24, 30, 28, 11, 0, 13, 4,
	  7, 17, 0, 25, 22, 31, 15, 29, 10, 12, 6, 0, 21, 14, 9, 5,
	  20, 8, 19, 18
	};
	r = Mod37BitPosition[(-v & v) % 37];

	out(r);
}

// ------------------------------------

void ch()
{
	unsigned int v;  // find the number of trailing zeros in 32-bit v 

	in(v);

	int r;           // result goes here
	static const int MultiplyDeBruijnBitPosition[32] = 
	{
	  0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8, 
	  31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
	};
	r = MultiplyDeBruijnBitPosition[((uint32_t)((v & -v) * 0x077CB531U)) >> 27];

	out(r);
}

// ------------------------------------

void ci()
{
	unsigned int v; // compute the next highest power of 2 of 32-bit v

	in(v);

	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;

	out(v);
}

// ------------------------------------

void cj()
{
	unsigned short x;   // Interleave bits of x and y, so that all of the
	unsigned short y;   // bits of x are in the even positions and y in the odd;

	in(x);
	in(y);

	unsigned int z = 0; // z gets the resulting Morton Number.

	for (int i = 0; i < sizeof(x) * CHAR_BIT; i++) // unroll for more speed...
	{
	  z |= (x & 1U << i) << i | (y & 1U << i) << (i + 1);
	}

	out(z);
}

// ------------------------------------

void ck()
{
	static const unsigned short MortonTable256[256] = 
	{
	  0x0000, 0x0001, 0x0004, 0x0005, 0x0010, 0x0011, 0x0014, 0x0015, 
	  0x0040, 0x0041, 0x0044, 0x0045, 0x0050, 0x0051, 0x0054, 0x0055, 
	  0x0100, 0x0101, 0x0104, 0x0105, 0x0110, 0x0111, 0x0114, 0x0115, 
	  0x0140, 0x0141, 0x0144, 0x0145, 0x0150, 0x0151, 0x0154, 0x0155, 
	  0x0400, 0x0401, 0x0404, 0x0405, 0x0410, 0x0411, 0x0414, 0x0415, 
	  0x0440, 0x0441, 0x0444, 0x0445, 0x0450, 0x0451, 0x0454, 0x0455, 
	  0x0500, 0x0501, 0x0504, 0x0505, 0x0510, 0x0511, 0x0514, 0x0515, 
	  0x0540, 0x0541, 0x0544, 0x0545, 0x0550, 0x0551, 0x0554, 0x0555, 
	  0x1000, 0x1001, 0x1004, 0x1005, 0x1010, 0x1011, 0x1014, 0x1015, 
	  0x1040, 0x1041, 0x1044, 0x1045, 0x1050, 0x1051, 0x1054, 0x1055, 
	  0x1100, 0x1101, 0x1104, 0x1105, 0x1110, 0x1111, 0x1114, 0x1115, 
	  0x1140, 0x1141, 0x1144, 0x1145, 0x1150, 0x1151, 0x1154, 0x1155, 
	  0x1400, 0x1401, 0x1404, 0x1405, 0x1410, 0x1411, 0x1414, 0x1415, 
	  0x1440, 0x1441, 0x1444, 0x1445, 0x1450, 0x1451, 0x1454, 0x1455, 
	  0x1500, 0x1501, 0x1504, 0x1505, 0x1510, 0x1511, 0x1514, 0x1515, 
	  0x1540, 0x1541, 0x1544, 0x1545, 0x1550, 0x1551, 0x1554, 0x1555, 
	  0x4000, 0x4001, 0x4004, 0x4005, 0x4010, 0x4011, 0x4014, 0x4015, 
	  0x4040, 0x4041, 0x4044, 0x4045, 0x4050, 0x4051, 0x4054, 0x4055, 
	  0x4100, 0x4101, 0x4104, 0x4105, 0x4110, 0x4111, 0x4114, 0x4115, 
	  0x4140, 0x4141, 0x4144, 0x4145, 0x4150, 0x4151, 0x4154, 0x4155, 
	  0x4400, 0x4401, 0x4404, 0x4405, 0x4410, 0x4411, 0x4414, 0x4415, 
	  0x4440, 0x4441, 0x4444, 0x4445, 0x4450, 0x4451, 0x4454, 0x4455, 
	  0x4500, 0x4501, 0x4504, 0x4505, 0x4510, 0x4511, 0x4514, 0x4515, 
	  0x4540, 0x4541, 0x4544, 0x4545, 0x4550, 0x4551, 0x4554, 0x4555, 
	  0x5000, 0x5001, 0x5004, 0x5005, 0x5010, 0x5011, 0x5014, 0x5015, 
	  0x5040, 0x5041, 0x5044, 0x5045, 0x5050, 0x5051, 0x5054, 0x5055, 
	  0x5100, 0x5101, 0x5104, 0x5105, 0x5110, 0x5111, 0x5114, 0x5115, 
	  0x5140, 0x5141, 0x5144, 0x5145, 0x5150, 0x5151, 0x5154, 0x5155, 
	  0x5400, 0x5401, 0x5404, 0x5405, 0x5410, 0x5411, 0x5414, 0x5415, 
	  0x5440, 0x5441, 0x5444, 0x5445, 0x5450, 0x5451, 0x5454, 0x5455, 
	  0x5500, 0x5501, 0x5504, 0x5505, 0x5510, 0x5511, 0x5514, 0x5515, 
	  0x5540, 0x5541, 0x5544, 0x5545, 0x5550, 0x5551, 0x5554, 0x5555
	};

	unsigned short x; // Interleave bits of x and y, so that all of the
	unsigned short y; // bits of x are in the even positions and y in the odd;

	in(x);
	in(y);

	unsigned int z;   // z gets the resulting 32-bit Morton Number.

	z = MortonTable256[y >> 8]   << 17 | 
	    MortonTable256[x >> 8]   << 16 |
	    MortonTable256[y & 0xFF] <<  1 | 
	    MortonTable256[x & 0xFF];
	
	out(z);
}

// ------------------------------------

void cl()
{
	unsigned char x;  // Interleave bits of (8-bit) x and y, so that all of the
	unsigned char y;  // bits of x are in the even positions and y in the odd;

	in(x);
	in(y);

	unsigned short z; // z gets the resulting 16-bit Morton Number.

	z = (((x * 0x0101010101010101ULL & 0x8040201008040201ULL) * 
	     0x0102040810204081ULL >> 49) & 0x5555) |
	    (((y * 0x0101010101010101ULL & 0x8040201008040201ULL) * 
	     0x0102040810204081ULL >> 48) & 0xAAAA);

	out(z);
}

// ------------------------------------

void cm()
{
	static const unsigned int B[] = {0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF};
	static const unsigned int S[] = {1, 2, 4, 8};

	unsigned int x; // Interleave lower 16 bits of x and y, so the bits of x
	unsigned int y; // are in the even positions and bits from y in the odd;

	in(x);
	in(y);

	unsigned int z; // z gets the resulting 32-bit Morton Number.  
			// x and y must initially be less than 65536.

	x = (x | (x << S[3])) & B[3];
	x = (x | (x << S[2])) & B[2];
	x = (x | (x << S[1])) & B[1];
	x = (x | (x << S[0])) & B[0];

	y = (y | (y << S[3])) & B[3];
	y = (y | (y << S[2])) & B[2];
	y = (y | (y << S[1])) & B[1];
	y = (y | (y << S[0])) & B[0];

	z = x | (y << 1);

	out(z);
}

// ------------------------------------

void cn()
{
	unsigned int v; // 32-bit word to check if any 8-bit byte in it is 0

	in(v);

	// Fewer operations:
	bool hasZeroByte = ~((((v & 0x7F7F7F7F) + 0x7F7F7F7F) | v) | 0x7F7F7F7F);

	out(hasZeroByte);
}

// ------------------------------------

void co()
{
	unsigned int v; // 32-bit word to check if any 8-bit byte in it is 0

	in(v);

	// More operations:
	bool hasNoZeroByte = ((v & 0xff) && (v & 0xff00) && (v & 0xff0000) && (v & 0xff000000));

	out(hasNoZeroByte);
}

// ------------------------------------

void cp()
{
	unsigned int v; // 32-bit word to check if any 8-bit byte in it is 0

	in(v);

	// OR:
	unsigned char * p = (unsigned char *) &v;  
	bool hasNoZeroByte = *p && *(p + 1) && *(p + 2) && *(p + 3);

	out(hasNoZeroByte);
}

// ------------------------------------

void cq()
{
	unsigned int v; // 32-bit word to check if any 8-bit byte in it is 0

	in(v);

	bool hasZeroByte = ((v + 0x7efefeff) ^ ~v) & 0x81010100;
	if (hasZeroByte) // or may just have 0x80 in the high byte
	{
	  hasZeroByte = ~((((v & 0x7F7F7F7F) + 0x7F7F7F7F) | v) | 0x7F7F7F7F);
	}

	out(hasZeroByte);
}

// ------------------------------------

void cr()
{
	unsigned int v; // current permutation of bits 

	in(v);

	unsigned int w; // next permutation of bits

	unsigned int t = v | (v - 1); // t gets v's least significant 0 bits set to 1
	// Next set to 1 the most significant bit to change, 
	// set to 0 the least significant ones, and add the necessary 1 bits.
	w = (t + 1) | (((~t & -~t) - 1) >> (__builtin_ctz(v) + 1));  

	out(w);
}

// ------------------------------------

void cs()
{
	unsigned int v; // current permutation of bits 

	in(v);

	unsigned int w; // next permutation of bits

	unsigned int t = (v | (v - 1)) + 1;  
	w = t | ((((t & -t) / (v & -v)) >> 1) - 1);  

	out(w);
}


// Source: http://programming.sirrida.de/


/*
  desc: SWAR routines for C++
  xlat: g++ -std=c++11 -O3 swartmps.cpp
*/


//////
// Intro.

// SWAR / SIMD library and test routines

// (c) 2011..2017 by Jasper L. Neumann
// www.sirrida.de / programming.sirrida.de
// E-Mail: info@sirrida.de

// Granted to the public domain
// First version: About 2014-07-10
// Last change: 2016-07-02

// Compile with
// g++ -std=c++11 -O3 swartmps.cpp


//////
// Defines.

#ifdef __SIZEOF_INT128__
#define has_128
#endif

#define CHAR_BIT 8


//////
// Base types.

typedef signed char t_8s;
typedef unsigned char t_8u;
typedef signed short int t_16s;
typedef unsigned short int t_16u;
typedef signed int t_32s;
typedef unsigned int t_32u;
typedef signed long long int t_64s;
typedef unsigned long long int t_64u;
#ifdef has_128
typedef __int128_t t_128s;
typedef __uint128_t t_128u;
#endif
typedef signed int t_int;
typedef unsigned int t_uint;


//////
// General functions which ought to be in std::

//////
// M_abs(T x)

// Absolute value.
// INLINE
template<typename T>
constexpr auto M_abs(T x) -> T {

  return (x >= 0) ? x : -x;
  }


//////
// M_gcd(T a, T b)

// Greatest common divisor.
template<typename T1, typename T2>
// constexpr
auto M_gcd(T1 x, T2 y) -> decltype(x+y) {

  typedef decltype(x+y) T;
  T t,a,b;

  a = x;
  b = y;
  // a = M_abs(a);
  // b = M_abs(b);
  while (b != 0) {
    t = a % b;
    a = b;
    b = t;
    }
  return a;
  }


//////
// M_odd(T x)

// Return true for odd values, false for even.
// INLINE
template<typename T>
constexpr auto M_odd(T x) -> bool {

  return (x & 1) != 0;
  }


//////
// M_sgn(x)

// Return the sign: <0:-1, 0:0, >0:1
// INLINE
template<typename T>
constexpr auto M_sgn(T x) -> t_int {

  return ((x>0) ? 1 : 0) - ((x<0) ? 1 : 0);
  }


//////
// M_min(x,y)

// Minimum value.
// INLINE
template<typename T1, typename T2>
constexpr auto M_min(T1 x, T2 y) -> decltype(x+y) {

  return (x<y) ? x : y;
  }


//////
// M_max(x,y)

// Maximum value.
// INLINE
template<typename T1, typename T2>
constexpr auto M_max(T1 x, T2 y) -> decltype(x+y) {

  return (x>y) ? x : y;
  }


//////
// M_abs_diff(x,y)

// Equivalent to abs(x-y) via a larger type;
// works for signed and unsigned.
// Result type ought to be unsigned.
template<typename T1, typename T2>
constexpr auto M_abs_diff(T1 x, T2 y) -> decltype(x+y) {

  return (x>y) ? (x-y) : (y-x);
  }


//////
// M_sgn_diff(x,y)

// Equivalent to sgn(x-y) via a larger type;
// works for signed and unsigned.
template<typename T1, typename T2>
constexpr auto M_sgn_diff(T1 x, T2 y) -> int {

  return ((x>y) ? 1 : 0) - ((x<y) ? 1 : 0);
  }


//////
// Enum types.

// Subword sizes.
enum class t_subword {
            // ld_bits bits
  bit,      //       0    1
  nyp,      //       1    2
  nibble,   //       2    4
  byte,     //       3    8
  word,     //       4   16
  dword,    //       5   32
  qword,    //       6   64
  dqword,   //       7  128
  };

// Modes for tbm (trailing bit modification operations), see t_bits_base.
enum t_tbm_mode {
  lo_0        = 0x00,    // Set lower bits to 0.
  lo_1        = 0x01,    // Set lower bits to 1.
  lo_mask     = 0x01,    // Set lower bits.

  tgt_0       = 0x00,    // Set found least significant bit to 0.
  tgt_1       = 0x02,    // Set found least significant bit to 1.
  tgt_mask    = 0x02,    // Set found least significant bit.

  hi_0        = 0x00,    // Set upper bits to 0.
  hi_1        = 0x04,    // Set upper bits to 1.
  hi_keep     = 0x08,    // Keep upper bits.
  hi_invert   = 0x0c,    // Invert upper bits.
  hi_mask     = 0x0c,    // Treat upper bits.

  search_0    = 0x00,    // Search for least significant 0 bit.
  search_1    = 0x10,    // Search for least significant 1 bit.
  search_mask = 0x10,    // Search for least significant.
  };

// Floating point rounding modes.
enum class t_round_mode {
  half_even,   // round, *.5 => next even, banker's rounding, x87:0
  floor,       // floor -> -infinity, sar, x87:1
  ceil,        // ceil -> infinity, x87:2
  down,        // trunc -> 0, chop, div, x87:3
  up,          // -> away from 0
  half_odd,    // round, *.5 => next odd
  half_floor,  // round, *.5 => floor
  half_ceil,   // round, *.5 => ceil
  half_down,   // round, *.5 => trunk
  half_up,     // round, *.5 => away from 0
  };

// Rounding modes, examples:
// *  mode \ value  -1.6 -1.5 -1.4 -1 -0.6 -0.5 -0.4 0 0.4 0.5 0.6 1 1.4 1.5 1.6
// e  half_even     -2   -2   -1   -1 -1    0    0   0 0   0   1   1 1   2   2
// f  floor         -2   -2   -2   -1 -1   -1   -1   0 0   0   0   1 1   1   1
// c  ceil          -1   -1   -1   -1  0    0    0   0 1   1   1   1 2   2   2
// d  down          -1   -1   -1   -1  0    0    0   0 0   0   0   1 1   1   1
// u  up            -2   -2   -2   -1 -1   -1   -1   0 1   1   1   1 2   2   2
// o  half_odd      -2   -1   -1   -1 -1   -1    0   0 0   1   1   1 1   1   2
// hf half_floor    -2   -2   -1   -1 -1   -1    0   0 0   0   1   1 1   1   2
// hc half_ceil     -2   -1   -1   -1 -1    0    0   0 0   1   1   1 1   2   2
// hd half_down     -2   -1   -1   -1 -1    0    0   0 0   0   1   1 1   1   2
// hu half_up       -2   -2   -1   -1 -1   -1    0   0 0   1   1   1 1   2   2
// * suffix for avg functions;
//   for unsigned functions u/d are preferred over f/c (same meaning).

// See http://docs.oracle.com/javase/6/docs/api/java/math/RoundingMode.html


//////
// Template prototypes.

template<typename T> class t_bits {};
template<typename T> class t_simd {};
template<typename T> class t_bfly {};
template<typename T> class t_cef {};
template<typename T> class t_ce_right {};
template<typename T> class t_ce_left {};
template<typename T> class t_vrot {};


//////
// Template implementations.

// Base implementation for bit operations acting on whole type.
// T: Base type, must be an unsigned integral type.
// TS_: must be the signed type corresponding to T (same size).
// ld_bits_: The log_2 of the number of bits of T.
template<
  typename T,
  typename TS_,
  t_uint ld_bits_,
  const T a_element_[],
  const T a_lo_[],
  const T a_hi_[],
  const T a_even_[],
  const T a_shuffle_[],
  const T a_prim_[]
  > class t_bits_base {
public:

  // Imports, types, asserts.
  typedef T TU;
  typedef TS_ TS;

  static constexpr const TU *a_element = a_element_;
    // Unfortunately [] and the info about array range is gone here.
  static constexpr const TU *a_lo = a_lo_;
  static constexpr const TU *a_hi = a_hi_;
  static constexpr const TU *a_even = a_even_;
  static constexpr const TU *a_shuffle = a_shuffle_;
  static constexpr const TU *a_prim = a_prim_;
  static constexpr const t_uint ld_bits = ld_bits_;

  static constexpr const t_uint bits = 1 << ld_bits;
  static constexpr const TU all_bits = ~(TU)(0);
  static constexpr const TU lo_bit = (TU)(1);
  static constexpr const TU hi_bit = (TU)(1) << (bits-1);

  static_assert(sizeof(TU) == sizeof(T), "sizeof(TU) == sizeof(T)");
  static_assert(sizeof(TS) == sizeof(T), "sizeof(TS) == sizeof(T)");
  static_assert(bits == sizeof(T)*CHAR_BIT, "bits == sizeof(T)*CHAR_BIT");
  static_assert((TU)(-1) > 0, "(TU)(-1) > 0");  // TU must be unsigned.
  static_assert((TS)(-1) < 0, "(TS)(-1) < 0");  // TS must be signed.

  // Rotating.

  // Rotate x left by rot.
  // Gives correct results for all values of rot.
  // Should usually be optimizable to one instruction (ROL).
  // O(1)
  // INLINE
  static constexpr auto rol(T x, t_uint rot) -> T {

    return (T)((TU)(x) << (rot & (bits-1))) |
           (T)((TU)(x) >> ((-rot) & (bits-1)));
    }

  // Rotate x right by rot.
  // Gives correct results for all values of rot.
  // Should usually be optimizable to one instruction (ROR).
  // O(1)
  // INLINE
  static constexpr auto ror(T x, t_uint rot) -> T {

    return (T)((TU)(x) >> (rot & (bits-1))) |
           (T)((TU)(x) << ((-rot) & (bits-1)));
    }

  // Shifting.
  // Shifting. TODO: Should *_safe or *_fast better be default?

  // Logically/arithmetically shift x left by shift.
  // Only as guaranteed by the C standard, i.e. 0 <= shift < bits.
  // Should usually be implementable by one instruction (SHL).
  // O(1)
  // INLINE
  static constexpr auto shl_fast(T x, t_uint shift) -> T {

    return (T)((TU)(x) << shift);
    }

  // Logically/arithmetically shift x left by shift.
  // Gives correct results for all values of shift;
  // please note that shift is unsigned.
  // O(1)
  static constexpr auto shl_safe(T x, t_uint shift) -> T {

    return (shift >= bits) ? 0 : shl_fast(x, shift);
    }

  // Logically shift x right by shift.
  // Only as guaranteed by the C standard, i.e. 0 <= shift < bits.
  // Should usually be implementable by one instruction (SHR).
  // O(1)
  // INLINE
  static constexpr auto shr_fast(T x, t_uint shift) -> T {

    return (T)((TU)(x) >> shift);
    }

  // Logically shift x right by shift.
  // Gives correct results for all values of shift;
  // please note that shift is unsigned.
  // O(1)
  static constexpr auto shr_safe(T x, t_uint shift) -> T {

    return (shift >= bits) ? 0 : shr_fast(x, shift);
    }

  // Arithmetically shift right, duplicating  the most significant (sign) bit.
  // Only as guaranteed by the C standard, i.e. 0 <= shift < bits.
  // Should usually be optimizable to one instruction (SAR).
  // O(1)
  // INLINE
  static constexpr auto sar_fast(T x, t_uint shift) -> T {

    if ((TS)(-1) >> 1 == (TS)(-1))  // Directly supported?
      return (T)((TS)(x) >> shift);  // Yes.
    else if ((TS)(x) >= 0)  // No; emulate it.
      return (T)((TU)(x) >> shift);  // Zero or positive.
    else
      return (T)(((TU)(x) >> shift) | ~((TU)(all_bits) >> shift));  // Negative.
    }

  // Arithmetically shift x right by shift.
  // Gives correct results for all values of shift;
  // please note that shift is unsigned.
  // O(1)
  static constexpr auto sar_safe(T x, t_uint shift) -> T {

    // return (shift >= bits) ? sar_fast(x, bits-1) : sar_fast(x, shift);
    return sar_fast(x, (shift >= bits) ? (bits-1) : shift);
    }

  // sal, sal, shr1, shl1 (not an instruction)
  // Missing: shl_sat, i.e. *2^shift saturated to max value, i.e. 0x7f../0xff..

  // Counting bits.

  // Count the set bits, aka population count.
  // On newer x86 implementable as POPCNT.
  // See Hacker's Delight, 5.1 "Counting 1-Bits"
  // O(ld_bits)
  static auto nr_1bits(T x) -> t_uint {

    t_int i, s;

    s = 1;
    for (i = 0; i <= ld_bits-1; ++i) {  // UNROLL
      // s = 1 << i;
      x = (x & a_even[i]) + (shr_fast(x, s) & a_even[i]);
      s = s << 1;
      }
    return (t_uint)(x);
    }

  // Count the reset bits.
  // O(ld_bits)
  static auto nr_0bits(T x) -> t_uint {

    return nr_1bits(~x);
    }

  // See Hacker's Delight, 5.4 "Counting Trailing 0's"
  // On x86 implementable as TZCNT or using BSF.
  // Also implementable by multiplying x&-x with a De Bruijn number,
  // shifting, and a lookup table.
  // Special case: 0 => bits.
  // O(ld_bits)
  static auto nr_trailing_0bits(T x) -> t_uint {

    t_int res, i, s;

    if (x == 0) {
      res = bits;
      }
    else {
      res = 0;
      s = 1 << (ld_bits-1);
      for (i = ld_bits-1; i >= 0; --i) {  // UNROLL
        // s = 1 << i;
        if ((x & a_element[i]) == 0) {
          x = shr_fast(x, s);
          res = res + s;
          }
        s = s >> 1;
        }
      }
    return (t_uint)(res);
    }

  // Number of contiguous least significant set bits
  // O(ld_bits)
  static auto nr_trailing_1bits(T x) -> t_uint {

    return nr_trailing_0bits(~x);
    }

  // See Hacker's Delight, 5.4 "Counting Trailing 0's"
  // On x86 implementable using BSR or as LZCNT.
  // Special case: 0 => bits.
  // O(ld_bits)
  static auto nr_leading_0bits(T x) -> t_uint {

    t_int res, i, s;

    if (x == 0) {
      res = bits;
      }
    else {
      res = bits - 1;
      s = 1 << (ld_bits-1);
      for (i = ld_bits-1; i >= 0; --i) {  // UNROLL
        // s = 1 << i;
        if ((x & ~a_element[i]) != 0) {
          x = shr_fast(x, s);
          res = res - s;
          }
        s = s >> 1;
        }
      }
    return (t_uint)(res);
    }

  // Number of contiguous most significant set bits
  // O(ld_bits)
  static auto nr_leading_1bits(T x) -> t_uint {

    return nr_leading_0bits(~x);
    }

  // Count the set bits and return whether this is an odd number.
  // O(ld_bits)
  static auto is_parity_odd(T x) -> bool {

    return M_odd(nr_1bits(x));
    }

  // Miscellaneous.

  // Sign extended the low b bits of x.
  static auto sign_extend(T x, t_uint b) -> TS {

    T mask;

    if (b == 0)
      return 0;
    else {
      mask = lo_bit << (b-1);
      return ((x & (mask*2-1)) ^ mask)-mask;
      }
    }

  // The bit equivalent to m?x:y.
  // O(1)
  // INLINE
  static auto blend(T m, T x, T y) -> T {

    return (m & x) | (~m & y);  // | can be replaced by ^ or +
    // return ((x | y) & m) ^ y;
    }

  // Gray code.
  // See Hacker's Delight, 13.1 "Gray Code"
  // O(1)
  static auto gray_code(T x) -> T {

    return x ^ shr(x, 1);
    }

  // Inverse Gray code.
  // See Hacker's Delight, 13.1 "Gray Code"
  // O(ld_bits)
  static auto inv_gray_code(T x) -> T {

    t_int i, s;

    s = 1;
    for (i = 0; i <= ld_bits-1; ++i) {
      // s = 1 << i;
      x = x ^ shr_fast(x, s);
      s = s << 1;
      }
    return x;
    }

  // Is x a contiguous string of 1 bits?
  // O(1)
  static auto is_contiguous_1bits(T x) -> bool {

    return ((((x - 1) | x) + 1) & x) == 0;
    }

  // General trailing bit modification operations.
  // 2014-02-11 by Jasper L. Neumann
  // Mode: The operating mode, see t_tbm_mode.
  // All of these operations can be performed by <= 3 instructions.
  // Some of these operations are realized as BMI1 or TBM instructions.
  // O(1)
  static auto tbm(T x, t_int mode) -> T {

    switch (mode & 0x1f) {
      case 0x00: return 0;
      case 0x01: return x & ~(x+1);
      case 0x02: return ~x & (x+1);
      case 0x03: return x ^ (x+1);
      case 0x04: return ~(x ^ (x+1));
      case 0x05: return x | ~(x+1);
      case 0x06: return ~x | (x+1);
      case 0x07: return all_bits;
      case 0x08: return x & (x+1);
      case 0x09: return x;
      case 0x0a: return x+1;
      case 0x0b: return x | (x+1);
      case 0x0c: return ~(x | (x+1));
      case 0x0d: return ~x-1;
      case 0x0e: return ~x;
      case 0x0f: return ~(x & (x+1));
      case 0x10: return 0;
      case 0x11: return ~x & (x-1);
      case 0x12: return x & -x;
      case 0x13: return x ^ (x-1);
      case 0x14: return x ^ -x;
      case 0x15: return ~x | (x-1);
      case 0x16: return x | -x;
      case 0x17: return all_bits;
      case 0x18: return x & (x-1);
      case 0x19: return x-1;
      case 0x1a: return x;
      case 0x1b: return x | (x-1);
      case 0x1c: return ~(x | (x-1));
      case 0x1d: return ~x;
      case 0x1e: return -x;
      case 0x1f: return ~(x & (x-1));
      default:   return 0;  // This can not happen.
      }
    }

  // and, or, xor
  // not
  // add, sub, neg, mul_lo, mul_hi_s, mul_hi_u

  // The Fortran function ISIGN: copy sign for integer values
  // (y>=0) ? abs(x) : -abs(x)
  // Should be very fast.
  // See Hacker's Delight, 2.9 "Transfer of Sign"

  // In x86 asm (eax,edx=>eax):
  //   xor   edx,eax
  //   sar   edx,31
  //   xor   eax,edx
  //   sub   eax,edx

  static auto isign(T x, T y) -> T {

    if ((TS)(y) >= 0)
      return M_abs(x);
    else
      return -M_abs(x);
    }

  // Simple transfer sign
  // (y>=0) ? x : -x
  // Should be very fast.

  // In x86 asm (eax,edx=>eax):
  //   sar   edx,31
  //   xor   eax,edx
  //   sub   eax,edx

  static auto xsign(T x, T y) -> T {

    if ((TS)(y) >= 0)
      return x;
    else
      return -x;
    }

  // Permuting.

  // Can be replaced by bit_permute_step_simple,
  // if for the relevant bits n the following holds:
  // nr_1bits(bit_permute_step_simple(n, m, shift)) = nr_1bits(n)
  // shift should be in 0..bits-1.
  // This is a candidate for a new instruction.
  // x86: >= 6/5 cycles
  // ARM: >= 4/4 cycles
  // INLINE
  static auto bit_permute_step(T x, T m, t_uint shift) -> T {

    T t;

    // assert((shl_fast(m, shift) & m) == 0);
    // assert(shr_fast(shl_fast(m, shift), shift) == m);
    t = (shr_fast(x, shift) ^ x) & m;
    // x = (x ^ t) ^ shl_fast(t, shift);
    x = x ^ t;  t = shl_fast(t, shift);  x = x ^ t;
    return x;
    }

  // Simplified replacement of bit_permute_step
  // Can always be replaced by bit_permute_step (not vice-versa).
  // shift should be in 0..bits-1.
  // x86: >= 5/4 (5/3) cycles
  // ARM: >= 3/2 cycles
  // INLINE
  static auto bit_permute_step_simple(T x, T m, t_uint shift) -> T {

    // assert((shl_fast(m, shift) & m) == 0);
    // assert(shr_fast(shl_fast(m, shift), shift) == m);
    // assert((shl_fast(m, shift) | m) == all_bits);  // for permutations
    x = shl_fast(x & m, shift) | (shr_fast(x, shift) & m);
    return x;
    }

  // Extended variant of bit_permute_step.
  // Will be slow if not inlined.
  // shift should be in 0..bits-1.
  // This is a candidate for a new instruction.
  // INLINE
  static void bit_permute_step2(T* x1, T* x2, T m, t_uint shift) {

    TU t;

    t = (shr_fast(*x2, shift) ^ *x1) & m;
    *x1 = *x1 ^ t;
    *x2 = *x2 ^ shl_fast(t, shift);
    }

  // This is similar to bit_permute_step but rotating instead of shifting.
  // shift should be in 0..bits-1.
  // This is a candidate for a new instruction.
  // x86: >= 6/5 cycles
  // ARM: >= 4/4 cycles
  // INLINE
  static auto bit_permute_step_rot(T x, T m, t_uint shift) -> T {

    T t;

    // assert((rol(m, shift) & m) == 0);
    // assert(ror(shl_fast(m, shift), shift) == m);
    t = (ror(x, shift) ^ x) & m;
    x = x ^ t;  t = rol(t, shift);  x = x ^ t;  // x = (x ^ t) ^ rol(t, shift);
    return x;
    }

  // This is similar to bit_permute_step2 but rotating instead of shifting.
  // Extended variant of bit_permute_step_rot.
  // Will be slow if not inlined.
  // shift should be in 0..bits-1.
  // This is a candidate for a new instruction.
  // INLINE
  static void bit_permute_step2_rot(T* x1, T* x2, T m, t_uint shift) {

    TU t;

    t = (ror(*x2, shift) ^ *x1) & m;
    *x1 = *x1 ^ t;
    *x2 = *x2 ^ rol(t, shift);
    }

  // Multiplication.

  // Calculate multiplicative inverse, i.e. mul_inv(x)*x==1.
  // The result is defined for all odd values of x.
  // For even x we simply return 0.
  // See Hacker's Delight, 10.16 "Exact division by constants"
  // Multiplicative inverse modulo 2**bits by Newton's method.
  static auto mul_inv(T x) -> T {

    T xn,t;

    if (!M_odd(x))
      return 0;  // only defined for odd numbers

    xn = x;
    while (true) {
      t = x * xn;
      if (t == 1)
        return xn;
      xn = xn * (2 - t);
      }
    }

  // Carry-less multiplication.
  typedef T t_bit_matrix[bits];

  // Bit matrix inversion by Gauß/Jordan elimination.
  // 2014-01-29 by Jasper L. Neumann
  static void cl_mat_inv(const t_bit_matrix& src, t_bit_matrix& inv) {

    t_bit_matrix org;
    T tmp, mask;
    t_int i, j;
    bool ok;

    mask = 1;
    for (i = 0; i <= bits-1; ++i) {
      org[i] = src[i];  // copy bit matrix
      inv[i] = mask;
      mask = shl_fast(mask, 1);
      }
    mask = 1;
    for (i = 0; i <= bits-1; ++i) {
      if ((org[i] & mask) == 0) {
        ok = false;
        for (j = i+1; j <= bits-1; ++j) {
          if ((org[j] & mask) != 0) {
            // std::swap(org[i], org[j]);
            // std::swap(inv[i], inv[j]);

            tmp = org[i];
            org[i] = org[j];
            org[j] = tmp;

            tmp = inv[i];
            inv[i] = inv[j];
            inv[j] = tmp;

            ok = true;
            break;
            }
          }
        if (!ok) {
          for (j = 0; j <= bits-1; ++j)
            inv[j] = 0;
          return;  // Not invertible!
          }
        }
      for (j = 0; j <= bits-1; ++j) {
        if (i != j) {
          if ((org[j] & mask) != 0) {
            org[j] = org[j] ^ org[i];
            inv[j] = inv[j] ^ inv[i];
            }
          }
        }
      mask = shl_fast(mask, 1);
      }
    return;
    }

  // "Carry-less multiplication" (cl_mul).
  // The properties are quite similar to the usual multiplication (and addition).
  //
  // Properties (^ is the XOR operator; * is the cl_mul operator):
  // - associative: a^(b^c) = (a^b)^c, a*(b*c) = (a*b)*c
  // - commutative: a^b = b^a, a*b = b*a
  // - distributive: a*(b^c) = (a*b)^(a*c), (a^b)*c = (a*c)^(b*c)
  // - neutral element: 0^a = a^0 = a, 1*a = a*1 = a
  // - zero element: 0*a = a*0 = 0
  // - ^ is always invertible: a^a = 0
  // - * is invertible for odd values: x*inv(x) = inv(x)*x = 1
  // - shift function: shl(a,x)*shl(b,y) = shl(a*b,x+y)
  // cl_mul and xor form a commutative ring.

  // Carry-less multiplication, lower bits of result.
  // Invertible by cl_mul_inv for odd numbers.
  static auto cl_mul(T x, T y) -> T {

    T res;

    res = 0;
    while (x != 0) {
      if (M_odd(x))
        res = res ^ y;
      y = shl_fast(y, 1);
      x = shr_fast(x, 1);
      }
    return res;
    }

  // Calculate carry-less multiplicative inverse,
  // i.e. cl_mul(cl_mul_inv(x),x) == 1.
  // The result is defined for all odd values of x.
  // For even x we simply return 0.
  static auto cl_mul_inv(T x) -> T {

    t_int i;
    T inv, rem;
    T mask;

    if (!M_odd(x))
      return 0;

    inv = 1;
    rem = x;
    mask = 1;
    for (i = 1; i <= bits-1; ++i) {
      mask = shl_fast(mask, 1);
      x = shl_fast(x, 1);
      if ((rem & mask) != 0) {
        rem = rem ^ x;
        inv = inv | mask;
        }
      }
    return inv;
    }

  // x to the y'th via cl_mul.
  // For negative y see cl_mul_inv.
  static auto cl_power(T x, t_int y) -> T {

    T res;

    if (y < 0) {
      x = cl_mul_inv(x);
      y = -y;
      }
    res = 1;
    while (y != 0) {
      while (!M_odd(y)) {
        y = shr_fast(y, 1);
        x = cl_mul(x, x);
        }
      res = cl_mul(res, x);
      --y;
      }
    return res;
    }

  // Cyclic carry-less multiplication.

  // 2014-01-29 by Jasper L. Neumann

  // Here is funny idea I've never seen elsewhere:
  // "Cyclic carry-less multiplication" (ccl_mul).
  // The properties are quite similar to the usual carry-less multiplication.
  //
  // Properties (^ is the XOR operator; * is the ccl_mul operator):
  // - associative: a^(b^c) = (a^b)^c, a*(b*c) = (a*b)*c
  // - commutative: a^b = b^a, a*b = b*a
  // - distributive: a*(b^c) = (a*b)^(a*c), (a^b)*c = (a*c)^(b*c)
  // - neutral element: 0^a = a^0 = a, 1*a = a*1 = a
  // - zero element: 0*a = a*0 = 0
  // - ^ is always invertible: a^a = 0
  // - * is invertible for odd popcnt: x*inv(x) = inv(x)*x = 1
  // - rotate function: rot(a,x)*rot(b,y) = rot(a*b,x+y)
  // ccl_mul and xor form a commutative ring.
  //
  // I've implemented the inverse function by a bit matrix inversion
  // by Gauß/Jordan elimination.
  // Not nice, not fast, but feasible.

  // Cyclic carry-less multiplication.
  // Invertible by ccl_mul_inv for numbers with odd parity.
  // 2014-01-29 by Jasper L. Neumann

  // In x86 asm (eax,edx=>eax):
  //   xor ecx,ecx
  //   jmp @@start
  // @@add:
  //   xor cl,dl
  // @@loop:
  //   rol dl,1
  // @@start:
  //   shr al,1
  //   jc @@add
  //   jnz @@loop
  //   mov eax,ecx

  static auto ccl_mul(T x, T y) -> T {

    T res;

    res = 0;
    while (x != 0) {
      if (M_odd(x))
        res = res ^ y;
      y = rol(y, 1);
      x = shr_fast(x, 1);
      }
    return res;
    }

  // Calculate cyclic carry-less multiplicative inverse,
  // i.e. ccl_mul(x, ccl_mul_inv(x))==1.
  // The result is defined for all values of x with odd popcount.
  // For x with even popcount we simply return 0.
  // 2014-01-29 by Jasper L. Neumann
  static auto ccl_mul_inv(T x) -> T {

    t_bit_matrix org, inv;
    t_int i;

    for (i = 0; i <= bits-1; ++i)
      org[i] = rol(x, i);
    cl_mat_inv(org,inv);
    return inv[0];
    }

  // x to the y'th via ccl_mul.
  // For negative y see ccl_mul_inv.
  static auto ccl_power(T x, t_int y) -> T {

    T res;

    if (y < 0) {
      x = ccl_mul_inv(x);
      y = -y;
      }
    res = 1;
    while (y != 0) {
      while (!M_odd(y)) {
        y = shr_fast(y, 1);
        x = ccl_mul(x, x);
        }
      res = ccl_mul(res, x);
      --y;
      }
    return res;
    }

  // Operations on low bits with specified bit count.

  // Calculate the mask for b bits to zero out the remaining bits.
  static auto mask_ex(t_uint b) -> T {

    if (b >= bits)
      return all_bits;
    else
      return ((T)(1) << b) - 1;
    }

  // Rotate right low b bits of x by r;
  // remaining bits are zeroed out.
  // = rol_ex(x,-r,b);
  static auto ror_ex(T x, t_int r, t_uint b) -> T {

    T mask;

    if (b == 0) {
      return 0;
      }
    else {
      r = r % (t_int)(b);
      if (r == 0) {
        // Prevent shifting by b-r >= b.
        return x;
        }
      if (r < 0)
        r = r + b;
      if (b == bits)
        return ((T)(x) >> r) | (x << (bits-r));
      else {
        mask = ((T)(1) << b)-1;
        x = x & mask;
        x = ((T)(x) >> r) | (x << (b-r));
        return x & mask;
        }
      }
    }

  // Rotate left low b bits of x by r;
  // remaining bits are zeroed out.
  // = ror_ex(x,-r,b);
  static auto rol_ex(T x, t_int r, t_uint b) -> T {

    T mask;

    if (b == 0) {
      return 0;
      }
    else {
      r = r % (t_int)(b);
      if (r == 0) {
        // Prevent shifting by b-r >= b.
        return x;
        }
      if (r < 0)
        r = r + b;
      if (b == bits)
        return (x << r) | ((T)(x) >> (bits-r));
      else {
        mask = ((T)(1) << b)-1;
        x = x & mask;
        x = (x << r) | ((T)(x) >> (b-r));
        return x & mask;
        }
      }
    }

  // Shift right low b bits of x by r;
  // remaining bits are zeroed out.
  static auto shr_ex(T x, t_uint r, t_uint b) -> T {

    if (r >= b)
      return 0;
    else {
      x = x & mask_ex(b);
      return (T)(x) >> r;
      }
    }

  // Shift left low b bits of x by r;
  // remaining bits are zeroed out.
  static auto shl_ex(T x, t_uint r, t_uint b) -> T {

    if (r >= b)
      return 0;
    else {
      x = x << r;
      x = x & mask_ex(b);
      return x;
      }
    }

  // Arithmetically (duplicating the MSB) shift right low b bits of x by r;
  // remaining bits are sign extended.
  static auto sar_ex(T x, t_uint r, t_uint b) -> T {

    T mask;

    if (b == 0)
      return 0;
    else if (r >= b) {
      if (M_odd(x >> (b-1)))
        return all_bits;
      else
        return 0;
      }
    else {
      mask = mask_ex(b);
      if (M_odd(x >> (b-1)))
        return ((T)(x) >> r) | ~((T)(mask) >> r);  // unsigned >>!
      else
        return ((T)(x) & mask) >> r;
      }
    }

  // Shift left low b bits of x by r duplicating the LSB;
  // remaining bits are zeroed out.
  static auto sal_ex(T x, t_uint r, t_uint b) -> T {

    T mask;

    if (b == 0)
      return 0;
    else {
      mask = mask_ex(b);
      if (r >= b) {
        if (M_odd(x))
          return mask;
        else
          return 0;
        }
      else if (M_odd(x))
        return (x << r) | ~(mask << r);
      else
        return (x & mask) << r;
      }
    }

  // Shift right low b bits of x by r shifting in 1 bits;
  // remaining bits are zeroed out.
  static auto shr1_ex(T x, t_uint r, t_uint b) -> T {

    if (r >= b)
      return mask_ex(b);
    else {
      if (r != 0) {
        x = (T)(x) >> r;
        x = x | ~(((T)(1) << (b-r))-1);
        }
      x = x & mask_ex(b);
      return x;
      }
    }

  // Shift left low b bits of x by r shifting in 1 bits;
  // remaining bits are zeroed out.
  static auto shl1_ex(T x, t_uint r, t_uint b) -> T {

    if (r >= b)
      return mask_ex(b);
    else {
      x = x << r;
      x = x | (((T)(1) << r)-1);
      x = x & mask_ex(b);
      return x;
      }
    }

  // Cyclic carry-less multiplication.
  // Invertible by ccl_mul_inv_ex for numbers with odd parity.
  // 2014-10-13 by Jasper L. Neumann
  static auto ccl_mul_ex(T x, T y, t_uint b) -> T {

    T res,mask;

    mask = mask_ex(b);
    x = x & mask;
    y = y & mask;
    res = 0;
    while (x != 0) {
      if (M_odd(x))
        res = res ^ y;
      y = rol_ex(y, 1, b);
      x = x >> 1;
      }
    return res;
    }

  // Calculate cyclic carry-less multiplicative inverse,
  // i.e. ccl_mul(x, ccl_mul_inv(x))==1.
  // The result is defined for all values of x with odd popcount.
  // For x with even popcount we simply return 0.
  // 2014-10-13 by Jasper L. Neumann
  static auto ccl_mul_inv_ex(T x, t_uint b) -> T {

    t_bit_matrix org, inv;
    t_int i;

    for (i = 0; i <= b-1; ++i)
      org[i] = rol_ex(x, i, b);
    for (i = b; i <= bits-1; ++i)
      org[i] = (T)(1) << i;
    cl_mat_inv(org,inv);
    return inv[0];
    }

  // x to the y'th via ccl_mul.
  // For negative y see ccl_mul_inv_ex.
  static auto ccl_power_ex(T x, t_int y, t_uint b) -> T {

    T res;

    if (y == 0)
      return x & mask_ex(b);
    if (y < 0) {
      x = ccl_mul_inv_ex(x,b);
      y = -y;
      }
    res = 1;
    while (y != 0) {
      while (!M_odd(y)) {
        y = (t_uint)(y) >> 1;
        x = ccl_mul_ex(x, x, b);
        }
      res = ccl_mul_ex(res, x, b);
      --y;
      }
    return res;
    }

  };

// Base implementation for SWAR operations using t_bits<T>.
// T: Base type, must be an unsigned integral type.
// T is splitted in subwords as described by the sw parameter.
template<typename T> class t_simd_base {
protected:
  typedef t_bits<T> t_my_bits;

  // Helper function to convert sign-only to mask,
  // i.e. mantissa must be 0,
  static auto sign2mask(t_subword sw, T x) -> T {

    t_int shift;

    shift = (1 << (t_int)(sw)) - 1;

    // x = x >> shift;  // shift high bits to low bits
    // x = x * a_element[(t_int)(sw)];  // transform 1 ==> -1
    x = (x << 1) - (x >> shift);  // transform sign: 1 ==> -1
    return x;
    }

  // Helper function to convert sign-only to LSB,
  // i.e. mantissa must be 0,
  static auto sign2lsb(t_subword sw, T x) -> T {

    t_int shift;

    shift = (1 << (t_int)(sw)) - 1;
    return x >> shift;
    }

public:

  // Imports, types, asserts.
  typedef typename t_my_bits::TU TU;
  typedef typename t_my_bits::TS TS;

  static constexpr const t_uint ld_bits = t_my_bits::ld_bits;
  static constexpr const t_uint bits = t_my_bits::bits;
  static constexpr const T all_bits = t_my_bits::all_bits;
  static constexpr const T lo_bit = t_my_bits::lo_bit;
  static constexpr const T hi_bit = t_my_bits::hi_bit;

  static constexpr const TU *a_element = t_my_bits::a_element;
  static constexpr const TU *a_lo = t_my_bits::a_lo;
  static constexpr const TU *a_hi = t_my_bits::a_hi;
  static constexpr const TU *a_even = t_my_bits::a_even;
  static constexpr const TU *a_shuffle = t_my_bits::a_shuffle;
  static constexpr const TU *a_prim = t_my_bits::a_prim;

  static_assert((T)(-1) > 0, "(T)(-1) > 0");  // T must be unsigned.

  // Basic functions.

  // Fill every subword with c, i.e. broadcast.
  // Also implementable by inductive doubling.
  // O(1)
  template<typename TE>
  static auto constant(t_subword sw, TE c) -> T {

    return ((T)(c) & a_element[(t_int)(sw)]) * a_lo[(t_int)(sw)];
    }

  // Create a mask for the subword #i.
  // O(1)
  static auto element_mask(t_subword sw, t_uint i) -> T {

    i = i << (t_int)(sw);  // index => bit
    return a_element[(t_int)(sw)] << i;
    }

  // Extract the subword #i of x.
  // O(1)
  template<typename TE = T>
  static auto extract(t_subword sw, T x, t_uint i) -> TE {

    T res;

    i = i << (t_int)(sw);  // index => bit
    res = (x >> i) & a_element[(t_int)(sw)];
    if ((TE)(-1) > 0)
      return (TE)(res);  // Result type is unsigned.
    else if ((res & a_hi[(t_int)(sw)]) == 0)
      return (TE)(res);  // Result is unsigned.
    else
      return (TE)(TS)(res | ~a_element[(t_int)(sw)]);  // Sign extend.
    }

  // Implant e into the subword #i of x.
  // O(1)
  template<typename TE>
  static auto implant(t_subword sw, T x, TE e, t_uint i) -> T {

    T mask;

    i = i << (t_int)(sw);  // index => bit
    mask = a_element[(t_int)(sw)];
    return
      (x & ~(mask << i)) |
      (((T)(e) & mask) << i);
    }

  // General SIMD functions.

  // Count bits/elements.

  // Count the number of set bits.
  // 2014-07-12 by Jasper L. Neumann
  // Equivalent to loop i in 0..sw-1: hadd_u(sw, x)
  // O(ld_bits)
  static auto nr_1bits(t_subword sw, T x) -> T {

    t_int i;
    T m;
    t_int s;

    s = 1;
    for (i = 0; i<=(t_int)(sw)-1; ++i) {
      // s = 1 << i;
      m = a_even[i];
      x = (x & m) + ((x >> s) & m);
      s = s << 1;
      }
    return x;
    }

  // Count the reset bits
  // O(ld_bits)
  static auto nr_0bits(t_subword sw, T x) -> T {

    return nr_1bits(sw, ~x);
    }

  // Count the number of trailing zero bits.
  // 2014-07-12 by Jasper L. Neumann
  // ~x&(x-1): smear least significant 1 bit to the left.
  // This is simpler than the parallel prefix method.
  // O(ld_bits)
  static auto nr_trailing_0bits(t_subword sw, T x) -> T {

    return nr_1bits(sw, ~x & dec(sw, x));
    }

  // Number of contiguous least significant set bits
  // O(ld_bits)
  static auto nr_trailing_1bits(t_subword sw, T x) -> T {

    return nr_trailing_0bits(sw, ~x);
    }

  // Count the number of leading zero bits.
  // 2014-07-12 by Jasper L. Neumann
  // O(ld_bits)
  static auto nr_leading_0bits(t_subword sw, T x) -> T {

    t_int i, s;
    T m0, m1;

    if ((t_int)(sw) > 0) {
      // Smear most significant 1 bit to the right by parallel suffix.
      m0 = a_lo[(t_int)(sw)];
      m1 = a_hi[(t_int)(sw)];
      s = 1;
      for (i = 0; i<=(t_int)(sw)-1; ++i) {
        // s = 1 << i;
        // x = x | ((x >> s) & ((m1 >> (s-1)) - m0));
        x = x | ((x >> s) & (m1 - m0));
        m1 = m1 >> s;
        s = s << 1;
        }
      }
    return nr_1bits(sw, ~x);  // Count the remaining (inverted) 0 bits.
    }

  // Number of contiguous most significant set bits
  // O(ld_bits)
  static auto nr_leading_1bits(t_subword sw, T x) -> T {

    return nr_leading_0bits(sw, ~x);
    }

  // True iff there is at least one subword with the value 0.
  // See http://graphics.stanford.edu/~seander/bithacks.html#ZeroInWord
  // See Paul Curtis, 2004-12-15 in http://hackersdelight.org/corres.txt
  // See also http://9.douban.com/subject/9065343/
  // See discussion on
  //   http://stackoverflow.com/questions/20021066/
  //        how-the-glibc-strlen-implementation-works
  // Thereby we can also find the lowest zero subword.
  // Not suitable to find all zeros, i.e. this can not replace eq0
  // (subwords left to the lowest zero might be flagged as zero).
  // 2014-07-10 by Jasper L. Neumann
  // O(1)
  static auto contains_0(t_subword sw, T x) -> bool {

    return ((x - a_lo[(t_int)(sw)]) & ~x & a_hi[(t_int)(sw)]) != 0;
    }

  // See http://graphics.stanford.edu/~seander/bithacks.html#ZeroInWord
  // See Paul Curtis, 2004-12-15 in http://hackersdelight.org/corres.txt
  // See contains_0.
  // 2014-07-12 by Jasper L. Neumann
  // O(1) / O(ld_bits)
  static auto nr_trailing_ne0(t_subword sw, T x) -> t_uint {

    x = (x - a_lo[(t_int)(sw)]) & ~x & a_hi[(t_int)(sw)];
    return t_my_bits::nr_trailing_0bits(x) >> (t_int)(sw);
    }

  // Count the set bits and return whether this is an odd number.
  // O(ld_bits)
  static auto is_parity_oddh(t_subword sw, T x) -> T {

    return oddh(sw, nr_1bits(sw, x));
    }

  // Count the set bits and return whether this is an odd number.
  // O(ld_bits)
  static auto is_parity_oddl(t_subword sw, T x) -> T {

    return oddl(sw, nr_1bits(sw, x));
    }

  // Count the set bits and return whether this is an odd number.
  // O(ld_bits)
  static auto is_parity_odd(t_subword sw, T x) -> T {

    return odd(sw, nr_1bits(sw, x));
    }

  // Math.

  // Wrap around, ignore overflows.

  // Add with wrap-around x+y.
  // See HD 2.18 "Multibyte Add, Subtract, Absolute Value"
  // O(1)
  static auto add(t_subword sw, T x, T y) -> T {

    T signmask;
    T signs;

    signmask = a_hi[(t_int)(sw)];
    signs = (x ^ y) & signmask;
    x = x & ~signmask;
    y = y & ~signmask;
    x = x + y;
    x = x ^ signs;
    return x;
    }

  // Subtract with wrap-around x-y.
  // See HD 2.18 "Multibyte Add, Subtract, Absolute Value"
  // O(1)
  static auto sub(t_subword sw, T x, T y) -> T {

    T signmask;
    T signs;

    signmask = a_hi[(t_int)(sw)];
    signs = (x ^ ~y) & signmask;
    x = x | signmask;
    y = y & ~signmask;
    x = x - y;
    x = x ^ signs;
    return x;
    }

  // Add with wrap-around x+y+1.
  // See HD 2.18 "Multibyte Add, Subtract, Absolute Value"
  // O(1)
  static auto add1(t_subword sw, T x, T y) -> T {

    T signmask;
    T signs;

    signmask = a_hi[(t_int)(sw)];
    signs = (x ^ y) & signmask;
    x = x & ~signmask;
    y = y & ~signmask;
    x = x + y + a_lo[(t_int)(sw)];
    x = x ^ signs;
    return x;
    }

  // Subtract with wrap-around x-y-1.
  // See HD 2.18 "Multibyte Add, Subtract, Absolute Value"
  // O(1)
  static auto sub1(t_subword sw, T x, T y) -> T {

    T signmask;
    T signs;

    signmask = a_hi[(t_int)(sw)];
    signs = (x ^ ~y) & signmask;
    x = x | signmask;
    y = y & ~signmask;
    x = x - y - a_lo[(t_int)(sw)];
    x = x ^ signs;
    return x;
    }

  // Add with wrap-around x+y+carry.
  // Only least significant bits of carry are used, i.e. carry may be 0 or 1.
  // See HD 2.18 "Multibyte Add, Subtract, Absolute Value"
  // O(1)
  static auto addc(t_subword sw, T x, T y, T carry) -> T {

    T signmask;
    T signs;

    signmask = a_hi[(t_int)(sw)];
    signs = (x ^ y) & signmask;
    x = x & ~signmask;
    y = y & ~signmask;
    x = x + y + (carry & a_lo[(t_int)(sw)]);
    x = x ^ signs;
    return x;
    }

  // Subtract with wrap-around x-y-carry.
  // Only least significant bits of carry are used, i.e. carry may be 0 or 1.
  // See HD 2.18 "Multibyte Add, Subtract, Absolute Value"
  // O(1)
  static auto subc(t_subword sw, T x, T y, T carry) -> T {

    T signmask;
    T signs;

    signmask = a_hi[(t_int)(sw)];
    signs = (x ^ ~y) & signmask;
    x = x | signmask;
    y = y & ~signmask;
    x = x - y - (carry & a_lo[(t_int)(sw)]);
    x = x ^ signs;
    return x;
    }

  // Increment with wrap-around.
  // =add(sw, x, constant(sw, 1))
  // See HD 2.18 "Multibyte Add, Subtract, Absolute Value"
  // 2014-07-11 by Jasper L. Neumann
  // O(1)
  static auto inc(t_subword sw, T x) -> T {

    T signmask;
    T signs;

    signmask = a_hi[(t_int)(sw)];
    signs = x & signmask;
    x = x & ~signmask;
    x = x + a_lo[(t_int)(sw)];
    x = x ^ signs;
    return x;
    }

  // Decrement with wrap-around.
  // =sub(sw, x, constant(sw, 1))
  // See HD 2.18 "Multibyte Add, Subtract, Absolute Value"
  // 2014-07-11 by Jasper L. Neumann
  // O(1)
  static auto dec(t_subword sw, T x) -> T {

    T signmask;
    T signs;

    signmask = a_hi[(t_int)(sw)];
    signs = ~x & signmask;
    x = x | signmask;
    x = x - a_lo[(t_int)(sw)];
    x = x ^ signs;
    return x;
    }

  // Negate with wrap-around.
  // =sub(sw, 0, x)
  // O(1)
  static auto neg(t_subword sw, T x) -> T {

    T signmask;

    signmask = a_hi[(t_int)(sw)];
    return (signmask - (x & ~signmask)) ^ (~x & signmask);
    }

  // Saturate overflows.

  // Unsigned saturated add x+y.
  // See Falk Hueffner 2004-01-23 "Multibyte (or SIMD) arithmetic"
  // in http://hackersdelight.org/corres.txt
  // O(1)
  static auto add_sat_u(t_subword sw, T x, T y) -> T {

    T signmask;
    T t0, t1;
    t_int shift;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];

    t0 = (y ^ x) & signmask;
    t1 = (y & x) & signmask;

    x = x & ~signmask;
    y = y & ~signmask;
    x = x + y;

    t1 = t1 | (t0 & x);
    t1 = (t1 << 1) - (t1 >> shift);
    return (x ^ t0) | t1;
    }

  // Unsigned saturated subtract x-y.
  // See Falk Hueffner 2004-01-23 "Multibyte (or SIMD) arithmetic"
  // in http://hackersdelight.org/corres.txt
  // O(1)
  static auto sub_sat_u(t_subword sw, T x, T y) -> T {

    T signmask;
    T t0, t1;
    t_int shift;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];

    t0 = (y ^ ~x) & signmask;
    t1 = (y & ~x) & signmask;

    x = x | signmask;
    y = y & ~signmask;
    x = x - y;

    t1 = t1 | (t0 & ~x);
    t1 = (t1 << 1) - (t1 >> shift);
    return (x ^ t0) & ~t1;
    }

  // Unsigned saturated add x+y+1.
  // See Falk Hueffner 2004-01-23 "Multibyte (or SIMD) arithmetic"
  // in http://hackersdelight.org/corres.txt
  // O(1)
  static auto add1_sat_u(t_subword sw, T x, T y) -> T {

    T signmask;
    T t0, t1;
    t_int shift;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];

    t0 = (y ^ x) & signmask;
    t1 = (y & x) & signmask;

    x = x & ~signmask;
    y = y & ~signmask;
    x = x + y + a_lo[(t_int)(sw)];

    t1 = t1 | (t0 & x);
    t1 = (t1 << 1) - (t1 >> shift);
    return (x ^ t0) | t1;
    }

  // Unsigned saturated subtract x-y-1.
  // See Falk Hueffner 2004-01-23 "Multibyte (or SIMD) arithmetic"
  // in http://hackersdelight.org/corres.txt
  // O(1)
  static auto sub1_sat_u(t_subword sw, T x, T y) -> T {

    T signmask;
    T t0, t1;
    t_int shift;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];

    t0 = (y ^ ~x) & signmask;
    t1 = (y & ~x) & signmask;

    x = x | signmask;
    y = y & ~signmask;
    x = x - y - a_lo[(t_int)(sw)];

    t1 = t1 | (t0 & ~x);
    t1 = (t1 << 1) - (t1 >> shift);
    return (x ^ t0) & ~t1;
    }

  // Unsigned saturated add x+y+carry.
  // Only least significant bits of carry are used, i.e. carry may be 0 or 1.
  // See Falk Hueffner 2004-01-23 "Multibyte (or SIMD) arithmetic"
  // in http://hackersdelight.org/corres.txt
  // O(1)
  static auto addc_sat_u(t_subword sw, T x, T y, T carry) -> T {

    T signmask;
    T t0, t1;
    t_int shift;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];

    t0 = (y ^ x) & signmask;
    t1 = (y & x) & signmask;

    x = x & ~signmask;
    y = y & ~signmask;
    x = x + y + (carry & a_lo[(t_int)(sw)]);

    t1 = t1 | (t0 & x);
    t1 = (t1 << 1) - (t1 >> shift);
    return (x ^ t0) | t1;
    }

  // Unsigned saturated subtract x-y-carry.
  // Only least significant bits of carry are used, i.e. carry may be 0 or 1.
  // See Falk Hueffner 2004-01-23 "Multibyte (or SIMD) arithmetic"
  // in http://hackersdelight.org/corres.txt
  // O(1)
  static auto subc_sat_u(t_subword sw, T x, T y, T carry) -> T {

    T signmask;
    T t0, t1;
    t_int shift;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];

    t0 = (y ^ ~x) & signmask;
    t1 = (y & ~x) & signmask;

    x = x | signmask;
    y = y & ~signmask;
    x = x - y - (carry & a_lo[(t_int)(sw)]);

    t1 = t1 | (t0 & ~x);
    t1 = (t1 << 1) - (t1 >> shift);
    return (x ^ t0) & ~t1;
    }

  // inc_sat_u
  // dec_sat_u

  // Signed saturated add x+y.
  // See Falk Hueffner 2004-01-23 "Multibyte (or SIMD) arithmetic"
  // in http://hackersdelight.org/corres.txt
  // O(1)
  static auto add_sat_s(t_subword sw, T x, T y) -> T {

    T signmask;
    T eq, xv, yv, satmask, satbits, satadd, t0, t1;
    t_int shift;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];

    eq = (x ^ ~y) & signmask;
    xv = x & ~signmask;
    yv = y & ~signmask;
    xv = xv + yv;
    satbits = (xv ^ y) & eq;
    satadd = satbits >> shift;
    satmask = (satbits << 1) - satadd;
    xv = xv ^ eq;
    t0 = (xv & ~satmask) ^ signmask;
    t1 = satadd & ~(xv >> shift);
    return t0 - t1;
    }

  // Signed saturated sub x-y.
  // Derived from add_sat_s.
  // 2014-07-10 by Jasper L. Neumann
  // O(1)
  static auto sub_sat_s(t_subword sw, T x, T y) -> T {

    T signmask;
    T eq, xv, yv, satmask, satbits, satadd, t0, t1;
    t_int shift;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];

    eq = (x ^ y) & signmask;
    xv = x | signmask;
    yv = y & ~signmask;
    xv = xv - yv;
    satbits = (xv ^ ~y) & eq;
    satadd = satbits >> shift;
    satmask = (satbits << 1) - satadd;
    xv = xv ^ eq;
    t0 = (xv & ~satmask) ^ signmask;
    t1 = satadd & ~(xv >> shift);
    return t0 - t1;
    }

  // Signed saturated add x+y+1.
  // See Falk Hueffner 2004-01-23 "Multibyte (or SIMD) arithmetic"
  // in http://hackersdelight.org/corres.txt
  // O(1)
  static auto add1_sat_s(t_subword sw, T x, T y) -> T {

    T signmask;
    T eq, xv, yv, satmask, satbits, satadd, t0, t1;
    t_int shift;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];

    eq = (x ^ ~y) & signmask;
    xv = x & ~signmask;
    yv = y & ~signmask;
    xv = xv + yv + a_lo[(t_int)(sw)];
    satbits = (xv ^ y) & eq;
    satadd = satbits >> shift;
    satmask = (satbits << 1) - satadd;
    xv = xv ^ eq;
    t0 = (xv & ~satmask) ^ signmask;
    t1 = satadd & ~(xv >> shift);
    return t0 - t1;
    }

  // Signed saturated sub x-y-1.
  // Only least significant bits of carry are used, i.e. carry may be 0 or 1.
  // Derived from add_sat_s.
  // 2014-07-10 by Jasper L. Neumann
  // O(1)
  static auto sub1_sat_s(t_subword sw, T x, T y) -> T {

    T signmask;
    T eq, xv, yv, satmask, satbits, satadd, t0, t1;
    t_int shift;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];

    eq = (x ^ y) & signmask;
    xv = x | signmask;
    yv = y & ~signmask;
    xv = xv - yv - a_lo[(t_int)(sw)];
    satbits = (xv ^ ~y) & eq;
    satadd = satbits >> shift;
    satmask = (satbits << 1) - satadd;
    xv = xv ^ eq;
    t0 = (xv & ~satmask) ^ signmask;
    t1 = satadd & ~(xv >> shift);
    return t0 - t1;
    }

  // Signed saturated add x+y+carry.
  // Only least significant bits of carry are used, i.e. carry may be 0 or 1.
  // See Falk Hueffner 2004-01-23 "Multibyte (or SIMD) arithmetic"
  // in http://hackersdelight.org/corres.txt
  // O(1)
  static auto addc_sat_s(t_subword sw, T x, T y, T carry) -> T {

    T signmask;
    T eq, xv, yv, satmask, satbits, satadd, t0, t1;
    t_int shift;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];

    eq = (x ^ ~y) & signmask;
    xv = x & ~signmask;
    yv = y & ~signmask;
    xv = xv + yv + (carry & a_lo[(t_int)(sw)]);
    satbits = (xv ^ y) & eq;
    satadd = satbits >> shift;
    satmask = (satbits << 1) - satadd;
    xv = xv ^ eq;
    t0 = (xv & ~satmask) ^ signmask;
    t1 = satadd & ~(xv >> shift);
    return t0 - t1;
    }

  // Signed saturated sub x-y-carry.
  // Only least significant bits of carry are used, i.e. carry may be 0 or 1.
  // Derived from add_sat_s.
  // 2014-07-10 by Jasper L. Neumann
  // O(1)
  static auto subc_sat_s(t_subword sw, T x, T y, T carry) -> T {

    T signmask;
    T eq, xv, yv, satmask, satbits, satadd, t0, t1;
    t_int shift;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];

    eq = (x ^ y) & signmask;
    xv = x | signmask;
    yv = y & ~signmask;
    xv = xv - yv - (carry & a_lo[(t_int)(sw)]);
    satbits = (xv ^ ~y) & eq;
    satadd = satbits >> shift;
    satmask = (satbits << 1) - satadd;
    xv = xv ^ eq;
    t0 = (xv & ~satmask) ^ signmask;
    t1 = satadd & ~(xv >> shift);
    return t0 - t1;
    }

  // inc_sat_s
  // dec_sat_s

  // Signed saturated negate;
  // the only value which will be saturated is the
  // smallest possible signed number; see abs for a similar problem.
  // =sub_sat_s(sw,0,x).
  // 2014-07-10 by Jasper L. Neumann
  // O(1)
  static auto neg_sat_s(t_subword sw, T x) -> T {

    T signmask;
    T eq, xv, yv, satmask, satbits, satadd, t0, t1;
    t_int shift;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];

    eq = x & signmask;
    yv = x & ~signmask;
    xv = signmask - yv;
    satbits = (xv ^ ~x) & eq;
    satadd = satbits >> shift;
    satmask = (satbits << 1) - satadd;
    xv = xv ^ eq;
    t0 = (xv & ~satmask) ^ signmask;
    t1 = satadd & ~(xv >> shift);
    return t0 - t1;
    }

  // Detect overflows.

  // Detect overflow in unsigned addition x+y.
  // See HD 2.18 "Multibyte Add, Subtract, Absolute Value"
  // See HD 2.13 "Overflow Detection" (Unsigned Add/Subtract)
  // 2016-06-03 by Jasper L. Neumann
  // O(1)
  static auto overflow_add_u(t_subword sw, T x, T y) -> T {
  // (x&y) | ((x|y) & ~(x+y+c))

    T signmask;
    T signs;
    t_int shift;
    T x1,y1,a,add,ov,t;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];
    signs = (x ^ y) & signmask;
    x1 = x & ~signmask;
    y1 = y & ~signmask;
    a = x1 + y1;
    add = a ^ signs;

    ov = (x&y) | ((x|y) & ~add);

    t = ov & signmask;  // result is now in sign bits

    return (t << 1) - (t >> shift);  // transform sign: 1 ==> -1
    }

  // Detect overflow in unsigned subtraction x-y.
  // See HD 2.18 "Multibyte Add, Subtract, Absolute Value"
  // See HD 2.13 "Overflow Detection" (Unsigned Add/Subtract)
  // 2016-06-03 by Jasper L. Neumann
  // O(1)
  static auto overflow_sub_u(t_subword sw, T x, T y) -> T {
  // (~x&y) | ((~x|y) & (x-y-c))

    T signmask;
    T signs;
    t_int shift;
    T x1,y1,s,sub,ov,t;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];
    signs = (x ^ ~y) & signmask;
    x1 = x | signmask;
    y1 = y & ~signmask;
    s = x1 - y1;
    sub = s ^ signs;

    ov = (~x&y) | ((~x|y) & sub);

    t = ov & signmask;  // result is now in sign bits

    return (t << 1) - (t >> shift);  // transform sign: 1 ==> -1
    }

  // Detect overflow in unsigned addition x+y+1.
  // See HD 2.18 "Multibyte Add, Subtract, Absolute Value"
  // See HD 2.13 "Overflow Detection" (Unsigned Add/Subtract)
  // 2016-06-03 by Jasper L. Neumann
  // O(1)
  static auto overflow_add1_u(t_subword sw, T x, T y) -> T {
  // (x&y) | ((x|y) & ~(x+y+c))

    T signmask;
    T signs;
    t_int shift;
    T x1,y1,a,add,ov,t;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];
    signs = (x ^ y) & signmask;
    x1 = x & ~signmask;
    y1 = y & ~signmask;
    a = x1 + y1 + a_lo[(t_int)(sw)];
    add = a ^ signs;

    ov = (x&y) | ((x|y) & ~add);

    t = ov & signmask;  // result is now in sign bits

    return (t << 1) - (t >> shift);  // transform sign: 1 ==> -1
    }

  // Detect overflow in unsigned subtraction x-y-1.
  // See HD 2.18 "Multibyte Add, Subtract, Absolute Value"
  // See HD 2.13 "Overflow Detection" (Unsigned Add/Subtract)
  // 2016-06-03 by Jasper L. Neumann
  // O(1)
  static auto overflow_sub1_u(t_subword sw, T x, T y) -> T {
  // (~x&y) | ((~x|y) & (x-y-c))

    T signmask;
    T signs;
    t_int shift;
    T x1,y1,s,sub,ov,t;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];
    signs = (x ^ ~y) & signmask;
    x1 = x | signmask;
    y1 = y & ~signmask;
    s = x1 - y1 - a_lo[(t_int)(sw)];
    sub = s ^ signs;

    ov = (~x&y) | ((~x|y) & sub);

    t = ov & signmask;  // result is now in sign bits

    return (t << 1) - (t >> shift);  // transform sign: 1 ==> -1
    }

  // Detect overflow in unsigned addition x+y+carry.
  // Only least significant bits of carry are used, i.e. carry may be 0 or 1.
  // See HD 2.18 "Multibyte Add, Subtract, Absolute Value"
  // See HD 2.13 "Overflow Detection" (Unsigned Add/Subtract)
  // 2016-06-03 by Jasper L. Neumann
  // O(1)
  static auto overflow_addc_u(t_subword sw, T x, T y, T carry) -> T {
  // (x&y) | ((x|y) & ~(x+y+c))

    T signmask;
    T signs;
    t_int shift;
    T x1,y1,a,add,ov,t;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];
    signs = (x ^ y) & signmask;
    x1 = x & ~signmask;
    y1 = y & ~signmask;
    a = x1 + y1 + (carry & a_lo[(t_int)(sw)]);
    add = a ^ signs;

    ov = (x&y) | ((x|y) & ~add);

    t = ov & signmask;  // result is now in sign bits

    return (t << 1) - (t >> shift);  // transform sign: 1 ==> -1
    }

  // Detect overflow in unsigned subtraction x-y-carry.
  // Only least significant bits of carry are used, i.e. carry may be 0 or 1.
  // See HD 2.18 "Multibyte Add, Subtract, Absolute Value"
  // See HD 2.13 "Overflow Detection" (Unsigned Add/Subtract)
  // 2016-06-03 by Jasper L. Neumann
  // O(1)
  static auto overflow_subc_u(t_subword sw, T x, T y, T carry) -> T {
  // (~x&y) | ((~x|y) & (x-y-c))

    T signmask;
    T signs;
    t_int shift;
    T x1,y1,s,sub,ov,t;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];
    signs = (x ^ ~y) & signmask;
    x1 = x | signmask;
    y1 = y & ~signmask;
    s = x1 - y1 - (carry & a_lo[(t_int)(sw)]);
    sub = s ^ signs;

    ov = (~x&y) | ((~x|y) & sub);

    t = ov & signmask;  // result is now in sign bits

    return (t << 1) - (t >> shift);  // transform sign: 1 ==> -1
    }

  // Detect overflow in signed addition x+y.
  // See HD 2.18 "Multibyte Add, Subtract, Absolute Value"
  // See HD 2.13 "Overflow Detection" (Signed Add/Subtract)
  // 2016-06-03 by Jasper L. Neumann
  // O(1)
  static auto overflow_add_s(t_subword sw, T x, T y) -> T {
  // ~(x^y) & ((x+y+c)^x)
  // Alternative: ((x+y+c)^x) & ((x+y+c)^y)

    T signmask;
    T signs;
    t_int shift;
    T x1,y1,a,add,ov,t;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];
    signs = (x ^ y) & signmask;
    x1 = x & ~signmask;
    y1 = y & ~signmask;
    a = x1 + y1;
    add = a ^ signs;

    ov = (add^x)&~signs;
    // ov = (add^x)&(add^y);

    t = ov & signmask;  // result is now in sign bits

    return (t << 1) - (t >> shift);  // transform sign: 1 ==> -1
    }

  // Detect overflow in signed subtraction x-y.
  // See HD 2.18 "Multibyte Add, Subtract, Absolute Value"
  // See HD 2.13 "Overflow Detection" (Signed Add/Subtract)
  // 2016-06-03 by Jasper L. Neumann
  // O(1)
  static auto overflow_sub_s(t_subword sw, T x, T y) -> T {
  // (x^y) & ((x-y-c)^x)
  // Alternative: ((x-y-c)^x) & ~((x-y-c)^y)

    T signmask;
    T signs;
    t_int shift;
    T x1,y1,s,sub,ov,t;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];
    signs = (x ^ ~y) & signmask;
    x1 = x | signmask;
    y1 = y & ~signmask;
    s = x1 - y1;
    sub = s ^ signs;

    ov = (x^y) & (sub^x);

    t = ov & signmask;  // result is now in sign bits

    return (t << 1) - (t >> shift);  // transform sign: 1 ==> -1
    }

  // Detect overflow in signed addition x+y+1.
  // See HD 2.18 "Multibyte Add, Subtract, Absolute Value"
  // See HD 2.13 "Overflow Detection" (Signed Add/Subtract)
  // 2016-06-03 by Jasper L. Neumann
  // O(1)
  static auto overflow_add1_s(t_subword sw, T x, T y) -> T {
  // ~(x^y) & ((x+y+c)^x)
  // Alternative: ((x+y+c)^x) & ((x+y+c)^y)

    T signmask;
    T signs;
    t_int shift;
    T x1,y1,a,add,ov,t;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];
    signs = (x ^ y) & signmask;
    x1 = x & ~signmask;
    y1 = y & ~signmask;
    a = x1 + y1 + a_lo[(t_int)(sw)];
    add = a ^ signs;

    ov = (add^x)&~signs;
    // ov = (add^x)&(add^y);

    t = ov & signmask;  // result is now in sign bits

    return (t << 1) - (t >> shift);  // transform sign: 1 ==> -1
    }

  // Detect overflow in signed subtraction x-y-1.
  // See HD 2.18 "Multibyte Add, Subtract, Absolute Value"
  // See HD 2.13 "Overflow Detection" (Signed Add/Subtract)
  // 2016-06-03 by Jasper L. Neumann
  // O(1)
  static auto overflow_sub1_s(t_subword sw, T x, T y) -> T {
  // (x^y) & ((x-y-c)^x)
  // Alternative: ((x-y-c)^x) & ~((x-y-c)^y)

    T signmask;
    T signs;
    t_int shift;
    T x1,y1,s,sub,ov,t;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];
    signs = (x ^ ~y) & signmask;
    x1 = x | signmask;
    y1 = y & ~signmask;
    s = x1 - y1 - a_lo[(t_int)(sw)];
    sub = s ^ signs;

    ov = (x^y) & (sub^x);

    t = ov & signmask;  // result is now in sign bits

    return (t << 1) - (t >> shift);  // transform sign: 1 ==> -1
    }

  // Detect overflow in signed addition x+y+carry.
  // Only least significant bits of carry are used, i.e. carry may be 0 or 1.
  // See HD 2.18 "Multibyte Add, Subtract, Absolute Value"
  // See HD 2.13 "Overflow Detection" (Signed Add/Subtract)
  // 2016-06-03 by Jasper L. Neumann
  // O(1)
  static auto overflow_addc_s(t_subword sw, T x, T y, T carry) -> T {
  // ~(x^y) & ((x+y+c)^x)
  // Alternative: ((x+y+c)^x) & ((x+y+c)^y)

    T signmask;
    T signs;
    t_int shift;
    T x1,y1,a,add,ov,t;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];
    signs = (x ^ y) & signmask;
    x1 = x & ~signmask;
    y1 = y & ~signmask;
    a = x1 + y1 + (carry & a_lo[(t_int)(sw)]);
    add = a ^ signs;

    ov = (add^x)&~signs;
    // ov = (add^x)&(add^y);

    t = ov & signmask;  // result is now in sign bits

    return (t << 1) - (t >> shift);  // transform sign: 1 ==> -1
    }

  // Detect overflow in signed subtraction x-y-carry.
  // Only least significant bits of carry are used, i.e. carry may be 0 or 1.
  // See HD 2.18 "Multibyte Add, Subtract, Absolute Value"
  // See HD 2.13 "Overflow Detection" (Signed Add/Subtract)
  // 2016-06-03 by Jasper L. Neumann
  // O(1)
  static auto overflow_subc_s(t_subword sw, T x, T y, T carry) -> T {
  // (x^y) & ((x-y-c)^x)
  // Alternative: ((x-y-c)^x) & ~((x-y-c)^y)

    T signmask;
    T signs;
    t_int shift;
    T x1,y1,s,sub,ov,t;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];
    signs = (x ^ ~y) & signmask;
    x1 = x | signmask;
    y1 = y & ~signmask;
    s = x1 - y1 - (carry & a_lo[(t_int)(sw)]);
    sub = s ^ signs;

    ov = (x^y) & (sub^x);

    t = ov & signmask;  // result is now in sign bits

    return (t << 1) - (t >> shift);  // transform sign: 1 ==> -1
    }

  // Averages.

  // Unsigned average rounding down, i.e. (x+y)>>1.
  // See Falk Hueffner 2004-01-23 "Multibyte (or SIMD) arithmetic"
  // in http://hackersdelight.org/corres.txt
  // O(1)
  static auto avgd_u(t_subword sw, T x, T y) -> T {

    return (x & y) + (((x ^ y) & ~a_lo[(t_int)(sw)]) >> 1);
    }

  // Unsigned average rounding up, i.e. (x+y+1)>>1.
  // See Falk Hueffner 2004-01-23 "Multibyte (or SIMD) arithmetic"
  // in http://hackersdelight.org/corres.txt
  // O(1)
  static auto avgu_u(t_subword sw, T x, T y) -> T {

    return (x | y) - (((x ^ y) & ~a_lo[(t_int)(sw)]) >> 1);
    }

  // Superfluous: avgf_u = avgd_u (floor, toward -oo)
  // Superfluous: avgc_u = avgu_u (ceiling, toward oo)

  // Signed average rounding toward -oo.
  // 2014-09-26 by Jasper L. Neumann
  // O(1)
  static auto avgf_s(t_subword sw, T x, T y) -> T {

    T signmask;

    signmask = a_hi[(t_int)(sw)];
    return avgd_u(sw, x ^ signmask, y ^ signmask) ^ signmask;
    }

  // Signed average rounding toward oo.
  // 2014-09-26 by Jasper L. Neumann
  // O(1)
  static auto avgc_s(t_subword sw, T x, T y) -> T {

    T signmask;

    signmask = a_hi[(t_int)(sw)];
    return avgu_u(sw, x ^ signmask, y ^ signmask) ^ signmask;
    }

  // Missing:     avgd_s (round down, abs toward 0)
  // Missing:     avgu_s (round up, abs toward oo)

  // Missing:     avge_* (round half to even)
  // Missing:     avgo_* (round half to odd)

  // Miscellaneous.

  // abs(x-y) for unsigned values.
  // =sub_sat_u(sw, x, y)+sub_sat_u(sw, y, x);
  // Alternative: ((x-y) xor (x<y)) - (x<y).
  // 2014-07-13 by Jasper L. Neumann
  // O(1)
  static auto abs_diff_u(t_subword sw, T x, T y) -> T {

    T signmask;
    T t0, t1, t2;
    T x1, x2, y1, y2;
    t_int shift;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];

    // t0 = (y ^ ~x) & signmask;
    t0 = ~(x ^ y) & signmask;
    t1 = (y & ~x) & signmask;
    t2 = (x & ~y) & signmask;

    x1 = x | signmask;
    y1 = y & ~signmask;
    x1 = x1 - y1;

    t1 = t1 | (t0 & ~x1);
    t1 = (t1 << 1) - (t1 >> shift);

    y2 = y | signmask;
    x2 = x & ~signmask;
    y2 = y2 - x2;

    t2 = t2 | (t0 & ~y2);
    t2 = (t2 << 1) - (t2 >> shift);

    return
      ((x1 ^ t0) & ~t1) +
      ((y2 ^ t0) & ~t2);
    }

  // sgn(x-y) for unsigned values resulting in unsigned values.
  // 2014-07-13 by Jasper L. Neumann
  // O(1)
  static auto sgn_diff_u(t_subword sw, T x, T y) -> T {

    return ~le_u(sw, y, x) | (~eq0(sw, x ^ y) & a_lo[(t_int)(sw)]);
    }

  // abs(x-y) for signed values resulting in unsigned values.
  // 2014-07-13 by Jasper L. Neumann
  // O(1)
  static auto abs_diff_s(t_subword sw, T x, T y) -> T {

    T signmask;

    signmask = a_hi[(t_int)(sw)];
    return abs_diff_u(sw, x ^ signmask, y ^ signmask);
    }

  // sgn(x-y) for signed values.
  // 2014-07-13 by Jasper L. Neumann
  // O(1)
  static auto sgn_diff_s(t_subword sw, T x, T y) -> T {

    T signmask;

    signmask = a_hi[(t_int)(sw)];
    return sgn_diff_u(sw, x ^ signmask, y ^ signmask);
    }

  // Multiply c with subwords of x yielding the lower half (scalar multiplication).
  // Suitable for signed and unsigned values.
  // 2016-06-27 by Jasper L. Neumann
  // O(1)
  static auto smul_lo(t_subword sw, T c, T x) -> T {

    T e, o;

    c = c & a_element[(t_int)(sw)];  // be on the safe side
    e = a_even[(t_int)(sw)];
    o = ~e;
    return
      (((x & e) * c) & e) |
      (((x & o) * c) & o);
    }

  // Multiply c with subwords of c yielding the upper half (scalar multiplication).
  // Suitable for unsigned values.
  // sw must not be the maximum value, i.e. there must be at least 2 subwords.
  // 2016-06-27 by Jasper L. Neumann
  // O(1)
  static auto smul_hi_u(t_subword sw, T c, T x) -> T {

    T e, o, xe, xo;
    t_int shift;

    c = c & a_element[(t_int)(sw)];  // be on the safe side
    e = a_even[(t_int)(sw)];
    o = ~e;
    shift = 1 << (t_int)(sw);
    xe = (x & e) * c;
    xo = ((x & o) >> shift) * c;
    return
      ((xe & o) >> shift) |
      (xo & o);
    }

  // Multiply c with subwords of c yielding the upper half (scalar multiplication).
  // Suitable for signed values.
  // sw must not be the maximum value, i.e. there must be at least 2 subwords.
  // See HD 8.1 "Multiword multiplication"
  // 2016-06-28 by Jasper L. Neumann
  // O(1)
  static auto smul_hi_s(t_subword sw, T c, T x) -> T {

    T e, o, xe, xo, x1, y1, xneg, yneg, xyneg, xnegy;
    t_int shift;

    c = c & a_element[(t_int)(sw)];  // be on the safe side
    if (c == 0) {
      return 0;
      }
    e = a_even[(t_int)(sw)];
    o = ~e;
    shift = 1 << (t_int)(sw);
    y1 = constant(sw,c);
    yneg = lt0_s(sw, y1);
    xneg = lt0_s(sw, x);
    xnegy = xneg & y1;
    xyneg = (xneg | yneg) & ne0h(sw, x);

    // even subwords
    x1 = x & e;
    xe = x1 * c
      + ((
         + ((xyneg & e) << 1)
         - (xnegy & e)
         - (yneg & x1)
        ) << shift);

    x = x >> shift;
    xyneg = xyneg >> shift;
    xnegy = xnegy >> shift;

    // odd subwords
    x1 = x & e;
    xo = x1 * c
      + ((
         + ((xyneg & e) << 1)
         - (xnegy & e)
         - (yneg & x1)
        ) << shift);

    return
      ((xe & o) >> shift) |
      (xo & o);
    }

  // Multiply c with even subwords of x (scalar multiplication).
  // Suitable for unsigned values.
  // sw must not be the maximum value, i.e. there must be at least 2 subwords.
  // 2016-07-02 by Jasper L. Neumann
  // O(1)
  static auto smule_u(t_subword sw, T c, T x) -> T {

    T e;

    c = c & a_element[(t_int)(sw)];  // be on the safe side
    e = a_even[(t_int)(sw)];
    return (x & e) * c;
    }

  // Multiply c with even subwords of x (scalar multiplication).
  // Suitable for signed values.
  // sw must not be the maximum value, i.e. there must be at least 2 subwords.
  // 2016-07-02 by Jasper L. Neumann
  // O(1)
  static auto smule_s(t_subword sw, T c, T x) -> T {

    T e, xe, x1, y1, xneg, yneg, xyneg, xnegy;
    t_int shift;

    c = c & a_element[(t_int)(sw)];  // be on the safe side
    if (c == 0) {
      return 0;
      }
    e = a_even[(t_int)(sw)];
    shift = 1 << (t_int)(sw);
    y1 = constant(sw,c);
    yneg = lt0_s(sw, y1);
    xneg = lt0_s(sw, x);
    xnegy = xneg & y1;
    xyneg = (xneg | yneg) & ne0h(sw, x);

    // even subwords
    x1 = x & e;
    xe = x1 * c
      + ((
         + ((xyneg & e) << 1)
         - (xnegy & e)
         - (yneg & x1)
        ) << shift);

    return xe;
    }

  // Multiply corresponding subwords of x and y yielding the lower half.
  // Suitable for signed and unsigned values.
  // Stupid emulation due to absence of a matching instruction/operation.
  // Emulation of the multiplication usually
  // is far too slow (O(subword size)).
  // 2014-09-12 by Jasper L. Neumann
  // O(#subwords)
  static auto mul_lo(t_subword sw, T x, T y) -> T {

    T m, res;
    t_int b, j, s;

    b = 1 << (t_int)(sw);
    m = a_element[(t_int)(sw)];
    res = 0;
    s = 0;
    for (j = (t_int)(bits >> (t_int)(sw))-1; j >= 0; --j) {
      res = res | ((((x >> s) * (y >> s)) & m) << s);
      s = s + b;
      }
    return res;
    }

  // mul_hi_s
  // mul_lo_sat_u
  // mul_lo_sat_s

  // Absolute values.
  // The result contains unsigned values,
  // however smallest possible signed number remains as is;
  // the similar function nabs does not have this problem.
  // See HD 2.18 "Multibyte Add, Subtract, Absolute Value"
  // O(1)
  static auto abs(t_subword sw, T x) -> T {

    T signmask;
    T a, b, m;
    t_int shift;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];
    a = x & signmask;
    b = a >> shift;
    //  m = (a - b) | a;
    m = a + a - b;  // mask of negative numbers

    return (x ^ m) + b;  // complement & inc if negative
    }

  // Negated absolute values, i.e. -abs(x).
  // The result contains signed values.
  // See HD 2.18 "Multibyte Add, Subtract, Absolute Value"
  // 2014-07-12 by Jasper L. Neumann
  // O(1)
  static auto nabs(t_subword sw, T x) -> T {

    T signmask;
    T a, b, m;
    t_int shift;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];
    a = ~x & signmask;
    b = a >> shift;
    //  m = (a - b) | a;
    m = a + a - b;  // mask of positive numbers

    // return ((x ^ m ^ signmask) + b) ^ signmask;
    return ((x ^ m ^ a) + b) ^ a;
      // complement & inc if positive
      // avoiding carry by "^ signmask" | "^ a" (~0+1 creates a carry)
    }

  // Sign of x, i.e. <0 => -1, =0 => 0, >0 => 1.
  // See eq0.
  // See HD 6.1 "Find First 0-Byte"
  // 2014-07-11 by Jasper L. Neumann
  // O(1)
  static auto sgn(t_subword sw, T x) -> T {

    T m;
    T y;
    T sign;
    t_int shift;

    shift = (1 << (t_int)(sw)) - 1;
    m = a_hi[(t_int)(sw)];
    sign = x & m;
    sign = (sign << 1) - (sign >> shift);  // transform sign: 1 ==> -1 (<0)
    y = (x & ~m) + ~m;
    y = (y | x) & m;
    y = y >> shift;  // 1 if !=0
    return y | sign;
    }

  // Sign of x enlarged to saturate.
  // <0 => minimum value, =0 => 0, >0 => maximum value
  // 2014-09-26 by Jasper L. Neumann
  // O(1)
  static auto sgn_sat(t_subword sw, T x) -> T {

    return ne0(sw,x) & (lt0_s(sw,x) ^ ~a_hi[(t_int)(sw)]);
    }

  // Comparisons.

  // Set MSB if x==0.
  // See HD 6.1 "Find First 0-Byte"
  // 2016-06-30 by Jasper L. Neumann
  // O(1)
  static auto eq0h(t_subword sw, T x) -> T {

    T m;
    T t;

    m = ~a_hi[(t_int)(sw)];
    t = (x & m) + m;
    t = ~(t | x | m);
    return t;
    }

  // Set MSB if x!=0.
  // See HD 6.1 "Find First 0-Byte"
  // 2016-06-30 by Jasper L. Neumann
  // O(1)
  static auto ne0h(t_subword sw, T x) -> T {

    T m;
    T t;

    m = a_hi[(t_int)(sw)];
    t = (x & ~m) + ~m;
    t = (t | x) & m;
    return t;
    }

  // Set mask denoting x==y.
  // O(1)
  static auto eqh(t_subword sw, T x, T y) -> T {

    return eq0h(sw, x ^ y);
    }

  // Set MSB if x!=y.
  // O(1)
  static auto neh(t_subword sw, T x, T y) -> T {

    return ne0h(sw, x ^ y);
    }

  // Set MSB if x>y (unsigned).
  // HD 6.1 "Searching for a Value in a Given Range"
  // 2014-07-10 by Jasper L. Neumann
  // O(1)
  static auto gth_u(t_subword sw, T x, T y) -> T {

    T signmask;
    T t;
    T big;
    t_int shift;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];
    t = ~signmask;
    t = t + (x & t) - (y & t);

    big = y & signmask;
    big = (big << 1) - (big >> shift);
    t =
      ((t | x) & ~big) |
      ((t & x) & big);
    t = t & signmask;  // result is now in sign bits

    return t;
    }

  // Set MSB if x<=y (unsigned).
  // HD 6.1 "Searching for a Value in a Given Range"
  // 2014-07-10 by Jasper L. Neumann
  // O(1)
  static auto leh_u(t_subword sw, T x, T y) -> T {

    return gth_u(sw, x, y) ^ a_hi[(t_int)(sw)];
    }

  // Set MSB if x<y (unsigned).
  // O(1)
  static auto lth_u(t_subword sw, T x, T y) -> T {

    return gth_u(sw, y, x);
    }

  // Set MSB if x>=y (unsigned).
  // O(1)
  static auto geh_u(t_subword sw, T x, T y) -> T {

    return leh_u(sw, y, x);
    }

  // Set MSB if x<0.
  // Set mask denoting x<0 (signed), i.e. mask high (sign) bit.
  // 2014-07-11 by Jasper L. Neumann
  // O(1)
  static auto lt0h_s(t_subword sw, T x) -> T {

    T m;
    T sign;

    m = a_hi[(t_int)(sw)];
    sign = x & m;
    return sign;
    }

  // Set MSB if x<=y (signed).
  // 2014-07-10 by Jasper L. Neumann
  // O(1)
  static auto leh_s(t_subword sw, T x, T y) -> T {

    T m;

    m = a_hi[(t_int)(sw)];
    return leh_u(sw, x ^ m, y ^ m);
    }

  // Set MSB if x<y (signed).
  // O(1)
  static auto lth_s(t_subword sw, T x, T y) -> T {

    T m;

    m = a_hi[(t_int)(sw)];
    return lth_u(sw, x ^ m, y ^ m);
    }

  // Set MSB if x>=y (signed).
  // O(1)
  static auto geh_s(t_subword sw, T x, T y) -> T {

    T m;

    m = a_hi[(t_int)(sw)];
    return geh_u(sw, x ^ m, y ^ m);
    }

  // Set MSB if x>y (signed).
  // O(1)
  static auto gth_s(t_subword sw, T x, T y) -> T {

    T m;

    m = a_hi[(t_int)(sw)];
    return gth_u(sw, x ^ m, y ^ m);
    }

  // Set MSB for odd values, shift propagate low bit to MSB.
  // 2016-06-30 by Jasper L. Neumann
  // O(1)
  static auto oddh(t_subword sw, T x) -> T {

    T m;
    T lsb;
    t_int shift;

    shift = (1 << (t_int)(sw)) - 1;
    m = a_lo[(t_int)(sw)];
    lsb = x & m;
    return lsb << shift;
    }

  // Set LSB if x==0.
  // See HD 6.1 "Find First 0-Byte"
  // 2016-06-30 by Jasper L. Neumann
  // O(1)
  static auto eq0l(t_subword sw, T x) -> T {

    return sign2lsb(sw, eq0h(sw, x));
    }

  // Set LSB if x!=0.
  // See HD 6.1 "Find First 0-Byte"
  // 2016-06-30 by Jasper L. Neumann
  // O(1)
  static auto ne0l(t_subword sw, T x) -> T {

    return sign2lsb(sw, ne0h(sw, x));
    }

  // Set mask denoting x==y.
  // O(1)
  static auto eql(t_subword sw, T x, T y) -> T {

    return sign2lsb(sw, eq0h(sw, x ^ y));
    }

  // Set LSB if x!=y.
  // O(1)
  static auto nel(t_subword sw, T x, T y) -> T {

    return sign2lsb(sw, ne0h(sw, x ^ y));
    }

  // Set LSB if x>y (unsigned).
  // HD 6.1 "Searching for a Value in a Given Range"
  // 2014-07-10 by Jasper L. Neumann
  // O(1)
  static auto gtl_u(t_subword sw, T x, T y) -> T {

    return sign2lsb(sw, gth_u(sw, x, y));
    }

  // Set LSB if x<=y (unsigned).
  // HD 6.1 "Searching for a Value in a Given Range"
  // 2014-07-10 by Jasper L. Neumann
  // O(1)
  static auto lel_u(t_subword sw, T x, T y) -> T {

    return gtl_u(sw, x, y) ^ a_lo[(t_int)(sw)];
    }

  // Set LSB if x<y (unsigned).
  // O(1)
  static auto ltl_u(t_subword sw, T x, T y) -> T {

    return gtl_u(sw, y, x);
    }

  // Set LSB if x>=y (unsigned).
  // O(1)
  static auto gel_u(t_subword sw, T x, T y) -> T {

    return lel_u(sw, y, x);
    }

  // Set LSB if x<0, i.e. shift high (sign) bit to LSB.
  // 2014-07-11 by Jasper L. Neumann
  // O(1)
  static auto lt0l_s(t_subword sw, T x) -> T {

    T m;
    T sign;

    m = a_hi[(t_int)(sw)];
    sign = x & m;
    return sign2lsb(sw, sign);
    }

  // Set LSB if x<=y (signed).
  // 2014-07-10 by Jasper L. Neumann
  // O(1)
  static auto lel_s(t_subword sw, T x, T y) -> T {

    T m;

    m = a_hi[(t_int)(sw)];
    return lel_u(sw, x ^ m, y ^ m);
    }

  // Set LSB if x<y (signed).
  // O(1)
  static auto ltl_s(t_subword sw, T x, T y) -> T {

    T m;

    m = a_hi[(t_int)(sw)];
    return ltl_u(sw, x ^ m, y ^ m);
    }

  // Set LSB if x>=y (signed).
  // O(1)
  static auto gel_s(t_subword sw, T x, T y) -> T {

    T m;

    m = a_hi[(t_int)(sw)];
    return gel_u(sw, x ^ m, y ^ m);
    }

  // Set LSB if x>y (signed).
  // O(1)
  static auto gtl_s(t_subword sw, T x, T y) -> T {

    T m;

    m = a_hi[(t_int)(sw)];
    return gtl_u(sw, x ^ m, y ^ m);
    }

  // Set LSB for odd values, i.e. mask out LSB.
  // 2016-06-30 by Jasper L. Neumann
  // O(1)
  static auto oddl(t_subword sw, T x) -> T {

    T m;
    T lsb;

    m = a_lo[(t_int)(sw)];
    lsb = x & m;
    return lsb;
    }

  // Set mask denoting x==0.
  // See HD 6.1 "Find First 0-Byte"
  // 2014-07-10 by Jasper L. Neumann
  // O(1)
  static auto eq0(t_subword sw, T x) -> T {

    return sign2mask(sw, eq0h(sw, x));
    }

  // Set mask denoting x!=0.
  // O(1)
  static auto ne0(t_subword sw, T x) -> T {

    return sign2mask(sw, ne0h(sw, x));
    }

  // Set mask denoting x==y.
  // O(1)
  static auto eq(t_subword sw, T x, T y) -> T {

    return eq0(sw, x ^ y);
    }

  // Set mask denoting x!=y.
  // O(1)
  static auto ne(t_subword sw, T x, T y) -> T {

    return ~eq0(sw, x ^ y);
    }

  // Set mask denoting x>y (unsigned).
  // O(1)
  static auto gt_u(t_subword sw, T x, T y) -> T {

    return sign2mask(sw, gth_u(sw, x, y));
    }

  // Set mask denoting x<=y (unsigned).
  // HD 6.1 "Searching for a Value in a Given Range"
  // 2014-07-10 by Jasper L. Neumann
  // O(1)
  static auto le_u(t_subword sw, T x, T y) -> T {

    return sign2mask(sw, leh_u(sw, x, y));
    }

  // Set mask denoting x<y (unsigned).
  // O(1)
  static auto lt_u(t_subword sw, T x, T y) -> T {

    return gt_u(sw, y, x);
    }

  // Set mask denoting x>=y (unsigned).
  // O(1)
  static auto ge_u(t_subword sw, T x, T y) -> T {

    return le_u(sw, y, x);
    }

  // Set mask denoting x<0 (signed), i.e. propagate high (sign) bit.
  // 2014-07-11 by Jasper L. Neumann
  // O(1)
  static auto lt0_s(t_subword sw, T x) -> T {

    T m;
    T sign;
    t_int shift;

    shift = (1 << (t_int)(sw)) - 1;
    m = a_hi[(t_int)(sw)];
    sign = x & m;
    sign = (sign << 1) - (sign >> shift);  // transform sign: 1 ==> -1
    return sign;
    }

  // Set mask denoting x<=y (signed).
  // 2014-07-10 by Jasper L. Neumann
  // O(1)
  static auto le_s(t_subword sw, T x, T y) -> T {

    T m;

    m = a_hi[(t_int)(sw)];
    return le_u(sw, x ^ m, y ^ m);
    }

  // Set mask denoting x<y (signed).
  // O(1)
  static auto lt_s(t_subword sw, T x, T y) -> T {

    T m;

    m = a_hi[(t_int)(sw)];
    return lt_u(sw, x ^ m, y ^ m);
    }

  // Set mask denoting x>=y (signed).
  // O(1)
  static auto ge_s(t_subword sw, T x, T y) -> T {

    T m;

    m = a_hi[(t_int)(sw)];
    return ge_u(sw, x ^ m, y ^ m);
    }

  // Set mask denoting x>y (signed).
  // O(1)
  static auto gt_s(t_subword sw, T x, T y) -> T {

    T m;

    m = a_hi[(t_int)(sw)];
    return gt_u(sw, x ^ m, y ^ m);
    }

  // Set mask denoting odd values, i.e. propagate low bit to the left.
  // 2014-07-12 by Jasper L. Neumann
  // O(1)
  static auto odd(t_subword sw, T x) -> T {

    T m;
    T lsb;
    t_int shift;

    shift = 1 << (t_int)(sw);
    m = a_lo[(t_int)(sw)];
    lsb = x & m;
    lsb = ((lsb << (shift-1)) << 1) - lsb;  // transform lsb: 1 ==> -1
    // A shift by >= bits is undefined by the C/C++ standard.
    // Here we do a double shift to avoid shift by bits.
    return lsb;
    }

  // Clamp negative signed numbers to 0.
  // =max_s(sw, x, 0)
  // 2014-07-16 by Jasper L. Neumann
  // O(1)
  static auto max0_s(t_subword sw, T x) -> T {

    return x & ~lt0_s(sw, x);
    }

  // Unsigned maximum.
  // =add(sw, sub_sat_u(sw, x, y), y);
  // HD 2.19 "Doz, Max, Min"
  // 2014-07-10 by Jasper L. Neumann
  // O(1)
  static auto max_u(t_subword sw, T x, T y) -> T {

    T signmask;
    T t0, t1;
    t_int shift;
    T signs;
    T x1, y1;

    shift = (1 << (t_int)(sw)) - 1;
    signmask = a_hi[(t_int)(sw)];

    t0 = (y ^ ~x) & signmask;
    t1 = (y & ~x) & signmask;

    x1 = x | signmask;
    y1 = y & ~signmask;
    x1 = x1 - y1;

    t1 = t1 | (t0 & ~x1);
    t1 = (t1 << 1) - (t1 >> shift);
    x1 = (x1 ^ t0) & ~t1;

    signs = (x1 ^ y) & signmask;
    x1 = x1 & ~signmask;
    x1 = x1 + y1;
    x1 = x1 ^ signs;
    return x1;
    }

  // Unsigned minimum.
  // HD 2.19 "Doz, Max, Min"
  // 2014-07-10 by Jasper L. Neumann
  // O(1)
  static auto min_u(t_subword sw, T x, T y) -> T {

    return sub(sw, x, sub_sat_u(sw, x, y));
    }

  // Signed maximum.
  // 2014-07-10 by Jasper L. Neumann
  // O(1)
  static auto max_s(t_subword sw, T x, T y) -> T {

    T signmask;

    signmask = a_hi[(t_int)(sw)];
    return max_u(sw, x ^ signmask, y ^ signmask) ^ signmask;
    }

  // Signed minimum.
  // 2014-07-10 by Jasper L. Neumann
  // O(1)
  static auto min_s(t_subword sw, T x, T y) -> T {

    T signmask;

    signmask = a_hi[(t_int)(sw)];
    return min_u(sw, x ^ signmask, y ^ signmask) ^ signmask;
    }

  // abs_sat
  // clamp_*(sw,x,l,h): max_*(sw,l,min_*(sw,h,x))
  // median_*(sw, x,y,z)

  // Shift by common amount.

  // Shift left, shifting in 0 bits; also known as logical shift.
  // Gives correct results for all values of shift;
  // please note that shift is unsigned.
  // 2014-07-11 by Jasper L. Neumann
  // O(1)
  static auto shl(t_subword sw, T x, t_uint shift) -> T {

    t_uint b;  // # affected bits
    T m;  // mask of affected bits

    b = 1 << (t_int)(sw);
    if (shift >= b)  // Limit shift.
      return 0;
    else {
      m = a_lo[(t_int)(sw)];
      m = ~((m << shift) - m);
      return (x << shift) & m;
      }
    }

  // Shift right, shifting in 0 bits; also known as logical shift.
  // Gives correct results for all values of shift;
  // please note that shift is unsigned.
  // 2014-07-11 by Jasper L. Neumann
  // O(1)
  static auto shr(t_subword sw, T x, t_uint shift) -> T {

    t_uint b;  // # affected bits
    T m;  // mask of affected bits

    b = 1 << (t_int)(sw);
    if (shift == 0)  // Prevent shifting by b-shift >= bits.
      return x;
    else if (shift >= b)  // Limit shift.
      return 0;
    else {
      m = a_lo[(t_int)(sw)];
      m = (m << (b - shift)) - m;
      return (x >> shift) & m;
      }
    }

  // Shift left, duplicating the least significant bit.
  // Gives correct results for all values of shift;
  // please note that shift is unsigned.
  // 2014-07-12 by Jasper L. Neumann
  // O(1)
  static auto sal(t_subword sw, T x, t_uint shift) -> T {

    t_uint b;  // # affected bits
    T m;  // mask of affected bits
    T sign;

    b = 1 << (t_int)(sw);
    if (shift >= b)  // Limit shift.
      shift = b - 1;
    m = a_lo[(t_int)(sw)];
    m = ~((m << shift) - m);
    sign = x & a_lo[(t_int)(sw)];
    sign = ((sign << shift) << 1) - sign;
      // A shift by >= bits is undefined by the C/C++ standard.
      // Double shift to avoid shift by bits
      // (should also be correct for maximum sw).
    return ((x << shift) & m) | sign;
    }

  // Arithmetically shift right, duplicating  the most significant (sign) bit.
  // Gives correct results for all values of shift;
  // please note that shift is unsigned.
  // This is also known as arithmetic shift.
  // 2014-07-11 by Jasper L. Neumann
  // O(1)
  static auto sar(t_subword sw, T x, t_uint shift) -> T {

    t_uint b;  // # affected bits
    T m;  // mask of affected bits
    T sign;

    b = 1 << (t_int)(sw);
    if (shift == 0)  // Prevent shifting by b-shift >= bits.
      return x;
    else {
      if (shift >= b)  // Limit shift.
        shift = b-1;
      m = a_lo[(t_int)(sw)];
      m = (m << (b - shift)) - m;
      sign = x & a_hi[(t_int)(sw)];
      sign = (sign << 1) - (sign >> shift);
      return ((x >> shift) & m) | sign;
      }
    }

  // Shift left, shifting in 1 bits.
  // Gives correct results for all values of shift;
  // please note that shift is unsigned.
  // 2014-07-13 by Jasper L. Neumann
  // O(1)
  static auto shl1(t_subword sw, T x, t_uint shift) -> T {

    t_uint b;  // # affected bits
    T m;  // mask of affected bits

    b = 1 << (t_int)(sw);
    if (shift >= b)  // Limit shift.
      return all_bits;
    else {
      m = a_lo[(t_int)(sw)];
      m = ((m << shift) - m);
      return (x << shift) | m;
      }
    }

  // Shift right, shifting in 1 bits.
  // Gives correct results for all values of shift;
  // please note that shift is unsigned.
  // 2014-07-13 by Jasper L. Neumann
  // O(1)
  static auto shr1(t_subword sw, T x, t_uint shift) -> T {

    t_uint b;  // # affected bits
    T m;  // mask of affected bits

    b = 1 << (t_int)(sw);
    if (shift == 0)  // Prevent shifting by b-shift >= bits.
      return x;
    else if (shift >= b)  // Limit shift..
      return all_bits;
    else {
      m = a_lo[(t_int)(sw)];
      m = (m << (b - shift)) - m;
      return (x >> shift) | ~m;
      }
    }

  // Shift left saturates if:
  // Unsigned: x != (x << s) >>u s  =>  all_bits
  // Signed: x != (x << s) >>s s  =>  sgn_sat

  // 2014-09-26 by Jasper L. Neumann
  // O(1)
  static auto shl_sat_u(t_subword sw, T x, t_uint shift) -> T {

    T res;

    res = shl(sw, x, shift);
    return blend(eq(sw,x,shr(sw,res,shift)), res, all_bits);
    }

  // 2014-09-26 by Jasper L. Neumann
  // O(1)
  static auto sal_sat_u(t_subword sw, T x, t_uint shift) -> T {

    T res;

    res = sal(sw, x, shift);
    return blend(eq(sw,x,shr(sw,res,shift)), res, all_bits);
    }

  // 2014-09-26 by Jasper L. Neumann
  // O(1)
  static auto shl1_sat_u(t_subword sw, T x, t_uint shift) -> T {

    T res;

    res = shl1(sw, x, shift);
    return blend(eq(sw,x,shr(sw,res,shift)), res, all_bits);
    }

  // 2014-09-26 by Jasper L. Neumann
  // O(1)
  static auto shl_sat_s(t_subword sw, T x, t_uint shift) -> T {

    T res;

    res = shl(sw, x, shift);
    return blend(eq(sw,x,sar(sw,res,shift)), res, sgn_sat(sw,x));
    }

  // 2014-09-26 by Jasper L. Neumann
  // O(1)
  static auto sal_sat_s(t_subword sw, T x, t_uint shift) -> T {

    T res;

    res = sal(sw, x, shift);
    return blend(eq(sw,x,sar(sw,res,shift)), res, sgn_sat(sw,x));
    }

  // 2014-09-26 by Jasper L. Neumann
  // O(1)
  static auto shl1_sat_s(t_subword sw, T x, t_uint shift) -> T {

    T res;

    res = shl1(sw, x, shift);
    return blend(eq(sw,x,sar(sw,res,shift)), res, sgn_sat(sw,x));
    }

  // Shift right with rounding (shr, sar); see t_round_mode
  // shr is always unsigned; rounding is d [=f]
  // sar is always signed; rounding is f
  // shr_u: x != shl(shr(x,s),s)  =>  inc  [=shr_c] (no overflow can occur)
  // sar_u: x != shl(sar(x,s),s) && ge0(x)  =>  inc (no overflow can occur)
  // sar_c: x != shl(sar(x,s),s)  =>  inc
  // sar_d: x != shl(sar(x,s),s) && lt0_s(x)  =>  inc

  //  half_even,   // round, *.5 => next even, banker's rounding, x87:0
  //  floor,       // floor -> -infinity, sar, x87:1
  //  ceil,        // ceil -> infinity, x87:2
  //  down,        // trunc -> 0, chop, div, x87:3
  //  up,          // -> away from 0
  //  half_odd,    // round, *.5 => next odd
  //  half_floor,  // round, *.5 => floor
  //  half_ceil,   // round, *.5 => ceil
  //  half_down,   // round, *.5 => trunk
  //  half_up      // round, *.5 => away from 0

  // Rotate by common amount.

  // Every (1 << (t_int)(sw)) bits are rotated left.
  // Gives correct results for all values of rot.
  // x: value
  // sw: log_2(#bits), must be <=ld_bits
  // rot: rotate count
  // Bit-parallel implementation: 2011-09-21 by Jasper L. Neumann
  // O(1)
  static auto rol(t_subword sw, T x, t_int rot) -> T {

    t_int b;  // # affected bits
    t_int r;  // rot % b
    T m;  // mask for affected bits

    b = 1 << (t_int)(sw);

    r = rot & (b - 1);
    if (r == 0) {
      // Prevent shifting by b-r >= bits.
      return x;
      }
    else {
      m = a_lo[(t_int)(sw)];
      m = (m << r) - m;

      return
        ((x << r) & ~m) |
        ((x >> (b - r)) & m);
      }
    }

  // Every (1 << (t_int)(sw)) bits are rotated right.
  // Gives correct results for all values of rot.
  // x: value
  // sw: log_2(#bits), must be <=ld_bits
  // rot: rotate count
  // O(1)
  static auto ror(t_subword sw, T x, t_int rot) -> T {

    return rol(sw, x, -rot);
    }

  // Every (1 << (t_int)(sw)) bits are rotated left
  // complementing the shifted in bits.
  // Gives correct results for all values of rot.
  // x: value
  // sw: log_2(#bits), must be <=ld_bits
  // rot: rotate count
  // Bit-parallel implementation: 2011-09-23 by Jasper L. Neumann
  // O(1)
  static auto rolc(t_subword sw, T x, t_int rot) -> T {

    t_int b;   // # affected bits
    t_int r;   // rot mod b
    T m;  // mask for affected bits

    b = 1 << (t_int)(sw);

    r = rot & (b-1);
    if (r == 0) {
      // Prevent shifting by b-r >= bits.
      }
    else {
      m = a_lo[(t_int)(sw)];
      m = (m << r) - m;

      x =
        ((x << r) & ~m) |
        ((x >> (b-r)) & m);

      // Until here essentially same code as rol.
      x = x ^ m;
      }

    if ((rot & b)!=0) {
      x = ~x;
      }

    return x;
    }

  // Every (1 << (t_int)(sw)) bits are rotated right
  // complementing the shifted in bits.
  // Gives correct results for all values of rot.
  // x: value
  // sw: log_2(#bits), must be <=ld_bits
  // rot: rotate count
  // O(1)
  static auto rorc(t_subword sw, T x, t_int rot) -> T {

    return rolc(sw, x, -rot);
    }

  // Rotate every (1 << (t_int)(sw)) bits of l and h
  // where the bit groups of l and h are considered to be one group.
  // This can be used to implement many kinds of shift and rotate.
  // Gives correct results for all values of rot.
  // l, h: value
  // sw: log_2(#bits), must be <=ld_bits
  // rot: rotate count
  // Example: l=hgfedcba, h=HGFEDCBA, sw=2, rot=1 => gfeHcbaD
  // Bit-parallel implementation: 2014-07-07 by Jasper L. Neumann
  // O(1)
  static auto rold(t_subword sw, T l, T h, t_int rot) -> T {

    t_uint b;  // # affected bits
    T m;  // mask of affected bits
    t_int r;  // rot % b

    b = 1 << (t_int)(sw);

    if ((rot & b) != 0) {
      // Exchange l and h.
      m = l;
      l = h;
      h = m;
      }

    r = rot & (b-1);
    if (r == 0) {
      // Prevent shifting by b-r >= bits.
      }
    else {
      m = a_lo[(t_int)(sw)];
      m = (m << r) - m;

      l =
        ((l << r) & ~m) |
        ((h >> (b-r)) & m);
      }
    return l;
    }

  // Rotate every (1 << (t_int)(sw)) bits of l and h
  // where the bit groups of l and h are considered to be one group.
  // This can be used to implement many kinds of shift and rotate.
  // Gives correct results for all values of rot.
  // l, h: value
  // sw: log_2(#bits), must be <=ld_bits
  // rot: rotate count
  // O(1)
  static auto rord(t_subword sw, T l, T h, t_int rot) -> T {

    return rold(sw, l, h, -rot);
    }

  // Variant of rold modifying *l and *h.
  // Rotate every (1 << (t_int)(sw)) bits of *l and *h
  // where the bit groups of l and h are considered to be one group.
  // This can be used to implement many kinds of shift and rotate.
  // Gives correct results for all values of rot.
  // *l, *h: value
  // sw: log_2(#bits), must be <=ld_bits
  // rot: rotate count
  // Example: l=hgfedcba, h=HGFEDCBA, sw=2, rot=1 => gfeHcbaD
  // Bit-parallel implementation: 2014-07-07 by Jasper L. Neumann
  // O(1)
  static void do_rold(t_subword sw, T* l, T* h, t_int rot) {

    t_uint b;  // # affected bits
    T m;  // mask of affected bits
    t_int r;  // rot % b
    T l0, h0;

    b = 1 << (t_int)(sw);

    if ((rot & b) != 0) {
      // Exchange *l and *h.
      m = *l;
      *l = *h;
      *h = m;
      }

    r = rot & (b-1);
    if (r == 0) {
      // Prevent shifting by b-r >= bits.
      }
    else {
      m = a_lo[(t_int)(sw)];
      m = (m << r) - m;

      l0 = *l;
      h0 = *h;
      *l =
        ((l0 << r) & ~m) |
        ((h0 >> (b-r)) & m);
      *h =
        ((h0 << r) & ~m) |
        ((l0 >> (b-r)) & m);
      }
    }

  // Variant of rord modifying *l and *h.
  // Rotate every (1 << (t_int)(sw)) bits of *l and *h
  // where the bit groups of l and h are considered to be one group.
  // This can be used to implement many kinds of shift and rotate.
  // Gives correct results for all values of rot.
  // *l, *h: value
  // sw: log_2(#bits), must be <=ld_bits
  // rot: rotate count
  // O(1)
  static void do_rord(t_subword sw, T* l, T* h, t_int rot) {

    do_rold(sw, l, h, -rot);
    }

  // Shift by fieldwise specified amount.

  // Field variable variant of shl.
  // Gives correct results for all values of shift;
  // please note that shift values are treated as unsigned.
  // 2014-07-14 by Jasper L. Neumann
  // O(ld_bits)
  static auto vshl(t_subword sw, T x, T shift) -> T {

    t_int i, s;
    T m;

    m = ~a_hi[(t_int)(sw)];
    s = 1;
    for (i = 0; i<=(t_int)(sw)-1; ++i) {
      // s = 1 << i;
      x = t_my_bits::blend(odd(sw, shift), shl(sw, x, s), x);
      shift = (shift >> 1) & m;
      s = s << 1;
      }
    x = x & eq0(sw, shift);
    return x;
    }

  // Field variable variant of shr.
  // Gives correct results for all values of shift;
  // please note that shift values are treated as unsigned.
  // 2014-07-14 by Jasper L. Neumann
  // O(ld_bits)
  static auto vshr(t_subword sw, T x, T shift) -> T {

    t_int i, s;
    T m;

    m = ~a_hi[(t_int)(sw)];
    s = 1;
    for (i = 0; i<=(t_int)(sw)-1; ++i) {
      // s = 1 << i;
      x = t_my_bits::blend(odd(sw, shift), shr(sw, x, s), x);
      shift = (shift >> 1) & m;
      s = s << 1;
      }
    x = x & eq0(sw, shift);
    return x;
    }

  // Field variable variant of sar.
  // Gives correct results for all values of shift;
  // please note that shift values are treated as unsigned.
  // 2014-07-14 by Jasper L. Neumann
  // O(ld_bits)
  static auto vsar(t_subword sw, T x, T shift) -> T {

    t_int i, s;

    shift = min_u(sw, shift, (T)((1 << (t_int)(sw))-1) * a_lo[(t_int)(sw)]);
    s = 1;
    for (i = 0; i<=(t_int)(sw)-1; ++i) {
      // s = 1 << i;
      x = t_my_bits::blend(odd(sw, shift), sar(sw, x, s), x);
      shift = shift >> 1;
      s = s << 1;
      }
    return x;
    }

  // Field variable variant of sal.
  // Gives correct results for all values of shift;
  // please note that shift values are treated as unsigned.
  // 2014-07-14 by Jasper L. Neumann
  // O(ld_bits)
  static auto vsal(t_subword sw, T x, T shift) -> T {

    t_int i, s;

    shift = min_u(sw, shift, (T)((1 << (t_int)(sw))-1) * a_lo[(t_int)(sw)]);
    s = 1;
    for (i = 0; i<=(t_int)(sw)-1; ++i) {
      // s = 1 << i;
      x = t_my_bits::blend(odd(sw, shift), sal(sw, x, s), x);
      shift = shift >> 1;
      s = s << 1;
      }
    return x;
    }

  // Field variable variant of shl1.
  // Gives correct results for all values of shift;
  // please note that shift values are treated as unsigned.
  // 2014-07-14 by Jasper L. Neumann
  // O(ld_bits)
  static auto vshl1(t_subword sw, T x, T shift) -> T {

    t_int i, s;
    T m;

    m = ~a_hi[(t_int)(sw)];
    s = 1;
    for (i = 0; i<=(t_int)(sw)-1; ++i) {
      // s = 1 << i;
      x = t_my_bits::blend(odd(sw, shift), shl1(sw, x, s), x);
      shift = (shift >> 1) & m;
      s = s << 1;
      }
    x = x | ~eq0(sw, shift);
    return x;
    }

  // Field variable variant of shr1.
  // Gives correct results for all values of shift;
  // please note that shift values are treated as unsigned.
  // 2014-07-14 by Jasper L. Neumann
  // O(ld_bits)
  static auto vshr1(t_subword sw, T x, T shift) -> T {

    t_int i, s;
    T m;

    m = ~a_hi[(t_int)(sw)];
    s = 1;
    for (i = 0; i<=(t_int)(sw)-1; ++i) {
      // s = 1 << i;
      x = t_my_bits::blend(odd(sw, shift), shr1(sw, x, s), x);
      shift = (shift >> 1) & m;
      s = s << 1;
      }
    x = x | ~eq0(sw, shift);
    return x;
    }

  // 2014-09-26 by Jasper L. Neumann
  // O(1)
  static auto vshl_sat_u(t_subword sw, T x, T shift) -> T {

    T res;

    res = vshl(sw, x, shift);
    return blend(eq(sw,x,vshr(sw,res,shift)), res, all_bits);
    }

  // 2014-09-26 by Jasper L. Neumann
  // O(1)
  static auto vsal_sat_u(t_subword sw, T x, T shift) -> T {

    T res;

    res = vsal(sw, x, shift);
    return blend(eq(sw,x,vshr(sw,res,shift)), res, all_bits);
    }

  // 2014-09-26 by Jasper L. Neumann
  // O(1)
  static auto vshl1_sat_u(t_subword sw, T x, T shift) -> T {

    T res;

    res = vshl1(sw, x, shift);
    return blend(eq(sw,x,vshr(sw,res,shift)), res, all_bits);
    }

  // 2014-09-26 by Jasper L. Neumann
  // O(1)
  static auto vshl_sat_s(t_subword sw, T x, T shift) -> T {

    T res;

    res = vshl(sw, x, shift);
    return blend(eq(sw,x,vsar(sw,res,shift)), res, sgn_sat(sw,x));
    }

  // 2014-09-26 by Jasper L. Neumann
  // O(1)
  static auto vsal_sat_s(t_subword sw, T x, T shift) -> T {

    T res;

    res = vsal(sw, x, shift);
    return blend(eq(sw,x,vsar(sw,res,shift)), res, sgn_sat(sw,x));
    }

  // 2014-09-26 by Jasper L. Neumann
  // O(1)
  static auto vshl1_sat_s(t_subword sw, T x, T shift) -> T {

    T res;

    res = vshl1(sw, x, shift);
    return blend(eq(sw,x,vsar(sw,res,shift)), res, sgn_sat(sw,x));
    }

  // Rotate by fieldwise specified amount.

  // Field variable variant of rol.
  // Gives correct results for all values of rot.
  // 2014-07-14 by Jasper L. Neumann
  // O(ld_bits)
  static auto vrol(t_subword sw, T x, T rot) -> T {

    t_int i, s;

    s = 1;
    for (i = 0; i<=(t_int)(sw)-1; ++i) {
      // s = 1 << i;
      x = t_my_bits::blend(odd(sw, rot), rol(sw, x, s), x);
      rot = rot >> 1;
      s = s << 1;
      }
    return x;
    }

  // Field variable variant of ror.
  // Gives correct results for all values of rot.
  // 2014-07-14 by Jasper L. Neumann
  // O(ld_bits)
  static auto vror(t_subword sw, T x, T rot) -> T {

    t_int i, s;

    s = 1;
    for (i = 0; i<=(t_int)(sw)-1; ++i) {
      // s = 1 << i;
      x = t_my_bits::blend(odd(sw, rot), ror(sw, x, s), x);
      rot = rot >> 1;
      s = s << 1;
      }
    return x;
    }

  // Field variable variant of rolc.
  // Gives correct results for all values of rot.
  // 2014-07-14 by Jasper L. Neumann
  // O(ld_bits)
  static auto vrolc(t_subword sw, T x, T rot) -> T {

    t_int i, s;

    s = 1;
    for (i = 0; i<=(t_int)(sw)-1; ++i) {
      // s = 1 << i;
      x = t_my_bits::blend(odd(sw, rot), rolc(sw, x, s), x);
      rot = rot >> 1;
      s = s << 1;
      }
    return x ^ odd(sw, rot);
    }

  // Field variable variant of rorc.
  // Gives correct results for all values of rot.
  // 2014-07-14 by Jasper L. Neumann
  // O(ld_bits)
  static auto vrorc(t_subword sw, T x, T rot) -> T {

    t_int i, s;

    s = 1;
    for (i = 0; i<=(t_int)(sw)-1; ++i) {
      // s = 1 << i;
      x = t_my_bits::blend(odd(sw, rot), rorc(sw, x, s), x);
      rot = rot >> 1;
      s = s << 1;
      }
    return x ^ odd(sw, rot);
    }

  // Field variable variant of rold.
  // Gives correct results for all values of rot.
  // 2014-07-14 by Jasper L. Neumann
  // O(ld_bits)
  static auto vrold(t_subword sw, T l, T h, T rot) -> T {

    t_int i;
    T l1, h1;
    T o;
    t_int b;  // # affected bits
    t_int s;
    T m0, m;  // mask for affected bits

    b = 1 << (t_int)(sw);
    m0 = a_lo[(t_int)(sw)];
    s = 1;
    for (i = 0; i<=(t_int)(sw)-1; ++i) {
      // s = 1 << i;
      m = (m0 << s) - m0;
      o = odd(sw, rot);

      // l1 = t_my_bits::blend(o, rold(sw, l, h, s), l);
      // h1 = t_my_bits::blend(o, rold(sw, h, l, s), h);
      // l = l1;  h = h1;
      l1 =
        ((l << s) & ~m) |
        ((h >> (b-s)) & m);
      h1 =
        ((h << s) & ~m) |
        ((l >> (b-s)) & m);
      l1 = t_my_bits::blend(o, l1, l);
      h1 = t_my_bits::blend(o, h1, h);
      l = l1;
      h = h1;

      rot = rot >> 1;
      s = s << 1;
      }

    o = odd(sw, rot);
    return t_my_bits::blend(o, h, l);
    }

  // Field variable variant of rord.
  // Gives correct results for all values of rot.
  // 2014-07-14 by Jasper L. Neumann
  // O(ld_bits)
  static auto vrord(t_subword sw, T l, T h, T rot) -> T {

    t_int i;
    T l1, h1;
    T o;
    t_int b;  // # affected bits
    t_int s;
    T m0, m;  // mask for affected bits

    b = 1 << (t_int)(sw);
    m0 = a_lo[(t_int)(sw)];
    s = 1;
    for (i = 0; i<=(t_int)(sw)-1; ++i) {
      // s = 1 << i;
      m = (m0 << (b-s)) - m0;
      o = odd(sw, rot);

      // l1 = t_my_bits::blend(o, rord(sw, l, h, s), l);
      // h1 = t_my_bits::blend(o, rord(sw, h, l, s), h);
      // l = l1;  h = h1;
      l1 =
        ((l >> s) & m) |
        ((h << (b-s)) & ~m);
      h1 =
        ((h >> s) & m) |
        ((l << (b-s)) & ~m);
      l1 = t_my_bits::blend(o, l1, l);
      h1 = t_my_bits::blend(o, h1, h);
      l = l1;
      h = h1;

      rot = rot >> 1;
      s = s << 1;
      }

    o = odd(sw, rot);
    return t_my_bits::blend(o, h, l);
    }

  // Variant of vrold modifying *l and *h.
  // Gives correct results for all values of rot.
  // 2014-07-14 by Jasper L. Neumann
  // O(ld_bits)
  static void do_vrold(t_subword sw, T* l, T* h, T rot) {

    t_int i;
    T l0, h0, l1, h1;
    T o;
    t_int b;  // # affected bits
    t_int s;
    T m0, m;  // mask for affected bits

    l0 = *l;
    h0 = *h;
    b = 1 << (t_int)(sw);
    m0 = a_lo[(t_int)(sw)];
    s = 1;
    for (i = 0; i<=(t_int)(sw)-1; ++i) {
      // s = 1 << i;
      m = (m0 << s) - m0;
      o = odd(sw, rot);

      // l1 = t_my_bits::blend(o, rold(sw, l0, h0, s), l0);
      // h1 = t_my_bits::blend(o, rold(sw, h0, l0, s), h0);
      // l0 = l1;  h0 = h1;
      l1 =
        ((l0 << s) & ~m) |
        ((h0 >> (b-s)) & m);
      h1 =
        ((h0 << s) & ~m) |
        ((l0 >> (b-s)) & m);
      l1 = t_my_bits::blend(o, l1, l0);
      h1 = t_my_bits::blend(o, h1, h0);
      l0 = l1;
      h0 = h1;

      rot = rot >> 1;
      s = s << 1;
      }

    o = odd(sw, rot);
    *l = t_my_bits::blend(o, h0, l0);
    *h = t_my_bits::blend(o, l0, h0);
    }

  // Variant of vrord modifying *l and *h.
  // Gives correct results for all values of rot.
  // 2014-07-14 by Jasper L. Neumann
  // O(ld_bits)
  static void do_vrord(t_subword sw, T* l, T* h, T rot) {

    t_int i;
    T l0, h0, l1, h1;
    T o;
    t_int b;  // # affected bits
    t_int s;
    T m0, m;  // mask for affected bits

    l0 = *l;
    h0 = *h;
    b = 1 << (t_int)(sw);
    m0 = a_lo[(t_int)(sw)];
    s = 1;
    for (i = 0; i<=(t_int)(sw)-1; ++i) {
      // s = 1 << i;
      m = (m0 << (b-s)) - m0;
      o = odd(sw, rot);

      // l1 = t_my_bits::blend(o, rord(sw, l0, h0, s), l0);
      // h1 = t_my_bits::blend(o, rord(sw, h0, l0, s), h0);
      // l0 = l1;  h0 = h1;
      l1 =
        ((l0 >> s) & m) |
        ((h0 << (b-s)) & ~m);
      h1 =
        ((h0 >> s) & m) |
        ((l0 << (b-s)) & ~m);
      l1 = t_my_bits::blend(o, l1, l0);
      h1 = t_my_bits::blend(o, h1, h0);
      l0 = l1;
      h0 = h1;

      rot = rot >> 1;
      s = s << 1;
      }

    o = odd(sw, rot);
    *l = t_my_bits::blend(o, h0, l0);
    *h = t_my_bits::blend(o, l0, h0);
    }

  // Special functions.

  // See Hacker's Delight, 13.1 "Gray Code"
  // x ^ (x >> 1)
  static auto gray_code(t_subword sw, T x) -> T {

    return x ^ ((x >> 1) & ~a_hi[(t_int)(sw)]);
    }

  // See Hacker's Delight, 13.1 "Gray Code"
  static auto inv_gray_code(t_subword sw, T x) -> T {

    t_int i, s;

    s = 1;
    for (i = 0; i <= (t_int)(sw)-1; ++i) {  // UNROLL
      // s = 1 << i;
      x = x ^ shr(sw, x, s);
      s = s << 1;
      }
    return x;
    }

  // General trailing bit modification operations.
  // 2014-02-11 by Jasper L. Neumann
  // Mode: The operating mode, see t_tbm_mode.
  // For whole words:
  // All of these operations can be performed by <= 3 instructions.
  // Some of these operations are realized as BMI1 or TBM instructions.
  // O(1)
  static auto tbm(t_subword sw, T x, t_int mode) -> T {

    switch (mode & 0x1f) {
      case 0x00: return 0;
      case 0x01: return x & ~inc(sw,x);
      case 0x02: return ~x & inc(sw,x);
      case 0x03: return x ^ inc(sw,x);
      case 0x04: return ~(x ^ inc(sw,x));
      case 0x05: return x | ~inc(sw,x);
      case 0x06: return ~x | inc(sw,x);
      case 0x07: return all_bits;
      case 0x08: return x & inc(sw,x);
      case 0x09: return x;
      case 0x0a: return inc(sw,x);
      case 0x0b: return x | inc(sw,x);
      case 0x0c: return ~(x | inc(sw,x));
      case 0x0d: return dec(sw,~x);
      case 0x0e: return ~x;
      case 0x0f: return ~(x & inc(sw,x));
      case 0x10: return 0;
      case 0x11: return ~(x | neg(sw, x));  // ~x & dec(sw,x);
      case 0x12: return x & neg(sw,x);
      case 0x13: return neg(sw, x) ^ ~x;  // x ^ dec(sw,x);
      case 0x14: return x ^ neg(sw,x);
      case 0x15: return ~x | dec(sw,x);
      case 0x16: return x | neg(sw,x);
      case 0x17: return all_bits;
      case 0x18: return x & dec(sw,x);
      case 0x19: return dec(sw,x);
      case 0x1a: return x;
      case 0x1b: return x | dec(sw,x);
      case 0x1c: return neg(sw, x) & ~x;  // ~(x | dec(sw,x));
      case 0x1d: return ~x;
      case 0x1e: return neg(sw,x);
      case 0x1f: return neg(sw, x) | ~x;  // ~(x & dec(sw,x));
      default:   return 0;  // This can not happen.
      }
    }

  // Horizontal math.

  // AND of neighboring subwords.
  // When processing a loop i in 0..(t_int)(sw): subword == -1.
  // O(1)
  static auto hand(t_subword sw, T x) -> T {

    T m;
    t_int s;

    m = a_even[(t_int)(sw)];
    s = 1 << (t_int)(sw);
    // x = (x & m) & ((x >> s) & m);
    x = (x & (x >> s)) & m;
    return x;
    }

  // OR of neighboring subwords.
  // When processing a loop i in 0..(t_int)(sw): subword != 0.
  // O(1)
  static auto hor(t_subword sw, T x) -> T {

    T m;
    t_int s;

    m = a_even[(t_int)(sw)];
    s = 1 << (t_int)(sw);
    // x = (x & m) | ((x >> s) & m);
    x = (x | (x >> s)) & m;
    return x;
    }

  // XOR of neighboring subwords.
  // When processing a loop i in 0..(t_int)(sw): Gray code / Parity.
  // O(1)
  static auto hxor(t_subword sw, T x) -> T {

    T m;
    t_int s;

    m = a_even[(t_int)(sw)];
    s = 1 << (t_int)(sw);
    // x = (x & m) ^ ((x >> s) & m);
    x = (x ^ (x >> s)) & m;
    return x;
    }

  // Sum of neighboring unsigned subwords.
  // See nr_1bits.
  // O(1)
  static auto hadd_u(t_subword sw, T x) -> T {

    T m;
    t_int s;

    m = a_even[(t_int)(sw)];
    s = 1 << (t_int)(sw);
    x = (x & m) + ((x >> s) & m);
    return x;
    }

  // Difference between neighboring unsigned subwords giving signed results.
  // Calculate lower - higher subword for all pairs.
  // 2014-07-10 by Jasper L. Neumann
  // O(1)
  static auto hsub_u(t_subword sw, T x) -> T {

    T m;
    T h;
    t_int s;
    T low, high;

    m = a_even[(t_int)(sw)];
    s = 1 << (t_int)(sw);
    low = x & m;
    high = (x >> s) & m;
    x = (a_lo[(t_int)(sw)+1] << s) + low - high;
      // This is the unmasked offsetted difference.
    // low: x & m
    // high can be 0 or 1: x & ~m
    h = x & ~m;
    // x = (x | ~m) - ((h << s)-h);
    x = (x | ~m) + h - (h << s);
    return x;
    }

  // Sum of neighboring signed subwords.
  // 2014-07-10 by Jasper L. Neumann
  // O(1)
  static auto hadd_s(t_subword sw, T x) -> T {

    T m;
    T h;
    t_int s;

    m = a_even[(t_int)(sw)];
    s = 1 << (t_int)(sw);
    x = x ^ a_hi[(t_int)(sw)];  // Signed => unsigned (offset)
    x = (x & m) + ((x >> s) & m);
    // low: x & m
    // high can be 0 or 1: x & ~m
    h = x & ~m;
    // x = (x | ~m) - ((h << s)-h);
    x = (x | ~m) + h - (h << s);
    return x;
    }

  // Difference between neighboring signed subwords giving signed results.
  // Calculate lower - higher subword for all pairs.
  // 2014-07-10 by Jasper L. Neumann
  // O(1)
  static auto hsub_s(t_subword sw, T x) -> T {

    T m;
    T h;
    t_int s;
    T low, high;

    m = a_even[(t_int)(sw)];
    s = 1 << (t_int)(sw);
    x = x ^ a_hi[(t_int)(sw)];  // Signed => unsigned (offset)
    low = x & m;
    high = (x >> s) & m;
    x = (a_lo[(t_int)(sw)+1] << s) + low - high;
      // This is the unmasked offsetted difference.
    // low: x & m
    // high can be 0 or 1: x & ~m
    h = x & ~m;
    // x = (x | ~m) - ((h << s)-h);
    x = (x | ~m) + h - (h << s);
    return x;
    }

  // Unsigned average rounding down,
  // i.e. (x+y)>>1, of neighboring unsigned subwords.
  // O(1)
  static auto havgd_u(t_subword sw, T x) -> T {

    T m;
    t_int s;

    m = a_even[(t_int)(sw)];
    s = 1 << (t_int)(sw);
    x = ((
      (x & m) +
      ((x >> s) & m)
      ) >> 1) & m;
    return x;
    }

  // Unsigned average rounding up,
  // i.e. (x+y+1)>>1, of neighboring unsigned subwords.
  // O(1)
  static auto havgu_u(t_subword sw, T x) -> T {

    T m;
    t_int s;

    m = a_even[(t_int)(sw)];
    s = 1 << (t_int)(sw);
    x = ((
      (x & m) +
      ((x >> s) & m) +
      a_lo[(t_int)(sw)+1]
      ) >> 1) & m;
    return x;
    }

  // Maximum between neighboring unsigned subwords.
  // HD 2.19 "Doz, Max, Min"
  // max(a, b) = max(0, a-b)+b
  // 2014-07-10 by Jasper L. Neumann
  // O(1)
  static auto hmax_u(t_subword sw, T x) -> T {

    T m;
    T h;
    t_int s;
    T low, high;

    m = a_even[(t_int)(sw)];
    s = 1 << (t_int)(sw);
    low = x & m;
    high = (x >> s) & m;
    x = (a_lo[(t_int)(sw)+1] << s) + low - high;
      // This is the unmasked offsetted difference.
    h = x & ~m;
    h = h - (h >> s);
    x = (x | ~m) & h;  // max0_s(low-high)
    x = x + high;
    return x;
    }

  // Minimum between neighboring unsigned subwords.
  // HD 2.19 "Doz, Max, Min"
  // min(a, b) = a-max(0, a-b)
  // 2014-07-10 by Jasper L. Neumann
  // O(1)
  static auto hmin_u(t_subword sw, T x) -> T {

    T m;
    T h;
    t_int s;
    T low, high;

    m = a_even[(t_int)(sw)];
    s = 1 << (t_int)(sw);
    low = x & m;
    high = (x >> s) & m;
    x = (a_lo[(t_int)(sw)+1] << s) + low - high;
      // This is the unmasked offsetted difference.
    h = x & ~m;
    h = h - (h >> s);
    x = (x | ~m) & h;  // max0_s(low-high)
    x = low - x;
    return x;
    }

  // Maximum between neighboring signed subwords.
  // HD 2.19 "Doz, Max, Min"
  // max(a, b) = max(0, a-b)+b
  // 2014-07-10 by Jasper L. Neumann
  // O(1)
  static auto hmax_s(t_subword sw, T x) -> T {

    T m;
    T h;
    t_int s;
    T low, high;
    T sign;

    m = a_even[(t_int)(sw)];
    s = 1 << (t_int)(sw);
    x = x ^ a_hi[(t_int)(sw)];  // Signed => unsigned (offset)
    low = x & m;
    high = (x >> s) & m;
    sign = a_lo[(t_int)(sw)+1] << s;
    x = sign + low - high;  // This is the unmasked offsetted difference.
    h = x & ~m;
    h = h - (h >> s);
    x = (x | ~m) & h;  // max0_s(low-high)
    x = x + high;
    // inverse sign extend
    x = x ^ (sign >> 1);
    h = (x << 1) & sign;
    h = (h << s)-h;
    x = x | h;
    return x;
    }

  // Minimum between neighboring signed subwords.
  // HD 2.19 "Doz, Max, Min"
  // min(a, b) = a-max(0, a-b)
  // 2014-07-10 by Jasper L. Neumann
  // O(1)
  static auto hmin_s(t_subword sw, T x) -> T {

    T m;
    T h;
    t_int s;
    T low, high;
    T sign;

    m = a_even[(t_int)(sw)];
    s = 1 << (t_int)(sw);
    x = x ^ a_hi[(t_int)(sw)];  // Signed => unsigned (offset)
    low = x & m;
    high = (x >> s) & m;
    sign = a_lo[(t_int)(sw)+1] << s;
    x = sign + low - high;  // This is the unmasked offsetted difference.
    h = x & ~m;
    h = h - (h >> s);
    x = (x | ~m) & h;  // max0_s(low-high)
    x = low - x;
    // inverse sign, sign extend
    x = x ^ (sign >> 1);
    h = (x << 1) & sign;
    h = (h << s)-h;
    x = x | h;
    return x;
    }

  // Zero extend even (lower) elements.
  // 2014-07-10 by Jasper L. Neumann
  // O(1)
  static auto hmovzx(t_subword sw, T x) -> T {

    T m;

    m = a_even[(t_int)(sw)];
    x = x & m;
    return x;
    }

  // Sign extend even (lower) elements.
  // 2014-07-10 by Jasper L. Neumann
  // O(1)
  static auto hmovsx(t_subword sw, T x) -> T {

    T m;
    T h;
    T sign;
    t_int s;

    s = 1 << (t_int)(sw);
    m = a_even[(t_int)(sw)];
    x = x & m;
    sign = a_lo[(t_int)(sw)+1] << s;
    h = (x << 1) & sign;
    h = (h << s)-h;
    x = x | h;
    return x;
    }

  // Product of neighboring unsigned subwords.
  // Stupid emulation due to absence of a matching instruction/operation.
  // 2014-09-12 by Jasper L. Neumann
  // O(#subwords)
  static auto hmul_u(t_subword sw, T x) -> T {

    T m1, m2, res, x1,x2;
    t_int b, j, s;

    b = 1 << (t_int)(sw);
    m1 = a_element[(t_int)(sw)];
    m2 = a_element[(t_int)(sw)+1];
    res = 0;
    s = 0;
    for (j = (t_int)(bits >> ((t_int)(sw)+1))-1; j >= 0; --j) {
      x1 = (x >> s) & m1;
      x2 = (x >> (s+b)) & m1;
      res = res | (((x1 * x2) & m2) << s);
      s = s + b*2;
      }
    return res;
    }

  // Product of neighboring signed subwords.
  // Stupid emulation due to absence of a matching instruction/operation.
  // 2014-09-12 by Jasper L. Neumann
  // O(#subwords)
  static auto hmul_s(t_subword sw, T x) -> T {

    T m2, res;
    TS x1,x2;
    t_int b, j, s;

    b = 1 << (t_int)(sw);
    m2 = a_element[(t_int)(sw)+1];
    res = 0;
    s = 0;
    for (j = (t_int)(bits >> ((t_int)(sw)+1))-1; j >= 0; --j) {
      x1 = t_my_bits::sign_extend(x >> s, b);
      x2 = t_my_bits::sign_extend(x >> (s+b), b);
      res = res | ((T)((x1 * x2) & m2) << s);
      s = s + b*2;
      }
    return res;
    }

  // swap_even_odd
  // unpack: sign/zero extend even (hmov?x) and odd
  // pack: 2 sources, saturate/clip
  // mul(even1,even2)+mul(odd1,odd2) (signed/unsigned) wie in x86:pmaddwd(signed)
  // mul(even1,even2) (signed/unsigned) wie in x86:pmuludq(unsigned)

  // Shuffle operations.

  // One butterfly step/stage.
  // sw: 0..ld_bits-1
  // m & a_bfly_mask[sw] should be == m
  // INLINE
  static auto butterfly(t_subword sw, T x, T m) -> T {

    return t_my_bits::bit_permute_step(x, m, 1 << (t_int)(sw));
    }

  // remaining shuffle functions => t_my_bits?

  // See ARM System Developer's Guide, 7.6.2 "Bit Permutations"
  // as used in the loop of general_reverse_bits.
  // INLINE
  static auto bit_index_complement(T x, t_subword k) -> T {

    t_int shift;
    T m;

    shift = 1 << (t_int)(k);
    m = a_even[(t_int)(k)];
    return t_my_bits::bit_permute_step_simple(x, m, shift);
    }

  // See ARM System Developer's Guide, 7.6.2 "Bit Permutations"
  // => shuffle, unshuffle
  // INLINE
  static auto bit_index_swap(T x, t_subword j, t_subword k) -> T {

    t_int shift;
    t_subword q;
    T m;

    if (j != k) {
      if (j < k) {
        q = j;
        j = k;
        k = q;
        }
      shift = (1 << (t_int)(j)) - (1 << (t_int)(k));
      m = a_even[(t_int)(j)] & ~a_even[(t_int)(k)];  // b_j==0 & b_k==1
      x = t_my_bits::bit_permute_step(x, m, shift);
      }
    return x;
    }

  // See ARM System Developer's Guide, 7.6.2 "Bit Permutations"
  // INLINE
  static auto bit_index_swap_complement(T x, t_subword j, t_subword k) -> T {

    t_int shift;
    t_subword q;
    T m;

    if (j != k) {
      if (j < k) {
        q = j;
        j = k;
        k = q;
        }
      shift = (1 << (t_int)(j)) + (1 << (t_int)(k));
      m = a_even[(t_int)(j)] & a_even[(t_int)(k)];  // b_j==0 & b_k==0
      x = t_my_bits::bit_permute_step(x, m, shift);
      }
    return x;
    }

  // Rotate an bit index field to the right by rot
  // by executing the necessary calls to bit_index_swap.
  // q+field+ofs=ld_bits
  // q: upper bit indexes (unchanged)
  // field: size of the affected bit string
  // ofs: lower bit indexes (unchanged)
  // Bit-parallel implementation: 2011-10-04 by Jasper L. Neumann
  static auto bit_index_ror(T x, t_int rot, t_subword ofs, t_subword field) -> T {

    t_int i, j, d, g, k, n;

    if ((t_int)(field) > 0) {
      rot = rot % (t_int)(field);  // rot might be negative
      if (rot != 0) {
        if (rot < 0) {
          // we need a real modulo operation yielding 0..field-1
          rot = rot + (t_int)(field);
          }
        g = M_gcd((t_int)(field), rot);
        d = (t_int)(field) / g;
        for (i = 0; i <= g-1; ++i) {
          k = i;
          for (j = 0; j <= d-2; ++j) {
            n = k + rot;
            if (n >= (t_int)(field)) {
              // avoid mod
              n = n - (t_int)(field);
              }
            x = bit_index_swap(x,
              (t_subword)(n+(t_int)(ofs)), (t_subword)(k+(t_int)(ofs)));
            k = n;
            }
          }
        }
      }
    return x;
    }

  // Transpose bit matrixes.
  // ld_fields: width of the bit fields
  // ld_col: ld(bit columns)
  // ld_row: ld(bit rows)
  // 2011-10-04 by Jasper L. Neumann
  static auto transpose(T x, t_subword ld_fields, t_subword ld_col, t_subword ld_row) -> T {

    return bit_index_ror(x, (t_int)(ld_col),
      ld_fields, (t_subword)((t_int)(ld_col)+(t_int)(ld_row)));
    }

  // pwr times shuffle(x, sw1, sw2)
  // See Hacker's Delight, 7.6/7.8 "Rearrangements and Index Transformations"
  // 2011-10-04 by Jasper L. Neumann
  static auto shuffle_power(T x, t_subword sw1, t_subword sw2, t_int pwr) -> T {

    return bit_index_ror(x, -pwr, sw1, (t_subword)((t_int)(sw2)-(t_int)(sw1)));
    }

  // pwr times unshuffle(x, sw1, sw2)
  // See Hacker's Delight, 7.6/7.8 "Rearrangements and Index Transformations"
  // 2011-10-04 by Jasper L. Neumann
  static auto unshuffle_power(T x, t_subword sw1, t_subword sw2, t_int pwr) -> T {

    return bit_index_ror(x, pwr, sw1, (t_subword)((t_int)(sw2)-(t_int)(sw1)));
    }

  // Swap all subwords of given levels.
  // See Hacker's Delight, 7.1 "Generalized Bit Reversal"
  // k: set of t_subword, i.e. one bit per subword size.
  // This is a candidate for a new instruction.
  static auto general_reverse_bits(T x, t_int k) -> T {

    t_int i, s;
    T m;

    s = 1;
    for (i = 0; i <= ld_bits-1; ++i) {  // UNROLL
      // s = 1 << i;
      if ((k & s) != 0) {
        // x = bit_index_complement(x, s);
        m = a_even[i];
        x = t_my_bits::bit_permute_step_simple(x, m, s);
        }
      s = s << 1;
      }
    return x;
    }

  // Exchange byte order.
  // This can be expressed in assembler:
  // bits = 8: n/a
  // bits = 16: "xchg al, ah" or "rol ax, 16"
  // bits = 32: "bswap eax"
  // bits = 64: "bswap rax"
  // bits = 128: "xchg rax, rdx; bswap rax; bswap rdx"
  // INLINE
  static auto bswap(T x) -> T {

    return general_reverse_bits(x, ~7);
    }

  // Swap by primitives.
  // Generalization of general_reverse_bits.
  // See "Matters Computational" by Joerg Arndt, "A restricted method"
  // See Hacker's Delight, 7.1 "Generalized Bit Reversal"
  // Bit-parallel implementation: 2011-02 by Jasper L. Neumann
  // O(ld_bits)
  static auto prim_swap(T x, T m) -> T {

    t_int i, s;
    T q;

    if ((m & hi_bit) != 0) {  // highest bit set?
      // normal operation
      for (i = ld_bits-1; i >= 0; --i) {  // UNROLL
        q = m & a_prim[i];
        s = 1 << i;
        q = (q << 1) - (q >> (s-1));  // broadcast bits
        x = t_my_bits::bit_permute_step(x, q, s);
        }
      }
    else {
      // inverse operation
      // same as above but with reversed loop
      for (i = 0; i <= ld_bits-1; ++i) {  // UNROLL
        q = m & a_prim[i];
        s = 1 << i;
        q = (q << 1) - (q >> (s-1));  // broadcast bits
        x = t_my_bits::bit_permute_step(x, q, s);
        }
      }
    return x;
    }

  // Shuffle/zip/interlace entities.
  // See Hacker's Delight, 7.2 "Shuffling Bits"
  // sw1: log_2(subword_length): entities to move
  // sw2: log_2(word_length): moving area
  // 0 <= sw1 < sw2 <= ld_bits
  // Example: sw1=2, sw2=5: Shuffle nibbles in dword
  // = shuffle_power(x, sw1, sw2, 1)
  // O(sw2-sw1)
  static auto shuffle(T x, t_subword sw1, t_subword sw2) -> T {

    t_int i, s;

    if ((t_int)(sw2) >= 2) {
      for (i = (t_int)(sw2)-2; i >= (t_int)(sw1); --i) {  // UNROLL?
        // x = bit_index_swap(x, i+1, i);
        s = 1 << i;
        x = t_my_bits::bit_permute_step(x, a_shuffle[i], s);
        }
      }
    return x;
    }

  // Unshuffle/unzip/uninterlace entities.
  // See Hacker's Delight, 7.2 "Shuffling Bits"
  // sw1: log_2(subword_length): entities to move
  // sw2: log_2(word_length): moving area
  // 0 <= sw1 < sw2 <= ld_bits
  // Example: sw1=0, sw2=3: Unshuffle bits in bytes
  // = unshuffle_power(x, sw1, sw2, 1)
  // O(sw2-sw1)
  static auto unshuffle(T x, t_subword sw1, t_subword sw2) -> T {

    t_int i, s;

    if ((t_int)(sw2) >= 2) {
      for (i = (t_int)(sw1); i <= (t_int)(sw2)-2; ++i) {  // UNROLL?
        // x = bit_index_swap(x, i+1, i);
        s = 1 << i;
        x = t_my_bits::bit_permute_step(x, a_shuffle[i], s);
        }
      }
    return x;
    }

  // CEF operations (via aux object).

  static auto compress_flip_right(t_subword sw, T x, T m) -> T {

    t_cef<T> cef;

    cef.gen_right(sw,m);
    return cef.compress(x);
    }

  static auto expand_flip_right(t_subword sw, T x, T m) -> T {

    t_cef<T> cef;

    cef.gen_right(sw,m);
    return cef.expand(x);
    }

  static auto compress_flip_left(t_subword sw, T x, T m) -> T {

    t_cef<T> cef;

    cef.gen_left(sw,m);
    return cef.compress(x);
    }

  static auto expand_flip_left(t_subword sw, T x, T m) -> T {

    t_cef<T> cef;

    cef.gen_left(sw,m);
    return cef.expand(x);
    }

  // CE operations (via aux object).

  static auto compress_right(t_subword sw, T x, T m) -> T {

    t_ce_right<T> ce;

    ce.gen(sw,m);
    return ce.compress(x);
    }

  static auto expand_right(t_subword sw, T x, T m) -> T {

    t_ce_right<T> ce;

    ce.gen(sw,m);
    return ce.expand(x);
    }

  static auto compress_left(t_subword sw, T x, T m) -> T {

    t_ce_left<T> ce;

    ce.gen(sw,m);
    return ce.compress(x);
    }

  static auto expand_left(t_subword sw, T x, T m) -> T {

    t_ce_left<T> ce;

    ce.gen(sw,m);
    return ce.expand(x);
    }

  };

// This structure is used to hold the configuration of
// butterfly-based operations as well as compress and expand.
template<typename T> class t_bfly_base
{
protected:
  typedef t_bits<T> t_my_bits;
public:

  // Imports, types, asserts.
  typedef typename t_my_bits::TU TU;
  typedef typename t_my_bits::TS TS;

  static constexpr const t_uint ld_bits = t_my_bits::ld_bits;
  static constexpr const t_uint bits = t_my_bits::bits;
  static constexpr const T all_bits = t_my_bits::all_bits;
  static constexpr const T lo_bit = t_my_bits::lo_bit;
  static constexpr const T hi_bit = t_my_bits::hi_bit;

  static constexpr const TU *a_element = t_my_bits::a_element;
  static constexpr const TU *a_lo = t_my_bits::a_lo;
  static constexpr const TU *a_hi = t_my_bits::a_hi;
  static constexpr const TU *a_even = t_my_bits::a_even;
  static constexpr const TU *a_shuffle = t_my_bits::a_shuffle;
  static constexpr const TU *a_prim = t_my_bits::a_prim;

  static_assert((T)(-1) > 0, "(T)(-1) > 0");  // T must be unsigned.

  T cfg[ld_bits];  // butterfly configuration
  t_int lo_stage, hi_stage;

  t_bfly_base() {
    lo_stage = 0;
    hi_stage = ld_bits-1;
    }

  // Apply butterfly configured network on x
  auto bfly(T x) -> T {

    t_int stage;

    for (stage = hi_stage; stage >= lo_stage; --stage) {  // UNROLL
      x = t_simd<T>::butterfly((t_subword)(stage), x, this->cfg[stage]);
      }

    return x;
    }

  // Apply inverse butterfly configured network on x
  auto ibfly(T x) -> T {

    t_int stage;

    for (stage = lo_stage; stage <= hi_stage; ++stage) {  // UNROLL
      x = t_simd<T>::butterfly((t_subword)(stage), x, this->cfg[stage]);
      }

    return x;
    }

  // Return the parity of a permutation given by a butterfly network.
  // This is false for even parity and true for odd parity.
  auto bfly_parity() -> bool {

    t_int stage;
    T x;

    x = 0;
    for (stage = lo_stage; stage <= hi_stage; ++stage) {  // UNROLL
      x = x ^ this->cfg[stage];
      }

    return (nr_1bits(x) & 1) != 0;
    }

  };

// Compress/Expand flip left/right
template<typename T> class t_cef_base
: public t_bfly_base<T>
{
protected:
  typedef t_bits<T> t_my_bits;
public:

  // Imports, types, asserts.
  typedef typename t_my_bits::TU TU;
  typedef typename t_my_bits::TS TS;

  static constexpr const t_uint ld_bits = t_my_bits::ld_bits;
  static constexpr const t_uint bits = t_my_bits::bits;
  static constexpr const T all_bits = t_my_bits::all_bits;
  static constexpr const T lo_bit = t_my_bits::lo_bit;
  static constexpr const T hi_bit = t_my_bits::hi_bit;

  static constexpr const TU *a_element = t_my_bits::a_element;
  static constexpr const TU *a_lo = t_my_bits::a_lo;
  static constexpr const TU *a_hi = t_my_bits::a_hi;
  static constexpr const TU *a_even = t_my_bits::a_even;
  static constexpr const TU *a_shuffle = t_my_bits::a_shuffle;
  static constexpr const TU *a_prim = t_my_bits::a_prim;

  static_assert((T)(-1) > 0, "(T)(-1) > 0");  // T must be unsigned.

  // T mask;
  // enum (virgin, cef_right, cef_left) kind;
  // t_subword sw;

  // t_cef_base() {}

  // Scatter/gather-flip, compress/expand+flip.
  // Generate configuration for [inverse] butterfly network.
  // To compress use ibfly.
  // To expand use bfly.
  // Bit-parallel implementation: 2011-02 by Jasper L. Neumann
  void gen_cef_right(t_subword sw, T m) {

    T t, mm, m0;
    t_int i, j, s;

    // this->mask = m;
    this->lo_stage = 0;
    this->hi_stage = (t_int)(sw) - 1;
    // for (i = 0; i <= ld_bits-1; ++i) {  // UNROLL
    //   this->cfg[i] = 0;
    //   }
    if ((t_int)(sw) > 0) {  // UNFOLD all cases of sw
      m = ~m;
      m0 = a_lo[(t_int)(sw)];

      for (i = 0; i <= (t_int)(sw)-1; ++i) {  // UNROLL
        t = m;
        for (j = i; j <= (t_int)(sw)-1; ++j) {  // UNROLL
          s = 1 << j;  // j ones; j=2: 1 + ~(1 << 4): 11101111+1=11110000
          mm = ~(m0 << s) + m0;  // mask to hinder shifting into other subwords
          m = m ^ ((m << s) & mm);
          }
        s = 1 << i;
        m = m & a_even[i];       // my bfly looks on low bits
        this->cfg[i] = m;
        m = (t ^ (t >> s)) & m;  // do a butterfly op
        m = (t ^ m) ^ (m << s);
        }
      }
    }

  // Scatter/gather-flip, compress/expand+flip.
  // Generate configuration for [inverse] butterfly network.
  // To compress use ibfly.
  // To expand use bfly.
  // Bit-parallel implementation: 2011-02 by Jasper L. Neumann
  void gen_cef_left(t_subword sw, T m) {

    T t, mm, m0, m1;
    t_int i, j, s;

    // this->mask = m;
    this->lo_stage = 0;
    this->hi_stage = (t_int)(sw) - 1;
    // for (i = 0; i <= ld_bits-1; ++i) {  // UNROLL
    //   this->cfg[i] = 0;
    //   }
    if ((t_int)(sw) > 0) {  // UNFOLD all cases of sw
      m = ~m;
      m1 = a_lo[(t_int)(sw)];
      m0 = a_hi[(t_int)(sw)];  // m0 = (m1 >> 1) + hi_bit;
      for (i = 0; i <= (t_int)(sw)-1; ++i) {  // UNROLL
        t = m;
        for (j = i; j <= (t_int)(sw)-1; ++j) {  // UNROLL
          s = 1 << j;  // j ones; j=2: 1 + ~(1 << 4): 11101111+1=11110000
          mm = (m0 >> (s-1)) - m1;  // mask to hinder shifting into other subwords
          m = m ^ ((m >> s) & mm);
          }
        s = 1 << i;
        m = (m >> s) & a_even[i];   // my bfly looks on low bits
        this->cfg[i] = m;           // so shift into place
        m = (t ^ (t >> s)) & m;     // do a butterfly op
        m = (t ^ m) ^ (m << s);
        }
      }
    }

  void gen_right(t_subword sw, T m) { gen_cef_right(sw,m); }
  void gen_left(t_subword sw, T m) { gen_cef_left(sw,m); }
  auto compress(T x) -> T { return this->ibfly(x); }
  auto expand(T x) -> T { return this->bfly(x); }
  };

// Compress/Expand right, rest filled with 0
template<typename T> class t_ce_right_base
: public t_bfly_base<T> {
protected:
  typedef t_bits<T> t_my_bits;
public:

  // Imports, types, asserts.
  typedef typename t_my_bits::TU TU;
  typedef typename t_my_bits::TS TS;

  static constexpr const t_uint ld_bits = t_my_bits::ld_bits;
  static constexpr const t_uint bits = t_my_bits::bits;
  static constexpr const T all_bits = t_my_bits::all_bits;
  static constexpr const T lo_bit = t_my_bits::lo_bit;
  static constexpr const T hi_bit = t_my_bits::hi_bit;

  static constexpr const TU *a_element = t_my_bits::a_element;
  static constexpr const TU *a_lo = t_my_bits::a_lo;
  static constexpr const TU *a_hi = t_my_bits::a_hi;
  static constexpr const TU *a_even = t_my_bits::a_even;
  static constexpr const TU *a_shuffle = t_my_bits::a_shuffle;
  static constexpr const TU *a_prim = t_my_bits::a_prim;

  static_assert((T)(-1) > 0, "(T)(-1) > 0");  // T must be unsigned.

  T mask;
  // enum (virgin, initialized) kind;
  // t_subword sw;

  // t_ce_right_base() {}

  // See Hacker's Delight, 7.4 "Compress, or Generalized Extract"
  // See Hacker's Delight, 7.5 "Expand, or Generalized Insert"
  // To compress use apply_compress_right.
  // To expand use apply_expand_right.
  // UNFOLD all cases of sw
  void gen_ce_right(t_subword sw, T m) {

    T mk, mp, mv, mm, m0;
    t_int i, j, s;

    this->mask = m;                       // Save original mask
    this->lo_stage = 0;
    this->hi_stage = (t_int)(sw) - 1;
    // for (i = 0; i <= ld_bits-1; ++i) {  // UNROLL
    //   this->cfg[i] = 0;
    //   }
    if ((t_int)(sw) > 0) {
      m0 = a_lo[(t_int)(sw)];
      mk = ((~m) << 1) & ~m0;             // We will count 0's to right
      for (i = 0; i <= (t_int)(sw)-1; ++i) {  // UNROLL
        mp = mk;                          // Parallel suffix
        for (j = 0; j <= (t_int)(sw)-1; ++j) {  // UNROLL
          s = 1 << j;
          mm = ~(m0 << s)+m0;
          mp = mp ^ ((mp << s) & mm);     // Masking not needed for sw=ld_bits
          }
        mv = mp & m;                      // Bits to move
        this->cfg[i] = mv;
        m = (m ^ mv) | (mv >> (1 << i));  // Compress m
        mk = mk & ~mp;
        }
      }
    }

  // See Hacker's Delight, 7.4 "Compress, or Generalized Extract"
  // This should be configured by gen_ce_right.
  auto apply_compress_right(T x) -> T {

    T t;
    t_int i, s;

    x = x & this->mask;  // Clear irrelevant bits

    s = 1 << this->lo_stage;
    for (i = this->lo_stage; i <= this->hi_stage; ++i) {  // UNROLL
      // s = 1 << i;
      t = x & this->cfg[i];
      x = (x ^ t) | (t >> s);  // Compress x (or ^)
      s = s << 1;
      }

    return x;
    }

  // See Hacker's Delight, 7.4 "Compress, or Generalized Extract"
  // See Hacker's Delight, 7.5 "Expand, or Generalized Insert"
  // This should be configured by gen_ce_right.
  // (a & b) | (~a & c) => ((b ^ c) & a) ^ c
  auto apply_expand_right(T x) -> T {

    t_int i, s;

    s = 1 << this->hi_stage;
    for (i = this->hi_stage; i >= this->lo_stage; --i) {  // UNROLL
      // s = 1 << i;
      x = (((x << s) ^ x) & this->cfg[i]) ^ x;
      s = s >> 1;
      }

    return x & this->mask;  // Clear out extraneous bits
    }

  void gen(t_subword sw, T m) { gen_ce_right(sw,m); }
  auto compress(T x) -> T { return apply_compress_right(x); }
  auto expand(T x) -> T { return apply_expand_right(x); }
  };

// Compress/Expand left, rest filled with 0
template<typename T> class t_ce_left_base
: public t_bfly_base<T> {
protected:
  typedef t_bits<T> t_my_bits;
public:

  // Imports, types, asserts.
  typedef typename t_my_bits::TU TU;
  typedef typename t_my_bits::TS TS;

  static constexpr const t_uint ld_bits = t_my_bits::ld_bits;
  static constexpr const t_uint bits = t_my_bits::bits;
  static constexpr const T all_bits = t_my_bits::all_bits;
  static constexpr const T lo_bit = t_my_bits::lo_bit;
  static constexpr const T hi_bit = t_my_bits::hi_bit;

  static constexpr const TU *a_element = t_my_bits::a_element;
  static constexpr const TU *a_lo = t_my_bits::a_lo;
  static constexpr const TU *a_hi = t_my_bits::a_hi;
  static constexpr const TU *a_even = t_my_bits::a_even;
  static constexpr const TU *a_shuffle = t_my_bits::a_shuffle;
  static constexpr const TU *a_prim = t_my_bits::a_prim;

  static_assert((T)(-1) > 0, "(T)(-1) > 0");  // T must be unsigned.

  T mask;
  // enum (virgin, initialized) kind;
  // t_subword sw;

  // t_ce_left_base() {}

  // See Hacker's Delight, 7.4 "Compress, or Generalized Extract"
  // See Hacker's Delight, 7.5 "Expand, or Generalized Insert"
  // To compress use apply_compress_left.
  // To expand use apply_expand_left.
  // UNFOLD all cases of sw
  void gen_ce_left(t_subword sw, T m) {

    T mk, mp, mv, mm, m0, m1;
    t_int i, j, s;

    this->mask = m;                       // Save original mask
    this->lo_stage = 0;
    this->hi_stage = (t_int)(sw) - 1;
    // for (i = 0; i <= ld_bits-1; ++i) {  // UNROLL
    //   this->cfg[i] = 0;
    //   }
    if ((t_int)(sw) > 0) {
      m1 = a_lo[(t_int)(sw)];
      m0 = a_hi[(t_int)(sw)];             // m0 = (m1 >> 1) + hi_bit;
      mk = ((~m) >> 1) & ~m0;             // We will count 0's to right
      for (i = 0; i <= (t_int)(sw)-1; ++i) {  // UNROLL
        mp = mk;                          // Parallel suffix
        for (j = 0; j <= (t_int)(sw)-1; ++j) {  // UNROLL
          s = 1 << j;
          mm = (m0 >> (s-1)) - m1;
          mp = mp ^ ((mp >> s) & mm);     // Masking not needed for sw=ld_bits
          }
        mv = mp & m;                      // Bits to move
        this->cfg[i] = mv;
        m = (m ^ mv) | (mv << (1 << i));  // Compress m
        mk = mk & ~mp;
        }
      }
    }

  // See Hacker's Delight, 7.4 "Compress, or Generalized Extract"
  // This should be configured by gen_ce_left.
  auto apply_compress_left(T x) -> T {

    T t;
    t_int i, s;

    x = x & this->mask;  // Clear irrelevant bits

    s = 1 << this->lo_stage;
    for (i = this->lo_stage; i <= this->hi_stage; ++i) {  // UNROLL
      // s = 1 << i;
      t = x & this->cfg[i];
      x = (x ^ t) | (t << s);  // Compress x (or ^)
      s = s << 1;
      }

    return x;
    }

  // See Hacker's Delight, 7.4 "Compress, or Generalized Extract"
  // See Hacker's Delight, 7.5 "Expand, or Generalized Insert"
  // This should be configured by gen_ce_left.
  // (a & b) | (~a & c) => ((b ^ c) & a) ^ c
  auto apply_expand_left(T x) -> T {

    t_int i, s;

    s = 1 << this->hi_stage;
    for (i = this->hi_stage; i >= this->lo_stage; --i) {  // UNROLL
      // s = 1 << i;
      x = (((x >> s) ^ x) & this->cfg[i]) ^ x;
      s = s >> 1;
      }

    return x & this->mask;  // Clear out extraneous bits
    }

  void gen(t_subword sw, T m) { gen_ce_left(sw,m); }
  auto compress(T x) -> T { return apply_compress_left(x); }
  auto expand(T x) -> T { return apply_expand_left(x); }
  };

// Faster variant of t_simd_base<T>::vrol/vror
// when using the same rotate counts repeatedly.
template<typename T> class t_vrot_base
: public t_bfly_base<T> {
protected:
  typedef t_bits<T> t_my_bits;
public:

  // Imports, types, asserts.
  typedef typename t_my_bits::TU TU;
  typedef typename t_my_bits::TS TS;

  static constexpr const t_uint ld_bits = t_my_bits::ld_bits;
  static constexpr const t_uint bits = t_my_bits::bits;
  static constexpr const T all_bits = t_my_bits::all_bits;
  static constexpr const T lo_bit = t_my_bits::lo_bit;
  static constexpr const T hi_bit = t_my_bits::hi_bit;

  static constexpr const TU *a_element = t_my_bits::a_element;
  static constexpr const TU *a_lo = t_my_bits::a_lo;
  static constexpr const TU *a_hi = t_my_bits::a_hi;
  static constexpr const TU *a_even = t_my_bits::a_even;
  static constexpr const TU *a_shuffle = t_my_bits::a_shuffle;
  static constexpr const TU *a_prim = t_my_bits::a_prim;

  static_assert((T)(-1) > 0, "(T)(-1) > 0");  // T must be unsigned.

  // T mask;
  // enum (virgin, initialized) kind;
  // t_subword sw;

  // t_ce_left_base() {}

  // Field variable rotate.
  // Generate configuration for [inverse] butterfly network.
  // Simulate rolc for every subword of the masks.
  // Bit-parallel implementation: 2011-09-14 by Jasper L. Neumann
  void gen_vrol(t_subword sw, T rot) {

    T t,x,y,lo;
    t_int i,s,my_sw;

    my_sw = (t_int)(sw);
    // this->mask = rot;
    this->lo_stage = 0;
    this->hi_stage = (t_int)(sw) - 1;
    // for (i = 0; i <= ld_bits-1; ++i) {  // UNROLL
    //   this->cfg[i] = 0;
    //   }
    switch (my_sw) {
      case 0: {
        break;  // nothing else to do
        }
      case 1: {
        this->cfg[0] = rot & a_even[0];
        break;
        }
      default: {  // UNFOLD all cases of my_sw
        // this code does not work for my_sw < 1
        // shift a single 1 to the left...
        lo = a_lo[my_sw];
        my_sw = my_sw-1;
        x = lo;
        for (i = 0; i <= (t_int)(my_sw)-1; ++i) {  // UNROLL
          y = (rot >> i) & lo;  // rot bit to #0
          s = 1 << i;
          // (lo_bit << s) - 1 == a_element[i]
          // t = x & (y * ((lo_bit << s) - 1));  // get offending bit(s)
          t = x & ((y << s) - y);  // get offending bit(s)
          x = (x ^ t) ^ (t << s);  // swap subwords if bit set
          }
        // x is e.g. 1000 here (1 << 3), we want 3 ones
        x = x - lo;  // sub 1 to yield 1-string, e.g. 1000-1=111
        y = (rot >> my_sw) & lo;
        s = 1 << my_sw;
        // (lo_bit << s) - 1 == a_element[my_sw]
        // x = x ^ (y * ((lo_bit << s) - 1));  // invert if rot<0
        x = x ^ ((y << s) - y);  // invert if rot<0
        x = x & a_even[my_sw];  // finalize rolc
        this->cfg[my_sw] = x;
        // and now for the lower stages...
        for (i = (t_int)(my_sw)-1; i >= 0; --i) {  // my_sw-1..0; UNROLL
          // xor 2 columns together to get new rolc for the stage...
          s = 1 << i;
          x = (x ^ (x >> s)) & a_even[i];
          x = x | (x << (s * 2));  // ...and spread into places
          this->cfg[i] = x;
          }
        break;
        }
      }
    }

  auto vrol(T x) -> T { return this->ibfly(x); }
  auto vror(T x) -> T { return this->bfly(x); }
  };


//////
// Concrete template classes.

constexpr const t_8u swar_element_8[] = {
  0x01, 0x03, 0x0f, 0xff };
constexpr const t_8u swar_lo_8[] = {
  0xff, 0x55, 0x11, 0x01 };
constexpr const t_8u swar_hi_8[] = {
  0xff, 0xaa, 0x88, 0x80 };
constexpr const t_8u swar_even_8[] = {
  0x55, 0x33, 0x0f, 0xff };
constexpr const t_8u swar_shuffle_8[] = {
  0x22, 0x0c };
constexpr const t_8u swar_prim_8[] = {
  0x55, 0x22, 0x08 };

template<> class t_bits<t_8u>:
public t_bits_base<
  t_8u, t_8s, 3,
  swar_element_8, swar_lo_8, swar_hi_8,
  swar_even_8, swar_shuffle_8, swar_prim_8> {};

template<> class t_simd<t_8u>:
public t_simd_base<t_8u> {};

template<> class t_bfly<t_8u>:
public t_bfly_base<t_8u> {};

template<> class t_cef<t_8u>:
public t_cef_base<t_8u> {};

template<> class t_ce_right<t_8u>:
public t_ce_right_base<t_8u> {};

template<> class t_ce_left<t_8u>:
public t_ce_left_base<t_8u> {};

template<> class t_vrot<t_8u>:
public t_vrot_base<t_8u> {};

typedef t_bits<t_8u>     t_bits_8;
typedef t_simd<t_8u>     t_simd_8;
typedef t_bfly<t_8u>     t_bfly_8;
typedef t_cef<t_8u>      t_cef_8;
typedef t_ce_right<t_8u> t_ce_right_8;
typedef t_ce_left<t_8u>  t_ce_left_8;
typedef t_vrot<t_8u>     t_vrot_8;

constexpr const t_16u swar_element_16[] = {
  0x0001, 0x0003, 0x000f, 0x00ff, 0xffff };
constexpr const t_16u swar_lo_16[] = {
  0xffff, 0x5555, 0x1111, 0x0101, 0x0001 };
constexpr const t_16u swar_hi_16[] = {
  0xffff, 0xaaaa, 0x8888, 0x8080, 0x8000 };
constexpr const t_16u swar_even_16[] = {
  0x5555, 0x3333, 0x0f0f, 0x00ff, 0xffff };
constexpr const t_16u swar_shuffle_16[] = {
  0x2222, 0x0c0c, 0x00f0 };
constexpr const t_16u swar_prim_16[] = {
  0x5555, 0x2222, 0x0808, 0x0080 };

template<> class t_bits<t_16u>:
public t_bits_base<
  t_16u, t_16s, 4,
  swar_element_16, swar_lo_16, swar_hi_16,
  swar_even_16, swar_shuffle_16, swar_prim_16> {};

template<> class t_simd<t_16u>:
public t_simd_base<t_16u> {};

template<> class t_bfly<t_16u>:
public t_bfly_base<t_16u> {};

template<> class t_cef<t_16u>:
public t_cef_base<t_16u> {};

template<> class t_ce_right<t_16u>:
public t_ce_right_base<t_16u> {};

template<> class t_ce_left<t_16u>:
public t_ce_left_base<t_16u> {};

template<> class t_vrot<t_16u>:
public t_vrot_base<t_16u> {};

typedef t_bits<t_16u>     t_bits_16;
typedef t_simd<t_16u>     t_simd_16;
typedef t_bfly<t_16u>     t_bfly_16;
typedef t_cef<t_16u>      t_cef_16;
typedef t_ce_right<t_16u> t_ce_right_16;
typedef t_ce_left<t_16u>  t_ce_left_16;
typedef t_vrot<t_16u>     t_vrot_16;

constexpr const t_32u swar_element_32[] = {
  0x00000001, 0x00000003, 0x0000000f, 0x000000ff, 0x0000ffff, 0xffffffff };
constexpr const t_32u swar_lo_32[] = {
  0xffffffff, 0x55555555, 0x11111111, 0x01010101, 0x00010001, 0x00000001 };
constexpr const t_32u swar_hi_32[] = {
  0xffffffff, 0xaaaaaaaa, 0x88888888, 0x80808080, 0x80008000, 0x80000000 };
constexpr const t_32u swar_even_32[] = {
  0x55555555, 0x33333333, 0x0f0f0f0f, 0x00ff00ff, 0x0000ffff, 0xffffffff };
constexpr const t_32u swar_shuffle_32[] = {
  0x22222222, 0x0c0c0c0c, 0x00f000f0, 0x0000ff00 };
constexpr const t_32u swar_prim_32[] = {
  0x55555555, 0x22222222, 0x08080808, 0x00800080, 0x00008000 };

template<> class t_bits<t_32u>:
public t_bits_base<
  t_32u, t_32s, 5,
  swar_element_32, swar_lo_32, swar_hi_32,
  swar_even_32, swar_shuffle_32, swar_prim_32> {};

template<> class t_simd<t_32u>:
public t_simd_base<t_32u> {};

template<> class t_bfly<t_32u>:
public t_bfly_base<t_32u> {};

template<> class t_cef<t_32u>:
public t_cef_base<t_32u> {};

template<> class t_ce_right<t_32u>:
public t_ce_right_base<t_32u> {};

template<> class t_ce_left<t_32u>:
public t_ce_left_base<t_32u> {};

template<> class t_vrot<t_32u>:
public t_vrot_base<t_32u> {};

typedef t_bits<t_32u>     t_bits_32;
typedef t_simd<t_32u>     t_simd_32;
typedef t_bfly<t_32u>     t_bfly_32;
typedef t_cef<t_32u>      t_cef_32;
typedef t_ce_right<t_32u> t_ce_right_32;
typedef t_ce_left<t_32u>  t_ce_left_32;
typedef t_vrot<t_32u>     t_vrot_32;

constexpr const t_64u swar_element_64[] = {
  0x0000000000000001ULL, 0x0000000000000003ULL, 0x000000000000000fULL,
  0x00000000000000ffULL, 0x000000000000ffffULL, 0x00000000ffffffffULL,
  0xffffffffffffffffULL };
constexpr const t_64u swar_lo_64[] = {
  0xffffffffffffffffULL, 0x5555555555555555ULL, 0x1111111111111111ULL,
  0x0101010101010101ULL, 0x0001000100010001ULL, 0x0000000100000001ULL,
  0x0000000000000001ULL };
constexpr const t_64u swar_hi_64[] = {
  0xffffffffffffffffULL, 0xaaaaaaaaaaaaaaaaULL, 0x8888888888888888ULL,
  0x8080808080808080ULL, 0x8000800080008000ULL, 0x8000000080000000ULL,
  0x8000000000000000ULL };
constexpr const t_64u swar_even_64[] = {
  0x5555555555555555ULL, 0x3333333333333333ULL, 0x0f0f0f0f0f0f0f0fULL,
  0x00ff00ff00ff00ffULL, 0x0000ffff0000ffffULL, 0x00000000ffffffffULL,
  0xffffffffffffffffULL };
constexpr const t_64u swar_shuffle_64[] = {
  0x2222222222222222ULL, 0x0c0c0c0c0c0c0c0cULL, 0x00f000f000f000f0ULL,
  0x0000ff000000ff00ULL, 0x00000000ffff0000 };
constexpr const t_64u swar_prim_64[] = {
  0x5555555555555555ULL, 0x2222222222222222ULL, 0x0808080808080808ULL,
  0x0080008000800080ULL, 0x0000800000008000ULL, 0x0000000080000000ULL };

template<> class t_bits<t_64u>:
public t_bits_base<
  t_64u, t_64s, 6,
  swar_element_64, swar_lo_64, swar_hi_64,
  swar_even_64, swar_shuffle_64, swar_prim_64> {};

template<> class t_simd<t_64u>:
public t_simd_base<t_64u> {};

template<> class t_bfly<t_64u>:
public t_bfly_base<t_64u> {};

template<> class t_cef<t_64u>:
public t_cef_base<t_64u> {};

template<> class t_ce_right<t_64u>:
public t_ce_right_base<t_64u> {};

template<> class t_ce_left<t_64u>:
public t_ce_left_base<t_64u> {};

template<> class t_vrot<t_64u>:
public t_vrot_base<t_64u> {};

typedef t_bits<t_64u>     t_bits_64;
typedef t_simd<t_64u>     t_simd_64;
typedef t_bfly<t_64u>     t_bfly_64;
typedef t_cef<t_64u>      t_cef_64;
typedef t_ce_right<t_64u> t_ce_right_64;
typedef t_ce_left<t_64u>  t_ce_left_64;
typedef t_vrot<t_64u>     t_vrot_64;

#ifdef has_128

constexpr const t_128u swar_element_128[] = {
  (((t_128u)(0x0000000000000000ULL))<<64)+0x0000000000000001ULL,   // 0
  (((t_128u)(0x0000000000000000ULL))<<64)+0x0000000000000003ULL,   // 1
  (((t_128u)(0x0000000000000000ULL))<<64)+0x000000000000000fULL,   // 2
  (((t_128u)(0x0000000000000000ULL))<<64)+0x00000000000000ffULL,   // 3
  (((t_128u)(0x0000000000000000ULL))<<64)+0x000000000000ffffULL,   // 4
  (((t_128u)(0x0000000000000000ULL))<<64)+0x00000000ffffffffULL,   // 5
  (((t_128u)(0x0000000000000000ULL))<<64)+0xffffffffffffffffULL,   // 6
  (((t_128u)(0xffffffffffffffffULL))<<64)+0xffffffffffffffffULL};  // 7
constexpr const t_128u swar_lo_128[] = {
  (((t_128u)(0xffffffffffffffffULL))<<64)+0xffffffffffffffffULL,   // 0
  (((t_128u)(0x5555555555555555ULL))<<64)+0x5555555555555555ULL,   // 1 => a_bfly_mask[0]
  (((t_128u)(0x1111111111111111ULL))<<64)+0x1111111111111111ULL,   // 2
  (((t_128u)(0x0101010101010101ULL))<<64)+0x0101010101010101ULL,   // 3
  (((t_128u)(0x0001000100010001ULL))<<64)+0x0001000100010001ULL,   // 4
  (((t_128u)(0x0000000100000001ULL))<<64)+0x0000000100000001ULL,   // 5
  (((t_128u)(0x0000000000000001ULL))<<64)+0x0000000000000001ULL,   // 6
  (((t_128u)(0x0000000000000000ULL))<<64)+0x0000000000000001ULL};  // 7
constexpr const t_128u swar_hi_128[] = {
  (((t_128u)(0xffffffffffffffffULL))<<64)+0xffffffffffffffffULL,   // 0
  (((t_128u)(0xaaaaaaaaaaaaaaaaULL))<<64)+0xaaaaaaaaaaaaaaaaULL,   // 1 => ~a_bfly_mask[0]
  (((t_128u)(0x8888888888888888ULL))<<64)+0x8888888888888888ULL,   // 2
  (((t_128u)(0x8080808080808080ULL))<<64)+0x8080808080808080ULL,   // 3
  (((t_128u)(0x8000800080008000ULL))<<64)+0x8000800080008000ULL,   // 4
  (((t_128u)(0x8000000080000000ULL))<<64)+0x8000000080000000ULL,   // 5
  (((t_128u)(0x8000000000000000ULL))<<64)+0x8000000000000000ULL,   // 6
  (((t_128u)(0x8000000000000000ULL))<<64)+0x0000000000000000ULL};  // 7
constexpr const t_128u swar_even_128[] = {
  (((t_128u)(0x5555555555555555ULL))<<64)+0x5555555555555555ULL,   // 0
  (((t_128u)(0x3333333333333333ULL))<<64)+0x3333333333333333ULL,   // 1
  (((t_128u)(0x0f0f0f0f0f0f0f0fULL))<<64)+0x0f0f0f0f0f0f0f0fULL,   // 2
  (((t_128u)(0x00ff00ff00ff00ffULL))<<64)+0x00ff00ff00ff00ffULL,   // 3
  (((t_128u)(0x0000ffff0000ffffULL))<<64)+0x0000ffff0000ffffULL,   // 4
  (((t_128u)(0x00000000ffffffffULL))<<64)+0x00000000ffffffffULL,   // 5
  (((t_128u)(0x0000000000000000ULL))<<64)+0xffffffffffffffffULL,   // 6
  (((t_128u)(0xffffffffffffffffULL))<<64)+0xffffffffffffffffULL};  // 7
constexpr const t_128u swar_shuffle_128[] = {
  (((t_128u)(0x2222222222222222ULL))<<64)+0x2222222222222222ULL,   // 0
  (((t_128u)(0x0c0c0c0c0c0c0c0cULL))<<64)+0x0c0c0c0c0c0c0c0cULL,   // 1
  (((t_128u)(0x00f000f000f000f0ULL))<<64)+0x00f000f000f000f0ULL,   // 2
  (((t_128u)(0x0000ff000000ff00ULL))<<64)+0x0000ff000000ff00ULL,   // 3
  (((t_128u)(0x00000000ffff0000ULL))<<64)+0x00000000ffff0000ULL,   // 4
  (((t_128u)(0x0000000000000000ULL))<<64)+0xffffffff00000000ULL};  // 5
constexpr const t_128u swar_prim_128[] = {
  (((t_128u)(0x5555555555555555ULL))<<64)+0x5555555555555555ULL,   // 0
  (((t_128u)(0x2222222222222222ULL))<<64)+0x2222222222222222ULL,   // 1
  (((t_128u)(0x0808080808080808ULL))<<64)+0x0808080808080808ULL,   // 2
  (((t_128u)(0x0080008000800080ULL))<<64)+0x0080008000800080ULL,   // 3
  (((t_128u)(0x0000800000008000ULL))<<64)+0x0000800000008000ULL,   // 5
  (((t_128u)(0x0000000080000000ULL))<<64)+0x0000000080000000ULL,   // 6
  (((t_128u)(0x0000000000000000ULL))<<64)+0x8000000000000000ULL};  // 7

template<> class t_bits<t_128u>:
public t_bits_base<
  t_128u, t_128s, 7,
  swar_element_128, swar_lo_128, swar_hi_128,
  swar_even_128, swar_shuffle_128, swar_prim_128> {};

template<> class t_simd<t_128u>:
public t_simd_base<t_128u> {};

template<> class t_bfly<t_128u>:
public t_bfly_base<t_128u> {};

template<> class t_cef<t_128u>:
public t_cef_base<t_128u> {};

template<> class t_ce_right<t_128u>:
public t_ce_right_base<t_128u> {};

template<> class t_ce_left<t_128u>:
public t_ce_left_base<t_128u> {};

template<> class t_vrot<t_128u>:
public t_vrot_base<t_128u> {};

typedef t_bits<t_128u>     t_bits_128;
typedef t_simd<t_128u>     t_simd_128;
typedef t_bfly<t_128u>     t_bfly_128;
typedef t_cef<t_128u>      t_cef_128;
typedef t_ce_right<t_128u> t_ce_right_128;
typedef t_ce_left<t_128u>  t_ce_left_128;
typedef t_vrot<t_128u>     t_vrot_128;

#endif


//////
// Tests.

/////// Tests ///////

#include <iostream>
#include <stdlib.h>

#define loop_count 10000

#ifdef has_128
typedef t_128s t_hugeint;
typedef t_128u t_hugeuint;
#else
typedef t_64s t_hugeint;
typedef t_64u t_hugeuint;
#endif

#define M_avgd_u(x, y) (((t_hugeuint)(x)+(t_hugeuint)(y))>>1)
                    // (((x) & (y)) + ((((x) ^ (y)) & ~1) >> 1))
#define M_avgu_u(x, y) (((t_hugeuint)(x)+(t_hugeuint)(y)+1)>>1)
                    // (((x) | (y)) - ((((x) ^ (y)) & ~1) >> 1))


//////
// Operations on low bits with specified bit count.

// Calculate the mask for b bits to zero out the remaining bits.
auto mask_ex(t_uint b) -> t_hugeint {

  if (b == sizeof(t_hugeint)*CHAR_BIT)
    return -1;
  else
    return ((t_hugeint)(1) << b) - 1;
  }

// Rotate right low b bits of x by r;
// remaining bits are zeroed out.
// = rol_ex(x,-r,b);
auto ror_ex(t_hugeint x, t_int r, t_uint b) -> t_hugeint {

  t_hugeint mask;

  if (b == 0) {
    return 0;
    }
  else {
    r = r % (t_int)(b);
    if (r == 0) {
      // Prevent shifting by b-r >= b.
      return x;
      }
    if (r < 0)
      r = r + b;
    if (b == sizeof(t_hugeint)*CHAR_BIT)
      return ((t_hugeuint)(x) >> r) | (x << (sizeof(t_hugeint)*CHAR_BIT-r));
    else {
      mask = ((t_hugeint)(1) << b)-1;
      x = x & mask;
      x = ((t_hugeuint)(x) >> r) | (x << (b-r));
      return x & mask;
      }
    }
  }

// Rotate left low b bits of x by r;
// remaining bits are zeroed out.
// = ror_ex(x,-r,b);
auto rol_ex(t_hugeint x, t_int r, t_uint b) -> t_hugeint {

  t_hugeint mask;

  if (b == 0) {
    return 0;
    }
  else {
    r = r % (t_int)(b);
    if (r == 0) {
      // Prevent shifting by b-r >= b.
      return x;
      }
    if (r < 0)
      r = r + b;
    if (b == sizeof(t_hugeint)*CHAR_BIT)
      return (x << r) | ((t_hugeuint)(x) >> (sizeof(t_hugeint)*CHAR_BIT-r));
    else {
      mask = ((t_hugeint)(1) << b)-1;
      x = x & mask;
      x = (x << r) | ((t_hugeuint)(x) >> (b-r));
      return x & mask;
      }
    }
  }

// Shift right low b bits of x by r;
// remaining bits are zeroed out.
auto shr_ex(t_hugeint x, t_uint r, t_uint b) -> t_hugeint {

  if (r >= b)
    return 0;
  else {
    x = x & mask_ex(b);
    return (t_hugeuint)(x) >> r;
    }
  }

// Shift left low b bits of x by r;
// remaining bits are zeroed out.
auto shl_ex(t_hugeint x, t_uint r, t_uint b) -> t_hugeint {

  if (r >= b)
    return 0;
  else {
    x = x << r;
    x = x & mask_ex(b);
    return x;
    }
  }

// Arithmetically (duplicating the MSB) shift right low b bits of x by r;
// remaining bits are sign extended.
auto sar_ex(t_hugeint x, t_uint r, t_uint b) -> t_hugeint {

  t_hugeint mask;

  if (b == 0)
    return 0;
  else if (r >= b) {
    if (M_odd(x >> (b-1)))
      return -1;
    else
      return 0;
    }
  else {
    mask = mask_ex(b);
    if (M_odd(x >> (b-1)))
      return ((t_hugeuint)(x) >> r) | ~((t_hugeuint)(mask) >> r);  // unsigned >>!
    else
      return ((t_hugeuint)(x) & mask) >> r;
    }
  }

// Shift left low b bits of x by r duplicating the LSB;
// remaining bits are zeroed out.
auto sal_ex(t_hugeint x, t_uint r, t_uint b) -> t_hugeint {

  t_hugeint mask;

  if (b == 0)
    return 0;
  else {
    mask = mask_ex(b);
    if (r >= b) {
      if (M_odd(x))
        return mask;
      else
        return 0;
      }
    else if (M_odd(x))
      return (x << r) | ~(mask << r);
    else
      return (x & mask) << r;
    }
  }

// Shift right low b bits of x by r shifting in 1 bits;
// remaining bits are zeroed out.
auto shr1_ex(t_hugeint x, t_uint r, t_uint b) -> t_hugeint {

  if (r >= b)
    return mask_ex(b);
  else {
    if (r != 0) {
      x = (t_hugeuint)(x) >> r;
      x = x | ~(((t_hugeint)(1) << (b-r))-1);
      }
    x = x & mask_ex(b);
    return x;
    }
  }

// Shift left low b bits of x by r shifting in 1 bits;
// remaining bits are zeroed out.
auto shl1_ex(t_hugeint x, t_uint r, t_uint b) -> t_hugeint {

  if (r >= b)
    return mask_ex(b);
  else {
    x = x << r;
    x = x | (((t_hugeint)(1) << r)-1);
    x = x & mask_ex(b);
    return x;
    }
  }

// Sign extended the low b bits of x.
auto sign_extend(t_hugeuint x, t_uint b) -> t_hugeint {
  return t_bits<t_hugeuint>::sign_extend(x, b);
  }


//////
// Extra functions.

auto clamp_u_sw(t_subword sw, t_hugeint x) -> t_hugeint {

  t_hugeint m;

  m = t_bits<t_hugeuint>::shl_safe(1, 1 << (t_int)(sw))-1;
  if (x<0)
    return 0;
  else if (x>m)
    return m;
  else
    return x;
  }

auto clamp_s_sw(t_subword sw, t_hugeint x) -> t_hugeint {

  t_hugeint m;

  m = (t_bits<t_hugeuint>::shl_safe(1, 1 << (t_int)(sw))-1)>>1;
  if (x>=m)
    return m;
  else if (x<-m-1)
    return -m-1;
  else
    return x;
  }

template<typename T>
auto random_bits() -> T {
  T res;
  res=0;
  for (int i=0; i<(t_int)(sizeof(T)); ++i) {
    res=res*256+rand();
    }
  return res;
  }

auto random_int(t_uint x) -> t_uint {
  return rand() % x;
  }

template<typename T>
auto nr_1bits(T x) -> int {
  int res;
  res = 0;
  while (x!=0) {
    x = x & (x-1);
    ++res;
    }
  return res;
  }

template<typename T>
auto nr_leading_0bits(T x) -> int {
  return t_bits<t_hugeuint>::nr_leading_0bits(x);
  }

// swap bit string
// swap corresponding bit string (double blend)

// Rotate right the bits o..f-o+1 of x by r;
// remaining bits are kept.
// o+f must be <= #bits of t_hugeint
auto ror_ex2(t_hugeint x, t_int r, t_uint o, t_uint f) -> t_hugeint {

  t_hugeint mask1, mask2;

  if (f == 0)
    return x;

  mask1 = ((t_hugeint)(1) << f) - 1;
  mask2 = mask1 << o;

  return
    (ror_ex(x >> o, r, f) << o) |
    (x & ~mask2);
  }

// using namespace std;
#define cout std::cout
#define endl std::endl
#define hex std::hex

template<typename T>
static auto test_bits_base() -> bool {
  typedef t_bits<T> t_my_bits;
  const t_int bits = t_my_bits::bits;

  t_int i;
  T x, y, z;
  t_uint us1, us2;
  t_int ss1, ss2;

  cout << "test_bits_base" << endl;
  x = 0;
  y = 0;
  z = 0;
  for (i = 1; i<=loop_count; ++i) {
    us1 = random_int(bits*3+2);  // positive shift/rotate amount
    us2 = random_int(bits*3+2);  // positive shift/rotate amount
    ss1 = random_int(bits*6) - (bits)*3;  // rotate amount
    ss2 = random_int(bits*6) - (bits)*3;  // rotate amount

    // odd
    // sign_extend
    // blend

    // Rotating.
    // rol
    // ror

    // Shifting.
    // shl_fast
    // shl_safe
    // shr_fast
    // shr_safe
    // sar_fast
    // sar_safe

    // Counting bits.
    // nr_1bits
    // nr_0bits
    // nr_trailing_0bits
    // nr_trailing_1bits
    // nr_leading_0bits
    // nr_leading_1bits

    // gray_code
    // inv_gray_code
    // is_contiguous_1bits
    // tbm

    // Permuting.
    // bit_permute_step
    // bit_permute_step_simple
    // bit_permute_step2
    // bit_permute_step_rot
    // bit_permute_step2_rot

    // isign
    // xsign

    if (M_odd(x)) {
      if ((T)(x * t_my_bits::mul_inv(x)) != 1) {
        cout << "t_my_bits::mul_inv" << endl;
        return false;
        }
      }
    else {
      if (t_my_bits::mul_inv(x) != 0) {
        cout << "t_my_bits::mul_inv" << endl;
        return false;
        }
      }

    // cl_mat_inv

    // Carry-less multiplication.

    // associative: a*(b*c) = (a*b)*c
    if (t_my_bits::cl_mul(x, t_my_bits::cl_mul(y, z)) !=
        t_my_bits::cl_mul(t_my_bits::cl_mul(x, y), z) ) {
      cout << "t_my_bits::cl_mul" << endl;
      return false;
      }

    // commutative: a*b = b*a
    if (t_my_bits::cl_mul(x, y) !=
        t_my_bits::cl_mul(y, x) ) {
      cout << "t_my_bits::cl_mul" << endl;
      return false;
      }

    // distributive: a*(b^c) = (a*b)^(a*c)
    if (t_my_bits::cl_mul(x, y ^ z) !=
        (T)(t_my_bits::cl_mul(x, y) ^ t_my_bits::cl_mul(x, z)) ) {
      cout << "t_my_bits::cl_mul" << endl;
      return false;
      }

    // - neutral element: 1*a = a
    if (t_my_bits::cl_mul(1, x) != x) {
      cout << "t_my_bits::cl_mul" << endl;
      return false;
      }

    // - zero element: 0*a = 0
    if (t_my_bits::cl_mul(0, x) != 0) {
      cout << "t_my_bits::cl_mul" << endl;
      return false;
      }

    // shift function: shl_safe(a,x)*shl_safe(b,y) = shl_safe(a*b,x+y)
    if (t_my_bits::cl_mul(t_my_bits::shl_safe(x, us1),
                          t_my_bits::shl_safe(y, us2)) !=
        t_my_bits::shl_safe(t_my_bits::cl_mul(x, y), us1 + us2) ) {
      cout << "t_my_bits::cl_mul" << endl;
      return false;
      }

    if (M_odd(x)) {
      if (t_my_bits::cl_mul(x, t_my_bits::cl_mul_inv(x)) != 1) {
        cout << "t_my_bits::cl_mul_inv" << endl;
        return false;
        }
      }
    else {
      if (t_my_bits::cl_mul_inv(x) != 0) {
        cout << "t_my_bits::cl_mul_inv" << endl;
        return false;
        }
      }

    // cl_power

    // Cyclic carry-less multiplication.

    // associative: a*(b*c) = (a*b)*c
    if (t_my_bits::ccl_mul(x, t_my_bits::ccl_mul(y, z)) !=
        t_my_bits::ccl_mul(t_my_bits::ccl_mul(x, y), z) ) {
      cout << "t_my_bits::ccl_mul" << endl;
      return false;
      }

    // commutative: a*b = b*a
    if (t_my_bits::ccl_mul(x, y) !=
        t_my_bits::ccl_mul(y, x) ) {
      cout << "t_my_bits::ccl_mul" << endl;
      return false;
      }

    // distributive: a*(b^c) = (a*b)^(a*c)
    if (t_my_bits::ccl_mul(x, y ^ z) !=
        (T)(t_my_bits::ccl_mul(x, y) ^ t_my_bits::ccl_mul(x, z)) ) {
      cout << "t_my_bits::ccl_mul" << endl;
      return false;
      }

    // - neutral element: 1*a = a
    if (t_my_bits::ccl_mul(1, x) != x) {
      cout << "t_my_bits::ccl_mul" << endl;
      return false;
      }

    // - zero element: 0*a = 0
    if (t_my_bits::ccl_mul(0, x) != 0) {
      cout << "t_my_bits::ccl_mul" << endl;
      return false;
      }

    // rotate function: rot(a,x)*rot(b,y) = rot(a*b,x+y)
    if (t_my_bits::ccl_mul(t_my_bits::rol(x, ss1),
                           t_my_bits::rol(y, ss2)) !=
        t_my_bits::rol(t_my_bits::ccl_mul(x, y), ss1 + ss2) ) {
      cout << "t_my_bits::ccl_mul" << endl;
      return false;
      }

    if (t_my_bits::is_parity_odd(x)) {
      if (t_my_bits::ccl_mul(x, t_my_bits::ccl_mul_inv(x)) != 1) {
        cout << "t_my_bits::ccl_mul_inv" << endl;
        return false;
        }
      }
    else {
      if (t_my_bits::ccl_mul_inv(x) != 0) {
        cout << "t_my_bits::ccl_mul_inv" << endl;
        return false;
        }
      }

    // ccl_power

    x = random_bits<T>();
    y = random_bits<T>();
    z = random_bits<T>();
    }
  return true;
  }

template<typename T>
static auto test_swar() -> bool {

  typedef t_simd<T> t_my_simd;
  typedef t_bits<T> t_my_bits;

  t_int i;
  T x, y, z, l, h, h1, res;
  T x1, x2, x3;
  t_subword sw, sw1, sw2;
  t_hugeint xx;
  t_int j, b, s, k;
  T m, m2;
  bool ok;
  t_int s1, s2;
  t_int p;

  cout << "test_swar" << endl;
  x = 0;
  y = 0;
  z = 0;
  for (i = 1; i<=loop_count; ++i) {
    sw = (t_subword)(random_int(t_my_simd::ld_bits+1));
    sw1 = (t_subword)(random_int(t_my_simd::ld_bits));
    sw2 = (t_subword)(random_int(t_my_simd::ld_bits));

    b = 1 << (t_int)(sw);
    m = t_my_simd::a_element[(t_int)(sw)];
    l = t_my_simd::a_lo[(t_int)(sw)];
    h = t_my_simd::a_hi[(t_int)(sw)];
    h1 = ((T)(1)) << (b-1);
    s1 = random_int((1 << b)*3+2);  // positive shift/rotate amount
    s2 = random_int((1 << b)*6) - (1 << b)*3;  // rotate amount
    p = random_int(t_my_simd::bits >> (t_int)(sw));  // element index

    // Basic functions.

    res = t_my_simd::constant(sw, x);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = x & m;
      if (x1 != x2) {
        cout << "t_my_simd::constant " << (t_int)(sw) << endl;
        return false;
        }
      }

    res = t_my_simd::element_mask(sw, p);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (j == p)
        x2 = m;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::element_mask " << (t_int)(sw) << endl;
        return false;
        }
      }

    res = t_my_simd::template extract<T>(sw, x, p);
    // res = t_my_simd::extract(sw, x, p);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = res;
      if (j == p)
        x2 = (x >> s) & m;
      else
        continue;
      if (x1 != x2) {
        cout << "t_my_simd::extract " << (t_int)(sw) << endl;
        return false;
        }
      }

    res = t_my_simd::implant(sw, x, y, p);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (j == p)
        x2 = y & m;
      else
        x2 = (x >> s) & m;
      if (x1 != x2) {
        cout << "t_my_simd::implant " << (t_int)(sw) << endl;
        return false;
        }
      }

    // General SIMD functions.

    // Counting.

    res = t_my_simd::nr_1bits(sw, x);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = nr_1bits((x >> s) & m);
      if (x1 != x2) {
        cout << "t_my_simd::nr_1bits " << (t_int)(sw) << endl;
        return false;
        }
      }

    if (t_my_simd::nr_1bits(sw, x) + t_my_simd::nr_0bits(sw, x) !=
        t_my_simd::constant(sw, 1<<(t_int)(sw))) {
      cout << "t_my_simd::nr_trailing_ne0 " << (t_int)(sw) << endl;
      return false;
      }

    res = t_my_simd::nr_trailing_0bits(sw, x);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = (x >> s) & m;
      x2 = nr_1bits((~x2 & (x2-1)) & m);
      if (x1 != x2) {
        cout << "t_my_simd::nr_trailing_0bits " << (t_int)(sw) << endl;
        return false;
        }
      }

    // t_my_simd::nr_trailing_1bits

    res = t_my_simd::nr_leading_0bits(sw, x);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = (x >> s) & m;
      if (x2 == 0)
        x2 = b;
      else
        // x2 = b-1-floor_log2_i(x2);
        x2 = nr_leading_0bits(x2)+b-sizeof(t_hugeint)*CHAR_BIT;
      if (x1 != x2) {
        cout << "t_my_simd::nr_leading_0bits " << (t_int)(sw) << endl;
        return false;
        }
      }

    // t_my_simd::nr_leading_1bits

    ok = false;
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      if (((x >> s) & m) == 0) {
        ok = true;
        break;
        }
      }

    if (t_my_simd::contains_0(sw, x) != ok) {
      cout << "t_my_simd::contains_0 " << (t_int)(sw) << endl;
      return false;
      }

    res = t_my_simd::nr_trailing_ne0(sw, x);
    x1 = 0;
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      if (((x >> s) & m) == 0) {
        break;
        }
      ++x1;
      }

    if (res != x1) {
      cout << "t_my_simd::nr_trailing_ne0 " << (t_int)(sw) << endl;
      return false;
      }

    // Math.

    res = t_my_simd::add(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = ((x >> s) + (y >> s)) & m;
      if (x1 != x2) {
        cout << "t_my_simd::add " << (t_int)(sw) << endl;
        return false;
        }
      }

    res = t_my_simd::sub(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = ((x >> s) - (y >> s)) & m;
      if (x1 != x2) {
        cout << "t_my_simd::sub " << (t_int)(sw) << endl;
        return false;
        }
      }

    res = t_my_simd::add1(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = ((x >> s) + (y >> s) + 1) & m;
      if (x1 != x2) {
        cout << "t_my_simd::add1 " << (t_int)(sw) << endl;
        return false;
        }
      }

    res = t_my_simd::sub1(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = ((x >> s) - (y >> s) - 1) & m;
      if (x1 != x2) {
        cout << "t_my_simd::sub1 " << (t_int)(sw) << endl;
        return false;
        }
      }

    res = t_my_simd::neg(sw, x);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = (-(x >> s)) & m;
      if (x1 != x2) {
        cout << "t_my_simd::neg " << (t_int)(sw) << endl;
        return false;
        }
      }

    if (t_my_simd::add(sw, x, y) != t_my_simd::sub1(sw, x, ~y)) {
      cout << "t_my_simd::add/sub1 " << (t_int)(sw) << endl;
      return false;
      }

    if (t_my_simd::add1(sw, x, y) != t_my_simd::sub(sw, x, ~y)) {
      cout << "t_my_simd::add1/sub " << (t_int)(sw) << endl;
      return false;
      }

    if (t_my_simd::addc(sw, x, y, z) != t_my_simd::subc(sw, x, ~y, ~z)) {
      cout << "t_my_simd::addc/subc " << (t_int)(sw) << endl;
      return false;
      }

    if (t_my_simd::add_sat_s(sw, x, y) != t_my_simd::sub1_sat_s(sw, x, ~y)) {
      cout << "t_my_simd::add_sat_s/sub1_sat_s " << (t_int)(sw) << endl;
      return false;
      }

    if (t_my_simd::add1_sat_s(sw, x, y) != t_my_simd::sub_sat_s(sw, x, ~y)) {
      cout << "t_my_simd::add1_sat_s/sub_sat_s " << (t_int)(sw) << endl;
      return false;
      }

    if (t_my_simd::addc_sat_s(sw, x, y, z) != t_my_simd::subc_sat_s(sw, x, ~y, ~z)) {
      cout << "t_my_simd::addc_sat_s/subc_sat_s " << (t_int)(sw) << endl;
      return false;
      }

    if (t_my_simd::overflow_add_s(sw, x, y) != t_my_simd::overflow_sub1_s(sw, x, ~y)) {
      cout << "t_my_simd::overflow_add_s/overflow_sub1_s " << (t_int)(sw) << endl;
      return false;
      }

    if (t_my_simd::overflow_add1_s(sw, x, y) != t_my_simd::overflow_sub_s(sw, x, ~y)) {
      cout << "t_my_simd::overflow_add1_s/overflow_sub_s " << (t_int)(sw) << endl;
      return false;
      }

    if (t_my_simd::overflow_addc_s(sw, x, y, z) != t_my_simd::overflow_subc_s(sw, x, ~y, ~z)) {
      cout << "t_my_simd::overflow_addc_s/overflow_subc_s " << (t_int)(sw) << endl;
      return false;
      }

    if (t_my_simd::inc(sw, x) != t_my_simd::add(sw, x, t_my_simd::a_lo[(t_int)(sw)])) {
      cout << "t_my_simd::inc " << (t_int)(sw) << endl;
      return false;
      }

    if (t_my_simd::inc(sw, x) != t_my_simd::sub(sw, x, (T)(-1))) {
      cout << "t_my_simd::inc " << (t_int)(sw) << endl;
      return false;
      }

    if (t_my_simd::dec(sw, x) != t_my_simd::sub(sw, x, t_my_simd::a_lo[(t_int)(sw)])) {
      cout << "t_my_simd::dec " << (t_int)(sw) << endl;
      return false;
      }

    if (t_my_simd::dec(sw, x) != t_my_simd::add(sw, x, (T)(-1))) {
      cout << "t_my_simd::dec " << (t_int)(sw) << endl;
      return false;
      }

    if (t_my_simd::neg(sw, x) != t_my_simd::sub(sw, (T)(0), x)) {
      cout << "t_my_simd::neg " << (t_int)(sw) << endl;
      return false;
      }

    res = t_my_simd::overflow_add_u(sw, x, y);
    x1 = t_my_simd::add(sw, x, y);
    x2 = t_my_simd::add_sat_u(sw, x, y);
    x3 = t_my_simd::ne(sw, x1, x2);
    if (res != x3) {
      cout << "t_my_simd::overflow_add_u " << (t_int)(sw) << endl;
      return false;
      }

    res = t_my_simd::overflow_sub_u(sw, x, y);
    x1 = t_my_simd::sub(sw, x, y);
    x2 = t_my_simd::sub_sat_u(sw, x, y);
    x3 = t_my_simd::ne(sw, x1, x2);
    if (res != x3) {
      cout << "t_my_simd::overflow_sub_u " << (t_int)(sw) << endl;
      return false;
      }

    res = t_my_simd::overflow_add_s(sw, x, y);
    x1 = t_my_simd::add(sw, x, y);
    x2 = t_my_simd::add_sat_s(sw, x, y);
    x3 = t_my_simd::ne(sw, x1, x2);
    if (res != x3) {
      cout << "t_my_simd::overflow_add_s " << (t_int)(sw) << endl;
      return false;
      }

    res = t_my_simd::overflow_sub_s(sw, x, y);
    x1 = t_my_simd::sub(sw, x, y);
    x2 = t_my_simd::sub_sat_s(sw, x, y);
    x3 = t_my_simd::ne(sw, x1, x2);
    if (res != x3) {
      cout << "t_my_simd::overflow_sub_s " << (t_int)(sw) << endl;
      return false;
      }

    res = t_my_simd::overflow_add1_u(sw, x, y);
    x1 = t_my_simd::add_sat_u(sw, x, y);
    x2 = t_my_simd::add1_sat_u(sw, x, y);
    x2 = t_my_simd::dec(sw, x2);
    x3 = t_my_simd::ne(sw, x1, x2);
    if (res != x3) {
      cout << "t_my_simd::overflow_add1_u " << (t_int)(sw) << endl;
      return false;
      }

    res = t_my_simd::overflow_sub1_u(sw, x, y);
    x1 = t_my_simd::sub_sat_u(sw, x, y);
    x2 = t_my_simd::sub1_sat_u(sw, x, y);
    x2 = t_my_simd::inc(sw, x2);
    x3 = t_my_simd::ne(sw, x1, x2);
    if (res != x3) {
      cout << "t_my_simd::overflow_sub1_u " << (t_int)(sw) << endl;
      return false;
      }

    if ((1 << (t_int)(sw)) < (t_int)(sizeof(t_hugeint))*CHAR_BIT) {
      // not testable for large T due to unavailability of larger types

      res = t_my_simd::add_sat_u(sw, x, y);
      for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
        s = b*j;
        x1 = (res >> s) & m;
        x2 = clamp_u_sw(sw, (t_hugeint)((x >> s) & m) + (t_hugeint)((y >> s) & m));
        if (x1 != x2) {
          cout << "t_my_simd::add_sat_u " << (t_int)(sw) << endl;
          return false;
          }
        }

      res = t_my_simd::sub_sat_u(sw, x, y);
      for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
        s = b*j;
        x1 = (res >> s) & m;
        x2 = clamp_u_sw(sw, (t_hugeint)((x >> s) & m) - (t_hugeint)((y >> s) & m));
        if (x1 != x2) {
          cout << "t_my_simd::sub_sat_u " << (t_int)(sw) << endl;
          return false;
          }
        }

      res = t_my_simd::add_sat_s(sw, x, y);
      for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
        s = b*j;
        x1 = (res >> s) & m;
        x2 = clamp_s_sw(sw, sign_extend(x >> s, b) + sign_extend(y >> s, b)) & m;
        if (x1 != x2) {
          cout << "t_my_simd::add_sat_s " << (t_int)(sw) << endl;
          return false;
          }
        }

      res = t_my_simd::sub_sat_s(sw, x, y);
      for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
        s = b*j;
        x1 = (res >> s) & m;
        x2 = clamp_s_sw(sw, sign_extend(x >> s, b) - sign_extend(y >> s, b)) & m;
        if (x1 != x2) {
          cout << "t_my_simd::sub_sat_s " << (t_int)(sw) << endl;
          return false;
          }
        }

      res = t_my_simd::add1_sat_u(sw, x, y);
      for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
        s = b*j;
        x1 = (res >> s) & m;
        x2 = clamp_u_sw(sw, (t_hugeint)((x >> s) & m) + (t_hugeint)((y >> s) & m) + 1);
        if (x1 != x2) {
          cout << "t_my_simd::add1_sat_u " << (t_int)(sw) << endl;
          return false;
          }
        }

      res = t_my_simd::sub1_sat_u(sw, x, y);
      for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
        s = b*j;
        x1 = (res >> s) & m;
        x2 = clamp_u_sw(sw, (t_hugeint)((x >> s) & m) - (t_hugeint)((y >> s) & m) - 1);
        if (x1 != x2) {
          cout << "t_my_simd::sub1_sat_u " << (t_int)(sw) << endl;
          return false;
          }
        }

      res = t_my_simd::add1_sat_s(sw, x, y);
      for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
        s = b*j;
        x1 = (res >> s) & m;
        x2 = clamp_s_sw(sw, sign_extend(x >> s, b) + sign_extend(y >> s, b) + 1) & m;
        if (x1 != x2) {
          cout << "t_my_simd::add1_sat_s " << (t_int)(sw) << endl;
          return false;
          }
        }

      res = t_my_simd::sub1_sat_s(sw, x, y);
      for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
        s = b*j;
        x1 = (res >> s) & m;
        x2 = clamp_s_sw(sw, sign_extend(x >> s, b) - sign_extend(y >> s, b) - 1) & m;
        if (x1 != x2) {
          cout << "t_my_simd::sub1_sat_s " << (t_int)(sw) << endl;
          return false;
          }
        }

      res = t_my_simd::addc_sat_u(sw, x, y, z);
      for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
        s = b*j;
        x1 = (res >> s) & m;
        x2 = clamp_u_sw(sw, (t_hugeint)((x >> s) & m) + (t_hugeint)((y >> s) & m) + (t_hugeint)((z >> s) & 1));
        if (x1 != x2) {
          cout << "t_my_simd::addc_sat_u " << (t_int)(sw) << endl;
          return false;
          }
        }

      res = t_my_simd::subc_sat_u(sw, x, y, z);
      for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
        s = b*j;
        x1 = (res >> s) & m;
        x2 = clamp_u_sw(sw, (t_hugeint)((x >> s) & m) - (t_hugeint)((y >> s) & m) - (t_hugeint)((z >> s) & 1));
        if (x1 != x2) {
          cout << "t_my_simd::subc_sat_u " << (t_int)(sw) << endl;
          return false;
          }
        }

      res = t_my_simd::addc_sat_s(sw, x, y, z);
      for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
        s = b*j;
        x1 = (res >> s) & m;
        x2 = clamp_s_sw(sw, sign_extend(x >> s, b) + sign_extend(y >> s, b) + (t_hugeint)((z >> s) & 1)) & m;
        if (x1 != x2) {
          cout << "t_my_simd::addc_sat_s " << (t_int)(sw) << endl;
          return false;
          }
        }

      res = t_my_simd::subc_sat_s(sw, x, y, z);
      for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
        s = b*j;
        x1 = (res >> s) & m;
        x2 = clamp_s_sw(sw, sign_extend(x >> s, b) - sign_extend(y >> s, b) - (t_hugeint)((z >> s) & 1)) & m;
        if (x1 != x2) {
          cout << "t_my_simd::subc_sat_s " << (t_int)(sw) << endl;
          return false;
          }
        }

      res = t_my_simd::overflow_addc_u(sw, x, y, z);
      for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
        s = b*j;
        x1 = (res >> s) & m;
        xx = (t_hugeint)((x >> s) & m) + (t_hugeint)((y >> s) & m) + (t_hugeint)((z >> s) & 1);
        if (xx == clamp_u_sw(sw, xx)) {
          x2 = 0;
          }
        else {
          x2 = m;
          }
        if (x1 != x2) {
          cout << "t_my_simd::overflow_addc_u " << (t_int)(sw) << endl;
          return false;
          }
        }

      res = t_my_simd::overflow_subc_u(sw, x, y, z);
      for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
        s = b*j;
        x1 = (res >> s) & m;
        xx = (t_hugeint)((x >> s) & m) - (t_hugeint)((y >> s) & m) - (t_hugeint)((z >> s) & 1);
        if (xx == clamp_u_sw(sw, xx)) {
          x2 = 0;
          }
        else {
          x2 = m;
          }
        if (x1 != x2) {
          cout << "t_my_simd::overflow_subc_u " << (t_int)(sw) << endl;
          return false;
          }
        }

      res = t_my_simd::overflow_add1_s(sw, x, y);
      for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
        s = b*j;
        x1 = (res >> s) & m;
        xx = sign_extend(x >> s, b) + sign_extend(y >> s, b) + 1;
        if (xx == clamp_s_sw(sw, xx)) {
          x2 = 0;
          }
        else {
          x2 = m;
          }
        if (x1 != x2) {
          cout << "t_my_simd::overflow_add1_s " << (t_int)(sw) << endl;
          return false;
          }
        }

      res = t_my_simd::overflow_sub1_s(sw, x, y);
      for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
        s = b*j;
        x1 = (res >> s) & m;
        xx = sign_extend(x >> s, b) - sign_extend(y >> s, b) - 1;
        if (xx == clamp_s_sw(sw, xx)) {
          x2 = 0;
          }
        else {
          x2 = m;
          }
        if (x1 != x2) {
          cout << "t_my_simd::overflow_sub1_s " << (t_int)(sw) << endl;
          return false;
          }
        }

      res = t_my_simd::overflow_addc_s(sw, x, y, z);
      for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
        s = b*j;
        x1 = (res >> s) & m;
        xx = sign_extend(x >> s, b) + sign_extend(y >> s, b) + (t_hugeint)((z >> s) & 1);
        if (xx == clamp_s_sw(sw, xx)) {
          x2 = 0;
          }
        else {
          x2 = m;
          }
        if (x1 != x2) {
          cout << "t_my_simd::overflow_addc_s " << (t_int)(sw) << endl;
          return false;
          }
        }

      res = t_my_simd::overflow_subc_s(sw, x, y, z);
      for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
        s = b*j;
        x1 = (res >> s) & m;
        xx = sign_extend(x >> s, b) - sign_extend(y >> s, b) - (t_hugeint)((z >> s) & 1);
        if (xx == clamp_s_sw(sw, xx)) {
          x2 = 0;
          }
        else {
          x2 = m;
          }
        if (x1 != x2) {
          cout << "t_my_simd::overflow_subc_s " << (t_int)(sw) << endl;
          return false;
          }
        }

      res = t_my_simd::neg_sat_s(sw, x);
      for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
        s = b*j;
        x1 = (res >> s) & m;
        x2 = clamp_s_sw(sw, -sign_extend(x >> s, b)) & m;
        if (x1 != x2) {
          cout << "t_my_simd::neg_sat_s " << (t_int)(sw) << endl;
          return false;
          }
        }

      res = t_my_simd::avgd_u(sw, x, y);
      for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
        s = b*j;
        x1 = (res >> s) & m;
        x2 = M_avgd_u(((x >> s) & m), ((y >> s) & m));
        if (x1 != x2) {
          cout << "t_my_simd::avgd_u " << (t_int)(sw) << endl;
          return false;
          }
        }

      res = t_my_simd::avgu_u(sw, x, y);
      for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
        s = b*j;
        x1 = (res >> s) & m;
        x2 = M_avgu_u(((x >> s) & m), ((y >> s) & m));
        if (x1 != x2) {
          cout << "t_my_simd::avgu_u " << (t_int)(sw) << endl;
          return false;
          }
        }

      // t_my_simd::avgf_s
      // t_my_simd::avgc_s
      }

    if ((t_int)(sw) < t_my_simd::ld_bits) {
      m2 = t_my_simd::a_element[(t_int)(sw)+1];

      res = t_my_simd::smul_hi_u(sw, x, y);
      for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
        s = b*j;
        x1 = (res >> s) & m;
        x2 = (((x & m) * ((y >> s) & m)) >> b) & m;
        if (x1 != x2) {
          cout << "t_my_simd::smul_hi_u " << (t_int)(sw) << endl;
          return false;
          }
        }

      res = t_my_simd::smul_hi_s(sw, x, y);
      for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
        s = b*j;
        x1 = (res >> s) & m;
        x2 = ((sign_extend(x & m, b) * sign_extend((y >> s) & m, b)) >> b) & m;
        if (x1 != x2) {
          cout << "t_my_simd::smul_hi_s " << (t_int)(sw) << endl;
          return false;
          }
        }

      res = t_my_simd::smule_u(sw, x, y);
      for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; j+=2) {
        s = b*j;
        x1 = (res >> s) & m2;
        x2 = (x & m) * ((y >> s) & m);
        if (x1 != x2) {
          cout << "t_my_simd::smule_u " << (t_int)(sw) << endl;
          return false;
          }
        }

      res = t_my_simd::smule_s(sw, x, y);
      for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; j+=2) {
        s = b*j;
        x1 = (res >> s) & m2;
        x2 = (sign_extend(x & m, b) * sign_extend((y >> s) & m, b)) & m2;
        if (x1 != x2) {
          cout << "t_my_simd::smule_s " << (t_int)(sw) << endl;
          return false;
          }
        }

      }

    res = t_my_simd::abs_diff_u(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = M_abs_diff((x >> s) & m, (y >> s) & m) & m;
      if (x1 != x2) {
        cout << "t_my_simd::abs_diff_u " << (t_int)(sw) << endl;
        return false;
        }
      }

    res = t_my_simd::abs_diff_s(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = M_abs_diff(sign_extend(x >> s, b), sign_extend(y >> s, b)) & m;
      if (x1 != x2) {
        cout << "t_my_simd::abs_diff_s " << (t_int)(sw) << endl;
        return false;
        }
      }

    res = t_my_simd::sgn_diff_u(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = M_sgn_diff((x >> s) & m, (y >> s) & m) & m;
      if (x1 != x2) {
        cout << "t_my_simd::sgn_diff_u " << (t_int)(sw) << endl;
        return false;
        }
      }

    res = t_my_simd::sgn_diff_s(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = M_sgn_diff(sign_extend(x >> s, b), sign_extend(y >> s, b)) & m;
      if (x1 != x2) {
        cout << "t_my_simd::sgn_diff_s " << (t_int)(sw) << endl;
        return false;
        }
      }

    res = t_my_simd::mul_lo(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = ((x >> s) * (y >> s)) & m;
      if (x1 != x2) {
        cout << "t_my_simd::mul_lo " << (t_int)(sw) << endl;
        // cout << "x = " << (t_hugeuint)(x) << endl;
        // cout << "y = " << (t_hugeuint)(y) << endl;
        // cout << "r = " << (t_hugeuint)(res) << endl;
        return false;
        }
      }

    res = t_my_simd::smul_lo(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = ((x & m) * ((y >> s) & m)) & m;
      if (x1 != x2) {
        cout << "t_my_simd::smul_lo " << (t_int)(sw) << endl;
        return false;
        }
      }

    // Comparing.

    res = t_my_simd::eq0h(sw, x);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (((x >> s) & m) == 0)
        x2 = h1;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::eq0h " << (t_int)(sw) << endl;
        return false;
        }
      }

    if (t_my_simd::ne0h(sw, x) != (T)(h ^ t_my_simd::eq0h(sw, x))) {
      cout << "t_my_simd::ne0h " << (t_int)(sw) << endl;
      return false;
      }

    res = t_my_simd::ne0h(sw, x);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (((x >> s) & m) != 0)
        x2 = h1;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::ne0h " << (t_int)(sw) << endl;
        return false;
        }
      }

    if (t_my_simd::eqh(sw, x, y) != t_my_simd::eq0h(sw, x^y)) {
      cout << "t_my_simd::eqh " << (t_int)(sw) << endl;
      return false;
      }

    res = t_my_simd::leh_u(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (((x >> s) & m) <= ((y >> s) & m))
        x2 = h1;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::leh_u " << (t_int)(sw) << endl;
        return false;
        }
      }

    if (t_my_simd::lth_u(sw, x, y) != (T)(h ^ t_my_simd::leh_u(sw, y, x))) {
      cout << "t_my_simd::lth_u " << (t_int)(sw) << " (1)" << endl;
      return false;
      }

    res = t_my_simd::lth_u(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (((x >> s) & m) < ((y >> s) & m))
        x2 = h1;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::lth_u " << (t_int)(sw) << " (2)" << endl;
        return false;
        }
      }

    if (t_my_simd::geh_u(sw, x, y) != t_my_simd::leh_u(sw, y, x)) {
      cout << "t_my_simd::geh_u " << (t_int)(sw) << " (1)" << endl;
      return false;
      }

    res = t_my_simd::geh_u(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (((x >> s) & m) >= ((y >> s) & m))
        x2 = h1;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::geh_u " << (t_int)(sw) << " (2)" << endl;
        return false;
        }
      }

    if (t_my_simd::gth_u(sw, x, y) != (T)(h ^ t_my_simd::leh_u(sw, x, y))) {
      cout << "t_my_simd::gth_u " << (t_int)(sw) << " (1)" << endl;
      return false;
      }

    res = t_my_simd::gth_u(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (((x >> s) & m) > ((y >> s) & m))
        x2 = h1;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::gth_u " << (t_int)(sw) << " (2)" << endl;
        return false;
        }
      }

    res = t_my_simd::lt0h_s(sw, x);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (sign_extend(((x >> s) & m), b) < 0)
        x2 = h1;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::lt0h_s " << (t_int)(sw) << endl;
        return false;
        }
      }

    res = t_my_simd::leh_s(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (sign_extend(((x >> s) & m), b) <= sign_extend(((y >> s) & m), b))
        x2 = h1;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::leh_s " << (t_int)(sw) << endl;
        return false;
        }
      }

    if (t_my_simd::lth_s(sw, x, y) != (T)(h ^ t_my_simd::leh_s(sw, y, x))) {
      cout << "t_my_simd::lth_s " << (t_int)(sw) << " (1)" << endl;
      return false;
      }

    res = t_my_simd::lth_s(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (sign_extend(((x >> s) & m), b) < sign_extend(((y >> s) & m), b))
        x2 = h1;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::lth_s " << (t_int)(sw) << " (2)" << endl;
        return false;
        }
      }

    if (t_my_simd::geh_s(sw, x, y) != t_my_simd::leh_s(sw, y, x)) {
      cout << "t_my_simd::geh_s " << (t_int)(sw) << " (1)" << endl;
      return false;
      }

    res = t_my_simd::geh_s(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (sign_extend(((x >> s) & m), b) >= sign_extend(((y >> s) & m), b))
        x2 = h1;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::geh_s " << (t_int)(sw) << " (2)" << endl;
        return false;
        }
      }

    if (t_my_simd::gth_s(sw, x, y) != (T)(h ^ t_my_simd::leh_s(sw, x, y))) {
      cout << "t_my_simd::gth_s " << (t_int)(sw) << " (1)" << endl;
      return false;
      }

    res = t_my_simd::gth_s(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (sign_extend(((x >> s) & m), b) > sign_extend(((y >> s) & m), b))
        x2 = h1;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::gth_s " << (t_int)(sw) << " (2)" << endl;
        return false;
        }
      }

    if (t_my_simd::oddh(sw, x) != t_my_simd::ne0h(sw, x & t_my_simd::a_lo[(t_int)(sw)])) {
      cout << "t_my_simd::oddh " << (t_int)(sw) << endl;
      return false;
      }

    res = t_my_simd::eq0l(sw, x);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (((x >> s) & m) == 0)
        x2 = 1;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::eq0l " << (t_int)(sw) << endl;
        return false;
        }
      }

    if (t_my_simd::ne0l(sw, x) != (T)(l ^ t_my_simd::eq0l(sw, x))) {
      cout << "t_my_simd::ne0l " << (t_int)(sw) << endl;
      return false;
      }

    res = t_my_simd::ne0l(sw, x);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (((x >> s) & m) != 0)
        x2 = 1;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::ne0l " << (t_int)(sw) << endl;
        return false;
        }
      }

    if (t_my_simd::eql(sw, x, y) != t_my_simd::eq0l(sw, x^y)) {
      cout << "t_my_simd::eql " << (t_int)(sw) << endl;
      return false;
      }

    res = t_my_simd::lel_u(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (((x >> s) & m) <= ((y >> s) & m))
        x2 = 1;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::lel_u " << (t_int)(sw) << endl;
        return false;
        }
      }

    if (t_my_simd::ltl_u(sw, x, y) != (T)(l ^ t_my_simd::lel_u(sw, y, x))) {
      cout << "t_my_simd::ltl_u " << (t_int)(sw) << " (1)" << endl;
      return false;
      }

    res = t_my_simd::ltl_u(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (((x >> s) & m) < ((y >> s) & m))
        x2 = 1;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::ltl_u " << (t_int)(sw) << " (2)" << endl;
        return false;
        }
      }

    if (t_my_simd::gel_u(sw, x, y) != t_my_simd::lel_u(sw, y, x)) {
      cout << "t_my_simd::gel_u " << (t_int)(sw) << " (1)" << endl;
      return false;
      }

    res = t_my_simd::gel_u(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (((x >> s) & m) >= ((y >> s) & m))
        x2 = 1;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::gel_u " << (t_int)(sw) << " (2)" << endl;
        return false;
        }
      }

    if (t_my_simd::gtl_u(sw, x, y) != (T)(l ^ t_my_simd::lel_u(sw, x, y))) {
      cout << "t_my_simd::gtl_u " << (t_int)(sw) << " (1)" << endl;
      return false;
      }

    res = t_my_simd::gtl_u(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (((x >> s) & m) > ((y >> s) & m))
        x2 = 1;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::gtl_u " << (t_int)(sw) << " (2)" << endl;
        return false;
        }
      }

    res = t_my_simd::lt0l_s(sw, x);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (sign_extend(((x >> s) & m), b) < 0)
        x2 = 1;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::lt0l_s " << (t_int)(sw) << endl;
        return false;
        }
      }

    res = t_my_simd::lel_s(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (sign_extend(((x >> s) & m), b) <= sign_extend(((y >> s) & m), b))
        x2 = 1;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::lel_s " << (t_int)(sw) << endl;
        return false;
        }
      }

    if (t_my_simd::ltl_s(sw, x, y) != (T)(l ^ t_my_simd::lel_s(sw, y, x))) {
      cout << "t_my_simd::ltl_s " << (t_int)(sw) << " (1)" << endl;
      return false;
      }

    res = t_my_simd::ltl_s(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (sign_extend(((x >> s) & m), b) < sign_extend(((y >> s) & m), b))
        x2 = 1;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::ltl_s " << (t_int)(sw) << " (2)" << endl;
        return false;
        }
      }

    if (t_my_simd::gel_s(sw, x, y) != t_my_simd::lel_s(sw, y, x)) {
      cout << "t_my_simd::gel_s " << (t_int)(sw) << " (1)" << endl;
      return false;
      }

    res = t_my_simd::gel_s(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (sign_extend(((x >> s) & m), b) >= sign_extend(((y >> s) & m), b))
        x2 = 1;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::gel_s " << (t_int)(sw) << " (2)" << endl;
        return false;
        }
      }

    if (t_my_simd::gtl_s(sw, x, y) != (T)(l ^ t_my_simd::lel_s(sw, x, y))) {
      cout << "t_my_simd::gtl_s " << (t_int)(sw) << " (1)" << endl;
      return false;
      }

    res = t_my_simd::gtl_s(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (sign_extend(((x >> s) & m), b) > sign_extend(((y >> s) & m), b))
        x2 = 1;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::gtl_s " << (t_int)(sw) << " (2)" << endl;
        return false;
        }
      }

    if (t_my_simd::oddl(sw, x) != t_my_simd::ne0l(sw, x & t_my_simd::a_lo[(t_int)(sw)])) {
      cout << "t_my_simd::oddl " << (t_int)(sw) << endl;
      return false;
      }

    res = t_my_simd::eq0(sw, x);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (((x >> s) & m) == 0)
        x2 = m;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::eq0 " << (t_int)(sw) << endl;
        return false;
        }
      }

    if (t_my_simd::ne0(sw, x) != (T)(~t_my_simd::eq0(sw, x))) {
      cout << "t_my_simd::ne0 " << (t_int)(sw) << endl;
      return false;
      }

    res = t_my_simd::ne0(sw, x);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (((x >> s) & m) != 0)
        x2 = m;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::ne0 " << (t_int)(sw) << endl;
        return false;
        }
      }

    if (t_my_simd::eq(sw, x, y) != t_my_simd::eq0(sw, x^y)) {
      cout << "t_my_simd::eq " << (t_int)(sw) << endl;
      return false;
      }

    res = t_my_simd::le_u(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (((x >> s) & m) <= ((y >> s) & m))
        x2 = m;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::le_u " << (t_int)(sw) << endl;
        return false;
        }
      }

    if (t_my_simd::lt_u(sw, x, y) != (T)(~t_my_simd::le_u(sw, y, x))) {
      cout << "t_my_simd::lt_u " << (t_int)(sw) << endl;
      return false;
      }

    res = t_my_simd::lt_u(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (((x >> s) & m) < ((y >> s) & m))
        x2 = m;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::lt_u " << (t_int)(sw) << endl;
        return false;
        }
      }

    if (t_my_simd::ge_u(sw, x, y) != t_my_simd::le_u(sw, y, x)) {
      cout << "t_my_simd::ge_u " << (t_int)(sw) << endl;
      return false;
      }

    res = t_my_simd::ge_u(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (((x >> s) & m) >= ((y >> s) & m))
        x2 = m;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::ge_u " << (t_int)(sw) << endl;
        return false;
        }
      }

    if (t_my_simd::gt_u(sw, x, y) != (T)(~t_my_simd::le_u(sw, x, y))) {
      cout << "t_my_simd::gt_u " << (t_int)(sw) << endl;
      return false;
      }

    res = t_my_simd::gt_u(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (((x >> s) & m) > ((y >> s) & m))
        x2 = m;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::gt_u " << (t_int)(sw) << endl;
        return false;
        }
      }

    res = t_my_simd::lt0_s(sw, x);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (sign_extend(((x >> s) & m), b) < 0)
        x2 = m;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::lt0_s " << (t_int)(sw) << endl;
        return false;
        }
      }

    res = t_my_simd::le_s(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (sign_extend(((x >> s) & m), b) <= sign_extend(((y >> s) & m), b))
        x2 = m;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::le_s " << (t_int)(sw) << endl;
        return false;
        }
      }

    if (t_my_simd::lt_s(sw, x, y) != (T)(~t_my_simd::le_s(sw, y, x))) {
      cout << "t_my_simd::lt_s " << (t_int)(sw) << endl;
      return false;
      }

    res = t_my_simd::lt_s(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (sign_extend(((x >> s) & m), b) < sign_extend(((y >> s) & m), b))
        x2 = m;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::lt_s " << (t_int)(sw) << endl;
        return false;
        }
      }

    if (t_my_simd::ge_s(sw, x, y) != t_my_simd::le_s(sw, y, x)) {
      cout << "t_my_simd::ge_s " << (t_int)(sw) << endl;
      return false;
      }

    res = t_my_simd::ge_s(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (sign_extend(((x >> s) & m), b) >= sign_extend(((y >> s) & m), b))
        x2 = m;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::ge_s " << (t_int)(sw) << endl;
        return false;
        }
      }

    if (t_my_simd::gt_s(sw, x, y) != (T)(~t_my_simd::le_s(sw, x, y))) {
      cout << "t_my_simd::gt_s " << (t_int)(sw) << endl;
      return false;
      }

    res = t_my_simd::gt_s(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      if (sign_extend(((x >> s) & m), b) > sign_extend(((y >> s) & m), b))
        x2 = m;
      else
        x2 = 0;
      if (x1 != x2) {
        cout << "t_my_simd::gt_s " << (t_int)(sw) << endl;
        return false;
        }
      }

    if (t_my_simd::odd(sw, x) != t_my_simd::ne0(sw, x & t_my_simd::a_lo[(t_int)(sw)])) {
      cout << "t_my_simd::odd " << (t_int)(sw) << endl;
      return false;
      }

    res = t_my_simd::max0_s(sw, x);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = M_max(0, sign_extend(x >> s, b)) & m;
      if (x1 != x2) {
        cout << "t_my_simd::max0_s " << (t_int)(sw) << endl;
        return false;
        }
      }

    res = t_my_simd::max_u(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = M_max((x >> s) & m, (y >> s) & m);
      if (x1 != x2) {
        cout << "t_my_simd::max_u " << (t_int)(sw) << endl;
        return false;
        }
      }

    res = t_my_simd::min_u(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = M_min((x >> s) & m, (y >> s) & m);
      if (x1 != x2) {
        cout << "t_my_simd::min_u " << (t_int)(sw) << endl;
        return false;
        }
      }

    res = t_my_simd::max_s(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = clamp_s_sw(sw, M_max(sign_extend(x >> s, b), sign_extend(y >> s, b))) & m;
      if (x1 != x2) {
        cout << "t_my_simd::max_s " << (t_int)(sw) << endl;
        return false;
        }
      }

    res = t_my_simd::min_s(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = clamp_s_sw(sw, M_min(sign_extend(x >> s, b), sign_extend(y >> s, b))) & m;
      if (x1 != x2) {
        cout << "t_my_simd::min_s " << (t_int)(sw) << endl;
        return false;
        }
      }

    if ((1 << (t_int)(sw)) < (t_int)(sizeof(t_hugeint))*CHAR_BIT) {
      // not testable for large T due to unavailability of larger types

      res = t_my_simd::abs(sw, x);
      for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
        s = b*j;
        x1 = (res >> s) & m;
        x2 = M_abs(sign_extend(x >> s, b)) & m;
        if (x1 != x2) {
          cout << "t_my_simd::abs " << (t_int)(sw) << endl;
          return false;
          }
        }

      res = t_my_simd::nabs(sw, x);
      for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
        s = b*j;
        x1 = (res >> s) & m;
        x2 = (-M_abs(sign_extend(x >> s, b))) & m;
        if (x1 != x2) {
          cout << "t_my_simd::nabs " << (t_int)(sw) << endl;
          return false;
          }
        }

      }

    res = t_my_simd::sgn(sw, x);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = M_sgn(sign_extend((x >> s) & m, b)) & m;
      if (x1 != x2) {
        cout << "t_my_simd::sgn " << (t_int)(sw) << endl;
        return false;
        }
      }

    res = t_my_simd::sgn_sat(sw, x);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = (x >> s) & m;
      if (x2 == 0)
        x2 = 0;
      else if (sign_extend(x2, b) < 0)
        x2 = (T)(1) << (b-1);  // smallest signed value
      else
        x2 = (((T)(1) << (b-1)) - 1) & m;  // largest signed value
      if (x1 != x2) {
        cout << "t_my_simd::sgn_sat " << (t_int)(sw) << endl;
        return false;
        }
      }

    // Shift by common amount.

    res = t_my_simd::shl(sw, x, s1);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = shl_ex((x >> s) & m, s1, b) & m;
      if (x1 != x2) {
        cout << "t_my_simd::shl " << (t_int)(sw) << endl;
        return false;
        }
      }

    if (t_my_simd::shl(sw, x, s1) != t_my_simd::rold(sw, x, 0, M_min(s1, b))) {
      cout << "t_my_simd::shl " << (t_int)(sw) << endl;
      return false;
      }

    res = t_my_simd::shr(sw, x, s1);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = shr_ex((x >> s) & m, s1, b);
      if (x1 != x2) {
        cout << "t_my_simd::shr " << (t_int)(sw) << endl;
        return false;
        }
      }

    if (t_my_simd::shr(sw, x, s1) != t_my_simd::rold(sw, x, 0, -M_min(s1, b))) {
      cout << "t_my_simd::shr " << (t_int)(sw) << endl;
      return false;
      }

    res = t_my_simd::sar(sw, x, s1);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = sar_ex((x >> s) & m, s1, b) & m;
      if (x1 != x2) {
        cout << "t_my_simd::sar " << (t_int)(sw) << endl;
        return false;
        }
      }

    if (t_my_simd::sar(sw, x, s1) != t_my_simd::rold(sw, x, t_my_simd::lt0_s(sw, x), -M_min(s1, b))) {
      cout << "t_my_simd::sar " << (t_int)(sw) << endl;
      return false;
      }

    if (t_my_simd::sal(sw, x, s1) != t_my_simd::rold(sw, x, t_my_simd::odd(sw, x), M_min(s1, b))) {
      cout << "t_my_simd::sal " << (t_int)(sw) << endl;
      return false;
      }

    if (t_my_simd::shl1(sw, x, s1) != t_my_simd::rold(sw, x, t_my_simd::all_bits, M_min(s1, b))) {
      cout << "t_my_simd::shl1 " << (t_int)(sw) << endl;
      return false;
      }

    if (t_my_simd::shr1(sw, x, s1) != t_my_simd::rold(sw, x, t_my_simd::all_bits, -M_min(s1, b))) {
      cout << "t_my_simd::shr1 " << (t_int)(sw) << endl;
      return false;
      }

    // t_my_simd::shl_sat_s
    // t_my_simd::shl_sat_u
    // t_my_simd::sal_sat_s
    // t_my_simd::sal_sat_u
    // t_my_simd::shl1_sat_s
    // t_my_simd::shl1_sat_u

    // Rotate by common amount.

    res = t_my_simd::rol(sw, x, s2);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = rol_ex((x >> s) & m, s2, b) & m;
      if (x1 != x2) {
        cout << "t_my_simd::rol " << (t_int)(sw) << endl;
        return false;
        }
      }

    if (t_my_simd::rol(sw, x, s2) != t_my_simd::rold(sw, x, x, s2)) {
      cout << "t_my_simd::rol " << (t_int)(sw) << endl;
      return false;
      }

    res = t_my_simd::ror(sw, x, s2);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = ror_ex((x >> s) & m, s2, b) & m;
      if (x1 != x2) {
        cout << "t_my_simd::ror " << (t_int)(sw) << endl;
        return false;
        }
      }

    if (t_my_simd::ror(sw, x, s2) != t_my_simd::rold(sw, x, x, -s2)) {
      cout << "t_my_simd::ror " << (t_int)(sw) << endl;
      return false;
      }

    if (t_my_simd::rolc(sw, x, s2) != t_my_simd::rold(sw, x, ~x, s2)) {
      cout << "t_my_simd::rolc " << (t_int)(sw) << endl;
      return false;
      }

    if (t_my_simd::rorc(sw, x, s2) != t_my_simd::rold(sw, x, ~x, -s2)) {
      cout << "t_my_simd::rorc " << (t_int)(sw) << endl;
      return false;
      }

    // t_my_simd::rold is implicitly tested

    if (t_my_simd::rord(sw, x, y, s2) != t_my_simd::rold(sw, x, y, -s2)) {
      cout << "t_my_simd::rord " << (t_int)(sw) << endl;
      return false;
      }

    x1 = x;
    x2 = y;
    t_my_simd::do_rold(sw, &x1, &x2, s2);
    if (x1 != t_my_simd::rold(sw, x, y, s2)) {
      cout << "t_my_simd::do_rold " << (t_int)(sw) << endl;
      return false;
      }
    if (x2 != t_my_simd::rold(sw, y, x, s2)) {
      cout << "t_my_simd::do_rold " << (t_int)(sw) << endl;
      return false;
      }

    x1 = x;
    x2 = y;
    t_my_simd::do_rord(sw, &x1, &x2, s2);
    if (x1 != t_my_simd::rord(sw, x, y, s2)) {
      cout << "t_my_simd::do_rord " << (t_int)(sw) << endl;
      return false;
      }
    if (x2 != t_my_simd::rord(sw, y, x, s2)) {
      cout << "t_my_simd::do_rord " << (t_int)(sw) << endl;
      return false;
      }

    // Shift by fieldwise specified amount.

    res = t_my_simd::vshl(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = shl_ex((x >> s) & m, (y >> s) & m, b) & m;
      if (x1 != x2) {
        cout << "t_my_simd::vshl " << (t_int)(sw) << endl;
        return false;
        }
      }

    res = t_my_simd::vshr(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = shr_ex((x >> s) & m, (y >> s) & m, b) & m;
      if (x1 != x2) {
        cout << "t_my_simd::vshr " << (t_int)(sw) << endl;
        return false;
        }
      }

    res = t_my_simd::vsar(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = sar_ex((x >> s) & m, (y >> s) & m, b) & m;
      if (x1 != x2) {
        cout << "t_my_simd::vsar " << (t_int)(sw) << endl;
        return false;
        }
      }

    res = t_my_simd::vsal(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = sal_ex((x >> s) & m, (y >> s) & m, b) & m;
      if (x1 != x2) {
        cout << "t_my_simd::vsal " << (t_int)(sw) << endl;
        return false;
        }
      }

    res = t_my_simd::vshl1(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = shl1_ex((x >> s) & m, (y >> s) & m, b) & m;
      if (x1 != x2) {
        cout << "t_my_simd::vshl1 " << (t_int)(sw) << endl;
        return false;
        }
      }

    res = t_my_simd::vshr1(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = shr1_ex((x >> s) & m, (y >> s) & m, b) & m;
      if (x1 != x2) {
        cout << "t_my_simd::vshr1 " << (t_int)(sw) << endl;
        return false;
        }
      }

    // t_my_simd::vshl_sat_s
    // t_my_simd::vshl_sat_u
    // t_my_simd::vsal_sat_s
    // t_my_simd::vsal_sat_u
    // t_my_simd::vshl1_sat_s
    // t_my_simd::vshl1_sat_u

    // Rotate by fieldwise specified amount.

    res = t_my_simd::vrol(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = rol_ex((x >> s) & m, (y >> s) & m, b) & m;
      if (x1 != x2) {
        cout << "t_my_simd::vrol " << (t_int)(sw) << endl;
        return false;
        }
      }

    res = t_my_simd::vror(sw, x, y);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = ror_ex((x >> s) & m, (y >> s) & m, b) & m;
      if (x1 != x2) {
        cout << "t_my_simd::vror " << (t_int)(sw) << endl;
        return false;
        }
      }

    if (t_my_simd::vrolc(sw, x, y) != t_my_simd::vrold(sw, x, ~x, y)) {
      cout << "t_my_simd::vrolc " << (t_int)(sw) << endl;
      return false;
      }

    if (t_my_simd::vrorc(sw, x, y) != t_my_simd::vrord(sw, x, ~x, y)) {
      cout << "t_my_simd::vrorc " << (t_int)(sw) << endl;
      return false;
      }

    res = t_my_simd::vrold(sw, x, y, z);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = t_my_simd::rold(sw, (x >> s) & m, (y >> s) & m, (z >> s) & m) & m;
      if (x1 != x2) {
        cout << "t_my_simd::vrold " << (t_int)(sw) << endl;
        return false;
        }
      }

    res = t_my_simd::vrord(sw, x, y, z);
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = t_my_simd::rord(sw, (x >> s) & m, (y >> s) & m, (z >> s) & m) & m;
      if (x1 != x2) {
        cout << "t_my_simd::vrord " << (t_int)(sw) << endl;
        return false;
        }
      }

    x1 = x;
    x2 = y;
    t_my_simd::do_vrold(sw, &x1, &x2, z);
    if (x1 != t_my_simd::vrold(sw, x, y, z)) {
      cout << "t_my_simd::do_vrold " << (t_int)(sw) << endl;
      return false;
      }
    if (x2 != t_my_simd::vrold(sw, y, x, z)) {
      cout << "t_my_simd::do_vrold " << (t_int)(sw) << endl;
      return false;
      }

    x1 = x;
    x2 = y;
    t_my_simd::do_vrord(sw, &x1, &x2, z);
    if (x1 != t_my_simd::vrord(sw, x, y, z)) {
      cout << "t_my_simd::do_vrord " << (t_int)(sw) << endl;
      return false;
      }
    if (x2 != t_my_simd::vrord(sw, y, x, z)) {
      cout << "t_my_simd::do_vrord " << (t_int)(sw) << endl;
      return false;
      }

    // Special functions.

    if (t_my_simd::gray_code(sw, x) != (T)(x ^ t_my_simd::shr(sw, x, 1))) {
      cout << "t_my_simd::gray_code " << (t_int)(sw) << endl;
      return false;
      }

    if (t_my_simd::gray_code(sw, t_my_simd::inv_gray_code(sw, x)) != x) {
      cout << "t_my_simd::inv_gray_code " << (t_int)(sw) << endl;
      return false;
      }

    // assuming y has at least 5 bits for mode
    res = t_my_simd::tbm(sw, x, (t_int)(y));
    for (j = 0; j<=(t_int)(t_my_simd::bits >> (t_int)(sw))-1; ++j) {
      s = b*j;
      x1 = (res >> s) & m;
      x2 = t_my_bits::tbm((x >> s) & m, (t_int)(y)) & m;
      if (x1 != x2) {
        cout << "t_my_simd::tbm " << (t_int)(sw) << " " << ((t_int)(y) & 0x1f) << endl;
        return false;
        }
      }

    if (res != (
        t_my_simd::tbm(sw, x, (t_int)(y) & 0x11) |
        t_my_simd::tbm(sw, x, (t_int)(y) & 0x12) |
        t_my_simd::tbm(sw, x, (t_int)(y) & 0x1c) )) {
      cout << "t_my_simd::tbm (a) " << (t_int)(sw) << " " << ((t_int)(y) & 0x1f) << endl;
      return false;
      }

    if (((t_int)(y) & 0x08) == 0) {
      if (res != (T)(~t_my_simd::tbm(sw, x, (t_int)(y) ^ 0x07))) {
        cout << "t_my_simd::tbm (b) " << (t_int)(sw) << " " << ((t_int)(y) & 0x1f) << endl;
        return false;
        }
      if (res != t_my_simd::tbm(sw, ~x, (t_int)(y) ^ 0x10)) {
        cout << "t_my_simd::tbm (c) " << (t_int)(sw) << " " << ((t_int)(y) & 0x1f) << endl;
        return false;
        }
      }
    else {
      if (res != (T)(~t_my_simd::tbm(sw, ~x, (t_int)(y) ^ 0x13))) {
        cout << "t_my_simd::tbm (d) " << (t_int)(sw) << " " << ((t_int)(y) & 0x1f) << endl;
        return false;
        }
      if (res != t_my_simd::tbm(sw, ~x, (t_int)(y) ^ 0x14)) {
        cout << "t_my_simd::tbm (e) " << (t_int)(sw) << " " << ((t_int)(y) & 0x1f) << endl;
        return false;
        }
      if (((t_int)(y) & 0x10) == 0) {
        if (res != t_my_simd::tbm(sw, t_my_simd::inc(sw, x), (t_int)(y) ^ 0x10)) {
          cout << "t_my_simd::tbm (f) " << (t_int)(sw) << " " << ((t_int)(y) & 0x1f) << endl;
          return false;
          }
        }
      else {
        if (res != t_my_simd::tbm(sw, t_my_simd::dec(sw, x), (t_int)(y) ^ 0x10)) {
          cout << "t_my_simd::tbm (g) " << (t_int)(sw) << " " << ((t_int)(y) & 0x1f) << endl;
          return false;
          }
        }
      }

    // Horizontal math.
    // => test_horizontal

    // butterfly

    res = t_my_simd::bit_index_complement(x, sw1);

    for (j = 0; j<=t_my_simd::bits-1; ++j) {
      k = j ^ (1 << (t_int)(sw1));
      if (t_my_simd::extract(t_subword::bit, x, j) !=
          t_my_simd::extract(t_subword::bit, res, k)) {
        cout << "t_my_simd::bit_index_complement " << (t_int)(sw1) << endl;
        return false;
        }
      }

    res = t_my_simd::bit_index_swap(x, sw1, sw2);

    {
      t_int i1, i2, delta, t1, t2, m;
      i1 = M_min((t_int)(sw1), (t_int)(sw2));
      i2 = M_max((t_int)(sw1), (t_int)(sw2));
      delta = i2 - i1;
      m = 1 << i1;

      for (j = 0; j<=t_my_simd::bits-1; ++j) {
        if (sw1 == sw2)
          k = j;
        else {
          // k: exchange bits sw1 and sw2 of j, see HD 2-20 "Exchanging Registers"
          t1 = (j ^ (j >> delta)) & m;
          t2 = t1 << delta;
          k = j ^ t1 ^ t2;
          }
        if (t_my_simd::extract(t_subword::bit, x, j) !=
            t_my_simd::extract(t_subword::bit, res, k)) {
          cout << "t_my_simd::bit_index_swap " << (t_int)(sw1) << " " << (t_int)(sw2) << endl;
          return false;
          }
        }
      }

    res = t_my_simd::bit_index_swap_complement(x, sw1, sw2);

    {
      t_int i1, i2, delta, t1, t2, m;
      i1 = M_min((t_int)(sw1), (t_int)(sw2));
      i2 = M_max((t_int)(sw1), (t_int)(sw2));
      delta = i2 - i1;
      m = 1 << i1;

      for (j = 0; j<=t_my_simd::bits-1; ++j) {
        if (sw1 == sw2)
          k = j;
        else {
          // k: exchange bits sw1 and sw2 of j, see HD 2-20 "Exchanging Registers"
          t1 = (j ^ (j >> delta)) & m;
          t2 = t1 << delta;
          k = j ^ t1 ^ t2;
          // complement bits sw1 and sw2
          k = k ^ (1 << (t_int)(sw1)) ^ (1 << (t_int)(sw2));
          }
        if (t_my_simd::extract(t_subword::bit, x, j) !=
            t_my_simd::extract(t_subword::bit, res, k)) {
          cout << "t_my_simd::bit_index_swap_complement " << (t_int)(sw1) << " " << (t_int)(sw2) << endl;
          return false;
          }
        }
      }

    res = t_my_simd::bit_index_ror(x, s2,
      t_subword::bit,
      (t_subword)(t_my_simd::ld_bits));

    for (j = 0; j<=t_my_simd::bits-1; ++j) {
      k = ror_ex(j, s2, t_my_simd::ld_bits);
      if (t_my_simd::extract(t_subword::bit, x, j) !=
          t_my_simd::extract(t_subword::bit, res, k)) {
        cout << "t_my_simd::bit_index_ror simple " << s2 <<
          " " << j <<
          " " << k <<
          endl;
        return false;
        }
      }

    if ((t_int)(sw1) + (t_int)(sw2) <= t_my_simd::ld_bits) {
      res = t_my_simd::bit_index_ror(x, s2, sw1, sw2);

      for (j = 0; j<=t_my_simd::bits-1; ++j) {
        k = ror_ex2(j, s2, (t_int)(sw1), (t_int)(sw2));
        if (t_my_simd::extract(t_subword::bit, x, j) !=
            t_my_simd::extract(t_subword::bit, res, k)) {
          cout << "t_my_simd::bit_index_ror " << s2 <<
            " " << j <<
            " " << k <<
            endl;
          return false;
          }
        }
      }

    // transpose

    x1 = x;
    for (j = 0; j<=t_my_simd::bits*2; ++j) {
      if (x1 != t_my_simd::shuffle_power(x, sw1, sw2, j)) {
        cout << "t_my_simd::shuffle_power " << (t_int)(sw1) << " " << (t_int)(sw2) << endl;
        return false;
        }
      x1 = t_my_simd::shuffle(x1, sw1, sw2);
      }

    x1 = x;
    for (j = 0; j<=t_my_simd::bits*2; ++j) {
      if (x1 != t_my_simd::unshuffle_power(x, sw1, sw2, j)) {
        cout << "t_my_simd::unshuffle_power " << (t_int)(sw1) << " " << (t_int)(sw2) << endl;
        return false;
        }
      x1 = t_my_simd::unshuffle(x1, sw1, sw2);
      }

    res = t_my_simd::general_reverse_bits(x, y);

    x1 = x;
    for (j = 0; j<=t_my_simd::ld_bits-1; ++j) {
      if (((y >> j) & 1) !=0) {
        x1 = t_my_simd::bit_index_complement(x1, (t_subword)(j));
        }
      }

    if (res != x1) {
      cout << "t_my_simd::general_reverse_bits " << endl;
      return false;
      }

    res = t_my_simd::bswap(x);

    if (res != t_my_simd::general_reverse_bits(x, ~7)) {
      cout << "t_my_simd::bswap " << endl;
      return false;
      }

    // prim_swap

    if (t_my_simd::unshuffle(t_my_simd::shuffle(x, sw1, sw2), sw1, sw2) != x) {
      cout << "t_my_simd::shuffle " << (t_int)(sw1) << " " << (t_int)(sw2) << endl;
      return false;
      }

    if (t_my_simd::shuffle(t_my_simd::unshuffle(x, sw1, sw2), sw1, sw2) != x) {
      cout << "t_my_simd::unshuffle " << (t_int)(sw1) << " " << (t_int)(sw2) << endl;
      return false;
      }

    // if (sw2 >= sw1)  // (un)shuffle

    // CEF operations.

    if (t_my_simd::compress_flip_right(sw, x, y) !=
       (T)(
         t_my_simd::compress_right(sw, x, y) |
         t_my_simd::general_reverse_bits(
           t_my_simd::compress_right(sw, x, ~y),
           (1 << (t_int)(sw)) - 1 )) ) {
      cout << "t_my_simd::compress_flip_right " << (t_int)(sw) << endl;
      return false;
      }

    if (t_my_simd::expand_flip_right(sw, t_my_simd::compress_flip_right(sw, x, y), y) != x) {
      cout << "t_my_simd::expand_flip_right " << (t_int)(sw) << endl;
      return false;
      }

    if (t_my_simd::compress_flip_left(sw, x, y) !=
       (T)(
         t_my_simd::compress_left(sw, x, y) |
         t_my_simd::general_reverse_bits(
           t_my_simd::compress_left(sw, x, ~y),
           (1 << (t_int)(sw)) - 1 )) ) {
      cout << "t_my_simd::compress_flip_left " << (t_int)(sw) << endl;
      return false;
      }

    if (t_my_simd::expand_flip_left(sw, t_my_simd::compress_flip_left(sw, x, y), y) != x) {
      cout << "t_my_simd::expand_flip_left " << (t_int)(sw) << endl;
      return false;
      }

    // CE operations.

    if (t_my_simd::compress_right(sw, x, y) != t_my_simd::compress_flip_right(sw, x & y, y)) {
      cout << "t_my_simd::compress_right " << (t_int)(sw) << endl;
      return false;
      }

    if (t_my_simd::expand_right(sw, t_my_simd::compress_right(sw, x, y), y) != (T)(x & y)) {
      cout << "t_my_simd::expand_right " << (t_int)(sw) << endl;
      return false;
      }

    if (t_my_simd::compress_left(sw, x, y) != t_my_simd::compress_flip_left(sw, x & y, y)) {
      cout << "t_my_simd::compress_left " << (t_int)(sw) << endl;
      return false;
      }

    if (t_my_simd::expand_left(sw, t_my_simd::compress_left(sw, x, y), y) != (T)(x & y)) {
      cout << "t_my_simd::expand_left " << (t_int)(sw) << endl;
      return false;
      }

    {
      t_vrot<T> vrol;
      vrol.gen_vrol(sw, y);
      x1 = vrol.vrol(x);
      x2 = t_my_simd::vrol(sw, x, y);

      if (x1 != x2) {
        cout << "t_vrot::vrol " << (t_int)(sw) << endl;
        return false;
        }
      }

    {
      t_vrot<T> vror;
      vror.gen_vrol(sw, y);
      x1 = vror.vror(x);
      x2 = t_my_simd::vror(sw, x, y);

      if (x1 != x2) {
        cout << "t_vrot::vror " << (t_int)(sw) << endl;
        return false;
        }
      }

    x = random_bits<T>();
    y = random_bits<T>();
    z = random_bits<T>();
    }
  return true;
  }

template<typename T>
static auto test_horizontal() -> bool {
  typedef t_simd<T> t_my_simd;

  t_int i;
  T x, z;
  T x1, x2;
  t_subword sw;
  t_int j, b, s;
  T m, m1;

  cout << "test_horizontal" << endl;
  for (i = 1; i<=loop_count; ++i) {
    x = random_bits<T>();
    sw = (t_subword)(random_int(t_my_simd::ld_bits));

    b = 1 << (t_int)(sw);
    m = t_my_simd::a_element[(t_int)(sw)];
    m1 = t_my_simd::a_element[(t_int)(sw)+1];

    z = t_my_simd::hand(sw, x);
    for (j = 0; j<=((t_int)(t_my_simd::bits) >> ((t_int)(sw)+1))-1; ++j) {
      s = b*(j*2);
      x1 = (z >> s) & m1;
      x2 = ((x >> s) & m) & ((x >> (s+b)) & m);
      if (x1 != x2) {
        cout << "hand " << (t_int)(sw) << endl;
        return false;
        }
      }

    z = t_my_simd::hor(sw, x);
    for (j = 0; j<=((t_int)(t_my_simd::bits) >> ((t_int)(sw)+1))-1; ++j) {
      s = b*(j*2);
      x1 = (z >> s) & m1;
      x2 = ((x >> s) & m) | ((x >> (s+b)) & m);
      if (x1 != x2) {
        cout << "hor " << (t_int)(sw) << endl;
        return false;
        }
      }

    z = t_my_simd::hxor(sw, x);
    for (j = 0; j<=((t_int)(t_my_simd::bits) >> ((t_int)(sw)+1))-1; ++j) {
      s = b*(j*2);
      x1 = (z >> s) & m1;
      x2 = ((x >> s) & m) ^ ((x >> (s+b)) & m);
      if (x1 != x2) {
        cout << "hxor " << (t_int)(sw) << endl;
        return false;
        }
      }

    z = t_my_simd::hadd_u(sw, x);
    for (j = 0; j<=((t_int)(t_my_simd::bits) >> ((t_int)(sw)+1))-1; ++j) {
      s = b*(j*2);
      x1 = (z >> s) & m1;
      x2 = ((x >> s) & m) + ((x >> (s+b)) & m);
      if (x1 != x2) {
        cout << "hadd_u " << (t_int)(sw) << endl;
        return false;
        }
      }

    z = t_my_simd::hadd_s(sw, x);
    for (j = 0; j<=((t_int)(t_my_simd::bits) >> ((t_int)(sw)+1))-1; ++j) {
      s = b*(j*2);
      x1 = sign_extend((z >> s) & m1, b*2);
      x2 = sign_extend((x >> s) & m, b) + sign_extend((x >> (s+b)) & m, b);
      if (x1 != x2) {
        cout << "hadd_s " << (t_int)(sw) << endl;
        return false;
        }
      }

    z = t_my_simd::hsub_u(sw, x);
    for (j = 0; j<=((t_int)(t_my_simd::bits) >> ((t_int)(sw)+1))-1; ++j) {
      s = b*(j*2);
      x1 = sign_extend((z >> s) & m1, b*2);
      x2 = ((x >> s) & m) - ((x >> (s+b)) & m);
      if (x1 != x2) {
        cout << "hsub_u " << (t_int)(sw) << endl;
        return false;
        }
      }

    z = t_my_simd::hsub_s(sw, x);
    for (j = 0; j<=((t_int)(t_my_simd::bits) >> ((t_int)(sw)+1))-1; ++j) {
      s = b*(j*2);
      x1 = sign_extend((z >> s) & m1, b*2);
      x2 = sign_extend((x >> s) & m, b) - sign_extend((x >> (s+b)) & m, b);
      if (x1 != x2) {
        cout << "hsub_s " << (t_int)(sw) << endl;
        return false;
        }
      }

    z = t_my_simd::havgd_u(sw, x);
    for (j = 0; j<=((t_int)(t_my_simd::bits) >> ((t_int)(sw)+1))-1; ++j) {
      s = b*(j*2);
      x1 = (z >> s) & m1;
      x2 = M_avgd_u(((x >> s) & m), ((x >> (s+b)) & m));
      if (x1 != x2) {
        cout << "havgd_u " << (t_int)(sw) << endl;
        return false;
        }
      }

    z = t_my_simd::havgu_u(sw, x);
    for (j = 0; j<=((t_int)(t_my_simd::bits) >> ((t_int)(sw)+1))-1; ++j) {
      s = b*(j*2);
      x1 = (z >> s) & m1;
      x2 = M_avgu_u(((x >> s) & m), ((x >> (s+b)) & m));
      if (x1 != x2) {
        cout << "havgu_u " << (t_int)(sw) << endl;
        return false;
        }
      }

    z = t_my_simd::hmax_u(sw, x);
    for (j = 0; j<=((t_int)(t_my_simd::bits) >> ((t_int)(sw)+1))-1; ++j) {
      s = b*(j*2);
      x1 = (z >> s) & m1;
      x2 = M_max( ((x >> s) & m) , ((x >> (s+b)) & m) );
      if (x1 != x2) {
        cout << "hmax_u " << (t_int)(sw) << endl;
        return false;
        }
      }

    z = t_my_simd::hmin_u(sw, x);
    for (j = 0; j<=((t_int)(t_my_simd::bits) >> ((t_int)(sw)+1))-1; ++j) {
      s = b*(j*2);
      x1 = (z >> s) & m1;
      x2 = M_min( ((x >> s) & m) , ((x >> (s+b)) & m) );
      if (x1 != x2) {
        cout << "hmin_u " << (t_int)(sw) << endl;
        return false;
        }
      }

    z = t_my_simd::hmax_s(sw, x);
    for (j = 0; j<=((t_int)(t_my_simd::bits) >> ((t_int)(sw)+1))-1; ++j) {
      s = b*(j*2);
      x1 = sign_extend((z >> s) & m1, b*2);
      x2 = M_max( sign_extend((x >> s) & m, b), sign_extend((x >> (s+b)) & m, b));
      if (x1 != x2) {
        cout << "hmax_s " << (t_int)(sw) << endl;
        return false;
        }
      }

    z = t_my_simd::hmin_s(sw, x);
    for (j = 0; j<=((t_int)(t_my_simd::bits) >> ((t_int)(sw)+1))-1; ++j) {
      s = b*(j*2);
      x1 = sign_extend((z >> s) & m1, b*2);
      x2 = M_min( sign_extend((x >> s) & m, b), sign_extend((x >> (s+b)) & m, b));
      if (x1 != x2) {
        cout << "hmin_s " << (t_int)(sw) << endl;
        return false;
        }
      }

    z = t_my_simd::hmovzx(sw, x);
    for (j = 0; j<=((t_int)(t_my_simd::bits) >> ((t_int)(sw)+1))-1; ++j) {
      s = b*(j*2);
      x1 = (z >> s) & m1;
      x2 = (x >> s) & m;
      if (x1 != x2) {
        cout << "hmovzx " << (t_int)(sw) << endl;
        return false;
        }
      }

    z = t_my_simd::hmovsx(sw, x);
    for (j = 0; j<=((t_int)(t_my_simd::bits) >> ((t_int)(sw)+1))-1; ++j) {
      s = b*(j*2);
      x1 = sign_extend((z >> s) & m1, b*2);
      x2 = sign_extend((x >> s) & m, b);
      if (x1 != x2) {
        cout << "hmovsx " << (t_int)(sw) << endl;
        return false;
        }
      }

    z = t_my_simd::hmul_u(sw, x);
    for (j = 0; j<=((t_int)(t_my_simd::bits) >> ((t_int)(sw)+1))-1; ++j) {
      s = b*(j*2);
      x1 = (z >> s) & m1;
      x2 = ((x >> s) & m) * ((x >> (s+b)) & m);
      if (x1 != x2) {
        cout << "hmul_u " << (t_int)(sw) << endl;
        // cout << "x = " << (t_hugeuint)(x) << endl;
        // cout << "r = " << (t_hugeuint)(z) << endl;
        return false;
        }
      }

    z = t_my_simd::hmul_s(sw, x);
    for (j = 0; j<=((t_int)(t_my_simd::bits) >> ((t_int)(sw)+1))-1; ++j) {
      s = b*(j*2);
      x1 = sign_extend((z >> s) & m1, b*2);
      x2 = sign_extend((x >> s) & m, b) * sign_extend((x >> (s+b)) & m, b);
      if (x1 != x2) {
        cout << "hmul_s " << (t_int)(sw) << endl;
        // cout << "x = " << (t_hugeuint)(x) << endl;
        // cout << "r = " << (t_hugeuint)(z) << endl;
        return false;
        }
      }

    }
  return true;
  }

// gen_vrol => test_swar

// vrol => test_swar
// vror => test_swar

template<typename T>
static auto self_test(const char* s) -> bool {
  cout << "Testing " << s << endl;
  if (!test_bits_base<T>())
    return false;
  if (!test_swar<T>())
    return false;
  if (!test_horizontal<T>())
    return false;
  cout << "OK" << endl;

  // Experiment for ccl_mul_ex / ccl_mul_inv_ex
  {
    typedef t_bits<T> t_my_bits;

    int i,n;
    T x,y;
    bool ok;
    for (n=1; n<=t_my_bits::bits; ++n) {
      ok=true;
      for (i=0; i<=10000; ++i) {
        x=random_bits<T>() & t_my_bits::mask_ex(n);
        y=t_my_bits::ccl_power_ex(x,n-1,n);
        if (t_my_bits::ccl_mul_ex(x,y,n)!=1)
          y=0;
        if (t_my_bits::ccl_mul_inv_ex(x,n)!=y) {
          ok=false;
          break;
          }
        }
      if (ok)
        cout << n << " ";  // All results are differences of 2 powers of 2.
      }
    cout << endl;
    }
  return true;
  }

//#include <stdio.h>
//#include <stdlib.h>
#include <time.h>

auto main() -> int {

  srand((unsigned int)(time(NULL)));

  if (!self_test<t_8u>("t_8u")) {
    cout << "ERROR";
    return 1;
    }
  if (!self_test<t_16u>("t_16u")) {
    cout << "ERROR";
    return 1;
    }
  if (!self_test<t_32u>("t_32u")) {
    cout << "ERROR";
    return 1;
    }
  if (!self_test<t_64u>("t_64u")) {
    cout << "ERROR";
    return 1;
    }
#ifdef has_128
  if (!self_test<t_128u>("t_128u")) {
    cout << "ERROR";
    return 1;
    }
#endif
  return 0;
  }

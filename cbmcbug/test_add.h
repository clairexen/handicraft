#include <stdint.h>
#include <stdbool.h>

#ifndef YOSYS_SIMPLEC_SIGNAL32_T
#define YOSYS_SIMPLEC_SIGNAL32_T
typedef struct {
  uint32_t value_31_0 : 32;
} signal32_t;
#endif

#ifndef YOSYS_SIMPLEC_SIGNAL1_T
#define YOSYS_SIMPLEC_SIGNAL1_T
typedef struct {
  uint8_t value_0_0 : 1;
} signal1_t;
#endif

#ifndef YOSYS_SIMPLEC_SIGNAL4_T
#define YOSYS_SIMPLEC_SIGNAL4_T
typedef struct {
  uint8_t value_3_0 : 4;
} signal4_t;
#endif

#ifndef YOSYS_SIMPLEC_SIGNAL5_T
#define YOSYS_SIMPLEC_SIGNAL5_T
typedef struct {
  uint8_t value_4_0 : 5;
} signal5_t;
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_0_OF_32
#define YOSYS_SIMPLEC_SET_BIT_0_OF_32
static inline void yosys_simplec_set_bit_0_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 0)) | ((uint64_t)value << 0);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_1_OF_32
#define YOSYS_SIMPLEC_SET_BIT_1_OF_32
static inline void yosys_simplec_set_bit_1_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 1)) | ((uint64_t)value << 1);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_2_OF_32
#define YOSYS_SIMPLEC_SET_BIT_2_OF_32
static inline void yosys_simplec_set_bit_2_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 2)) | ((uint64_t)value << 2);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_3_OF_32
#define YOSYS_SIMPLEC_SET_BIT_3_OF_32
static inline void yosys_simplec_set_bit_3_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 3)) | ((uint64_t)value << 3);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_4_OF_32
#define YOSYS_SIMPLEC_SET_BIT_4_OF_32
static inline void yosys_simplec_set_bit_4_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 4)) | ((uint64_t)value << 4);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_5_OF_32
#define YOSYS_SIMPLEC_SET_BIT_5_OF_32
static inline void yosys_simplec_set_bit_5_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 5)) | ((uint64_t)value << 5);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_6_OF_32
#define YOSYS_SIMPLEC_SET_BIT_6_OF_32
static inline void yosys_simplec_set_bit_6_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 6)) | ((uint64_t)value << 6);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_7_OF_32
#define YOSYS_SIMPLEC_SET_BIT_7_OF_32
static inline void yosys_simplec_set_bit_7_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 7)) | ((uint64_t)value << 7);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_8_OF_32
#define YOSYS_SIMPLEC_SET_BIT_8_OF_32
static inline void yosys_simplec_set_bit_8_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 8)) | ((uint64_t)value << 8);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_9_OF_32
#define YOSYS_SIMPLEC_SET_BIT_9_OF_32
static inline void yosys_simplec_set_bit_9_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 9)) | ((uint64_t)value << 9);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_10_OF_32
#define YOSYS_SIMPLEC_SET_BIT_10_OF_32
static inline void yosys_simplec_set_bit_10_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 10)) | ((uint64_t)value << 10);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_11_OF_32
#define YOSYS_SIMPLEC_SET_BIT_11_OF_32
static inline void yosys_simplec_set_bit_11_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 11)) | ((uint64_t)value << 11);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_12_OF_32
#define YOSYS_SIMPLEC_SET_BIT_12_OF_32
static inline void yosys_simplec_set_bit_12_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 12)) | ((uint64_t)value << 12);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_13_OF_32
#define YOSYS_SIMPLEC_SET_BIT_13_OF_32
static inline void yosys_simplec_set_bit_13_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 13)) | ((uint64_t)value << 13);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_14_OF_32
#define YOSYS_SIMPLEC_SET_BIT_14_OF_32
static inline void yosys_simplec_set_bit_14_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 14)) | ((uint64_t)value << 14);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_15_OF_32
#define YOSYS_SIMPLEC_SET_BIT_15_OF_32
static inline void yosys_simplec_set_bit_15_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 15)) | ((uint64_t)value << 15);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_16_OF_32
#define YOSYS_SIMPLEC_SET_BIT_16_OF_32
static inline void yosys_simplec_set_bit_16_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 16)) | ((uint64_t)value << 16);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_17_OF_32
#define YOSYS_SIMPLEC_SET_BIT_17_OF_32
static inline void yosys_simplec_set_bit_17_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 17)) | ((uint64_t)value << 17);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_18_OF_32
#define YOSYS_SIMPLEC_SET_BIT_18_OF_32
static inline void yosys_simplec_set_bit_18_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 18)) | ((uint64_t)value << 18);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_19_OF_32
#define YOSYS_SIMPLEC_SET_BIT_19_OF_32
static inline void yosys_simplec_set_bit_19_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 19)) | ((uint64_t)value << 19);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_20_OF_32
#define YOSYS_SIMPLEC_SET_BIT_20_OF_32
static inline void yosys_simplec_set_bit_20_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 20)) | ((uint64_t)value << 20);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_21_OF_32
#define YOSYS_SIMPLEC_SET_BIT_21_OF_32
static inline void yosys_simplec_set_bit_21_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 21)) | ((uint64_t)value << 21);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_22_OF_32
#define YOSYS_SIMPLEC_SET_BIT_22_OF_32
static inline void yosys_simplec_set_bit_22_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 22)) | ((uint64_t)value << 22);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_23_OF_32
#define YOSYS_SIMPLEC_SET_BIT_23_OF_32
static inline void yosys_simplec_set_bit_23_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 23)) | ((uint64_t)value << 23);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_24_OF_32
#define YOSYS_SIMPLEC_SET_BIT_24_OF_32
static inline void yosys_simplec_set_bit_24_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 24)) | ((uint64_t)value << 24);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_25_OF_32
#define YOSYS_SIMPLEC_SET_BIT_25_OF_32
static inline void yosys_simplec_set_bit_25_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 25)) | ((uint64_t)value << 25);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_26_OF_32
#define YOSYS_SIMPLEC_SET_BIT_26_OF_32
static inline void yosys_simplec_set_bit_26_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 26)) | ((uint64_t)value << 26);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_27_OF_32
#define YOSYS_SIMPLEC_SET_BIT_27_OF_32
static inline void yosys_simplec_set_bit_27_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 27)) | ((uint64_t)value << 27);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_28_OF_32
#define YOSYS_SIMPLEC_SET_BIT_28_OF_32
static inline void yosys_simplec_set_bit_28_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 28)) | ((uint64_t)value << 28);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_29_OF_32
#define YOSYS_SIMPLEC_SET_BIT_29_OF_32
static inline void yosys_simplec_set_bit_29_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 29)) | ((uint64_t)value << 29);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_30_OF_32
#define YOSYS_SIMPLEC_SET_BIT_30_OF_32
static inline void yosys_simplec_set_bit_30_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 30)) | ((uint64_t)value << 30);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_31_OF_32
#define YOSYS_SIMPLEC_SET_BIT_31_OF_32
static inline void yosys_simplec_set_bit_31_of_32(signal32_t *sig, bool value)
{
    sig->value_31_0 = (sig->value_31_0 & ~((uint64_t)1 << 31)) | ((uint64_t)value << 31);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_0_OF_4
#define YOSYS_SIMPLEC_SET_BIT_0_OF_4
static inline void yosys_simplec_set_bit_0_of_4(signal4_t *sig, bool value)
{
    sig->value_3_0 = (sig->value_3_0 & ~((uint64_t)1 << 0)) | ((uint64_t)value << 0);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_1_OF_4
#define YOSYS_SIMPLEC_SET_BIT_1_OF_4
static inline void yosys_simplec_set_bit_1_of_4(signal4_t *sig, bool value)
{
    sig->value_3_0 = (sig->value_3_0 & ~((uint64_t)1 << 1)) | ((uint64_t)value << 1);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_2_OF_4
#define YOSYS_SIMPLEC_SET_BIT_2_OF_4
static inline void yosys_simplec_set_bit_2_of_4(signal4_t *sig, bool value)
{
    sig->value_3_0 = (sig->value_3_0 & ~((uint64_t)1 << 2)) | ((uint64_t)value << 2);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_3_OF_4
#define YOSYS_SIMPLEC_SET_BIT_3_OF_4
static inline void yosys_simplec_set_bit_3_of_4(signal4_t *sig, bool value)
{
    sig->value_3_0 = (sig->value_3_0 & ~((uint64_t)1 << 3)) | ((uint64_t)value << 3);
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_26_OF_32
#define YOSYS_SIMPLEC_GET_BIT_26_OF_32
static inline bool yosys_simplec_get_bit_26_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 26) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_25_OF_32
#define YOSYS_SIMPLEC_GET_BIT_25_OF_32
static inline bool yosys_simplec_get_bit_25_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 25) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_28_OF_32
#define YOSYS_SIMPLEC_GET_BIT_28_OF_32
static inline bool yosys_simplec_get_bit_28_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 28) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_27_OF_32
#define YOSYS_SIMPLEC_GET_BIT_27_OF_32
static inline bool yosys_simplec_get_bit_27_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 27) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_30_OF_32
#define YOSYS_SIMPLEC_GET_BIT_30_OF_32
static inline bool yosys_simplec_get_bit_30_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 30) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_29_OF_32
#define YOSYS_SIMPLEC_GET_BIT_29_OF_32
static inline bool yosys_simplec_get_bit_29_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 29) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_31_OF_32
#define YOSYS_SIMPLEC_GET_BIT_31_OF_32
static inline bool yosys_simplec_get_bit_31_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 31) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_13_OF_32
#define YOSYS_SIMPLEC_GET_BIT_13_OF_32
static inline bool yosys_simplec_get_bit_13_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 13) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_12_OF_32
#define YOSYS_SIMPLEC_GET_BIT_12_OF_32
static inline bool yosys_simplec_get_bit_12_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 12) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_14_OF_32
#define YOSYS_SIMPLEC_GET_BIT_14_OF_32
static inline bool yosys_simplec_get_bit_14_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 14) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_1_OF_32
#define YOSYS_SIMPLEC_GET_BIT_1_OF_32
static inline bool yosys_simplec_get_bit_1_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 1) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_0_OF_32
#define YOSYS_SIMPLEC_GET_BIT_0_OF_32
static inline bool yosys_simplec_get_bit_0_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 0) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_3_OF_32
#define YOSYS_SIMPLEC_GET_BIT_3_OF_32
static inline bool yosys_simplec_get_bit_3_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 3) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_2_OF_32
#define YOSYS_SIMPLEC_GET_BIT_2_OF_32
static inline bool yosys_simplec_get_bit_2_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 2) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_5_OF_32
#define YOSYS_SIMPLEC_GET_BIT_5_OF_32
static inline bool yosys_simplec_get_bit_5_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 5) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_4_OF_32
#define YOSYS_SIMPLEC_GET_BIT_4_OF_32
static inline bool yosys_simplec_get_bit_4_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 4) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_6_OF_32
#define YOSYS_SIMPLEC_GET_BIT_6_OF_32
static inline bool yosys_simplec_get_bit_6_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 6) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_8_OF_32
#define YOSYS_SIMPLEC_GET_BIT_8_OF_32
static inline bool yosys_simplec_get_bit_8_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 8) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_7_OF_32
#define YOSYS_SIMPLEC_GET_BIT_7_OF_32
static inline bool yosys_simplec_get_bit_7_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 7) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_10_OF_32
#define YOSYS_SIMPLEC_GET_BIT_10_OF_32
static inline bool yosys_simplec_get_bit_10_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 10) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_9_OF_32
#define YOSYS_SIMPLEC_GET_BIT_9_OF_32
static inline bool yosys_simplec_get_bit_9_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 9) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_11_OF_32
#define YOSYS_SIMPLEC_GET_BIT_11_OF_32
static inline bool yosys_simplec_get_bit_11_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 11) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_15_OF_32
#define YOSYS_SIMPLEC_GET_BIT_15_OF_32
static inline bool yosys_simplec_get_bit_15_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 15) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_16_OF_32
#define YOSYS_SIMPLEC_GET_BIT_16_OF_32
static inline bool yosys_simplec_get_bit_16_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 16) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_21_OF_32
#define YOSYS_SIMPLEC_GET_BIT_21_OF_32
static inline bool yosys_simplec_get_bit_21_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 21) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_17_OF_32
#define YOSYS_SIMPLEC_GET_BIT_17_OF_32
static inline bool yosys_simplec_get_bit_17_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 17) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_18_OF_32
#define YOSYS_SIMPLEC_GET_BIT_18_OF_32
static inline bool yosys_simplec_get_bit_18_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 18) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_22_OF_32
#define YOSYS_SIMPLEC_GET_BIT_22_OF_32
static inline bool yosys_simplec_get_bit_22_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 22) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_20_OF_32
#define YOSYS_SIMPLEC_GET_BIT_20_OF_32
static inline bool yosys_simplec_get_bit_20_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 20) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_19_OF_32
#define YOSYS_SIMPLEC_GET_BIT_19_OF_32
static inline bool yosys_simplec_get_bit_19_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 19) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_24_OF_32
#define YOSYS_SIMPLEC_GET_BIT_24_OF_32
static inline bool yosys_simplec_get_bit_24_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 24) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_GET_BIT_23_OF_32
#define YOSYS_SIMPLEC_GET_BIT_23_OF_32
static inline bool yosys_simplec_get_bit_23_of_32(const signal32_t *sig)
{
  return (sig->value_31_0 >> 23) & 1;
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_0_OF_5
#define YOSYS_SIMPLEC_SET_BIT_0_OF_5
static inline void yosys_simplec_set_bit_0_of_5(signal5_t *sig, bool value)
{
    sig->value_4_0 = (sig->value_4_0 & ~((uint64_t)1 << 0)) | ((uint64_t)value << 0);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_1_OF_5
#define YOSYS_SIMPLEC_SET_BIT_1_OF_5
static inline void yosys_simplec_set_bit_1_of_5(signal5_t *sig, bool value)
{
    sig->value_4_0 = (sig->value_4_0 & ~((uint64_t)1 << 1)) | ((uint64_t)value << 1);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_2_OF_5
#define YOSYS_SIMPLEC_SET_BIT_2_OF_5
static inline void yosys_simplec_set_bit_2_of_5(signal5_t *sig, bool value)
{
    sig->value_4_0 = (sig->value_4_0 & ~((uint64_t)1 << 2)) | ((uint64_t)value << 2);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_3_OF_5
#define YOSYS_SIMPLEC_SET_BIT_3_OF_5
static inline void yosys_simplec_set_bit_3_of_5(signal5_t *sig, bool value)
{
    sig->value_4_0 = (sig->value_4_0 & ~((uint64_t)1 << 3)) | ((uint64_t)value << 3);
}
#endif

#ifndef YOSYS_SIMPLEC_SET_BIT_4_OF_5
#define YOSYS_SIMPLEC_SET_BIT_4_OF_5
static inline void yosys_simplec_set_bit_4_of_5(signal5_t *sig, bool value)
{
    sig->value_4_0 = (sig->value_4_0 & ~((uint64_t)1 << 4)) | ((uint64_t)value << 4);
}
#endif

#ifndef YOSYS_SIMPLEC_RVFI_INSN_ADD_STATE_T
#define YOSYS_SIMPLEC_RVFI_INSN_ADD_STATE_T
struct rvfi_insn_add_state_t
{
  // Input Ports
  signal32_t rvfi_insn; // rvfi_insn
  signal32_t rvfi_mem_rdata; // rvfi_mem_rdata
  signal32_t rvfi_pc_rdata; // rvfi_pc_rdata
  signal32_t rvfi_rs1_rdata; // rvfi_rs1_rdata
  signal32_t rvfi_rs2_rdata; // rvfi_rs2_rdata
  signal1_t rvfi_valid; // rvfi_valid

  // Output Ports
  signal32_t spec_mem_addr; // spec_mem_addr
  signal4_t spec_mem_rmask; // spec_mem_rmask
  signal32_t spec_mem_wdata; // spec_mem_wdata
  signal4_t spec_mem_wmask; // spec_mem_wmask
  signal32_t spec_pc_wdata; // spec_pc_wdata
  signal5_t spec_rd_addr; // spec_rd_addr
  signal32_t spec_rd_wdata; // spec_rd_wdata
  signal5_t spec_rs1_addr; // spec_rs1_addr
  signal5_t spec_rs2_addr; // spec_rs2_addr
  signal1_t spec_trap; // spec_trap
  signal1_t spec_valid; // spec_valid

  // Internal Wires
  signal1_t _abc_1000_n181_1; // $abc$1000$n181_1
  signal1_t _abc_1000_n182; // $abc$1000$n182
  signal1_t _abc_1000_n183_1; // $abc$1000$n183_1
  signal1_t _abc_1000_n184_1; // $abc$1000$n184_1
  signal1_t _abc_1000_n185; // $abc$1000$n185
  signal1_t _abc_1000_n186_1; // $abc$1000$n186_1
  signal1_t _abc_1000_n187_1; // $abc$1000$n187_1
  signal1_t _abc_1000_n188; // $abc$1000$n188
  signal1_t _abc_1000_n189_1; // $abc$1000$n189_1
  signal1_t _abc_1000_n190_1; // $abc$1000$n190_1
  signal1_t _abc_1000_n191; // $abc$1000$n191
  signal1_t _abc_1000_n192_1; // $abc$1000$n192_1
  signal1_t _abc_1000_n193_1; // $abc$1000$n193_1
  signal1_t _abc_1000_n194; // $abc$1000$n194
  signal1_t _abc_1000_n195_1; // $abc$1000$n195_1
  signal1_t _abc_1000_n196_1; // $abc$1000$n196_1
  signal1_t _abc_1000_n198_1; // $abc$1000$n198_1
  signal1_t _abc_1000_n199_1; // $abc$1000$n199_1
  signal1_t _abc_1000_n200; // $abc$1000$n200
  signal1_t _abc_1000_n201_1; // $abc$1000$n201_1
  signal1_t _abc_1000_n202_1; // $abc$1000$n202_1
  signal1_t _abc_1000_n204_1; // $abc$1000$n204_1
  signal1_t _abc_1000_n205_1; // $abc$1000$n205_1
  signal1_t _abc_1000_n206; // $abc$1000$n206
  signal1_t _abc_1000_n208; // $abc$1000$n208
  signal1_t _abc_1000_n209; // $abc$1000$n209
  signal1_t _abc_1000_n210; // $abc$1000$n210
  signal1_t _abc_1000_n211; // $abc$1000$n211
  signal1_t _abc_1000_n213; // $abc$1000$n213
  signal1_t _abc_1000_n214; // $abc$1000$n214
  signal1_t _abc_1000_n215; // $abc$1000$n215
  signal1_t _abc_1000_n216; // $abc$1000$n216
  signal1_t _abc_1000_n217; // $abc$1000$n217
  signal1_t _abc_1000_n219; // $abc$1000$n219
  signal1_t _abc_1000_n220; // $abc$1000$n220
  signal1_t _abc_1000_n221; // $abc$1000$n221
  signal1_t _abc_1000_n222; // $abc$1000$n222
  signal1_t _abc_1000_n223; // $abc$1000$n223
  signal1_t _abc_1000_n224; // $abc$1000$n224
  signal1_t _abc_1000_n226; // $abc$1000$n226
  signal1_t _abc_1000_n227; // $abc$1000$n227
  signal1_t _abc_1000_n228; // $abc$1000$n228
  signal1_t _abc_1000_n229; // $abc$1000$n229
  signal1_t _abc_1000_n231; // $abc$1000$n231
  signal1_t _abc_1000_n232; // $abc$1000$n232
  signal1_t _abc_1000_n233; // $abc$1000$n233
  signal1_t _abc_1000_n234; // $abc$1000$n234
  signal1_t _abc_1000_n235; // $abc$1000$n235
  signal1_t _abc_1000_n236; // $abc$1000$n236
  signal1_t _abc_1000_n237; // $abc$1000$n237
  signal1_t _abc_1000_n239; // $abc$1000$n239
  signal1_t _abc_1000_n240; // $abc$1000$n240
  signal1_t _abc_1000_n241; // $abc$1000$n241
  signal1_t _abc_1000_n242; // $abc$1000$n242
  signal1_t _abc_1000_n243; // $abc$1000$n243
  signal1_t _abc_1000_n244; // $abc$1000$n244
  signal1_t _abc_1000_n246; // $abc$1000$n246
  signal1_t _abc_1000_n247; // $abc$1000$n247
  signal1_t _abc_1000_n248; // $abc$1000$n248
  signal1_t _abc_1000_n249; // $abc$1000$n249
  signal1_t _abc_1000_n250; // $abc$1000$n250
  signal1_t _abc_1000_n251; // $abc$1000$n251
  signal1_t _abc_1000_n252; // $abc$1000$n252
  signal1_t _abc_1000_n253; // $abc$1000$n253
  signal1_t _abc_1000_n255; // $abc$1000$n255
  signal1_t _abc_1000_n256; // $abc$1000$n256
  signal1_t _abc_1000_n257; // $abc$1000$n257
  signal1_t _abc_1000_n258; // $abc$1000$n258
  signal1_t _abc_1000_n259; // $abc$1000$n259
  signal1_t _abc_1000_n261; // $abc$1000$n261
  signal1_t _abc_1000_n262; // $abc$1000$n262
  signal1_t _abc_1000_n263; // $abc$1000$n263
  signal1_t _abc_1000_n264; // $abc$1000$n264
  signal1_t _abc_1000_n265; // $abc$1000$n265
  signal1_t _abc_1000_n266; // $abc$1000$n266
  signal1_t _abc_1000_n267; // $abc$1000$n267
  signal1_t _abc_1000_n268; // $abc$1000$n268
  signal1_t _abc_1000_n270_1; // $abc$1000$n270_1
  signal1_t _abc_1000_n271_1; // $abc$1000$n271_1
  signal1_t _abc_1000_n272_1; // $abc$1000$n272_1
  signal1_t _abc_1000_n273_1; // $abc$1000$n273_1
  signal1_t _abc_1000_n274; // $abc$1000$n274
  signal1_t _abc_1000_n276_1; // $abc$1000$n276_1
  signal1_t _abc_1000_n277; // $abc$1000$n277
  signal1_t _abc_1000_n278_1; // $abc$1000$n278_1
  signal1_t _abc_1000_n279_1; // $abc$1000$n279_1
  signal1_t _abc_1000_n280; // $abc$1000$n280
  signal1_t _abc_1000_n281_1; // $abc$1000$n281_1
  signal1_t _abc_1000_n282_1; // $abc$1000$n282_1
  signal1_t _abc_1000_n283; // $abc$1000$n283
  signal1_t _abc_1000_n284_1; // $abc$1000$n284_1
  signal1_t _abc_1000_n286; // $abc$1000$n286
  signal1_t _abc_1000_n287_1; // $abc$1000$n287_1
  signal1_t _abc_1000_n288_1; // $abc$1000$n288_1
  signal1_t _abc_1000_n289; // $abc$1000$n289
  signal1_t _abc_1000_n291_1; // $abc$1000$n291_1
  signal1_t _abc_1000_n292; // $abc$1000$n292
  signal1_t _abc_1000_n293_1; // $abc$1000$n293_1
  signal1_t _abc_1000_n294_1; // $abc$1000$n294_1
  signal1_t _abc_1000_n295; // $abc$1000$n295
  signal1_t _abc_1000_n296_1; // $abc$1000$n296_1
  signal1_t _abc_1000_n297_1; // $abc$1000$n297_1
  signal1_t _abc_1000_n299_1; // $abc$1000$n299_1
  signal1_t _abc_1000_n300_1; // $abc$1000$n300_1
  signal1_t _abc_1000_n301; // $abc$1000$n301
  signal1_t _abc_1000_n302_1; // $abc$1000$n302_1
  signal1_t _abc_1000_n303_1; // $abc$1000$n303_1
  signal1_t _abc_1000_n304; // $abc$1000$n304
  signal1_t _abc_1000_n306_1; // $abc$1000$n306_1
  signal1_t _abc_1000_n307; // $abc$1000$n307
  signal1_t _abc_1000_n308_1; // $abc$1000$n308_1
  signal1_t _abc_1000_n309_1; // $abc$1000$n309_1
  signal1_t _abc_1000_n310; // $abc$1000$n310
  signal1_t _abc_1000_n311_1; // $abc$1000$n311_1
  signal1_t _abc_1000_n312_1; // $abc$1000$n312_1
  signal1_t _abc_1000_n313; // $abc$1000$n313
  signal1_t _abc_1000_n314_1; // $abc$1000$n314_1
  signal1_t _abc_1000_n315_1; // $abc$1000$n315_1
  signal1_t _abc_1000_n317_1; // $abc$1000$n317_1
  signal1_t _abc_1000_n318_1; // $abc$1000$n318_1
  signal1_t _abc_1000_n319; // $abc$1000$n319
  signal1_t _abc_1000_n320_1; // $abc$1000$n320_1
  signal1_t _abc_1000_n322; // $abc$1000$n322
  signal1_t _abc_1000_n323_1; // $abc$1000$n323_1
  signal1_t _abc_1000_n324_1; // $abc$1000$n324_1
  signal1_t _abc_1000_n325; // $abc$1000$n325
  signal1_t _abc_1000_n326_1; // $abc$1000$n326_1
  signal1_t _abc_1000_n327_1; // $abc$1000$n327_1
  signal1_t _abc_1000_n329_1; // $abc$1000$n329_1
  signal1_t _abc_1000_n330_1; // $abc$1000$n330_1
  signal1_t _abc_1000_n331; // $abc$1000$n331
  signal1_t _abc_1000_n332_1; // $abc$1000$n332_1
  signal1_t _abc_1000_n333_1; // $abc$1000$n333_1
  signal1_t _abc_1000_n335_1; // $abc$1000$n335_1
  signal1_t _abc_1000_n336_1; // $abc$1000$n336_1
  signal1_t _abc_1000_n337; // $abc$1000$n337
  signal1_t _abc_1000_n338_1; // $abc$1000$n338_1
  signal1_t _abc_1000_n339_1; // $abc$1000$n339_1
  signal1_t _abc_1000_n340; // $abc$1000$n340
  signal1_t _abc_1000_n341_1; // $abc$1000$n341_1
  signal1_t _abc_1000_n342_1; // $abc$1000$n342_1
  signal1_t _abc_1000_n343; // $abc$1000$n343
  signal1_t _abc_1000_n345_1; // $abc$1000$n345_1
  signal1_t _abc_1000_n346; // $abc$1000$n346
  signal1_t _abc_1000_n347_1; // $abc$1000$n347_1
  signal1_t _abc_1000_n348_1; // $abc$1000$n348_1
  signal1_t _abc_1000_n349; // $abc$1000$n349
  signal1_t _abc_1000_n351_1; // $abc$1000$n351_1
  signal1_t _abc_1000_n352; // $abc$1000$n352
  signal1_t _abc_1000_n353_1; // $abc$1000$n353_1
  signal1_t _abc_1000_n354_1; // $abc$1000$n354_1
  signal1_t _abc_1000_n355; // $abc$1000$n355
  signal1_t _abc_1000_n356_1; // $abc$1000$n356_1
  signal1_t _abc_1000_n357; // $abc$1000$n357
  signal1_t _abc_1000_n358; // $abc$1000$n358
  signal1_t _abc_1000_n360; // $abc$1000$n360
  signal1_t _abc_1000_n361; // $abc$1000$n361
  signal1_t _abc_1000_n362; // $abc$1000$n362
  signal1_t _abc_1000_n363; // $abc$1000$n363
  signal1_t _abc_1000_n364; // $abc$1000$n364
  signal1_t _abc_1000_n365; // $abc$1000$n365
  signal1_t _abc_1000_n367; // $abc$1000$n367
  signal1_t _abc_1000_n368; // $abc$1000$n368
  signal1_t _abc_1000_n369; // $abc$1000$n369
  signal1_t _abc_1000_n370; // $abc$1000$n370
  signal1_t _abc_1000_n371; // $abc$1000$n371
  signal1_t _abc_1000_n372; // $abc$1000$n372
  signal1_t _abc_1000_n373; // $abc$1000$n373
  signal1_t _abc_1000_n374; // $abc$1000$n374
  signal1_t _abc_1000_n375; // $abc$1000$n375
  signal1_t _abc_1000_n376; // $abc$1000$n376
  signal1_t _abc_1000_n377; // $abc$1000$n377
  signal1_t _abc_1000_n379; // $abc$1000$n379
  signal1_t _abc_1000_n380; // $abc$1000$n380
  signal1_t _abc_1000_n381; // $abc$1000$n381
  signal1_t _abc_1000_n382; // $abc$1000$n382
  signal1_t _abc_1000_n383; // $abc$1000$n383
  signal1_t _abc_1000_n384; // $abc$1000$n384
  signal1_t _abc_1000_n385; // $abc$1000$n385
  signal1_t _abc_1000_n387; // $abc$1000$n387
  signal1_t _abc_1000_n388; // $abc$1000$n388
  signal1_t _abc_1000_n389; // $abc$1000$n389
  signal1_t _abc_1000_n390; // $abc$1000$n390
  signal1_t _abc_1000_n391; // $abc$1000$n391
  signal1_t _abc_1000_n392; // $abc$1000$n392
  signal1_t _abc_1000_n393; // $abc$1000$n393
  signal1_t _abc_1000_n394; // $abc$1000$n394
  signal1_t _abc_1000_n396; // $abc$1000$n396
  signal1_t _abc_1000_n397; // $abc$1000$n397
  signal1_t _abc_1000_n398; // $abc$1000$n398
  signal1_t _abc_1000_n399; // $abc$1000$n399
  signal1_t _abc_1000_n400; // $abc$1000$n400
  signal1_t _abc_1000_n401; // $abc$1000$n401
  signal1_t _abc_1000_n403; // $abc$1000$n403
  signal1_t _abc_1000_n404; // $abc$1000$n404
  signal1_t _abc_1000_n405; // $abc$1000$n405
  signal1_t _abc_1000_n406; // $abc$1000$n406
  signal1_t _abc_1000_n407; // $abc$1000$n407
  signal1_t _abc_1000_n408; // $abc$1000$n408
  signal1_t _abc_1000_n409; // $abc$1000$n409
  signal1_t _abc_1000_n410; // $abc$1000$n410
  signal1_t _abc_1000_n411; // $abc$1000$n411
  signal1_t _abc_1000_n412; // $abc$1000$n412
  signal1_t _abc_1000_n414; // $abc$1000$n414
  signal1_t _abc_1000_n415; // $abc$1000$n415
  signal1_t _abc_1000_n416; // $abc$1000$n416
  signal1_t _abc_1000_n417; // $abc$1000$n417
  signal1_t _abc_1000_n418; // $abc$1000$n418
  signal1_t _abc_1000_n420; // $abc$1000$n420
  signal1_t _abc_1000_n421; // $abc$1000$n421
  signal1_t _abc_1000_n422; // $abc$1000$n422
  signal1_t _abc_1000_n423; // $abc$1000$n423
  signal1_t _abc_1000_n424; // $abc$1000$n424
  signal1_t _abc_1000_n425; // $abc$1000$n425
  signal1_t _abc_1000_n426; // $abc$1000$n426
  signal1_t _abc_1000_n428; // $abc$1000$n428
  signal1_t _abc_1000_n429; // $abc$1000$n429
  signal1_t _abc_1000_n430; // $abc$1000$n430
  signal1_t _abc_1000_n431; // $abc$1000$n431
  signal1_t _abc_1000_n435; // $abc$1000$n435
  signal1_t _abc_1000_n437; // $abc$1000$n437
  signal1_t _abc_1000_n439; // $abc$1000$n439
  signal1_t _abc_1000_n440; // $abc$1000$n440
  signal1_t _abc_1000_n442; // $abc$1000$n442
  signal1_t _abc_1000_n444; // $abc$1000$n444
  signal1_t _abc_1000_n445; // $abc$1000$n445
  signal1_t _abc_1000_n446; // $abc$1000$n446
  signal1_t _abc_1000_n448; // $abc$1000$n448
  signal1_t _abc_1000_n450; // $abc$1000$n450
  signal1_t _abc_1000_n451; // $abc$1000$n451
  signal1_t _abc_1000_n453_1; // $abc$1000$n453_1
  signal1_t _abc_1000_n455; // $abc$1000$n455
  signal1_t _abc_1000_n456_1; // $abc$1000$n456_1
  signal1_t _abc_1000_n457; // $abc$1000$n457
  signal1_t _abc_1000_n459; // $abc$1000$n459
  signal1_t _abc_1000_n461; // $abc$1000$n461
  signal1_t _abc_1000_n462; // $abc$1000$n462
  signal1_t _abc_1000_n464; // $abc$1000$n464
  signal1_t _abc_1000_n466; // $abc$1000$n466
  signal1_t _abc_1000_n467; // $abc$1000$n467
  signal1_t _abc_1000_n468; // $abc$1000$n468
  signal1_t _abc_1000_n469; // $abc$1000$n469
  signal1_t _abc_1000_n471; // $abc$1000$n471
  signal1_t _abc_1000_n473; // $abc$1000$n473
  signal1_t _abc_1000_n474; // $abc$1000$n474
  signal1_t _abc_1000_n476_1; // $abc$1000$n476_1
  signal1_t _abc_1000_n478; // $abc$1000$n478
  signal1_t _abc_1000_n479; // $abc$1000$n479
  signal1_t _abc_1000_n480; // $abc$1000$n480
  signal1_t _abc_1000_n482; // $abc$1000$n482
  signal1_t _abc_1000_n484; // $abc$1000$n484
  signal1_t _abc_1000_n485; // $abc$1000$n485
  signal1_t _abc_1000_n487; // $abc$1000$n487
  signal1_t _abc_1000_n489; // $abc$1000$n489
  signal1_t _abc_1000_n490; // $abc$1000$n490
  signal1_t _abc_1000_n491; // $abc$1000$n491
  signal1_t _abc_1000_n492; // $abc$1000$n492
  signal1_t _abc_1000_n494; // $abc$1000$n494
  signal1_t _abc_1000_n496; // $abc$1000$n496
  signal1_t _abc_1000_n497; // $abc$1000$n497
  signal1_t _abc_1000_n499; // $abc$1000$n499
  signal1_t _abc_1000_n501; // $abc$1000$n501
  signal1_t _abc_1000_n502; // $abc$1000$n502
  signal1_t _abc_1000_n503; // $abc$1000$n503
  signal1_t _abc_1000_n505; // $abc$1000$n505
  signal1_t _abc_1000_n507; // $abc$1000$n507
  signal1_t _abc_1000_n508; // $abc$1000$n508
  signal1_t _abc_1000_n510; // $abc$1000$n510
};
#endif

static void rvfi_insn_add_init(struct rvfi_insn_add_state_t *state)
{
  yosys_simplec_set_bit_0_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_1_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_2_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_3_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_4_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_5_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_6_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_7_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_8_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_9_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_10_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_11_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_12_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_13_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_14_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_15_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_16_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_17_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_18_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_19_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_20_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_21_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_22_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_23_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_24_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_25_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_26_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_27_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_28_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_29_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_30_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_31_of_32(&state->spec_mem_addr, false);
  yosys_simplec_set_bit_0_of_4(&state->spec_mem_rmask, false);
  yosys_simplec_set_bit_1_of_4(&state->spec_mem_rmask, false);
  yosys_simplec_set_bit_2_of_4(&state->spec_mem_rmask, false);
  yosys_simplec_set_bit_3_of_4(&state->spec_mem_rmask, false);
  yosys_simplec_set_bit_0_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_1_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_2_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_3_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_4_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_5_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_6_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_7_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_8_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_9_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_10_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_11_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_12_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_13_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_14_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_15_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_16_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_17_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_18_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_19_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_20_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_21_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_22_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_23_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_24_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_25_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_26_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_27_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_28_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_29_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_30_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_31_of_32(&state->spec_mem_wdata, false);
  yosys_simplec_set_bit_0_of_4(&state->spec_mem_wmask, false);
  yosys_simplec_set_bit_1_of_4(&state->spec_mem_wmask, false);
  yosys_simplec_set_bit_2_of_4(&state->spec_mem_wmask, false);
  yosys_simplec_set_bit_3_of_4(&state->spec_mem_wmask, false);
  state->spec_trap.value_0_0 = false;
  // Updated signal in rvfi_insn_add: \rvfi_valid
  // Updated signal in rvfi_insn_add: \rvfi_insn
  // Updated signal in rvfi_insn_add: \rvfi_pc_rdata
  // Updated signal in rvfi_insn_add: \rvfi_rs1_rdata
  // Updated signal in rvfi_insn_add: \rvfi_rs2_rdata
  // Updated signal in rvfi_insn_add: \rvfi_mem_rdata
  state->_abc_1000_n181_1.value_0_0 = yosys_simplec_get_bit_26_of_32(&state->rvfi_insn) | yosys_simplec_get_bit_25_of_32(&state->rvfi_insn); // $abc$1000$auto$blifparse.cc:346:parse_blif$1001 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n181_1
  state->_abc_1000_n182.value_0_0 = yosys_simplec_get_bit_28_of_32(&state->rvfi_insn) | yosys_simplec_get_bit_27_of_32(&state->rvfi_insn); // $abc$1000$auto$blifparse.cc:346:parse_blif$1002 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n182
  state->_abc_1000_n183_1.value_0_0 = state->_abc_1000_n182.value_0_0 | state->_abc_1000_n181_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1003 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n183_1
  state->_abc_1000_n184_1.value_0_0 = yosys_simplec_get_bit_30_of_32(&state->rvfi_insn) | yosys_simplec_get_bit_29_of_32(&state->rvfi_insn); // $abc$1000$auto$blifparse.cc:346:parse_blif$1004 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n184_1
  state->_abc_1000_n185.value_0_0 = state->_abc_1000_n184_1.value_0_0 | yosys_simplec_get_bit_31_of_32(&state->rvfi_insn); // $abc$1000$auto$blifparse.cc:346:parse_blif$1005 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n185
  state->_abc_1000_n186_1.value_0_0 = state->_abc_1000_n185.value_0_0 | state->_abc_1000_n183_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1006 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n186_1
  state->_abc_1000_n187_1.value_0_0 = state->rvfi_valid.value_0_0 & (!state->_abc_1000_n186_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1007 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n187_1
  state->_abc_1000_n188.value_0_0 = yosys_simplec_get_bit_13_of_32(&state->rvfi_insn) | yosys_simplec_get_bit_12_of_32(&state->rvfi_insn); // $abc$1000$auto$blifparse.cc:346:parse_blif$1008 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n188
  state->_abc_1000_n189_1.value_0_0 = state->_abc_1000_n188.value_0_0 | yosys_simplec_get_bit_14_of_32(&state->rvfi_insn); // $abc$1000$auto$blifparse.cc:346:parse_blif$1009 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n189_1
  state->_abc_1000_n190_1.value_0_0 = state->_abc_1000_n187_1.value_0_0 & (!state->_abc_1000_n189_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1010 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n190_1
  state->_abc_1000_n191.value_0_0 = !(yosys_simplec_get_bit_1_of_32(&state->rvfi_insn) & yosys_simplec_get_bit_0_of_32(&state->rvfi_insn)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1011 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n191
  state->_abc_1000_n192_1.value_0_0 = yosys_simplec_get_bit_3_of_32(&state->rvfi_insn) | yosys_simplec_get_bit_2_of_32(&state->rvfi_insn); // $abc$1000$auto$blifparse.cc:346:parse_blif$1012 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n192_1
  state->_abc_1000_n193_1.value_0_0 = state->_abc_1000_n192_1.value_0_0 | state->_abc_1000_n191.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1013 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n193_1
  state->_abc_1000_n194.value_0_0 = !(yosys_simplec_get_bit_5_of_32(&state->rvfi_insn) & yosys_simplec_get_bit_4_of_32(&state->rvfi_insn)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1014 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n194
  state->_abc_1000_n195_1.value_0_0 = state->_abc_1000_n194.value_0_0 | yosys_simplec_get_bit_6_of_32(&state->rvfi_insn); // $abc$1000$auto$blifparse.cc:346:parse_blif$1015 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n195_1
  state->_abc_1000_n196_1.value_0_0 = state->_abc_1000_n195_1.value_0_0 | state->_abc_1000_n193_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1016 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n196_1
  state->spec_valid.value_0_0 = state->_abc_1000_n190_1.value_0_0 & (!state->_abc_1000_n196_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1017 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_valid
  state->_abc_1000_n199_1.value_0_0 = !(yosys_simplec_get_bit_8_of_32(&state->rvfi_insn) | yosys_simplec_get_bit_7_of_32(&state->rvfi_insn)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1019 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n199_1
  state->_abc_1000_n200.value_0_0 = yosys_simplec_get_bit_10_of_32(&state->rvfi_insn) | yosys_simplec_get_bit_9_of_32(&state->rvfi_insn); // $abc$1000$auto$blifparse.cc:346:parse_blif$1020 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n200
  state->_abc_1000_n201_1.value_0_0 = state->_abc_1000_n200.value_0_0 | (!state->_abc_1000_n199_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1021 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n201_1
  state->_abc_1000_n202_1.value_0_0 = state->_abc_1000_n201_1.value_0_0 | yosys_simplec_get_bit_11_of_32(&state->rvfi_insn); // $abc$1000$auto$blifparse.cc:346:parse_blif$1022 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n202_1
  state->_abc_1000_n205_1.value_0_0 = !(yosys_simplec_get_bit_0_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_0_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1025 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n205_1
  state->_abc_1000_n198_1.value_0_0 = !(yosys_simplec_get_bit_0_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_0_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1018 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n198_1
  state->_abc_1000_n232.value_0_0 = !(yosys_simplec_get_bit_5_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_5_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1052 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n232
  state->_abc_1000_n226.value_0_0 = !(yosys_simplec_get_bit_5_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_5_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1046 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n226
  state->_abc_1000_n292.value_0_0 = !(yosys_simplec_get_bit_13_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_13_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1112 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n292
  state->_abc_1000_n286.value_0_0 = !(yosys_simplec_get_bit_13_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_13_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1106 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n286
  state->_abc_1000_n241.value_0_0 = !(yosys_simplec_get_bit_6_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_6_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1061 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n241
  state->_abc_1000_n240.value_0_0 = yosys_simplec_get_bit_6_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_6_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1060 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n240
  state->_abc_1000_n231.value_0_0 = !(yosys_simplec_get_bit_6_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_6_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1051 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n231
  state->_abc_1000_n301.value_0_0 = !(yosys_simplec_get_bit_14_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_14_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1121 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n301
  state->_abc_1000_n300_1.value_0_0 = yosys_simplec_get_bit_14_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_14_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1120 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n300_1
  state->_abc_1000_n291_1.value_0_0 = !(yosys_simplec_get_bit_14_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_14_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1111 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n291_1
  state->_abc_1000_n415.value_0_0 = yosys_simplec_get_bit_28_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_28_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1235 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n415
  state->_abc_1000_n403.value_0_0 = !(yosys_simplec_get_bit_28_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_28_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1223 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n403
  state->_abc_1000_n247.value_0_0 = !(yosys_simplec_get_bit_7_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_7_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1067 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n247
  state->_abc_1000_n239.value_0_0 = !(yosys_simplec_get_bit_7_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_7_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1059 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n239
  state->_abc_1000_n249.value_0_0 = state->_abc_1000_n240.value_0_0 & (!state->_abc_1000_n239.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1069 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n249
  state->_abc_1000_n243.value_0_0 = !((state->_abc_1000_n236.value_0_0 & state->_abc_1000_n240.value_0_0) | state->_abc_1000_n242.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1063 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n243
  state->_abc_1000_n237.value_0_0 = state->_abc_1000_n236.value_0_0 ^ state->_abc_1000_n231.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1057 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n237
  state->_abc_1000_n248.value_0_0 = !((state->_abc_1000_n241.value_0_0 | state->_abc_1000_n239.value_0_0) & state->_abc_1000_n247.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1068 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n248
  state->_abc_1000_n242.value_0_0 = !state->_abc_1000_n241.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1062 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n242
  state->_abc_1000_n243.value_0_0 = !((state->_abc_1000_n236.value_0_0 & state->_abc_1000_n240.value_0_0) | state->_abc_1000_n242.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1063 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n243
  state->_abc_1000_n244.value_0_0 = !(state->_abc_1000_n243.value_0_0 ^ state->_abc_1000_n239.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1064 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n244
  state->_abc_1000_n307.value_0_0 = !(yosys_simplec_get_bit_15_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_15_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1127 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n307
  state->_abc_1000_n299_1.value_0_0 = !(yosys_simplec_get_bit_15_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_15_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1119 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n299_1
  state->_abc_1000_n309_1.value_0_0 = state->_abc_1000_n300_1.value_0_0 & (!state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1129 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n309_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n308_1.value_0_0 = !((state->_abc_1000_n301.value_0_0 | state->_abc_1000_n299_1.value_0_0) & state->_abc_1000_n307.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1128 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n308_1
  state->_abc_1000_n302_1.value_0_0 = !state->_abc_1000_n301.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1122 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n302_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n422.value_0_0 = !((yosys_simplec_get_bit_29_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_29_of_32(&state->rvfi_rs1_rdata)) | state->_abc_1000_n421.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1242 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n422
  state->_abc_1000_n414.value_0_0 = !(yosys_simplec_get_bit_29_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_29_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1234 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n414
  state->_abc_1000_n257.value_0_0 = !(yosys_simplec_get_bit_8_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_8_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1077 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n257
  state->_abc_1000_n246.value_0_0 = !(yosys_simplec_get_bit_8_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_8_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1066 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n246
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n423.value_0_0 = !(state->_abc_1000_n414.value_0_0 | state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1243 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n423
  state->_abc_1000_n424.value_0_0 = !state->_abc_1000_n423.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1244 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n424
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n421.value_0_0 = state->_abc_1000_n415.value_0_0 & (!state->_abc_1000_n414.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1241 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n421
  state->_abc_1000_n422.value_0_0 = !((yosys_simplec_get_bit_29_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_29_of_32(&state->rvfi_rs1_rdata)) | state->_abc_1000_n421.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1242 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n422
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n416.value_0_0 = !state->_abc_1000_n415.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1236 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n416
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n455.value_0_0 = !(yosys_simplec_get_bit_11_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_10_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1275 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n455
  state->_abc_1000_n453_1.value_0_0 = state->_abc_1000_n451.value_0_0 & yosys_simplec_get_bit_10_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1273 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n453_1
  yosys_simplec_set_bit_10_of_32(&state->spec_pc_wdata, state->_abc_1000_n451.value_0_0 ^ yosys_simplec_get_bit_10_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1272 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [10]
  yosys_simplec_set_bit_11_of_32(&state->spec_pc_wdata, state->_abc_1000_n453_1.value_0_0 ^ yosys_simplec_get_bit_11_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1274 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [11]
  state->_abc_1000_n428.value_0_0 = yosys_simplec_get_bit_31_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_31_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1248 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n428
  state->_abc_1000_n429.value_0_0 = yosys_simplec_get_bit_30_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_30_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1249 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n429
  state->_abc_1000_n420.value_0_0 = yosys_simplec_get_bit_30_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_30_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1240 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n420
  state->_abc_1000_n318_1.value_0_0 = !(yosys_simplec_get_bit_16_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_16_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1138 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n318_1
  state->_abc_1000_n306_1.value_0_0 = !(yosys_simplec_get_bit_16_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_16_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1126 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n306_1
  state->_abc_1000_n263.value_0_0 = !(yosys_simplec_get_bit_9_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_9_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1083 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n263
  state->_abc_1000_n255.value_0_0 = !(yosys_simplec_get_bit_9_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_9_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1075 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n255
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n352.value_0_0 = yosys_simplec_get_bit_21_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_21_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1172 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n352
  state->_abc_1000_n345_1.value_0_0 = yosys_simplec_get_bit_21_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_21_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1165 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n345_1
  state->_abc_1000_n435.value_0_0 = yosys_simplec_get_bit_3_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_2_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1255 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n435
  yosys_simplec_set_bit_3_of_32(&state->spec_pc_wdata, yosys_simplec_get_bit_3_of_32(&state->rvfi_pc_rdata) ^ yosys_simplec_get_bit_2_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1254 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [3]
  yosys_simplec_set_bit_2_of_32(&state->spec_pc_wdata, !yosys_simplec_get_bit_2_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1253 ($_NOT_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [2]
  state->_abc_1000_n324_1.value_0_0 = !((yosys_simplec_get_bit_17_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_17_of_32(&state->rvfi_rs1_rdata)) | state->_abc_1000_n323_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1144 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n324_1
  state->_abc_1000_n317_1.value_0_0 = !(yosys_simplec_get_bit_17_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_17_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1137 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n317_1
  state->_abc_1000_n354_1.value_0_0 = !(state->_abc_1000_n353_1.value_0_0 | state->_abc_1000_n352.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1174 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n354_1
  state->_abc_1000_n271_1.value_0_0 = !(yosys_simplec_get_bit_10_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_10_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1091 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n271_1
  state->_abc_1000_n261.value_0_0 = yosys_simplec_get_bit_10_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_10_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1081 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n261
  state->_abc_1000_n466.value_0_0 = !(yosys_simplec_get_bit_15_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_14_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1286 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n466
  state->_abc_1000_n464.value_0_0 = yosys_simplec_get_bit_14_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n462.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1284 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n464
  yosys_simplec_set_bit_14_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n462.value_0_0 ^ yosys_simplec_get_bit_14_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1283 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [14]
  yosys_simplec_set_bit_15_of_32(&state->spec_pc_wdata, state->_abc_1000_n464.value_0_0 ^ yosys_simplec_get_bit_15_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1285 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [15]
  state->_abc_1000_n467.value_0_0 = state->_abc_1000_n466.value_0_0 | state->_abc_1000_n461.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1287 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n467
  state->_abc_1000_n459.value_0_0 = yosys_simplec_get_bit_12_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n457.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1279 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n459
  state->_abc_1000_n461.value_0_0 = !(yosys_simplec_get_bit_13_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_12_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1281 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n461
  state->_abc_1000_n467.value_0_0 = state->_abc_1000_n466.value_0_0 | state->_abc_1000_n461.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1287 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n467
  state->_abc_1000_n462.value_0_0 = state->_abc_1000_n461.value_0_0 | state->_abc_1000_n457.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1282 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n462
  state->_abc_1000_n464.value_0_0 = yosys_simplec_get_bit_14_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n462.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1284 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n464
  yosys_simplec_set_bit_14_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n462.value_0_0 ^ yosys_simplec_get_bit_14_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1283 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [14]
  yosys_simplec_set_bit_15_of_32(&state->spec_pc_wdata, state->_abc_1000_n464.value_0_0 ^ yosys_simplec_get_bit_15_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1285 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [15]
  yosys_simplec_set_bit_12_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n457.value_0_0 ^ yosys_simplec_get_bit_12_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1278 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [12]
  yosys_simplec_set_bit_13_of_32(&state->spec_pc_wdata, state->_abc_1000_n459.value_0_0 ^ yosys_simplec_get_bit_13_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1280 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [13]
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n372.value_0_0 = !((state->_abc_1000_n371.value_0_0 | state->_abc_1000_n354_1.value_0_0) & state->_abc_1000_n370.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1192 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n372
  state->_abc_1000_n355.value_0_0 = state->_abc_1000_n335_1.value_0_0 | (!state->_abc_1000_n345_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1175 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n355
  state->_abc_1000_n373.value_0_0 = !(state->_abc_1000_n371.value_0_0 | state->_abc_1000_n355.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1193 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n373
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n353_1.value_0_0 = state->_abc_1000_n345_1.value_0_0 & (!state->_abc_1000_n346.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1173 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n353_1
  state->_abc_1000_n354_1.value_0_0 = !(state->_abc_1000_n353_1.value_0_0 | state->_abc_1000_n352.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1174 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n354_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n372.value_0_0 = !((state->_abc_1000_n371.value_0_0 | state->_abc_1000_n354_1.value_0_0) & state->_abc_1000_n370.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1192 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n372
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n387.value_0_0 = !(yosys_simplec_get_bit_26_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_26_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1207 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n387
  state->_abc_1000_n399.value_0_0 = yosys_simplec_get_bit_26_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_26_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1219 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n399
  state->_abc_1000_n388.value_0_0 = yosys_simplec_get_bit_25_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_25_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1208 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n388
  state->_abc_1000_n379.value_0_0 = yosys_simplec_get_bit_25_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_25_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1199 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n379
  state->_abc_1000_n330_1.value_0_0 = !(yosys_simplec_get_bit_18_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_18_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1150 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n330_1
  state->_abc_1000_n322.value_0_0 = !(yosys_simplec_get_bit_18_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_18_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1142 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n322
  state->_abc_1000_n276_1.value_0_0 = !(yosys_simplec_get_bit_12_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_12_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1096 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n276_1
  state->_abc_1000_n287_1.value_0_0 = !(yosys_simplec_get_bit_12_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_12_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1107 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n287_1
  state->_abc_1000_n390.value_0_0 = state->_abc_1000_n389.value_0_0 | state->_abc_1000_n388.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1210 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n390
  state->_abc_1000_n277.value_0_0 = !(yosys_simplec_get_bit_11_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_11_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1097 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n277
  state->_abc_1000_n270_1.value_0_0 = !(yosys_simplec_get_bit_11_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_11_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1090 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n270_1
  state->_abc_1000_n362.value_0_0 = !(yosys_simplec_get_bit_22_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_22_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1182 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n362
  state->_abc_1000_n351_1.value_0_0 = !(yosys_simplec_get_bit_22_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_22_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1171 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n351_1
  state->_abc_1000_n444.value_0_0 = !(yosys_simplec_get_bit_7_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_6_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1264 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n444
  state->_abc_1000_n442.value_0_0 = state->_abc_1000_n440.value_0_0 & yosys_simplec_get_bit_6_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1262 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n442
  yosys_simplec_set_bit_6_of_32(&state->spec_pc_wdata, state->_abc_1000_n440.value_0_0 ^ yosys_simplec_get_bit_6_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1261 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [6]
  yosys_simplec_set_bit_7_of_32(&state->spec_pc_wdata, state->_abc_1000_n442.value_0_0 ^ yosys_simplec_get_bit_7_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1263 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [7]
  state->_abc_1000_n279_1.value_0_0 = state->_abc_1000_n261.value_0_0 & (!state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1099 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n279_1
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n262.value_0_0 = !state->_abc_1000_n261.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1082 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n262
  state->_abc_1000_n268.value_0_0 = state->_abc_1000_n267.value_0_0 ^ state->_abc_1000_n262.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1088 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n268
  state->_abc_1000_n278_1.value_0_0 = !((state->_abc_1000_n271_1.value_0_0 | state->_abc_1000_n270_1.value_0_0) & state->_abc_1000_n277.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1098 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n278_1
  state->_abc_1000_n272_1.value_0_0 = !state->_abc_1000_n271_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1092 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n272_1
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n274.value_0_0 = !(state->_abc_1000_n273_1.value_0_0 ^ state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1094 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n274
  state->_abc_1000_n363.value_0_0 = !state->_abc_1000_n362.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1183 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n363
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n408.value_0_0 = state->_abc_1000_n390.value_0_0 & (!state->_abc_1000_n407.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1228 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n408
  state->_abc_1000_n409.value_0_0 = state->_abc_1000_n408.value_0_0 | state->_abc_1000_n406.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1229 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n409
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n391.value_0_0 = state->_abc_1000_n367.value_0_0 | (!state->_abc_1000_n379.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1211 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n391
  state->_abc_1000_n410.value_0_0 = !(state->_abc_1000_n407.value_0_0 | state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1230 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n410
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n389.value_0_0 = state->_abc_1000_n379.value_0_0 & (!state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1209 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n389
  state->_abc_1000_n390.value_0_0 = state->_abc_1000_n389.value_0_0 | state->_abc_1000_n388.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1210 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n390
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n408.value_0_0 = state->_abc_1000_n390.value_0_0 & (!state->_abc_1000_n407.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1228 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n408
  state->_abc_1000_n409.value_0_0 = state->_abc_1000_n408.value_0_0 | state->_abc_1000_n406.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1229 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n409
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n380.value_0_0 = !state->_abc_1000_n379.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1200 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n380
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n446.value_0_0 = state->_abc_1000_n435.value_0_0 & (!state->_abc_1000_n445.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1266 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n446
  state->_abc_1000_n440.value_0_0 = state->_abc_1000_n435.value_0_0 & (!state->_abc_1000_n439.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1260 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n440
  state->_abc_1000_n442.value_0_0 = state->_abc_1000_n440.value_0_0 & yosys_simplec_get_bit_6_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1262 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n442
  yosys_simplec_set_bit_6_of_32(&state->spec_pc_wdata, state->_abc_1000_n440.value_0_0 ^ yosys_simplec_get_bit_6_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1261 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [6]
  yosys_simplec_set_bit_7_of_32(&state->spec_pc_wdata, state->_abc_1000_n442.value_0_0 ^ yosys_simplec_get_bit_7_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1263 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [7]
  state->_abc_1000_n437.value_0_0 = state->_abc_1000_n435.value_0_0 & yosys_simplec_get_bit_4_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1257 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n437
  state->_abc_1000_n439.value_0_0 = !(yosys_simplec_get_bit_5_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_4_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1259 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n439
  state->_abc_1000_n440.value_0_0 = state->_abc_1000_n435.value_0_0 & (!state->_abc_1000_n439.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1260 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n440
  state->_abc_1000_n442.value_0_0 = state->_abc_1000_n440.value_0_0 & yosys_simplec_get_bit_6_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1262 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n442
  yosys_simplec_set_bit_6_of_32(&state->spec_pc_wdata, state->_abc_1000_n440.value_0_0 ^ yosys_simplec_get_bit_6_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1261 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [6]
  yosys_simplec_set_bit_7_of_32(&state->spec_pc_wdata, state->_abc_1000_n442.value_0_0 ^ yosys_simplec_get_bit_7_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1263 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [7]
  yosys_simplec_set_bit_4_of_32(&state->spec_pc_wdata, state->_abc_1000_n435.value_0_0 ^ yosys_simplec_get_bit_4_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1256 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [4]
  yosys_simplec_set_bit_5_of_32(&state->spec_pc_wdata, state->_abc_1000_n437.value_0_0 ^ yosys_simplec_get_bit_5_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1258 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [5]
  state->_abc_1000_n445.value_0_0 = state->_abc_1000_n444.value_0_0 | state->_abc_1000_n439.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1265 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n445
  state->_abc_1000_n446.value_0_0 = state->_abc_1000_n435.value_0_0 & (!state->_abc_1000_n445.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1266 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n446
  state->_abc_1000_n335_1.value_0_0 = !(yosys_simplec_get_bit_20_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_20_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1155 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n335_1
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n355.value_0_0 = state->_abc_1000_n335_1.value_0_0 | (!state->_abc_1000_n345_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1175 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n355
  state->_abc_1000_n373.value_0_0 = !(state->_abc_1000_n371.value_0_0 | state->_abc_1000_n355.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1193 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n373
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n346.value_0_0 = !(yosys_simplec_get_bit_20_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_20_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1166 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n346
  state->_abc_1000_n353_1.value_0_0 = state->_abc_1000_n345_1.value_0_0 & (!state->_abc_1000_n346.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1173 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n353_1
  state->_abc_1000_n354_1.value_0_0 = !(state->_abc_1000_n353_1.value_0_0 | state->_abc_1000_n352.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1174 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n354_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n372.value_0_0 = !((state->_abc_1000_n371.value_0_0 | state->_abc_1000_n354_1.value_0_0) & state->_abc_1000_n370.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1192 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n372
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n407.value_0_0 = state->_abc_1000_n396.value_0_0 | state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1227 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n407
  state->_abc_1000_n408.value_0_0 = state->_abc_1000_n390.value_0_0 & (!state->_abc_1000_n407.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1228 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n408
  state->_abc_1000_n409.value_0_0 = state->_abc_1000_n408.value_0_0 | state->_abc_1000_n406.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1229 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n409
  state->_abc_1000_n410.value_0_0 = !(state->_abc_1000_n407.value_0_0 | state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1230 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n410
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n398.value_0_0 = !state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1218 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n398
  state->_abc_1000_n405.value_0_0 = state->_abc_1000_n399.value_0_0 & (!state->_abc_1000_n396.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1225 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n405
  state->_abc_1000_n406.value_0_0 = state->_abc_1000_n405.value_0_0 | state->_abc_1000_n404.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1226 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n406
  state->_abc_1000_n409.value_0_0 = state->_abc_1000_n408.value_0_0 | state->_abc_1000_n406.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1229 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n409
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n336_1.value_0_0 = state->_abc_1000_n329_1.value_0_0 & (!state->_abc_1000_n330_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1156 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n336_1
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n338_1.value_0_0 = state->_abc_1000_n322.value_0_0 | (!state->_abc_1000_n329_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1158 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n338_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n448.value_0_0 = state->_abc_1000_n446.value_0_0 & yosys_simplec_get_bit_8_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1268 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n448
  yosys_simplec_set_bit_8_of_32(&state->spec_pc_wdata, state->_abc_1000_n446.value_0_0 ^ yosys_simplec_get_bit_8_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1267 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [8]
  state->_abc_1000_n450.value_0_0 = !(yosys_simplec_get_bit_9_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_8_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1270 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n450
  yosys_simplec_set_bit_9_of_32(&state->spec_pc_wdata, state->_abc_1000_n448.value_0_0 ^ yosys_simplec_get_bit_9_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1269 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [9]
  state->_abc_1000_n469.value_0_0 = state->_abc_1000_n446.value_0_0 & (!state->_abc_1000_n468.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1289 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n469
  state->_abc_1000_n457.value_0_0 = state->_abc_1000_n456_1.value_0_0 | (!state->_abc_1000_n446.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1277 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n457
  state->_abc_1000_n462.value_0_0 = state->_abc_1000_n461.value_0_0 | state->_abc_1000_n457.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1282 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n462
  state->_abc_1000_n464.value_0_0 = yosys_simplec_get_bit_14_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n462.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1284 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n464
  yosys_simplec_set_bit_14_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n462.value_0_0 ^ yosys_simplec_get_bit_14_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1283 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [14]
  yosys_simplec_set_bit_15_of_32(&state->spec_pc_wdata, state->_abc_1000_n464.value_0_0 ^ yosys_simplec_get_bit_15_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1285 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [15]
  state->_abc_1000_n459.value_0_0 = yosys_simplec_get_bit_12_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n457.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1279 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n459
  yosys_simplec_set_bit_12_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n457.value_0_0 ^ yosys_simplec_get_bit_12_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1278 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [12]
  yosys_simplec_set_bit_13_of_32(&state->spec_pc_wdata, state->_abc_1000_n459.value_0_0 ^ yosys_simplec_get_bit_13_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1280 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [13]
  state->_abc_1000_n451.value_0_0 = state->_abc_1000_n446.value_0_0 & (!state->_abc_1000_n450.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1271 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n451
  state->_abc_1000_n453_1.value_0_0 = state->_abc_1000_n451.value_0_0 & yosys_simplec_get_bit_10_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1273 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n453_1
  yosys_simplec_set_bit_10_of_32(&state->spec_pc_wdata, state->_abc_1000_n451.value_0_0 ^ yosys_simplec_get_bit_10_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1272 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [10]
  yosys_simplec_set_bit_11_of_32(&state->spec_pc_wdata, state->_abc_1000_n453_1.value_0_0 ^ yosys_simplec_get_bit_11_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1274 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [11]
  state->_abc_1000_n456_1.value_0_0 = state->_abc_1000_n455.value_0_0 | state->_abc_1000_n450.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1276 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n456_1
  state->_abc_1000_n457.value_0_0 = state->_abc_1000_n456_1.value_0_0 | (!state->_abc_1000_n446.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1277 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n457
  state->_abc_1000_n462.value_0_0 = state->_abc_1000_n461.value_0_0 | state->_abc_1000_n457.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1282 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n462
  state->_abc_1000_n464.value_0_0 = yosys_simplec_get_bit_14_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n462.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1284 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n464
  yosys_simplec_set_bit_14_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n462.value_0_0 ^ yosys_simplec_get_bit_14_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1283 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [14]
  yosys_simplec_set_bit_15_of_32(&state->spec_pc_wdata, state->_abc_1000_n464.value_0_0 ^ yosys_simplec_get_bit_15_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1285 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [15]
  state->_abc_1000_n459.value_0_0 = yosys_simplec_get_bit_12_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n457.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1279 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n459
  yosys_simplec_set_bit_12_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n457.value_0_0 ^ yosys_simplec_get_bit_12_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1278 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [12]
  yosys_simplec_set_bit_13_of_32(&state->spec_pc_wdata, state->_abc_1000_n459.value_0_0 ^ yosys_simplec_get_bit_13_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1280 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [13]
  state->_abc_1000_n468.value_0_0 = state->_abc_1000_n467.value_0_0 | state->_abc_1000_n456_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1288 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n468
  state->_abc_1000_n469.value_0_0 = state->_abc_1000_n446.value_0_0 & (!state->_abc_1000_n468.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1289 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n469
  state->_abc_1000_n337.value_0_0 = !((yosys_simplec_get_bit_19_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_19_of_32(&state->rvfi_rs1_rdata)) | state->_abc_1000_n336_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1157 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n337
  state->_abc_1000_n329_1.value_0_0 = yosys_simplec_get_bit_19_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_19_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1149 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n329_1
  state->_abc_1000_n336_1.value_0_0 = state->_abc_1000_n329_1.value_0_0 & (!state->_abc_1000_n330_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1156 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n336_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n338_1.value_0_0 = state->_abc_1000_n322.value_0_0 | (!state->_abc_1000_n329_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1158 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n338_1
  state->_abc_1000_n337.value_0_0 = !((yosys_simplec_get_bit_19_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_19_of_32(&state->rvfi_rs1_rdata)) | state->_abc_1000_n336_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1157 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n337
  state->_abc_1000_n396.value_0_0 = !(yosys_simplec_get_bit_27_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_27_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1216 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n396
  state->_abc_1000_n407.value_0_0 = state->_abc_1000_n396.value_0_0 | state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1227 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n407
  state->_abc_1000_n408.value_0_0 = state->_abc_1000_n390.value_0_0 & (!state->_abc_1000_n407.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1228 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n408
  state->_abc_1000_n409.value_0_0 = state->_abc_1000_n408.value_0_0 | state->_abc_1000_n406.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1229 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n409
  state->_abc_1000_n410.value_0_0 = !(state->_abc_1000_n407.value_0_0 | state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1230 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n410
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n405.value_0_0 = state->_abc_1000_n399.value_0_0 & (!state->_abc_1000_n396.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1225 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n405
  state->_abc_1000_n406.value_0_0 = state->_abc_1000_n405.value_0_0 | state->_abc_1000_n404.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1226 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n406
  state->_abc_1000_n409.value_0_0 = state->_abc_1000_n408.value_0_0 | state->_abc_1000_n406.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1229 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n409
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n397.value_0_0 = !state->_abc_1000_n396.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1217 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n397
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n404.value_0_0 = yosys_simplec_get_bit_27_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_27_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1224 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n404
  state->_abc_1000_n406.value_0_0 = state->_abc_1000_n405.value_0_0 | state->_abc_1000_n404.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1226 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n406
  state->_abc_1000_n409.value_0_0 = state->_abc_1000_n408.value_0_0 | state->_abc_1000_n406.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1229 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n409
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n253.value_0_0 = state->_abc_1000_n252.value_0_0 ^ state->_abc_1000_n246.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1073 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n253
  state->_abc_1000_n265.value_0_0 = state->_abc_1000_n255.value_0_0 | state->_abc_1000_n246.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1085 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n265
  state->_abc_1000_n266.value_0_0 = state->_abc_1000_n252.value_0_0 & (!state->_abc_1000_n265.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1086 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n266
  state->_abc_1000_n264.value_0_0 = !((state->_abc_1000_n257.value_0_0 | state->_abc_1000_n255.value_0_0) & state->_abc_1000_n263.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1084 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n264
  state->_abc_1000_n258.value_0_0 = !((state->_abc_1000_n256.value_0_0 | state->_abc_1000_n246.value_0_0) & state->_abc_1000_n257.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1078 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n258
  state->_abc_1000_n259.value_0_0 = state->_abc_1000_n258.value_0_0 ^ state->_abc_1000_n255.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1079 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n259
  state->_abc_1000_n282_1.value_0_0 = state->_abc_1000_n279_1.value_0_0 & (!state->_abc_1000_n265.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1102 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n282_1
  state->_abc_1000_n283.value_0_0 = !((state->_abc_1000_n282_1.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n281_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1103 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n283
  state->_abc_1000_n313.value_0_0 = state->_abc_1000_n282_1.value_0_0 & (!state->_abc_1000_n311_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1133 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n313
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n280.value_0_0 = !((state->_abc_1000_n279_1.value_0_0 & state->_abc_1000_n264.value_0_0) | state->_abc_1000_n278_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1100 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n280
  state->_abc_1000_n312_1.value_0_0 = !((state->_abc_1000_n311_1.value_0_0 | state->_abc_1000_n280.value_0_0) & state->_abc_1000_n310.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1132 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n312_1
  state->_abc_1000_n281_1.value_0_0 = !state->_abc_1000_n280.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1101 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n281_1
  state->_abc_1000_n283.value_0_0 = !((state->_abc_1000_n282_1.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n281_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1103 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n283
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n267.value_0_0 = state->_abc_1000_n266.value_0_0 | state->_abc_1000_n264.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1087 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n267
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n268.value_0_0 = state->_abc_1000_n267.value_0_0 ^ state->_abc_1000_n262.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1088 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n268
  state->_abc_1000_n274.value_0_0 = !(state->_abc_1000_n273_1.value_0_0 ^ state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1094 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n274
  state->_abc_1000_n478.value_0_0 = !(yosys_simplec_get_bit_19_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_18_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1298 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n478
  state->_abc_1000_n476_1.value_0_0 = yosys_simplec_get_bit_18_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n474.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1296 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n476_1
  yosys_simplec_set_bit_18_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n474.value_0_0 ^ yosys_simplec_get_bit_18_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1295 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [18]
  yosys_simplec_set_bit_19_of_32(&state->spec_pc_wdata, state->_abc_1000_n476_1.value_0_0 ^ yosys_simplec_get_bit_19_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1297 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [19]
  state->_abc_1000_n367.value_0_0 = !(yosys_simplec_get_bit_24_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_24_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1187 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n367
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n391.value_0_0 = state->_abc_1000_n367.value_0_0 | (!state->_abc_1000_n379.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1211 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n391
  state->_abc_1000_n410.value_0_0 = !(state->_abc_1000_n407.value_0_0 | state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1230 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n410
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n383.value_0_0 = !(yosys_simplec_get_bit_24_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_24_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1203 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n383
  state->_abc_1000_n389.value_0_0 = state->_abc_1000_n379.value_0_0 & (!state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1209 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n389
  state->_abc_1000_n390.value_0_0 = state->_abc_1000_n389.value_0_0 | state->_abc_1000_n388.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1210 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n390
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n408.value_0_0 = state->_abc_1000_n390.value_0_0 & (!state->_abc_1000_n407.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1228 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n408
  state->_abc_1000_n409.value_0_0 = state->_abc_1000_n408.value_0_0 | state->_abc_1000_n406.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1229 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n409
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n284_1.value_0_0 = !(state->_abc_1000_n283.value_0_0 ^ state->_abc_1000_n276_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1104 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n284_1
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n288_1.value_0_0 = !((state->_abc_1000_n283.value_0_0 | state->_abc_1000_n276_1.value_0_0) & state->_abc_1000_n287_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1108 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n288_1
  state->_abc_1000_n295.value_0_0 = state->_abc_1000_n286.value_0_0 | state->_abc_1000_n276_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1115 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n295
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n293_1.value_0_0 = !((state->_abc_1000_n287_1.value_0_0 | state->_abc_1000_n286.value_0_0) & state->_abc_1000_n292.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1113 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n293_1
  state->_abc_1000_n289.value_0_0 = state->_abc_1000_n288_1.value_0_0 ^ state->_abc_1000_n286.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1109 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n289
  state->_abc_1000_n311_1.value_0_0 = state->_abc_1000_n295.value_0_0 | (!state->_abc_1000_n309_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1131 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n311_1
  state->_abc_1000_n313.value_0_0 = state->_abc_1000_n282_1.value_0_0 & (!state->_abc_1000_n311_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1133 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n313
  state->_abc_1000_n312_1.value_0_0 = !((state->_abc_1000_n311_1.value_0_0 | state->_abc_1000_n280.value_0_0) & state->_abc_1000_n310.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1132 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n312_1
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n310.value_0_0 = !((state->_abc_1000_n309_1.value_0_0 & state->_abc_1000_n293_1.value_0_0) | state->_abc_1000_n308_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1130 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n310
  state->_abc_1000_n312_1.value_0_0 = !((state->_abc_1000_n311_1.value_0_0 | state->_abc_1000_n280.value_0_0) & state->_abc_1000_n310.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1132 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n312_1
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n294_1.value_0_0 = !state->_abc_1000_n293_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1114 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n294_1
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n368.value_0_0 = !((state->_abc_1000_n338_1.value_0_0 | state->_abc_1000_n324_1.value_0_0) & state->_abc_1000_n337.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1188 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n368
  state->_abc_1000_n340.value_0_0 = state->_abc_1000_n339_1.value_0_0 & state->_abc_1000_n337.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1160 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n340
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n341_1.value_0_0 = state->_abc_1000_n338_1.value_0_0 | state->_abc_1000_n325.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1161 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n341_1
  state->_abc_1000_n339_1.value_0_0 = state->_abc_1000_n338_1.value_0_0 | state->_abc_1000_n324_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1159 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n339_1
  state->_abc_1000_n340.value_0_0 = state->_abc_1000_n339_1.value_0_0 & state->_abc_1000_n337.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1160 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n340
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n319.value_0_0 = !((state->_abc_1000_n314_1.value_0_0 | state->_abc_1000_n306_1.value_0_0) & state->_abc_1000_n318_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1139 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n319
  state->_abc_1000_n325.value_0_0 = state->_abc_1000_n317_1.value_0_0 | state->_abc_1000_n306_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1145 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n325
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n341_1.value_0_0 = state->_abc_1000_n338_1.value_0_0 | state->_abc_1000_n325.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1161 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n341_1
  state->_abc_1000_n315_1.value_0_0 = !(state->_abc_1000_n314_1.value_0_0 ^ state->_abc_1000_n306_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1135 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n315_1
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n342_1.value_0_0 = !((state->_abc_1000_n341_1.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n340.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1162 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n342_1
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n375.value_0_0 = state->_abc_1000_n341_1.value_0_0 | (!state->_abc_1000_n373.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1195 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n375
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n374.value_0_0 = !((state->_abc_1000_n373.value_0_0 & state->_abc_1000_n368.value_0_0) | state->_abc_1000_n372.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1194 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n374
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n323_1.value_0_0 = !(state->_abc_1000_n318_1.value_0_0 | state->_abc_1000_n317_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1143 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n323_1
  state->_abc_1000_n324_1.value_0_0 = !((yosys_simplec_get_bit_17_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_17_of_32(&state->rvfi_rs1_rdata)) | state->_abc_1000_n323_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1144 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n324_1
  state->_abc_1000_n368.value_0_0 = !((state->_abc_1000_n338_1.value_0_0 | state->_abc_1000_n324_1.value_0_0) & state->_abc_1000_n337.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1188 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n368
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n339_1.value_0_0 = state->_abc_1000_n338_1.value_0_0 | state->_abc_1000_n324_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1159 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n339_1
  state->_abc_1000_n340.value_0_0 = state->_abc_1000_n339_1.value_0_0 & state->_abc_1000_n337.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1160 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n340
  state->_abc_1000_n342_1.value_0_0 = !((state->_abc_1000_n341_1.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n340.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1162 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n342_1
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n374.value_0_0 = !((state->_abc_1000_n373.value_0_0 & state->_abc_1000_n368.value_0_0) | state->_abc_1000_n372.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1194 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n374
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n320_1.value_0_0 = state->_abc_1000_n319.value_0_0 ^ state->_abc_1000_n317_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1140 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n320_1
  state->_abc_1000_n370.value_0_0 = !((state->_abc_1000_n363.value_0_0 & state->_abc_1000_n360.value_0_0) | state->_abc_1000_n369.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1190 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n370
  state->_abc_1000_n372.value_0_0 = !((state->_abc_1000_n371.value_0_0 | state->_abc_1000_n354_1.value_0_0) & state->_abc_1000_n370.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1192 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n372
  state->_abc_1000_n374.value_0_0 = !((state->_abc_1000_n373.value_0_0 & state->_abc_1000_n368.value_0_0) | state->_abc_1000_n372.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1194 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n374
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n371.value_0_0 = state->_abc_1000_n351_1.value_0_0 | (!state->_abc_1000_n360.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1191 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n371
  state->_abc_1000_n372.value_0_0 = !((state->_abc_1000_n371.value_0_0 | state->_abc_1000_n354_1.value_0_0) & state->_abc_1000_n370.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1192 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n372
  state->_abc_1000_n373.value_0_0 = !(state->_abc_1000_n371.value_0_0 | state->_abc_1000_n355.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1193 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n373
  state->_abc_1000_n375.value_0_0 = state->_abc_1000_n341_1.value_0_0 | (!state->_abc_1000_n373.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1195 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n375
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n374.value_0_0 = !((state->_abc_1000_n373.value_0_0 & state->_abc_1000_n368.value_0_0) | state->_abc_1000_n372.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1194 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n374
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n361.value_0_0 = !state->_abc_1000_n351_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1181 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n361
  state->_abc_1000_n358.value_0_0 = state->_abc_1000_n357.value_0_0 ^ state->_abc_1000_n351_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1178 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n358
  state->_abc_1000_n364.value_0_0 = !((state->_abc_1000_n357.value_0_0 & state->_abc_1000_n361.value_0_0) | state->_abc_1000_n363.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1184 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n364
  state->_abc_1000_n365.value_0_0 = state->_abc_1000_n364.value_0_0 ^ state->_abc_1000_n360.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1185 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n365
  state->_abc_1000_n369.value_0_0 = yosys_simplec_get_bit_23_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_23_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1189 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n369
  state->_abc_1000_n370.value_0_0 = !((state->_abc_1000_n363.value_0_0 & state->_abc_1000_n360.value_0_0) | state->_abc_1000_n369.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1190 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n370
  state->_abc_1000_n372.value_0_0 = !((state->_abc_1000_n371.value_0_0 | state->_abc_1000_n354_1.value_0_0) & state->_abc_1000_n370.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1192 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n372
  state->_abc_1000_n374.value_0_0 = !((state->_abc_1000_n373.value_0_0 & state->_abc_1000_n368.value_0_0) | state->_abc_1000_n372.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1194 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n374
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n360.value_0_0 = yosys_simplec_get_bit_23_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_23_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1180 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n360
  state->_abc_1000_n370.value_0_0 = !((state->_abc_1000_n363.value_0_0 & state->_abc_1000_n360.value_0_0) | state->_abc_1000_n369.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1190 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n370
  state->_abc_1000_n372.value_0_0 = !((state->_abc_1000_n371.value_0_0 | state->_abc_1000_n354_1.value_0_0) & state->_abc_1000_n370.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1192 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n372
  state->_abc_1000_n374.value_0_0 = !((state->_abc_1000_n373.value_0_0 & state->_abc_1000_n368.value_0_0) | state->_abc_1000_n372.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1194 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n374
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n365.value_0_0 = state->_abc_1000_n364.value_0_0 ^ state->_abc_1000_n360.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1185 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n365
  state->_abc_1000_n371.value_0_0 = state->_abc_1000_n351_1.value_0_0 | (!state->_abc_1000_n360.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1191 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n371
  state->_abc_1000_n372.value_0_0 = !((state->_abc_1000_n371.value_0_0 | state->_abc_1000_n354_1.value_0_0) & state->_abc_1000_n370.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1192 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n372
  state->_abc_1000_n373.value_0_0 = !(state->_abc_1000_n371.value_0_0 | state->_abc_1000_n355.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1193 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n373
  state->_abc_1000_n375.value_0_0 = state->_abc_1000_n341_1.value_0_0 | (!state->_abc_1000_n373.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1195 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n375
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n374.value_0_0 = !((state->_abc_1000_n373.value_0_0 & state->_abc_1000_n368.value_0_0) | state->_abc_1000_n372.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1194 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n374
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n235.value_0_0 = state->_abc_1000_n226.value_0_0 | state->_abc_1000_n219.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1055 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n235
  state->_abc_1000_n236.value_0_0 = !((state->_abc_1000_n235.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n234.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1056 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n236
  state->_abc_1000_n243.value_0_0 = !((state->_abc_1000_n236.value_0_0 & state->_abc_1000_n240.value_0_0) | state->_abc_1000_n242.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1063 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n243
  state->_abc_1000_n237.value_0_0 = state->_abc_1000_n236.value_0_0 ^ state->_abc_1000_n231.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1057 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n237
  state->_abc_1000_n244.value_0_0 = !(state->_abc_1000_n243.value_0_0 ^ state->_abc_1000_n239.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1064 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n244
  state->_abc_1000_n233.value_0_0 = !((state->_abc_1000_n227.value_0_0 | state->_abc_1000_n226.value_0_0) & state->_abc_1000_n232.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1053 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n233
  state->_abc_1000_n229.value_0_0 = state->_abc_1000_n228.value_0_0 ^ state->_abc_1000_n226.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1049 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n229
  state->_abc_1000_n251.value_0_0 = state->_abc_1000_n235.value_0_0 | (!state->_abc_1000_n249.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1071 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n251
  state->_abc_1000_n252.value_0_0 = !((state->_abc_1000_n251.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n250.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1072 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n252
  state->_abc_1000_n283.value_0_0 = !((state->_abc_1000_n282_1.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n281_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1103 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n283
  state->_abc_1000_n266.value_0_0 = state->_abc_1000_n252.value_0_0 & (!state->_abc_1000_n265.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1086 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n266
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n256.value_0_0 = !state->_abc_1000_n252.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1076 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n256
  state->_abc_1000_n253.value_0_0 = state->_abc_1000_n252.value_0_0 ^ state->_abc_1000_n246.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1073 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n253
  state->_abc_1000_n258.value_0_0 = !((state->_abc_1000_n256.value_0_0 | state->_abc_1000_n246.value_0_0) & state->_abc_1000_n257.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1078 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n258
  state->_abc_1000_n259.value_0_0 = state->_abc_1000_n258.value_0_0 ^ state->_abc_1000_n255.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1079 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n259
  state->_abc_1000_n267.value_0_0 = state->_abc_1000_n266.value_0_0 | state->_abc_1000_n264.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1087 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n267
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n268.value_0_0 = state->_abc_1000_n267.value_0_0 ^ state->_abc_1000_n262.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1088 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n268
  state->_abc_1000_n274.value_0_0 = !(state->_abc_1000_n273_1.value_0_0 ^ state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1094 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n274
  state->_abc_1000_n284_1.value_0_0 = !(state->_abc_1000_n283.value_0_0 ^ state->_abc_1000_n276_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1104 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n284_1
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n288_1.value_0_0 = !((state->_abc_1000_n283.value_0_0 | state->_abc_1000_n276_1.value_0_0) & state->_abc_1000_n287_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1108 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n288_1
  state->_abc_1000_n289.value_0_0 = state->_abc_1000_n288_1.value_0_0 ^ state->_abc_1000_n286.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1109 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n289
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n319.value_0_0 = !((state->_abc_1000_n314_1.value_0_0 | state->_abc_1000_n306_1.value_0_0) & state->_abc_1000_n318_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1139 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n319
  state->_abc_1000_n315_1.value_0_0 = !(state->_abc_1000_n314_1.value_0_0 ^ state->_abc_1000_n306_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1135 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n315_1
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n342_1.value_0_0 = !((state->_abc_1000_n341_1.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n340.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1162 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n342_1
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n320_1.value_0_0 = state->_abc_1000_n319.value_0_0 ^ state->_abc_1000_n317_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1140 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n320_1
  state->_abc_1000_n358.value_0_0 = state->_abc_1000_n357.value_0_0 ^ state->_abc_1000_n351_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1178 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n358
  state->_abc_1000_n364.value_0_0 = !((state->_abc_1000_n357.value_0_0 & state->_abc_1000_n361.value_0_0) | state->_abc_1000_n363.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1184 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n364
  state->_abc_1000_n365.value_0_0 = state->_abc_1000_n364.value_0_0 ^ state->_abc_1000_n360.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1185 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n365
  state->_abc_1000_n250.value_0_0 = !((state->_abc_1000_n249.value_0_0 & state->_abc_1000_n233.value_0_0) | state->_abc_1000_n248.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1070 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n250
  state->_abc_1000_n252.value_0_0 = !((state->_abc_1000_n251.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n250.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1072 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n252
  state->_abc_1000_n283.value_0_0 = !((state->_abc_1000_n282_1.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n281_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1103 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n283
  state->_abc_1000_n266.value_0_0 = state->_abc_1000_n252.value_0_0 & (!state->_abc_1000_n265.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1086 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n266
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n256.value_0_0 = !state->_abc_1000_n252.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1076 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n256
  state->_abc_1000_n253.value_0_0 = state->_abc_1000_n252.value_0_0 ^ state->_abc_1000_n246.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1073 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n253
  state->_abc_1000_n258.value_0_0 = !((state->_abc_1000_n256.value_0_0 | state->_abc_1000_n246.value_0_0) & state->_abc_1000_n257.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1078 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n258
  state->_abc_1000_n259.value_0_0 = state->_abc_1000_n258.value_0_0 ^ state->_abc_1000_n255.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1079 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n259
  state->_abc_1000_n267.value_0_0 = state->_abc_1000_n266.value_0_0 | state->_abc_1000_n264.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1087 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n267
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n268.value_0_0 = state->_abc_1000_n267.value_0_0 ^ state->_abc_1000_n262.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1088 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n268
  state->_abc_1000_n274.value_0_0 = !(state->_abc_1000_n273_1.value_0_0 ^ state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1094 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n274
  state->_abc_1000_n284_1.value_0_0 = !(state->_abc_1000_n283.value_0_0 ^ state->_abc_1000_n276_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1104 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n284_1
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n288_1.value_0_0 = !((state->_abc_1000_n283.value_0_0 | state->_abc_1000_n276_1.value_0_0) & state->_abc_1000_n287_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1108 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n288_1
  state->_abc_1000_n289.value_0_0 = state->_abc_1000_n288_1.value_0_0 ^ state->_abc_1000_n286.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1109 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n289
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n319.value_0_0 = !((state->_abc_1000_n314_1.value_0_0 | state->_abc_1000_n306_1.value_0_0) & state->_abc_1000_n318_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1139 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n319
  state->_abc_1000_n315_1.value_0_0 = !(state->_abc_1000_n314_1.value_0_0 ^ state->_abc_1000_n306_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1135 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n315_1
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n342_1.value_0_0 = !((state->_abc_1000_n341_1.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n340.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1162 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n342_1
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n320_1.value_0_0 = state->_abc_1000_n319.value_0_0 ^ state->_abc_1000_n317_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1140 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n320_1
  state->_abc_1000_n358.value_0_0 = state->_abc_1000_n357.value_0_0 ^ state->_abc_1000_n351_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1178 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n358
  state->_abc_1000_n364.value_0_0 = !((state->_abc_1000_n357.value_0_0 & state->_abc_1000_n361.value_0_0) | state->_abc_1000_n363.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1184 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n364
  state->_abc_1000_n365.value_0_0 = state->_abc_1000_n364.value_0_0 ^ state->_abc_1000_n360.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1185 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n365
  state->_abc_1000_n234.value_0_0 = !state->_abc_1000_n233.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1054 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n234
  state->_abc_1000_n236.value_0_0 = !((state->_abc_1000_n235.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n234.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1056 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n236
  state->_abc_1000_n243.value_0_0 = !((state->_abc_1000_n236.value_0_0 & state->_abc_1000_n240.value_0_0) | state->_abc_1000_n242.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1063 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n243
  state->_abc_1000_n237.value_0_0 = state->_abc_1000_n236.value_0_0 ^ state->_abc_1000_n231.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1057 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n237
  state->_abc_1000_n244.value_0_0 = !(state->_abc_1000_n243.value_0_0 ^ state->_abc_1000_n239.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1064 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n244
  state->_abc_1000_n219.value_0_0 = !(yosys_simplec_get_bit_4_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_4_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1039 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n219
  state->_abc_1000_n224.value_0_0 = !(state->_abc_1000_n223.value_0_0 ^ state->_abc_1000_n219.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1044 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n224
  state->_abc_1000_n228.value_0_0 = !((state->_abc_1000_n223.value_0_0 | state->_abc_1000_n219.value_0_0) & state->_abc_1000_n227.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1048 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n228
  state->_abc_1000_n235.value_0_0 = state->_abc_1000_n226.value_0_0 | state->_abc_1000_n219.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1055 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n235
  state->_abc_1000_n236.value_0_0 = !((state->_abc_1000_n235.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n234.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1056 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n236
  state->_abc_1000_n243.value_0_0 = !((state->_abc_1000_n236.value_0_0 & state->_abc_1000_n240.value_0_0) | state->_abc_1000_n242.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1063 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n243
  state->_abc_1000_n237.value_0_0 = state->_abc_1000_n236.value_0_0 ^ state->_abc_1000_n231.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1057 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n237
  state->_abc_1000_n244.value_0_0 = !(state->_abc_1000_n243.value_0_0 ^ state->_abc_1000_n239.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1064 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n244
  state->_abc_1000_n229.value_0_0 = state->_abc_1000_n228.value_0_0 ^ state->_abc_1000_n226.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1049 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n229
  state->_abc_1000_n251.value_0_0 = state->_abc_1000_n235.value_0_0 | (!state->_abc_1000_n249.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1071 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n251
  state->_abc_1000_n252.value_0_0 = !((state->_abc_1000_n251.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n250.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1072 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n252
  state->_abc_1000_n283.value_0_0 = !((state->_abc_1000_n282_1.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n281_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1103 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n283
  state->_abc_1000_n266.value_0_0 = state->_abc_1000_n252.value_0_0 & (!state->_abc_1000_n265.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1086 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n266
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n256.value_0_0 = !state->_abc_1000_n252.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1076 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n256
  state->_abc_1000_n253.value_0_0 = state->_abc_1000_n252.value_0_0 ^ state->_abc_1000_n246.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1073 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n253
  state->_abc_1000_n258.value_0_0 = !((state->_abc_1000_n256.value_0_0 | state->_abc_1000_n246.value_0_0) & state->_abc_1000_n257.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1078 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n258
  state->_abc_1000_n259.value_0_0 = state->_abc_1000_n258.value_0_0 ^ state->_abc_1000_n255.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1079 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n259
  state->_abc_1000_n267.value_0_0 = state->_abc_1000_n266.value_0_0 | state->_abc_1000_n264.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1087 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n267
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n268.value_0_0 = state->_abc_1000_n267.value_0_0 ^ state->_abc_1000_n262.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1088 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n268
  state->_abc_1000_n274.value_0_0 = !(state->_abc_1000_n273_1.value_0_0 ^ state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1094 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n274
  state->_abc_1000_n284_1.value_0_0 = !(state->_abc_1000_n283.value_0_0 ^ state->_abc_1000_n276_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1104 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n284_1
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n288_1.value_0_0 = !((state->_abc_1000_n283.value_0_0 | state->_abc_1000_n276_1.value_0_0) & state->_abc_1000_n287_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1108 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n288_1
  state->_abc_1000_n289.value_0_0 = state->_abc_1000_n288_1.value_0_0 ^ state->_abc_1000_n286.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1109 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n289
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n319.value_0_0 = !((state->_abc_1000_n314_1.value_0_0 | state->_abc_1000_n306_1.value_0_0) & state->_abc_1000_n318_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1139 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n319
  state->_abc_1000_n315_1.value_0_0 = !(state->_abc_1000_n314_1.value_0_0 ^ state->_abc_1000_n306_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1135 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n315_1
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n342_1.value_0_0 = !((state->_abc_1000_n341_1.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n340.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1162 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n342_1
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n320_1.value_0_0 = state->_abc_1000_n319.value_0_0 ^ state->_abc_1000_n317_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1140 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n320_1
  state->_abc_1000_n358.value_0_0 = state->_abc_1000_n357.value_0_0 ^ state->_abc_1000_n351_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1178 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n358
  state->_abc_1000_n364.value_0_0 = !((state->_abc_1000_n357.value_0_0 & state->_abc_1000_n361.value_0_0) | state->_abc_1000_n363.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1184 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n364
  state->_abc_1000_n365.value_0_0 = state->_abc_1000_n364.value_0_0 ^ state->_abc_1000_n360.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1185 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n365
  state->_abc_1000_n227.value_0_0 = !(yosys_simplec_get_bit_4_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_4_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1047 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n227
  state->_abc_1000_n228.value_0_0 = !((state->_abc_1000_n223.value_0_0 | state->_abc_1000_n219.value_0_0) & state->_abc_1000_n227.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1048 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n228
  state->_abc_1000_n233.value_0_0 = !((state->_abc_1000_n227.value_0_0 | state->_abc_1000_n226.value_0_0) & state->_abc_1000_n232.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1053 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n233
  state->_abc_1000_n229.value_0_0 = state->_abc_1000_n228.value_0_0 ^ state->_abc_1000_n226.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1049 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n229
  state->_abc_1000_n250.value_0_0 = !((state->_abc_1000_n249.value_0_0 & state->_abc_1000_n233.value_0_0) | state->_abc_1000_n248.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1070 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n250
  state->_abc_1000_n252.value_0_0 = !((state->_abc_1000_n251.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n250.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1072 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n252
  state->_abc_1000_n283.value_0_0 = !((state->_abc_1000_n282_1.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n281_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1103 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n283
  state->_abc_1000_n266.value_0_0 = state->_abc_1000_n252.value_0_0 & (!state->_abc_1000_n265.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1086 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n266
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n256.value_0_0 = !state->_abc_1000_n252.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1076 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n256
  state->_abc_1000_n253.value_0_0 = state->_abc_1000_n252.value_0_0 ^ state->_abc_1000_n246.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1073 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n253
  state->_abc_1000_n258.value_0_0 = !((state->_abc_1000_n256.value_0_0 | state->_abc_1000_n246.value_0_0) & state->_abc_1000_n257.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1078 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n258
  state->_abc_1000_n259.value_0_0 = state->_abc_1000_n258.value_0_0 ^ state->_abc_1000_n255.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1079 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n259
  state->_abc_1000_n267.value_0_0 = state->_abc_1000_n266.value_0_0 | state->_abc_1000_n264.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1087 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n267
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n268.value_0_0 = state->_abc_1000_n267.value_0_0 ^ state->_abc_1000_n262.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1088 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n268
  state->_abc_1000_n274.value_0_0 = !(state->_abc_1000_n273_1.value_0_0 ^ state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1094 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n274
  state->_abc_1000_n284_1.value_0_0 = !(state->_abc_1000_n283.value_0_0 ^ state->_abc_1000_n276_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1104 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n284_1
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n288_1.value_0_0 = !((state->_abc_1000_n283.value_0_0 | state->_abc_1000_n276_1.value_0_0) & state->_abc_1000_n287_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1108 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n288_1
  state->_abc_1000_n289.value_0_0 = state->_abc_1000_n288_1.value_0_0 ^ state->_abc_1000_n286.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1109 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n289
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n319.value_0_0 = !((state->_abc_1000_n314_1.value_0_0 | state->_abc_1000_n306_1.value_0_0) & state->_abc_1000_n318_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1139 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n319
  state->_abc_1000_n315_1.value_0_0 = !(state->_abc_1000_n314_1.value_0_0 ^ state->_abc_1000_n306_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1135 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n315_1
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n342_1.value_0_0 = !((state->_abc_1000_n341_1.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n340.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1162 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n342_1
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n320_1.value_0_0 = state->_abc_1000_n319.value_0_0 ^ state->_abc_1000_n317_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1140 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n320_1
  state->_abc_1000_n358.value_0_0 = state->_abc_1000_n357.value_0_0 ^ state->_abc_1000_n351_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1178 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n358
  state->_abc_1000_n364.value_0_0 = !((state->_abc_1000_n357.value_0_0 & state->_abc_1000_n361.value_0_0) | state->_abc_1000_n363.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1184 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n364
  state->_abc_1000_n365.value_0_0 = state->_abc_1000_n364.value_0_0 ^ state->_abc_1000_n360.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1185 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n365
  state->_abc_1000_n234.value_0_0 = !state->_abc_1000_n233.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1054 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n234
  state->_abc_1000_n236.value_0_0 = !((state->_abc_1000_n235.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n234.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1056 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n236
  state->_abc_1000_n243.value_0_0 = !((state->_abc_1000_n236.value_0_0 & state->_abc_1000_n240.value_0_0) | state->_abc_1000_n242.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1063 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n243
  state->_abc_1000_n237.value_0_0 = state->_abc_1000_n236.value_0_0 ^ state->_abc_1000_n231.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1057 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n237
  state->_abc_1000_n244.value_0_0 = !(state->_abc_1000_n243.value_0_0 ^ state->_abc_1000_n239.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1064 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n244
  yosys_simplec_set_bit_1_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n206.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1027 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [1]
  yosys_simplec_set_bit_2_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n211.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1032 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [2]
  yosys_simplec_set_bit_3_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n217.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1038 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [3]
  yosys_simplec_set_bit_4_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n224.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1045 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [4]
  yosys_simplec_set_bit_12_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n284_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1105 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [12]
  yosys_simplec_set_bit_23_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n365.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1186 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [23]
  yosys_simplec_set_bit_26_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n394.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1215 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [26]
  yosys_simplec_set_bit_19_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n333_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1154 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [19]
  yosys_simplec_set_bit_22_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n358.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1179 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [22]
  yosys_simplec_set_bit_11_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n274.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1095 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [11]
  yosys_simplec_set_bit_18_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n327_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1148 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [18]
  yosys_simplec_set_bit_25_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n385.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1206 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [25]
  yosys_simplec_set_bit_10_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n268.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1089 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [10]
  yosys_simplec_set_bit_21_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n349.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1170 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [21]
  yosys_simplec_set_bit_17_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n320_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1141 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [17]
  yosys_simplec_set_bit_31_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n431.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1252 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [31]
  yosys_simplec_set_bit_24_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n377.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1198 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [24]
  yosys_simplec_set_bit_16_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n315_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1136 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [16]
  yosys_simplec_set_bit_30_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n426.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1247 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [30]
  yosys_simplec_set_bit_20_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n343.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1164 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [20]
  yosys_simplec_set_bit_9_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n259.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1080 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [9]
  yosys_simplec_set_bit_8_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n253.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1074 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [8]
  yosys_simplec_set_bit_29_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n418.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1239 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [29]
  yosys_simplec_set_bit_15_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n304.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1125 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [15]
  yosys_simplec_set_bit_7_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n244.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1065 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [7]
  yosys_simplec_set_bit_28_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n412.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1233 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [28]
  yosys_simplec_set_bit_14_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n297_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1118 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [14]
  yosys_simplec_set_bit_6_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n237.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1058 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [6]
  yosys_simplec_set_bit_27_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n401.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1222 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [27]
  yosys_simplec_set_bit_13_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n289.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1110 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [13]
  yosys_simplec_set_bit_5_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n229.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1050 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [5]
  yosys_simplec_set_bit_0_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n198_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1023 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [0]
  state->_abc_1000_n210.value_0_0 = !((state->_abc_1000_n205_1.value_0_0 | state->_abc_1000_n204_1.value_0_0) & state->_abc_1000_n209.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1030 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n210
  state->_abc_1000_n211.value_0_0 = !(state->_abc_1000_n210.value_0_0 ^ state->_abc_1000_n208.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1031 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n211
  yosys_simplec_set_bit_2_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n211.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1032 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [2]
  state->_abc_1000_n223.value_0_0 = !((state->_abc_1000_n222.value_0_0 & state->_abc_1000_n210.value_0_0) | state->_abc_1000_n221.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1043 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n223
  state->_abc_1000_n224.value_0_0 = !(state->_abc_1000_n223.value_0_0 ^ state->_abc_1000_n219.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1044 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n224
  state->_abc_1000_n252.value_0_0 = !((state->_abc_1000_n251.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n250.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1072 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n252
  state->_abc_1000_n283.value_0_0 = !((state->_abc_1000_n282_1.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n281_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1103 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n283
  state->_abc_1000_n266.value_0_0 = state->_abc_1000_n252.value_0_0 & (!state->_abc_1000_n265.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1086 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n266
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n256.value_0_0 = !state->_abc_1000_n252.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1076 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n256
  state->_abc_1000_n253.value_0_0 = state->_abc_1000_n252.value_0_0 ^ state->_abc_1000_n246.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1073 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n253
  state->_abc_1000_n258.value_0_0 = !((state->_abc_1000_n256.value_0_0 | state->_abc_1000_n246.value_0_0) & state->_abc_1000_n257.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1078 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n258
  state->_abc_1000_n259.value_0_0 = state->_abc_1000_n258.value_0_0 ^ state->_abc_1000_n255.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1079 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n259
  state->_abc_1000_n267.value_0_0 = state->_abc_1000_n266.value_0_0 | state->_abc_1000_n264.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1087 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n267
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n268.value_0_0 = state->_abc_1000_n267.value_0_0 ^ state->_abc_1000_n262.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1088 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n268
  state->_abc_1000_n274.value_0_0 = !(state->_abc_1000_n273_1.value_0_0 ^ state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1094 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n274
  state->_abc_1000_n284_1.value_0_0 = !(state->_abc_1000_n283.value_0_0 ^ state->_abc_1000_n276_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1104 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n284_1
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n288_1.value_0_0 = !((state->_abc_1000_n283.value_0_0 | state->_abc_1000_n276_1.value_0_0) & state->_abc_1000_n287_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1108 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n288_1
  state->_abc_1000_n289.value_0_0 = state->_abc_1000_n288_1.value_0_0 ^ state->_abc_1000_n286.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1109 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n289
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n319.value_0_0 = !((state->_abc_1000_n314_1.value_0_0 | state->_abc_1000_n306_1.value_0_0) & state->_abc_1000_n318_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1139 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n319
  state->_abc_1000_n315_1.value_0_0 = !(state->_abc_1000_n314_1.value_0_0 ^ state->_abc_1000_n306_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1135 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n315_1
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n342_1.value_0_0 = !((state->_abc_1000_n341_1.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n340.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1162 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n342_1
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n320_1.value_0_0 = state->_abc_1000_n319.value_0_0 ^ state->_abc_1000_n317_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1140 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n320_1
  state->_abc_1000_n358.value_0_0 = state->_abc_1000_n357.value_0_0 ^ state->_abc_1000_n351_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1178 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n358
  state->_abc_1000_n364.value_0_0 = !((state->_abc_1000_n357.value_0_0 & state->_abc_1000_n361.value_0_0) | state->_abc_1000_n363.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1184 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n364
  state->_abc_1000_n365.value_0_0 = state->_abc_1000_n364.value_0_0 ^ state->_abc_1000_n360.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1185 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n365
  state->_abc_1000_n236.value_0_0 = !((state->_abc_1000_n235.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n234.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1056 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n236
  state->_abc_1000_n243.value_0_0 = !((state->_abc_1000_n236.value_0_0 & state->_abc_1000_n240.value_0_0) | state->_abc_1000_n242.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1063 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n243
  state->_abc_1000_n237.value_0_0 = state->_abc_1000_n236.value_0_0 ^ state->_abc_1000_n231.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1057 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n237
  state->_abc_1000_n244.value_0_0 = !(state->_abc_1000_n243.value_0_0 ^ state->_abc_1000_n239.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1064 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n244
  state->_abc_1000_n228.value_0_0 = !((state->_abc_1000_n223.value_0_0 | state->_abc_1000_n219.value_0_0) & state->_abc_1000_n227.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1048 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n228
  state->_abc_1000_n229.value_0_0 = state->_abc_1000_n228.value_0_0 ^ state->_abc_1000_n226.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1049 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n229
  yosys_simplec_set_bit_4_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n224.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1045 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [4]
  yosys_simplec_set_bit_12_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n284_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1105 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [12]
  yosys_simplec_set_bit_23_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n365.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1186 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [23]
  yosys_simplec_set_bit_26_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n394.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1215 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [26]
  yosys_simplec_set_bit_19_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n333_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1154 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [19]
  yosys_simplec_set_bit_22_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n358.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1179 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [22]
  yosys_simplec_set_bit_11_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n274.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1095 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [11]
  yosys_simplec_set_bit_18_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n327_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1148 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [18]
  yosys_simplec_set_bit_25_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n385.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1206 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [25]
  yosys_simplec_set_bit_10_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n268.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1089 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [10]
  yosys_simplec_set_bit_21_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n349.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1170 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [21]
  yosys_simplec_set_bit_17_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n320_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1141 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [17]
  yosys_simplec_set_bit_31_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n431.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1252 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [31]
  yosys_simplec_set_bit_24_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n377.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1198 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [24]
  yosys_simplec_set_bit_16_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n315_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1136 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [16]
  yosys_simplec_set_bit_30_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n426.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1247 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [30]
  yosys_simplec_set_bit_20_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n343.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1164 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [20]
  yosys_simplec_set_bit_9_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n259.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1080 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [9]
  yosys_simplec_set_bit_8_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n253.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1074 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [8]
  yosys_simplec_set_bit_29_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n418.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1239 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [29]
  yosys_simplec_set_bit_15_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n304.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1125 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [15]
  yosys_simplec_set_bit_7_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n244.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1065 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [7]
  yosys_simplec_set_bit_28_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n412.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1233 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [28]
  yosys_simplec_set_bit_14_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n297_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1118 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [14]
  yosys_simplec_set_bit_6_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n237.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1058 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [6]
  yosys_simplec_set_bit_27_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n401.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1222 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [27]
  yosys_simplec_set_bit_13_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n289.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1110 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [13]
  yosys_simplec_set_bit_5_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n229.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1050 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [5]
  state->_abc_1000_n216.value_0_0 = !((state->_abc_1000_n210.value_0_0 & state->_abc_1000_n208.value_0_0) | state->_abc_1000_n215.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1036 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n216
  state->_abc_1000_n217.value_0_0 = !(state->_abc_1000_n216.value_0_0 ^ state->_abc_1000_n213.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1037 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n217
  yosys_simplec_set_bit_3_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n217.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1038 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [3]
  state->_abc_1000_n206.value_0_0 = !(state->_abc_1000_n205_1.value_0_0 ^ state->_abc_1000_n204_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1026 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n206
  yosys_simplec_set_bit_1_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n206.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1027 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [1]
  state->_abc_1000_n213.value_0_0 = !(yosys_simplec_get_bit_3_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_3_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1033 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n213
  state->_abc_1000_n221.value_0_0 = !((state->_abc_1000_n214.value_0_0 | state->_abc_1000_n213.value_0_0) & state->_abc_1000_n220.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1041 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n221
  state->_abc_1000_n217.value_0_0 = !(state->_abc_1000_n216.value_0_0 ^ state->_abc_1000_n213.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1037 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n217
  yosys_simplec_set_bit_3_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n217.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1038 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [3]
  state->_abc_1000_n222.value_0_0 = state->_abc_1000_n208.value_0_0 & (!state->_abc_1000_n213.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1042 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n222
  state->_abc_1000_n223.value_0_0 = !((state->_abc_1000_n222.value_0_0 & state->_abc_1000_n210.value_0_0) | state->_abc_1000_n221.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1043 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n223
  state->_abc_1000_n224.value_0_0 = !(state->_abc_1000_n223.value_0_0 ^ state->_abc_1000_n219.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1044 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n224
  state->_abc_1000_n252.value_0_0 = !((state->_abc_1000_n251.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n250.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1072 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n252
  state->_abc_1000_n283.value_0_0 = !((state->_abc_1000_n282_1.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n281_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1103 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n283
  state->_abc_1000_n266.value_0_0 = state->_abc_1000_n252.value_0_0 & (!state->_abc_1000_n265.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1086 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n266
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n256.value_0_0 = !state->_abc_1000_n252.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1076 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n256
  state->_abc_1000_n253.value_0_0 = state->_abc_1000_n252.value_0_0 ^ state->_abc_1000_n246.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1073 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n253
  state->_abc_1000_n258.value_0_0 = !((state->_abc_1000_n256.value_0_0 | state->_abc_1000_n246.value_0_0) & state->_abc_1000_n257.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1078 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n258
  state->_abc_1000_n259.value_0_0 = state->_abc_1000_n258.value_0_0 ^ state->_abc_1000_n255.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1079 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n259
  state->_abc_1000_n267.value_0_0 = state->_abc_1000_n266.value_0_0 | state->_abc_1000_n264.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1087 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n267
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n268.value_0_0 = state->_abc_1000_n267.value_0_0 ^ state->_abc_1000_n262.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1088 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n268
  state->_abc_1000_n274.value_0_0 = !(state->_abc_1000_n273_1.value_0_0 ^ state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1094 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n274
  state->_abc_1000_n284_1.value_0_0 = !(state->_abc_1000_n283.value_0_0 ^ state->_abc_1000_n276_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1104 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n284_1
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n288_1.value_0_0 = !((state->_abc_1000_n283.value_0_0 | state->_abc_1000_n276_1.value_0_0) & state->_abc_1000_n287_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1108 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n288_1
  state->_abc_1000_n289.value_0_0 = state->_abc_1000_n288_1.value_0_0 ^ state->_abc_1000_n286.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1109 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n289
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n319.value_0_0 = !((state->_abc_1000_n314_1.value_0_0 | state->_abc_1000_n306_1.value_0_0) & state->_abc_1000_n318_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1139 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n319
  state->_abc_1000_n315_1.value_0_0 = !(state->_abc_1000_n314_1.value_0_0 ^ state->_abc_1000_n306_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1135 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n315_1
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n342_1.value_0_0 = !((state->_abc_1000_n341_1.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n340.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1162 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n342_1
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n320_1.value_0_0 = state->_abc_1000_n319.value_0_0 ^ state->_abc_1000_n317_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1140 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n320_1
  state->_abc_1000_n358.value_0_0 = state->_abc_1000_n357.value_0_0 ^ state->_abc_1000_n351_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1178 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n358
  state->_abc_1000_n364.value_0_0 = !((state->_abc_1000_n357.value_0_0 & state->_abc_1000_n361.value_0_0) | state->_abc_1000_n363.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1184 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n364
  state->_abc_1000_n365.value_0_0 = state->_abc_1000_n364.value_0_0 ^ state->_abc_1000_n360.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1185 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n365
  state->_abc_1000_n236.value_0_0 = !((state->_abc_1000_n235.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n234.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1056 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n236
  state->_abc_1000_n243.value_0_0 = !((state->_abc_1000_n236.value_0_0 & state->_abc_1000_n240.value_0_0) | state->_abc_1000_n242.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1063 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n243
  state->_abc_1000_n237.value_0_0 = state->_abc_1000_n236.value_0_0 ^ state->_abc_1000_n231.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1057 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n237
  state->_abc_1000_n244.value_0_0 = !(state->_abc_1000_n243.value_0_0 ^ state->_abc_1000_n239.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1064 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n244
  state->_abc_1000_n228.value_0_0 = !((state->_abc_1000_n223.value_0_0 | state->_abc_1000_n219.value_0_0) & state->_abc_1000_n227.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1048 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n228
  state->_abc_1000_n229.value_0_0 = state->_abc_1000_n228.value_0_0 ^ state->_abc_1000_n226.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1049 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n229
  yosys_simplec_set_bit_4_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n224.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1045 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [4]
  yosys_simplec_set_bit_12_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n284_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1105 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [12]
  yosys_simplec_set_bit_23_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n365.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1186 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [23]
  yosys_simplec_set_bit_26_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n394.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1215 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [26]
  yosys_simplec_set_bit_19_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n333_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1154 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [19]
  yosys_simplec_set_bit_22_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n358.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1179 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [22]
  yosys_simplec_set_bit_11_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n274.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1095 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [11]
  yosys_simplec_set_bit_18_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n327_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1148 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [18]
  yosys_simplec_set_bit_25_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n385.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1206 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [25]
  yosys_simplec_set_bit_10_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n268.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1089 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [10]
  yosys_simplec_set_bit_21_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n349.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1170 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [21]
  yosys_simplec_set_bit_17_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n320_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1141 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [17]
  yosys_simplec_set_bit_31_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n431.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1252 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [31]
  yosys_simplec_set_bit_24_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n377.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1198 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [24]
  yosys_simplec_set_bit_16_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n315_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1136 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [16]
  yosys_simplec_set_bit_30_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n426.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1247 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [30]
  yosys_simplec_set_bit_20_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n343.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1164 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [20]
  yosys_simplec_set_bit_9_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n259.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1080 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [9]
  yosys_simplec_set_bit_8_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n253.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1074 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [8]
  yosys_simplec_set_bit_29_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n418.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1239 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [29]
  yosys_simplec_set_bit_15_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n304.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1125 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [15]
  yosys_simplec_set_bit_7_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n244.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1065 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [7]
  yosys_simplec_set_bit_28_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n412.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1233 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [28]
  yosys_simplec_set_bit_14_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n297_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1118 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [14]
  yosys_simplec_set_bit_6_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n237.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1058 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [6]
  yosys_simplec_set_bit_27_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n401.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1222 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [27]
  yosys_simplec_set_bit_13_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n289.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1110 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [13]
  yosys_simplec_set_bit_5_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n229.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1050 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [5]
  state->_abc_1000_n220.value_0_0 = !(yosys_simplec_get_bit_3_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_3_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1040 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n220
  state->_abc_1000_n221.value_0_0 = !((state->_abc_1000_n214.value_0_0 | state->_abc_1000_n213.value_0_0) & state->_abc_1000_n220.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1041 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n221
  state->_abc_1000_n223.value_0_0 = !((state->_abc_1000_n222.value_0_0 & state->_abc_1000_n210.value_0_0) | state->_abc_1000_n221.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1043 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n223
  state->_abc_1000_n224.value_0_0 = !(state->_abc_1000_n223.value_0_0 ^ state->_abc_1000_n219.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1044 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n224
  state->_abc_1000_n252.value_0_0 = !((state->_abc_1000_n251.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n250.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1072 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n252
  state->_abc_1000_n283.value_0_0 = !((state->_abc_1000_n282_1.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n281_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1103 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n283
  state->_abc_1000_n266.value_0_0 = state->_abc_1000_n252.value_0_0 & (!state->_abc_1000_n265.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1086 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n266
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n256.value_0_0 = !state->_abc_1000_n252.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1076 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n256
  state->_abc_1000_n253.value_0_0 = state->_abc_1000_n252.value_0_0 ^ state->_abc_1000_n246.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1073 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n253
  state->_abc_1000_n258.value_0_0 = !((state->_abc_1000_n256.value_0_0 | state->_abc_1000_n246.value_0_0) & state->_abc_1000_n257.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1078 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n258
  state->_abc_1000_n259.value_0_0 = state->_abc_1000_n258.value_0_0 ^ state->_abc_1000_n255.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1079 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n259
  state->_abc_1000_n267.value_0_0 = state->_abc_1000_n266.value_0_0 | state->_abc_1000_n264.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1087 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n267
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n268.value_0_0 = state->_abc_1000_n267.value_0_0 ^ state->_abc_1000_n262.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1088 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n268
  state->_abc_1000_n274.value_0_0 = !(state->_abc_1000_n273_1.value_0_0 ^ state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1094 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n274
  state->_abc_1000_n284_1.value_0_0 = !(state->_abc_1000_n283.value_0_0 ^ state->_abc_1000_n276_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1104 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n284_1
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n288_1.value_0_0 = !((state->_abc_1000_n283.value_0_0 | state->_abc_1000_n276_1.value_0_0) & state->_abc_1000_n287_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1108 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n288_1
  state->_abc_1000_n289.value_0_0 = state->_abc_1000_n288_1.value_0_0 ^ state->_abc_1000_n286.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1109 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n289
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n319.value_0_0 = !((state->_abc_1000_n314_1.value_0_0 | state->_abc_1000_n306_1.value_0_0) & state->_abc_1000_n318_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1139 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n319
  state->_abc_1000_n315_1.value_0_0 = !(state->_abc_1000_n314_1.value_0_0 ^ state->_abc_1000_n306_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1135 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n315_1
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n342_1.value_0_0 = !((state->_abc_1000_n341_1.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n340.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1162 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n342_1
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n320_1.value_0_0 = state->_abc_1000_n319.value_0_0 ^ state->_abc_1000_n317_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1140 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n320_1
  state->_abc_1000_n358.value_0_0 = state->_abc_1000_n357.value_0_0 ^ state->_abc_1000_n351_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1178 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n358
  state->_abc_1000_n364.value_0_0 = !((state->_abc_1000_n357.value_0_0 & state->_abc_1000_n361.value_0_0) | state->_abc_1000_n363.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1184 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n364
  state->_abc_1000_n365.value_0_0 = state->_abc_1000_n364.value_0_0 ^ state->_abc_1000_n360.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1185 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n365
  state->_abc_1000_n236.value_0_0 = !((state->_abc_1000_n235.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n234.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1056 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n236
  state->_abc_1000_n243.value_0_0 = !((state->_abc_1000_n236.value_0_0 & state->_abc_1000_n240.value_0_0) | state->_abc_1000_n242.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1063 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n243
  state->_abc_1000_n237.value_0_0 = state->_abc_1000_n236.value_0_0 ^ state->_abc_1000_n231.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1057 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n237
  state->_abc_1000_n244.value_0_0 = !(state->_abc_1000_n243.value_0_0 ^ state->_abc_1000_n239.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1064 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n244
  state->_abc_1000_n228.value_0_0 = !((state->_abc_1000_n223.value_0_0 | state->_abc_1000_n219.value_0_0) & state->_abc_1000_n227.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1048 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n228
  state->_abc_1000_n229.value_0_0 = state->_abc_1000_n228.value_0_0 ^ state->_abc_1000_n226.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1049 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n229
  yosys_simplec_set_bit_4_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n224.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1045 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [4]
  yosys_simplec_set_bit_12_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n284_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1105 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [12]
  yosys_simplec_set_bit_23_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n365.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1186 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [23]
  yosys_simplec_set_bit_26_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n394.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1215 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [26]
  yosys_simplec_set_bit_19_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n333_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1154 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [19]
  yosys_simplec_set_bit_22_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n358.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1179 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [22]
  yosys_simplec_set_bit_11_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n274.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1095 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [11]
  yosys_simplec_set_bit_18_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n327_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1148 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [18]
  yosys_simplec_set_bit_25_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n385.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1206 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [25]
  yosys_simplec_set_bit_10_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n268.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1089 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [10]
  yosys_simplec_set_bit_21_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n349.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1170 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [21]
  yosys_simplec_set_bit_17_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n320_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1141 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [17]
  yosys_simplec_set_bit_31_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n431.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1252 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [31]
  yosys_simplec_set_bit_24_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n377.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1198 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [24]
  yosys_simplec_set_bit_16_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n315_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1136 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [16]
  yosys_simplec_set_bit_30_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n426.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1247 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [30]
  yosys_simplec_set_bit_20_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n343.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1164 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [20]
  yosys_simplec_set_bit_9_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n259.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1080 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [9]
  yosys_simplec_set_bit_8_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n253.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1074 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [8]
  yosys_simplec_set_bit_29_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n418.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1239 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [29]
  yosys_simplec_set_bit_15_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n304.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1125 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [15]
  yosys_simplec_set_bit_7_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n244.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1065 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [7]
  yosys_simplec_set_bit_28_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n412.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1233 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [28]
  yosys_simplec_set_bit_14_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n297_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1118 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [14]
  yosys_simplec_set_bit_6_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n237.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1058 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [6]
  yosys_simplec_set_bit_27_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n401.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1222 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [27]
  yosys_simplec_set_bit_13_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n289.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1110 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [13]
  yosys_simplec_set_bit_5_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n229.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1050 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [5]
  state->_abc_1000_n214.value_0_0 = !(yosys_simplec_get_bit_2_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_2_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1034 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n214
  state->_abc_1000_n215.value_0_0 = !state->_abc_1000_n214.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1035 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n215
  state->_abc_1000_n221.value_0_0 = !((state->_abc_1000_n214.value_0_0 | state->_abc_1000_n213.value_0_0) & state->_abc_1000_n220.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1041 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n221
  state->_abc_1000_n223.value_0_0 = !((state->_abc_1000_n222.value_0_0 & state->_abc_1000_n210.value_0_0) | state->_abc_1000_n221.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1043 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n223
  state->_abc_1000_n224.value_0_0 = !(state->_abc_1000_n223.value_0_0 ^ state->_abc_1000_n219.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1044 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n224
  state->_abc_1000_n252.value_0_0 = !((state->_abc_1000_n251.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n250.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1072 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n252
  state->_abc_1000_n283.value_0_0 = !((state->_abc_1000_n282_1.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n281_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1103 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n283
  state->_abc_1000_n266.value_0_0 = state->_abc_1000_n252.value_0_0 & (!state->_abc_1000_n265.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1086 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n266
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n256.value_0_0 = !state->_abc_1000_n252.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1076 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n256
  state->_abc_1000_n253.value_0_0 = state->_abc_1000_n252.value_0_0 ^ state->_abc_1000_n246.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1073 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n253
  state->_abc_1000_n258.value_0_0 = !((state->_abc_1000_n256.value_0_0 | state->_abc_1000_n246.value_0_0) & state->_abc_1000_n257.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1078 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n258
  state->_abc_1000_n259.value_0_0 = state->_abc_1000_n258.value_0_0 ^ state->_abc_1000_n255.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1079 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n259
  state->_abc_1000_n267.value_0_0 = state->_abc_1000_n266.value_0_0 | state->_abc_1000_n264.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1087 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n267
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n268.value_0_0 = state->_abc_1000_n267.value_0_0 ^ state->_abc_1000_n262.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1088 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n268
  state->_abc_1000_n274.value_0_0 = !(state->_abc_1000_n273_1.value_0_0 ^ state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1094 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n274
  state->_abc_1000_n284_1.value_0_0 = !(state->_abc_1000_n283.value_0_0 ^ state->_abc_1000_n276_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1104 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n284_1
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n288_1.value_0_0 = !((state->_abc_1000_n283.value_0_0 | state->_abc_1000_n276_1.value_0_0) & state->_abc_1000_n287_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1108 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n288_1
  state->_abc_1000_n289.value_0_0 = state->_abc_1000_n288_1.value_0_0 ^ state->_abc_1000_n286.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1109 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n289
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n319.value_0_0 = !((state->_abc_1000_n314_1.value_0_0 | state->_abc_1000_n306_1.value_0_0) & state->_abc_1000_n318_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1139 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n319
  state->_abc_1000_n315_1.value_0_0 = !(state->_abc_1000_n314_1.value_0_0 ^ state->_abc_1000_n306_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1135 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n315_1
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n342_1.value_0_0 = !((state->_abc_1000_n341_1.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n340.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1162 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n342_1
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n320_1.value_0_0 = state->_abc_1000_n319.value_0_0 ^ state->_abc_1000_n317_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1140 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n320_1
  state->_abc_1000_n358.value_0_0 = state->_abc_1000_n357.value_0_0 ^ state->_abc_1000_n351_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1178 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n358
  state->_abc_1000_n364.value_0_0 = !((state->_abc_1000_n357.value_0_0 & state->_abc_1000_n361.value_0_0) | state->_abc_1000_n363.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1184 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n364
  state->_abc_1000_n365.value_0_0 = state->_abc_1000_n364.value_0_0 ^ state->_abc_1000_n360.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1185 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n365
  state->_abc_1000_n236.value_0_0 = !((state->_abc_1000_n235.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n234.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1056 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n236
  state->_abc_1000_n243.value_0_0 = !((state->_abc_1000_n236.value_0_0 & state->_abc_1000_n240.value_0_0) | state->_abc_1000_n242.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1063 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n243
  state->_abc_1000_n237.value_0_0 = state->_abc_1000_n236.value_0_0 ^ state->_abc_1000_n231.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1057 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n237
  state->_abc_1000_n244.value_0_0 = !(state->_abc_1000_n243.value_0_0 ^ state->_abc_1000_n239.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1064 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n244
  state->_abc_1000_n228.value_0_0 = !((state->_abc_1000_n223.value_0_0 | state->_abc_1000_n219.value_0_0) & state->_abc_1000_n227.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1048 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n228
  state->_abc_1000_n229.value_0_0 = state->_abc_1000_n228.value_0_0 ^ state->_abc_1000_n226.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1049 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n229
  yosys_simplec_set_bit_4_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n224.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1045 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [4]
  yosys_simplec_set_bit_12_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n284_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1105 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [12]
  yosys_simplec_set_bit_23_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n365.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1186 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [23]
  yosys_simplec_set_bit_26_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n394.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1215 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [26]
  yosys_simplec_set_bit_19_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n333_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1154 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [19]
  yosys_simplec_set_bit_22_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n358.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1179 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [22]
  yosys_simplec_set_bit_11_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n274.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1095 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [11]
  yosys_simplec_set_bit_18_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n327_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1148 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [18]
  yosys_simplec_set_bit_25_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n385.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1206 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [25]
  yosys_simplec_set_bit_10_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n268.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1089 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [10]
  yosys_simplec_set_bit_21_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n349.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1170 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [21]
  yosys_simplec_set_bit_17_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n320_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1141 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [17]
  yosys_simplec_set_bit_31_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n431.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1252 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [31]
  yosys_simplec_set_bit_24_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n377.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1198 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [24]
  yosys_simplec_set_bit_16_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n315_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1136 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [16]
  yosys_simplec_set_bit_30_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n426.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1247 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [30]
  yosys_simplec_set_bit_20_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n343.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1164 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [20]
  yosys_simplec_set_bit_9_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n259.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1080 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [9]
  yosys_simplec_set_bit_8_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n253.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1074 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [8]
  yosys_simplec_set_bit_29_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n418.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1239 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [29]
  yosys_simplec_set_bit_15_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n304.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1125 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [15]
  yosys_simplec_set_bit_7_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n244.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1065 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [7]
  yosys_simplec_set_bit_28_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n412.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1233 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [28]
  yosys_simplec_set_bit_14_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n297_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1118 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [14]
  yosys_simplec_set_bit_6_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n237.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1058 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [6]
  yosys_simplec_set_bit_27_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n401.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1222 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [27]
  yosys_simplec_set_bit_13_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n289.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1110 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [13]
  yosys_simplec_set_bit_5_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n229.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1050 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [5]
  state->_abc_1000_n216.value_0_0 = !((state->_abc_1000_n210.value_0_0 & state->_abc_1000_n208.value_0_0) | state->_abc_1000_n215.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1036 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n216
  state->_abc_1000_n217.value_0_0 = !(state->_abc_1000_n216.value_0_0 ^ state->_abc_1000_n213.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1037 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n217
  yosys_simplec_set_bit_3_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n217.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1038 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [3]
  state->_abc_1000_n208.value_0_0 = yosys_simplec_get_bit_2_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_2_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1028 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n208
  state->_abc_1000_n222.value_0_0 = state->_abc_1000_n208.value_0_0 & (!state->_abc_1000_n213.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1042 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n222
  state->_abc_1000_n211.value_0_0 = !(state->_abc_1000_n210.value_0_0 ^ state->_abc_1000_n208.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1031 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n211
  yosys_simplec_set_bit_2_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n211.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1032 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [2]
  state->_abc_1000_n223.value_0_0 = !((state->_abc_1000_n222.value_0_0 & state->_abc_1000_n210.value_0_0) | state->_abc_1000_n221.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1043 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n223
  state->_abc_1000_n224.value_0_0 = !(state->_abc_1000_n223.value_0_0 ^ state->_abc_1000_n219.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1044 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n224
  state->_abc_1000_n252.value_0_0 = !((state->_abc_1000_n251.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n250.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1072 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n252
  state->_abc_1000_n283.value_0_0 = !((state->_abc_1000_n282_1.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n281_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1103 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n283
  state->_abc_1000_n266.value_0_0 = state->_abc_1000_n252.value_0_0 & (!state->_abc_1000_n265.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1086 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n266
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n256.value_0_0 = !state->_abc_1000_n252.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1076 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n256
  state->_abc_1000_n253.value_0_0 = state->_abc_1000_n252.value_0_0 ^ state->_abc_1000_n246.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1073 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n253
  state->_abc_1000_n258.value_0_0 = !((state->_abc_1000_n256.value_0_0 | state->_abc_1000_n246.value_0_0) & state->_abc_1000_n257.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1078 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n258
  state->_abc_1000_n259.value_0_0 = state->_abc_1000_n258.value_0_0 ^ state->_abc_1000_n255.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1079 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n259
  state->_abc_1000_n267.value_0_0 = state->_abc_1000_n266.value_0_0 | state->_abc_1000_n264.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1087 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n267
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n268.value_0_0 = state->_abc_1000_n267.value_0_0 ^ state->_abc_1000_n262.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1088 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n268
  state->_abc_1000_n274.value_0_0 = !(state->_abc_1000_n273_1.value_0_0 ^ state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1094 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n274
  state->_abc_1000_n284_1.value_0_0 = !(state->_abc_1000_n283.value_0_0 ^ state->_abc_1000_n276_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1104 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n284_1
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n288_1.value_0_0 = !((state->_abc_1000_n283.value_0_0 | state->_abc_1000_n276_1.value_0_0) & state->_abc_1000_n287_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1108 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n288_1
  state->_abc_1000_n289.value_0_0 = state->_abc_1000_n288_1.value_0_0 ^ state->_abc_1000_n286.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1109 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n289
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n319.value_0_0 = !((state->_abc_1000_n314_1.value_0_0 | state->_abc_1000_n306_1.value_0_0) & state->_abc_1000_n318_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1139 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n319
  state->_abc_1000_n315_1.value_0_0 = !(state->_abc_1000_n314_1.value_0_0 ^ state->_abc_1000_n306_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1135 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n315_1
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n342_1.value_0_0 = !((state->_abc_1000_n341_1.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n340.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1162 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n342_1
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n320_1.value_0_0 = state->_abc_1000_n319.value_0_0 ^ state->_abc_1000_n317_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1140 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n320_1
  state->_abc_1000_n358.value_0_0 = state->_abc_1000_n357.value_0_0 ^ state->_abc_1000_n351_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1178 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n358
  state->_abc_1000_n364.value_0_0 = !((state->_abc_1000_n357.value_0_0 & state->_abc_1000_n361.value_0_0) | state->_abc_1000_n363.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1184 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n364
  state->_abc_1000_n365.value_0_0 = state->_abc_1000_n364.value_0_0 ^ state->_abc_1000_n360.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1185 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n365
  state->_abc_1000_n236.value_0_0 = !((state->_abc_1000_n235.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n234.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1056 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n236
  state->_abc_1000_n243.value_0_0 = !((state->_abc_1000_n236.value_0_0 & state->_abc_1000_n240.value_0_0) | state->_abc_1000_n242.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1063 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n243
  state->_abc_1000_n237.value_0_0 = state->_abc_1000_n236.value_0_0 ^ state->_abc_1000_n231.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1057 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n237
  state->_abc_1000_n244.value_0_0 = !(state->_abc_1000_n243.value_0_0 ^ state->_abc_1000_n239.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1064 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n244
  state->_abc_1000_n228.value_0_0 = !((state->_abc_1000_n223.value_0_0 | state->_abc_1000_n219.value_0_0) & state->_abc_1000_n227.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1048 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n228
  state->_abc_1000_n229.value_0_0 = state->_abc_1000_n228.value_0_0 ^ state->_abc_1000_n226.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1049 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n229
  yosys_simplec_set_bit_4_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n224.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1045 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [4]
  yosys_simplec_set_bit_12_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n284_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1105 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [12]
  yosys_simplec_set_bit_23_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n365.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1186 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [23]
  yosys_simplec_set_bit_26_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n394.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1215 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [26]
  yosys_simplec_set_bit_19_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n333_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1154 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [19]
  yosys_simplec_set_bit_22_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n358.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1179 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [22]
  yosys_simplec_set_bit_11_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n274.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1095 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [11]
  yosys_simplec_set_bit_18_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n327_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1148 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [18]
  yosys_simplec_set_bit_25_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n385.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1206 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [25]
  yosys_simplec_set_bit_10_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n268.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1089 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [10]
  yosys_simplec_set_bit_21_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n349.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1170 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [21]
  yosys_simplec_set_bit_17_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n320_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1141 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [17]
  yosys_simplec_set_bit_31_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n431.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1252 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [31]
  yosys_simplec_set_bit_24_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n377.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1198 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [24]
  yosys_simplec_set_bit_16_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n315_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1136 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [16]
  yosys_simplec_set_bit_30_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n426.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1247 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [30]
  yosys_simplec_set_bit_20_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n343.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1164 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [20]
  yosys_simplec_set_bit_9_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n259.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1080 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [9]
  yosys_simplec_set_bit_8_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n253.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1074 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [8]
  yosys_simplec_set_bit_29_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n418.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1239 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [29]
  yosys_simplec_set_bit_15_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n304.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1125 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [15]
  yosys_simplec_set_bit_7_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n244.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1065 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [7]
  yosys_simplec_set_bit_28_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n412.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1233 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [28]
  yosys_simplec_set_bit_14_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n297_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1118 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [14]
  yosys_simplec_set_bit_6_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n237.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1058 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [6]
  yosys_simplec_set_bit_27_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n401.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1222 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [27]
  yosys_simplec_set_bit_13_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n289.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1110 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [13]
  yosys_simplec_set_bit_5_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n229.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1050 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [5]
  state->_abc_1000_n216.value_0_0 = !((state->_abc_1000_n210.value_0_0 & state->_abc_1000_n208.value_0_0) | state->_abc_1000_n215.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1036 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n216
  state->_abc_1000_n217.value_0_0 = !(state->_abc_1000_n216.value_0_0 ^ state->_abc_1000_n213.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1037 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n217
  yosys_simplec_set_bit_3_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n217.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1038 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [3]
  state->_abc_1000_n209.value_0_0 = !(yosys_simplec_get_bit_1_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_1_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1029 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n209
  state->_abc_1000_n210.value_0_0 = !((state->_abc_1000_n205_1.value_0_0 | state->_abc_1000_n204_1.value_0_0) & state->_abc_1000_n209.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1030 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n210
  state->_abc_1000_n211.value_0_0 = !(state->_abc_1000_n210.value_0_0 ^ state->_abc_1000_n208.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1031 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n211
  yosys_simplec_set_bit_2_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n211.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1032 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [2]
  state->_abc_1000_n223.value_0_0 = !((state->_abc_1000_n222.value_0_0 & state->_abc_1000_n210.value_0_0) | state->_abc_1000_n221.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1043 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n223
  state->_abc_1000_n224.value_0_0 = !(state->_abc_1000_n223.value_0_0 ^ state->_abc_1000_n219.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1044 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n224
  state->_abc_1000_n252.value_0_0 = !((state->_abc_1000_n251.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n250.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1072 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n252
  state->_abc_1000_n283.value_0_0 = !((state->_abc_1000_n282_1.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n281_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1103 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n283
  state->_abc_1000_n266.value_0_0 = state->_abc_1000_n252.value_0_0 & (!state->_abc_1000_n265.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1086 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n266
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n256.value_0_0 = !state->_abc_1000_n252.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1076 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n256
  state->_abc_1000_n253.value_0_0 = state->_abc_1000_n252.value_0_0 ^ state->_abc_1000_n246.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1073 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n253
  state->_abc_1000_n258.value_0_0 = !((state->_abc_1000_n256.value_0_0 | state->_abc_1000_n246.value_0_0) & state->_abc_1000_n257.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1078 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n258
  state->_abc_1000_n259.value_0_0 = state->_abc_1000_n258.value_0_0 ^ state->_abc_1000_n255.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1079 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n259
  state->_abc_1000_n267.value_0_0 = state->_abc_1000_n266.value_0_0 | state->_abc_1000_n264.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1087 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n267
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n268.value_0_0 = state->_abc_1000_n267.value_0_0 ^ state->_abc_1000_n262.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1088 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n268
  state->_abc_1000_n274.value_0_0 = !(state->_abc_1000_n273_1.value_0_0 ^ state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1094 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n274
  state->_abc_1000_n284_1.value_0_0 = !(state->_abc_1000_n283.value_0_0 ^ state->_abc_1000_n276_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1104 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n284_1
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n288_1.value_0_0 = !((state->_abc_1000_n283.value_0_0 | state->_abc_1000_n276_1.value_0_0) & state->_abc_1000_n287_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1108 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n288_1
  state->_abc_1000_n289.value_0_0 = state->_abc_1000_n288_1.value_0_0 ^ state->_abc_1000_n286.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1109 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n289
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n319.value_0_0 = !((state->_abc_1000_n314_1.value_0_0 | state->_abc_1000_n306_1.value_0_0) & state->_abc_1000_n318_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1139 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n319
  state->_abc_1000_n315_1.value_0_0 = !(state->_abc_1000_n314_1.value_0_0 ^ state->_abc_1000_n306_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1135 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n315_1
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n342_1.value_0_0 = !((state->_abc_1000_n341_1.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n340.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1162 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n342_1
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n320_1.value_0_0 = state->_abc_1000_n319.value_0_0 ^ state->_abc_1000_n317_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1140 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n320_1
  state->_abc_1000_n358.value_0_0 = state->_abc_1000_n357.value_0_0 ^ state->_abc_1000_n351_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1178 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n358
  state->_abc_1000_n364.value_0_0 = !((state->_abc_1000_n357.value_0_0 & state->_abc_1000_n361.value_0_0) | state->_abc_1000_n363.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1184 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n364
  state->_abc_1000_n365.value_0_0 = state->_abc_1000_n364.value_0_0 ^ state->_abc_1000_n360.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1185 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n365
  state->_abc_1000_n236.value_0_0 = !((state->_abc_1000_n235.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n234.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1056 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n236
  state->_abc_1000_n243.value_0_0 = !((state->_abc_1000_n236.value_0_0 & state->_abc_1000_n240.value_0_0) | state->_abc_1000_n242.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1063 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n243
  state->_abc_1000_n237.value_0_0 = state->_abc_1000_n236.value_0_0 ^ state->_abc_1000_n231.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1057 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n237
  state->_abc_1000_n244.value_0_0 = !(state->_abc_1000_n243.value_0_0 ^ state->_abc_1000_n239.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1064 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n244
  state->_abc_1000_n228.value_0_0 = !((state->_abc_1000_n223.value_0_0 | state->_abc_1000_n219.value_0_0) & state->_abc_1000_n227.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1048 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n228
  state->_abc_1000_n229.value_0_0 = state->_abc_1000_n228.value_0_0 ^ state->_abc_1000_n226.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1049 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n229
  yosys_simplec_set_bit_4_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n224.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1045 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [4]
  yosys_simplec_set_bit_12_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n284_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1105 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [12]
  yosys_simplec_set_bit_23_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n365.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1186 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [23]
  yosys_simplec_set_bit_26_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n394.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1215 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [26]
  yosys_simplec_set_bit_19_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n333_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1154 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [19]
  yosys_simplec_set_bit_22_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n358.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1179 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [22]
  yosys_simplec_set_bit_11_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n274.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1095 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [11]
  yosys_simplec_set_bit_18_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n327_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1148 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [18]
  yosys_simplec_set_bit_25_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n385.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1206 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [25]
  yosys_simplec_set_bit_10_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n268.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1089 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [10]
  yosys_simplec_set_bit_21_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n349.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1170 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [21]
  yosys_simplec_set_bit_17_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n320_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1141 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [17]
  yosys_simplec_set_bit_31_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n431.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1252 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [31]
  yosys_simplec_set_bit_24_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n377.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1198 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [24]
  yosys_simplec_set_bit_16_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n315_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1136 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [16]
  yosys_simplec_set_bit_30_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n426.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1247 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [30]
  yosys_simplec_set_bit_20_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n343.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1164 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [20]
  yosys_simplec_set_bit_9_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n259.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1080 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [9]
  yosys_simplec_set_bit_8_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n253.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1074 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [8]
  yosys_simplec_set_bit_29_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n418.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1239 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [29]
  yosys_simplec_set_bit_15_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n304.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1125 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [15]
  yosys_simplec_set_bit_7_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n244.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1065 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [7]
  yosys_simplec_set_bit_28_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n412.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1233 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [28]
  yosys_simplec_set_bit_14_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n297_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1118 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [14]
  yosys_simplec_set_bit_6_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n237.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1058 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [6]
  yosys_simplec_set_bit_27_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n401.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1222 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [27]
  yosys_simplec_set_bit_13_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n289.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1110 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [13]
  yosys_simplec_set_bit_5_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n229.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1050 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [5]
  state->_abc_1000_n216.value_0_0 = !((state->_abc_1000_n210.value_0_0 & state->_abc_1000_n208.value_0_0) | state->_abc_1000_n215.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1036 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n216
  state->_abc_1000_n217.value_0_0 = !(state->_abc_1000_n216.value_0_0 ^ state->_abc_1000_n213.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1037 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n217
  yosys_simplec_set_bit_3_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n217.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1038 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [3]
  state->_abc_1000_n204_1.value_0_0 = !(yosys_simplec_get_bit_1_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_1_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1024 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n204_1
  state->_abc_1000_n210.value_0_0 = !((state->_abc_1000_n205_1.value_0_0 | state->_abc_1000_n204_1.value_0_0) & state->_abc_1000_n209.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1030 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n210
  state->_abc_1000_n211.value_0_0 = !(state->_abc_1000_n210.value_0_0 ^ state->_abc_1000_n208.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1031 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n211
  yosys_simplec_set_bit_2_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n211.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1032 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [2]
  state->_abc_1000_n223.value_0_0 = !((state->_abc_1000_n222.value_0_0 & state->_abc_1000_n210.value_0_0) | state->_abc_1000_n221.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1043 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n223
  state->_abc_1000_n224.value_0_0 = !(state->_abc_1000_n223.value_0_0 ^ state->_abc_1000_n219.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1044 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n224
  state->_abc_1000_n252.value_0_0 = !((state->_abc_1000_n251.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n250.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1072 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n252
  state->_abc_1000_n283.value_0_0 = !((state->_abc_1000_n282_1.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n281_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1103 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n283
  state->_abc_1000_n266.value_0_0 = state->_abc_1000_n252.value_0_0 & (!state->_abc_1000_n265.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1086 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n266
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n256.value_0_0 = !state->_abc_1000_n252.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1076 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n256
  state->_abc_1000_n253.value_0_0 = state->_abc_1000_n252.value_0_0 ^ state->_abc_1000_n246.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1073 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n253
  state->_abc_1000_n258.value_0_0 = !((state->_abc_1000_n256.value_0_0 | state->_abc_1000_n246.value_0_0) & state->_abc_1000_n257.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1078 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n258
  state->_abc_1000_n259.value_0_0 = state->_abc_1000_n258.value_0_0 ^ state->_abc_1000_n255.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1079 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n259
  state->_abc_1000_n267.value_0_0 = state->_abc_1000_n266.value_0_0 | state->_abc_1000_n264.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1087 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n267
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n268.value_0_0 = state->_abc_1000_n267.value_0_0 ^ state->_abc_1000_n262.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1088 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n268
  state->_abc_1000_n274.value_0_0 = !(state->_abc_1000_n273_1.value_0_0 ^ state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1094 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n274
  state->_abc_1000_n284_1.value_0_0 = !(state->_abc_1000_n283.value_0_0 ^ state->_abc_1000_n276_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1104 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n284_1
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n288_1.value_0_0 = !((state->_abc_1000_n283.value_0_0 | state->_abc_1000_n276_1.value_0_0) & state->_abc_1000_n287_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1108 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n288_1
  state->_abc_1000_n289.value_0_0 = state->_abc_1000_n288_1.value_0_0 ^ state->_abc_1000_n286.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1109 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n289
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n319.value_0_0 = !((state->_abc_1000_n314_1.value_0_0 | state->_abc_1000_n306_1.value_0_0) & state->_abc_1000_n318_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1139 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n319
  state->_abc_1000_n315_1.value_0_0 = !(state->_abc_1000_n314_1.value_0_0 ^ state->_abc_1000_n306_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1135 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n315_1
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n342_1.value_0_0 = !((state->_abc_1000_n341_1.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n340.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1162 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n342_1
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n320_1.value_0_0 = state->_abc_1000_n319.value_0_0 ^ state->_abc_1000_n317_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1140 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n320_1
  state->_abc_1000_n358.value_0_0 = state->_abc_1000_n357.value_0_0 ^ state->_abc_1000_n351_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1178 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n358
  state->_abc_1000_n364.value_0_0 = !((state->_abc_1000_n357.value_0_0 & state->_abc_1000_n361.value_0_0) | state->_abc_1000_n363.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1184 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n364
  state->_abc_1000_n365.value_0_0 = state->_abc_1000_n364.value_0_0 ^ state->_abc_1000_n360.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1185 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n365
  state->_abc_1000_n236.value_0_0 = !((state->_abc_1000_n235.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n234.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1056 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n236
  state->_abc_1000_n243.value_0_0 = !((state->_abc_1000_n236.value_0_0 & state->_abc_1000_n240.value_0_0) | state->_abc_1000_n242.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1063 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n243
  state->_abc_1000_n237.value_0_0 = state->_abc_1000_n236.value_0_0 ^ state->_abc_1000_n231.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1057 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n237
  state->_abc_1000_n244.value_0_0 = !(state->_abc_1000_n243.value_0_0 ^ state->_abc_1000_n239.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1064 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n244
  state->_abc_1000_n228.value_0_0 = !((state->_abc_1000_n223.value_0_0 | state->_abc_1000_n219.value_0_0) & state->_abc_1000_n227.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1048 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n228
  state->_abc_1000_n229.value_0_0 = state->_abc_1000_n228.value_0_0 ^ state->_abc_1000_n226.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1049 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n229
  yosys_simplec_set_bit_4_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n224.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1045 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [4]
  yosys_simplec_set_bit_12_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n284_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1105 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [12]
  yosys_simplec_set_bit_23_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n365.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1186 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [23]
  yosys_simplec_set_bit_26_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n394.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1215 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [26]
  yosys_simplec_set_bit_19_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n333_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1154 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [19]
  yosys_simplec_set_bit_22_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n358.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1179 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [22]
  yosys_simplec_set_bit_11_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n274.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1095 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [11]
  yosys_simplec_set_bit_18_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n327_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1148 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [18]
  yosys_simplec_set_bit_25_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n385.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1206 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [25]
  yosys_simplec_set_bit_10_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n268.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1089 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [10]
  yosys_simplec_set_bit_21_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n349.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1170 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [21]
  yosys_simplec_set_bit_17_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n320_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1141 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [17]
  yosys_simplec_set_bit_31_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n431.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1252 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [31]
  yosys_simplec_set_bit_24_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n377.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1198 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [24]
  yosys_simplec_set_bit_16_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n315_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1136 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [16]
  yosys_simplec_set_bit_30_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n426.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1247 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [30]
  yosys_simplec_set_bit_20_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n343.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1164 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [20]
  yosys_simplec_set_bit_9_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n259.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1080 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [9]
  yosys_simplec_set_bit_8_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n253.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1074 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [8]
  yosys_simplec_set_bit_29_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n418.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1239 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [29]
  yosys_simplec_set_bit_15_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n304.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1125 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [15]
  yosys_simplec_set_bit_7_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n244.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1065 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [7]
  yosys_simplec_set_bit_28_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n412.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1233 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [28]
  yosys_simplec_set_bit_14_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n297_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1118 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [14]
  yosys_simplec_set_bit_6_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n237.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1058 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [6]
  yosys_simplec_set_bit_27_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n401.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1222 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [27]
  yosys_simplec_set_bit_13_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n289.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1110 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [13]
  yosys_simplec_set_bit_5_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n229.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1050 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [5]
  state->_abc_1000_n216.value_0_0 = !((state->_abc_1000_n210.value_0_0 & state->_abc_1000_n208.value_0_0) | state->_abc_1000_n215.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1036 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n216
  state->_abc_1000_n217.value_0_0 = !(state->_abc_1000_n216.value_0_0 ^ state->_abc_1000_n213.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1037 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n217
  yosys_simplec_set_bit_3_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n217.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1038 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [3]
  state->_abc_1000_n206.value_0_0 = !(state->_abc_1000_n205_1.value_0_0 ^ state->_abc_1000_n204_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1026 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n206
  yosys_simplec_set_bit_1_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n206.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1027 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [1]
  state->_abc_1000_n492.value_0_0 = state->_abc_1000_n469.value_0_0 & (!state->_abc_1000_n491.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1312 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n492
  state->_abc_1000_n474.value_0_0 = state->_abc_1000_n473.value_0_0 | (!state->_abc_1000_n469.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1294 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n474
  state->_abc_1000_n476_1.value_0_0 = yosys_simplec_get_bit_18_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n474.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1296 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n476_1
  yosys_simplec_set_bit_18_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n474.value_0_0 ^ yosys_simplec_get_bit_18_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1295 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [18]
  yosys_simplec_set_bit_19_of_32(&state->spec_pc_wdata, state->_abc_1000_n476_1.value_0_0 ^ yosys_simplec_get_bit_19_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1297 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [19]
  state->_abc_1000_n471.value_0_0 = state->_abc_1000_n469.value_0_0 & yosys_simplec_get_bit_16_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1291 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n471
  state->_abc_1000_n473.value_0_0 = !(yosys_simplec_get_bit_17_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_16_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1293 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n473
  state->_abc_1000_n474.value_0_0 = state->_abc_1000_n473.value_0_0 | (!state->_abc_1000_n469.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1294 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n474
  state->_abc_1000_n476_1.value_0_0 = yosys_simplec_get_bit_18_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n474.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1296 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n476_1
  yosys_simplec_set_bit_18_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n474.value_0_0 ^ yosys_simplec_get_bit_18_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1295 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [18]
  yosys_simplec_set_bit_19_of_32(&state->spec_pc_wdata, state->_abc_1000_n476_1.value_0_0 ^ yosys_simplec_get_bit_19_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1297 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [19]
  yosys_simplec_set_bit_16_of_32(&state->spec_pc_wdata, state->_abc_1000_n469.value_0_0 ^ yosys_simplec_get_bit_16_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1290 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [16]
  state->_abc_1000_n480.value_0_0 = state->_abc_1000_n469.value_0_0 & (!state->_abc_1000_n479.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1300 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n480
  yosys_simplec_set_bit_17_of_32(&state->spec_pc_wdata, state->_abc_1000_n471.value_0_0 ^ yosys_simplec_get_bit_17_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1292 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [17]
  state->_abc_1000_n479.value_0_0 = state->_abc_1000_n478.value_0_0 | state->_abc_1000_n473.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1299 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n479
  state->_abc_1000_n480.value_0_0 = state->_abc_1000_n469.value_0_0 & (!state->_abc_1000_n479.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1300 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n480
  state->_abc_1000_n491.value_0_0 = state->_abc_1000_n490.value_0_0 | state->_abc_1000_n479.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1311 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n491
  state->_abc_1000_n492.value_0_0 = state->_abc_1000_n469.value_0_0 & (!state->_abc_1000_n491.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1312 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n492
  state->_abc_1000_n510.value_0_0 = yosys_simplec_get_bit_30_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n508.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1330 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n510
  yosys_simplec_set_bit_30_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n508.value_0_0 ^ yosys_simplec_get_bit_30_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1329 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [30]
  state->_abc_1000_n485.value_0_0 = state->_abc_1000_n484.value_0_0 | (!state->_abc_1000_n480.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1305 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n485
  yosys_simplec_set_bit_20_of_32(&state->spec_pc_wdata, state->_abc_1000_n480.value_0_0 ^ yosys_simplec_get_bit_20_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1301 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [20]
  state->_abc_1000_n484.value_0_0 = !(yosys_simplec_get_bit_21_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_20_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1304 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n484
  state->_abc_1000_n485.value_0_0 = state->_abc_1000_n484.value_0_0 | (!state->_abc_1000_n480.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1305 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n485
  state->_abc_1000_n490.value_0_0 = state->_abc_1000_n489.value_0_0 | state->_abc_1000_n484.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1310 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n490
  state->_abc_1000_n491.value_0_0 = state->_abc_1000_n490.value_0_0 | state->_abc_1000_n479.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1311 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n491
  state->_abc_1000_n492.value_0_0 = state->_abc_1000_n469.value_0_0 & (!state->_abc_1000_n491.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1312 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n492
  state->_abc_1000_n482.value_0_0 = state->_abc_1000_n480.value_0_0 & yosys_simplec_get_bit_20_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1302 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n482
  yosys_simplec_set_bit_21_of_32(&state->spec_pc_wdata, state->_abc_1000_n482.value_0_0 ^ yosys_simplec_get_bit_21_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1303 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [21]
  yosys_simplec_set_bit_29_of_32(&state->spec_pc_wdata, state->_abc_1000_n505.value_0_0 ^ yosys_simplec_get_bit_29_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1326 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [29]
  state->_abc_1000_n507.value_0_0 = yosys_simplec_get_bit_29_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_28_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1327 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n507
  state->_abc_1000_n505.value_0_0 = state->_abc_1000_n503.value_0_0 & yosys_simplec_get_bit_28_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1325 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n505
  yosys_simplec_set_bit_29_of_32(&state->spec_pc_wdata, state->_abc_1000_n505.value_0_0 ^ yosys_simplec_get_bit_29_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1326 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [29]
  yosys_simplec_set_bit_28_of_32(&state->spec_pc_wdata, state->_abc_1000_n503.value_0_0 ^ yosys_simplec_get_bit_28_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1324 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [28]
  state->_abc_1000_n508.value_0_0 = !(state->_abc_1000_n507.value_0_0 & state->_abc_1000_n503.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1328 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n508
  state->_abc_1000_n510.value_0_0 = yosys_simplec_get_bit_30_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n508.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1330 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n510
  yosys_simplec_set_bit_30_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n508.value_0_0 ^ yosys_simplec_get_bit_30_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1329 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [30]
  state->_abc_1000_n489.value_0_0 = !(yosys_simplec_get_bit_23_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_22_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1309 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n489
  state->_abc_1000_n490.value_0_0 = state->_abc_1000_n489.value_0_0 | state->_abc_1000_n484.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1310 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n490
  state->_abc_1000_n491.value_0_0 = state->_abc_1000_n490.value_0_0 | state->_abc_1000_n479.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1311 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n491
  state->_abc_1000_n492.value_0_0 = state->_abc_1000_n469.value_0_0 & (!state->_abc_1000_n491.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1312 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n492
  state->_abc_1000_n487.value_0_0 = yosys_simplec_get_bit_22_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n485.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1307 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n487
  yosys_simplec_set_bit_22_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n485.value_0_0 ^ yosys_simplec_get_bit_22_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1306 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [22]
  yosys_simplec_set_bit_23_of_32(&state->spec_pc_wdata, state->_abc_1000_n487.value_0_0 ^ yosys_simplec_get_bit_23_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1308 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [23]
  yosys_simplec_set_bit_31_of_32(&state->spec_pc_wdata, state->_abc_1000_n510.value_0_0 ^ yosys_simplec_get_bit_31_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1331 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [31]
  state->_abc_1000_n497.value_0_0 = state->_abc_1000_n496.value_0_0 | (!state->_abc_1000_n492.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1317 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n497
  state->_abc_1000_n494.value_0_0 = state->_abc_1000_n492.value_0_0 & yosys_simplec_get_bit_24_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1314 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n494
  state->_abc_1000_n496.value_0_0 = !(yosys_simplec_get_bit_25_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_24_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1316 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n496
  state->_abc_1000_n497.value_0_0 = state->_abc_1000_n496.value_0_0 | (!state->_abc_1000_n492.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1317 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n497
  state->_abc_1000_n502.value_0_0 = state->_abc_1000_n501.value_0_0 | state->_abc_1000_n496.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1322 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n502
  yosys_simplec_set_bit_24_of_32(&state->spec_pc_wdata, state->_abc_1000_n492.value_0_0 ^ yosys_simplec_get_bit_24_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1313 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [24]
  state->_abc_1000_n503.value_0_0 = state->_abc_1000_n492.value_0_0 & (!state->_abc_1000_n502.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1323 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n503
  state->_abc_1000_n505.value_0_0 = state->_abc_1000_n503.value_0_0 & yosys_simplec_get_bit_28_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1325 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n505
  yosys_simplec_set_bit_29_of_32(&state->spec_pc_wdata, state->_abc_1000_n505.value_0_0 ^ yosys_simplec_get_bit_29_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1326 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [29]
  yosys_simplec_set_bit_28_of_32(&state->spec_pc_wdata, state->_abc_1000_n503.value_0_0 ^ yosys_simplec_get_bit_28_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1324 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [28]
  state->_abc_1000_n508.value_0_0 = !(state->_abc_1000_n507.value_0_0 & state->_abc_1000_n503.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1328 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n508
  state->_abc_1000_n510.value_0_0 = yosys_simplec_get_bit_30_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n508.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1330 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n510
  yosys_simplec_set_bit_30_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n508.value_0_0 ^ yosys_simplec_get_bit_30_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1329 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [30]
  yosys_simplec_set_bit_31_of_32(&state->spec_pc_wdata, state->_abc_1000_n510.value_0_0 ^ yosys_simplec_get_bit_31_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1331 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [31]
  yosys_simplec_set_bit_25_of_32(&state->spec_pc_wdata, state->_abc_1000_n494.value_0_0 ^ yosys_simplec_get_bit_25_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1315 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [25]
  yosys_simplec_set_bit_26_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n497.value_0_0 ^ yosys_simplec_get_bit_26_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1318 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [26]
  state->_abc_1000_n501.value_0_0 = !(yosys_simplec_get_bit_27_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_26_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1321 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n501
  state->_abc_1000_n502.value_0_0 = state->_abc_1000_n501.value_0_0 | state->_abc_1000_n496.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1322 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n502
  state->_abc_1000_n503.value_0_0 = state->_abc_1000_n492.value_0_0 & (!state->_abc_1000_n502.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1323 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n503
  state->_abc_1000_n505.value_0_0 = state->_abc_1000_n503.value_0_0 & yosys_simplec_get_bit_28_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1325 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n505
  yosys_simplec_set_bit_29_of_32(&state->spec_pc_wdata, state->_abc_1000_n505.value_0_0 ^ yosys_simplec_get_bit_29_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1326 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [29]
  yosys_simplec_set_bit_28_of_32(&state->spec_pc_wdata, state->_abc_1000_n503.value_0_0 ^ yosys_simplec_get_bit_28_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1324 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [28]
  state->_abc_1000_n508.value_0_0 = !(state->_abc_1000_n507.value_0_0 & state->_abc_1000_n503.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1328 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n508
  state->_abc_1000_n510.value_0_0 = yosys_simplec_get_bit_30_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n508.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1330 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n510
  yosys_simplec_set_bit_30_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n508.value_0_0 ^ yosys_simplec_get_bit_30_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1329 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [30]
  yosys_simplec_set_bit_31_of_32(&state->spec_pc_wdata, state->_abc_1000_n510.value_0_0 ^ yosys_simplec_get_bit_31_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1331 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [31]
  state->_abc_1000_n499.value_0_0 = yosys_simplec_get_bit_26_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n497.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1319 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n499
  yosys_simplec_set_bit_27_of_32(&state->spec_pc_wdata, state->_abc_1000_n499.value_0_0 ^ yosys_simplec_get_bit_27_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1320 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [27]
  yosys_simplec_set_bit_0_of_32(&state->spec_pc_wdata, yosys_simplec_get_bit_0_of_32(&state->rvfi_pc_rdata));
  yosys_simplec_set_bit_1_of_32(&state->spec_pc_wdata, yosys_simplec_get_bit_1_of_32(&state->rvfi_pc_rdata));
  yosys_simplec_set_bit_0_of_5(&state->spec_rd_addr, yosys_simplec_get_bit_7_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_1_of_5(&state->spec_rd_addr, yosys_simplec_get_bit_8_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_2_of_5(&state->spec_rd_addr, yosys_simplec_get_bit_9_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_3_of_5(&state->spec_rd_addr, yosys_simplec_get_bit_10_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_4_of_5(&state->spec_rd_addr, yosys_simplec_get_bit_11_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_0_of_5(&state->spec_rs1_addr, yosys_simplec_get_bit_15_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_1_of_5(&state->spec_rs1_addr, yosys_simplec_get_bit_16_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_2_of_5(&state->spec_rs1_addr, yosys_simplec_get_bit_17_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_3_of_5(&state->spec_rs1_addr, yosys_simplec_get_bit_18_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_4_of_5(&state->spec_rs1_addr, yosys_simplec_get_bit_19_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_0_of_5(&state->spec_rs2_addr, yosys_simplec_get_bit_20_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_1_of_5(&state->spec_rs2_addr, yosys_simplec_get_bit_21_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_2_of_5(&state->spec_rs2_addr, yosys_simplec_get_bit_22_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_3_of_5(&state->spec_rs2_addr, yosys_simplec_get_bit_23_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_4_of_5(&state->spec_rs2_addr, yosys_simplec_get_bit_24_of_32(&state->rvfi_insn));
}

static void rvfi_insn_add_eval(struct rvfi_insn_add_state_t *state)
{
  // Updated signal in rvfi_insn_add: \rvfi_valid
  // Updated signal in rvfi_insn_add: \rvfi_insn
  // Updated signal in rvfi_insn_add: \rvfi_pc_rdata
  // Updated signal in rvfi_insn_add: \rvfi_rs1_rdata
  // Updated signal in rvfi_insn_add: \rvfi_rs2_rdata
  // Updated signal in rvfi_insn_add: \rvfi_mem_rdata
  state->_abc_1000_n181_1.value_0_0 = yosys_simplec_get_bit_26_of_32(&state->rvfi_insn) | yosys_simplec_get_bit_25_of_32(&state->rvfi_insn); // $abc$1000$auto$blifparse.cc:346:parse_blif$1001 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n181_1
  state->_abc_1000_n182.value_0_0 = yosys_simplec_get_bit_28_of_32(&state->rvfi_insn) | yosys_simplec_get_bit_27_of_32(&state->rvfi_insn); // $abc$1000$auto$blifparse.cc:346:parse_blif$1002 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n182
  state->_abc_1000_n183_1.value_0_0 = state->_abc_1000_n182.value_0_0 | state->_abc_1000_n181_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1003 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n183_1
  state->_abc_1000_n184_1.value_0_0 = yosys_simplec_get_bit_30_of_32(&state->rvfi_insn) | yosys_simplec_get_bit_29_of_32(&state->rvfi_insn); // $abc$1000$auto$blifparse.cc:346:parse_blif$1004 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n184_1
  state->_abc_1000_n185.value_0_0 = state->_abc_1000_n184_1.value_0_0 | yosys_simplec_get_bit_31_of_32(&state->rvfi_insn); // $abc$1000$auto$blifparse.cc:346:parse_blif$1005 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n185
  state->_abc_1000_n186_1.value_0_0 = state->_abc_1000_n185.value_0_0 | state->_abc_1000_n183_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1006 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n186_1
  state->_abc_1000_n187_1.value_0_0 = state->rvfi_valid.value_0_0 & (!state->_abc_1000_n186_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1007 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n187_1
  state->_abc_1000_n188.value_0_0 = yosys_simplec_get_bit_13_of_32(&state->rvfi_insn) | yosys_simplec_get_bit_12_of_32(&state->rvfi_insn); // $abc$1000$auto$blifparse.cc:346:parse_blif$1008 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n188
  state->_abc_1000_n189_1.value_0_0 = state->_abc_1000_n188.value_0_0 | yosys_simplec_get_bit_14_of_32(&state->rvfi_insn); // $abc$1000$auto$blifparse.cc:346:parse_blif$1009 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n189_1
  state->_abc_1000_n190_1.value_0_0 = state->_abc_1000_n187_1.value_0_0 & (!state->_abc_1000_n189_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1010 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n190_1
  state->_abc_1000_n191.value_0_0 = !(yosys_simplec_get_bit_1_of_32(&state->rvfi_insn) & yosys_simplec_get_bit_0_of_32(&state->rvfi_insn)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1011 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n191
  state->_abc_1000_n192_1.value_0_0 = yosys_simplec_get_bit_3_of_32(&state->rvfi_insn) | yosys_simplec_get_bit_2_of_32(&state->rvfi_insn); // $abc$1000$auto$blifparse.cc:346:parse_blif$1012 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n192_1
  state->_abc_1000_n193_1.value_0_0 = state->_abc_1000_n192_1.value_0_0 | state->_abc_1000_n191.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1013 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n193_1
  state->_abc_1000_n194.value_0_0 = !(yosys_simplec_get_bit_5_of_32(&state->rvfi_insn) & yosys_simplec_get_bit_4_of_32(&state->rvfi_insn)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1014 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n194
  state->_abc_1000_n195_1.value_0_0 = state->_abc_1000_n194.value_0_0 | yosys_simplec_get_bit_6_of_32(&state->rvfi_insn); // $abc$1000$auto$blifparse.cc:346:parse_blif$1015 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n195_1
  state->_abc_1000_n196_1.value_0_0 = state->_abc_1000_n195_1.value_0_0 | state->_abc_1000_n193_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1016 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n196_1
  state->spec_valid.value_0_0 = state->_abc_1000_n190_1.value_0_0 & (!state->_abc_1000_n196_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1017 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_valid
  state->_abc_1000_n199_1.value_0_0 = !(yosys_simplec_get_bit_8_of_32(&state->rvfi_insn) | yosys_simplec_get_bit_7_of_32(&state->rvfi_insn)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1019 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n199_1
  state->_abc_1000_n200.value_0_0 = yosys_simplec_get_bit_10_of_32(&state->rvfi_insn) | yosys_simplec_get_bit_9_of_32(&state->rvfi_insn); // $abc$1000$auto$blifparse.cc:346:parse_blif$1020 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n200
  state->_abc_1000_n201_1.value_0_0 = state->_abc_1000_n200.value_0_0 | (!state->_abc_1000_n199_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1021 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n201_1
  state->_abc_1000_n202_1.value_0_0 = state->_abc_1000_n201_1.value_0_0 | yosys_simplec_get_bit_11_of_32(&state->rvfi_insn); // $abc$1000$auto$blifparse.cc:346:parse_blif$1022 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n202_1
  state->_abc_1000_n205_1.value_0_0 = !(yosys_simplec_get_bit_0_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_0_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1025 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n205_1
  state->_abc_1000_n198_1.value_0_0 = !(yosys_simplec_get_bit_0_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_0_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1018 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n198_1
  state->_abc_1000_n232.value_0_0 = !(yosys_simplec_get_bit_5_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_5_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1052 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n232
  state->_abc_1000_n226.value_0_0 = !(yosys_simplec_get_bit_5_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_5_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1046 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n226
  state->_abc_1000_n292.value_0_0 = !(yosys_simplec_get_bit_13_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_13_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1112 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n292
  state->_abc_1000_n286.value_0_0 = !(yosys_simplec_get_bit_13_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_13_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1106 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n286
  state->_abc_1000_n241.value_0_0 = !(yosys_simplec_get_bit_6_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_6_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1061 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n241
  state->_abc_1000_n240.value_0_0 = yosys_simplec_get_bit_6_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_6_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1060 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n240
  state->_abc_1000_n231.value_0_0 = !(yosys_simplec_get_bit_6_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_6_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1051 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n231
  state->_abc_1000_n301.value_0_0 = !(yosys_simplec_get_bit_14_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_14_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1121 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n301
  state->_abc_1000_n300_1.value_0_0 = yosys_simplec_get_bit_14_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_14_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1120 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n300_1
  state->_abc_1000_n291_1.value_0_0 = !(yosys_simplec_get_bit_14_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_14_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1111 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n291_1
  state->_abc_1000_n415.value_0_0 = yosys_simplec_get_bit_28_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_28_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1235 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n415
  state->_abc_1000_n403.value_0_0 = !(yosys_simplec_get_bit_28_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_28_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1223 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n403
  state->_abc_1000_n247.value_0_0 = !(yosys_simplec_get_bit_7_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_7_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1067 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n247
  state->_abc_1000_n239.value_0_0 = !(yosys_simplec_get_bit_7_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_7_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1059 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n239
  state->_abc_1000_n249.value_0_0 = state->_abc_1000_n240.value_0_0 & (!state->_abc_1000_n239.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1069 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n249
  state->_abc_1000_n243.value_0_0 = !((state->_abc_1000_n236.value_0_0 & state->_abc_1000_n240.value_0_0) | state->_abc_1000_n242.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1063 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n243
  state->_abc_1000_n237.value_0_0 = state->_abc_1000_n236.value_0_0 ^ state->_abc_1000_n231.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1057 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n237
  state->_abc_1000_n248.value_0_0 = !((state->_abc_1000_n241.value_0_0 | state->_abc_1000_n239.value_0_0) & state->_abc_1000_n247.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1068 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n248
  state->_abc_1000_n242.value_0_0 = !state->_abc_1000_n241.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1062 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n242
  state->_abc_1000_n243.value_0_0 = !((state->_abc_1000_n236.value_0_0 & state->_abc_1000_n240.value_0_0) | state->_abc_1000_n242.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1063 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n243
  state->_abc_1000_n244.value_0_0 = !(state->_abc_1000_n243.value_0_0 ^ state->_abc_1000_n239.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1064 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n244
  state->_abc_1000_n307.value_0_0 = !(yosys_simplec_get_bit_15_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_15_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1127 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n307
  state->_abc_1000_n299_1.value_0_0 = !(yosys_simplec_get_bit_15_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_15_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1119 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n299_1
  state->_abc_1000_n309_1.value_0_0 = state->_abc_1000_n300_1.value_0_0 & (!state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1129 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n309_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n308_1.value_0_0 = !((state->_abc_1000_n301.value_0_0 | state->_abc_1000_n299_1.value_0_0) & state->_abc_1000_n307.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1128 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n308_1
  state->_abc_1000_n302_1.value_0_0 = !state->_abc_1000_n301.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1122 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n302_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n422.value_0_0 = !((yosys_simplec_get_bit_29_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_29_of_32(&state->rvfi_rs1_rdata)) | state->_abc_1000_n421.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1242 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n422
  state->_abc_1000_n414.value_0_0 = !(yosys_simplec_get_bit_29_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_29_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1234 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n414
  state->_abc_1000_n257.value_0_0 = !(yosys_simplec_get_bit_8_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_8_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1077 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n257
  state->_abc_1000_n246.value_0_0 = !(yosys_simplec_get_bit_8_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_8_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1066 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n246
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n423.value_0_0 = !(state->_abc_1000_n414.value_0_0 | state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1243 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n423
  state->_abc_1000_n424.value_0_0 = !state->_abc_1000_n423.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1244 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n424
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n421.value_0_0 = state->_abc_1000_n415.value_0_0 & (!state->_abc_1000_n414.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1241 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n421
  state->_abc_1000_n422.value_0_0 = !((yosys_simplec_get_bit_29_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_29_of_32(&state->rvfi_rs1_rdata)) | state->_abc_1000_n421.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1242 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n422
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n416.value_0_0 = !state->_abc_1000_n415.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1236 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n416
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n455.value_0_0 = !(yosys_simplec_get_bit_11_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_10_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1275 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n455
  state->_abc_1000_n453_1.value_0_0 = state->_abc_1000_n451.value_0_0 & yosys_simplec_get_bit_10_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1273 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n453_1
  yosys_simplec_set_bit_10_of_32(&state->spec_pc_wdata, state->_abc_1000_n451.value_0_0 ^ yosys_simplec_get_bit_10_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1272 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [10]
  yosys_simplec_set_bit_11_of_32(&state->spec_pc_wdata, state->_abc_1000_n453_1.value_0_0 ^ yosys_simplec_get_bit_11_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1274 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [11]
  state->_abc_1000_n428.value_0_0 = yosys_simplec_get_bit_31_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_31_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1248 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n428
  state->_abc_1000_n429.value_0_0 = yosys_simplec_get_bit_30_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_30_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1249 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n429
  state->_abc_1000_n420.value_0_0 = yosys_simplec_get_bit_30_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_30_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1240 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n420
  state->_abc_1000_n318_1.value_0_0 = !(yosys_simplec_get_bit_16_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_16_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1138 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n318_1
  state->_abc_1000_n306_1.value_0_0 = !(yosys_simplec_get_bit_16_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_16_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1126 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n306_1
  state->_abc_1000_n263.value_0_0 = !(yosys_simplec_get_bit_9_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_9_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1083 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n263
  state->_abc_1000_n255.value_0_0 = !(yosys_simplec_get_bit_9_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_9_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1075 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n255
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n352.value_0_0 = yosys_simplec_get_bit_21_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_21_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1172 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n352
  state->_abc_1000_n345_1.value_0_0 = yosys_simplec_get_bit_21_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_21_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1165 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n345_1
  state->_abc_1000_n435.value_0_0 = yosys_simplec_get_bit_3_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_2_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1255 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n435
  yosys_simplec_set_bit_3_of_32(&state->spec_pc_wdata, yosys_simplec_get_bit_3_of_32(&state->rvfi_pc_rdata) ^ yosys_simplec_get_bit_2_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1254 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [3]
  yosys_simplec_set_bit_2_of_32(&state->spec_pc_wdata, !yosys_simplec_get_bit_2_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1253 ($_NOT_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [2]
  state->_abc_1000_n324_1.value_0_0 = !((yosys_simplec_get_bit_17_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_17_of_32(&state->rvfi_rs1_rdata)) | state->_abc_1000_n323_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1144 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n324_1
  state->_abc_1000_n317_1.value_0_0 = !(yosys_simplec_get_bit_17_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_17_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1137 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n317_1
  state->_abc_1000_n354_1.value_0_0 = !(state->_abc_1000_n353_1.value_0_0 | state->_abc_1000_n352.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1174 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n354_1
  state->_abc_1000_n271_1.value_0_0 = !(yosys_simplec_get_bit_10_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_10_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1091 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n271_1
  state->_abc_1000_n261.value_0_0 = yosys_simplec_get_bit_10_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_10_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1081 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n261
  state->_abc_1000_n466.value_0_0 = !(yosys_simplec_get_bit_15_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_14_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1286 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n466
  state->_abc_1000_n464.value_0_0 = yosys_simplec_get_bit_14_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n462.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1284 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n464
  yosys_simplec_set_bit_14_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n462.value_0_0 ^ yosys_simplec_get_bit_14_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1283 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [14]
  yosys_simplec_set_bit_15_of_32(&state->spec_pc_wdata, state->_abc_1000_n464.value_0_0 ^ yosys_simplec_get_bit_15_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1285 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [15]
  state->_abc_1000_n467.value_0_0 = state->_abc_1000_n466.value_0_0 | state->_abc_1000_n461.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1287 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n467
  state->_abc_1000_n459.value_0_0 = yosys_simplec_get_bit_12_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n457.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1279 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n459
  state->_abc_1000_n461.value_0_0 = !(yosys_simplec_get_bit_13_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_12_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1281 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n461
  state->_abc_1000_n467.value_0_0 = state->_abc_1000_n466.value_0_0 | state->_abc_1000_n461.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1287 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n467
  state->_abc_1000_n462.value_0_0 = state->_abc_1000_n461.value_0_0 | state->_abc_1000_n457.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1282 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n462
  state->_abc_1000_n464.value_0_0 = yosys_simplec_get_bit_14_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n462.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1284 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n464
  yosys_simplec_set_bit_14_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n462.value_0_0 ^ yosys_simplec_get_bit_14_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1283 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [14]
  yosys_simplec_set_bit_15_of_32(&state->spec_pc_wdata, state->_abc_1000_n464.value_0_0 ^ yosys_simplec_get_bit_15_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1285 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [15]
  yosys_simplec_set_bit_12_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n457.value_0_0 ^ yosys_simplec_get_bit_12_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1278 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [12]
  yosys_simplec_set_bit_13_of_32(&state->spec_pc_wdata, state->_abc_1000_n459.value_0_0 ^ yosys_simplec_get_bit_13_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1280 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [13]
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n372.value_0_0 = !((state->_abc_1000_n371.value_0_0 | state->_abc_1000_n354_1.value_0_0) & state->_abc_1000_n370.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1192 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n372
  state->_abc_1000_n355.value_0_0 = state->_abc_1000_n335_1.value_0_0 | (!state->_abc_1000_n345_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1175 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n355
  state->_abc_1000_n373.value_0_0 = !(state->_abc_1000_n371.value_0_0 | state->_abc_1000_n355.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1193 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n373
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n353_1.value_0_0 = state->_abc_1000_n345_1.value_0_0 & (!state->_abc_1000_n346.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1173 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n353_1
  state->_abc_1000_n354_1.value_0_0 = !(state->_abc_1000_n353_1.value_0_0 | state->_abc_1000_n352.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1174 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n354_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n372.value_0_0 = !((state->_abc_1000_n371.value_0_0 | state->_abc_1000_n354_1.value_0_0) & state->_abc_1000_n370.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1192 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n372
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n387.value_0_0 = !(yosys_simplec_get_bit_26_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_26_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1207 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n387
  state->_abc_1000_n399.value_0_0 = yosys_simplec_get_bit_26_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_26_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1219 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n399
  state->_abc_1000_n388.value_0_0 = yosys_simplec_get_bit_25_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_25_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1208 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n388
  state->_abc_1000_n379.value_0_0 = yosys_simplec_get_bit_25_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_25_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1199 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n379
  state->_abc_1000_n330_1.value_0_0 = !(yosys_simplec_get_bit_18_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_18_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1150 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n330_1
  state->_abc_1000_n322.value_0_0 = !(yosys_simplec_get_bit_18_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_18_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1142 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n322
  state->_abc_1000_n276_1.value_0_0 = !(yosys_simplec_get_bit_12_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_12_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1096 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n276_1
  state->_abc_1000_n287_1.value_0_0 = !(yosys_simplec_get_bit_12_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_12_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1107 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n287_1
  state->_abc_1000_n390.value_0_0 = state->_abc_1000_n389.value_0_0 | state->_abc_1000_n388.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1210 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n390
  state->_abc_1000_n277.value_0_0 = !(yosys_simplec_get_bit_11_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_11_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1097 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n277
  state->_abc_1000_n270_1.value_0_0 = !(yosys_simplec_get_bit_11_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_11_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1090 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n270_1
  state->_abc_1000_n362.value_0_0 = !(yosys_simplec_get_bit_22_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_22_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1182 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n362
  state->_abc_1000_n351_1.value_0_0 = !(yosys_simplec_get_bit_22_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_22_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1171 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n351_1
  state->_abc_1000_n444.value_0_0 = !(yosys_simplec_get_bit_7_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_6_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1264 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n444
  state->_abc_1000_n442.value_0_0 = state->_abc_1000_n440.value_0_0 & yosys_simplec_get_bit_6_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1262 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n442
  yosys_simplec_set_bit_6_of_32(&state->spec_pc_wdata, state->_abc_1000_n440.value_0_0 ^ yosys_simplec_get_bit_6_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1261 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [6]
  yosys_simplec_set_bit_7_of_32(&state->spec_pc_wdata, state->_abc_1000_n442.value_0_0 ^ yosys_simplec_get_bit_7_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1263 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [7]
  state->_abc_1000_n279_1.value_0_0 = state->_abc_1000_n261.value_0_0 & (!state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1099 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n279_1
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n262.value_0_0 = !state->_abc_1000_n261.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1082 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n262
  state->_abc_1000_n268.value_0_0 = state->_abc_1000_n267.value_0_0 ^ state->_abc_1000_n262.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1088 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n268
  state->_abc_1000_n278_1.value_0_0 = !((state->_abc_1000_n271_1.value_0_0 | state->_abc_1000_n270_1.value_0_0) & state->_abc_1000_n277.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1098 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n278_1
  state->_abc_1000_n272_1.value_0_0 = !state->_abc_1000_n271_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1092 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n272_1
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n274.value_0_0 = !(state->_abc_1000_n273_1.value_0_0 ^ state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1094 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n274
  state->_abc_1000_n363.value_0_0 = !state->_abc_1000_n362.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1183 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n363
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n408.value_0_0 = state->_abc_1000_n390.value_0_0 & (!state->_abc_1000_n407.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1228 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n408
  state->_abc_1000_n409.value_0_0 = state->_abc_1000_n408.value_0_0 | state->_abc_1000_n406.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1229 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n409
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n391.value_0_0 = state->_abc_1000_n367.value_0_0 | (!state->_abc_1000_n379.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1211 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n391
  state->_abc_1000_n410.value_0_0 = !(state->_abc_1000_n407.value_0_0 | state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1230 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n410
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n389.value_0_0 = state->_abc_1000_n379.value_0_0 & (!state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1209 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n389
  state->_abc_1000_n390.value_0_0 = state->_abc_1000_n389.value_0_0 | state->_abc_1000_n388.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1210 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n390
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n408.value_0_0 = state->_abc_1000_n390.value_0_0 & (!state->_abc_1000_n407.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1228 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n408
  state->_abc_1000_n409.value_0_0 = state->_abc_1000_n408.value_0_0 | state->_abc_1000_n406.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1229 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n409
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n380.value_0_0 = !state->_abc_1000_n379.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1200 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n380
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n446.value_0_0 = state->_abc_1000_n435.value_0_0 & (!state->_abc_1000_n445.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1266 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n446
  state->_abc_1000_n440.value_0_0 = state->_abc_1000_n435.value_0_0 & (!state->_abc_1000_n439.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1260 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n440
  state->_abc_1000_n442.value_0_0 = state->_abc_1000_n440.value_0_0 & yosys_simplec_get_bit_6_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1262 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n442
  yosys_simplec_set_bit_6_of_32(&state->spec_pc_wdata, state->_abc_1000_n440.value_0_0 ^ yosys_simplec_get_bit_6_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1261 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [6]
  yosys_simplec_set_bit_7_of_32(&state->spec_pc_wdata, state->_abc_1000_n442.value_0_0 ^ yosys_simplec_get_bit_7_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1263 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [7]
  state->_abc_1000_n437.value_0_0 = state->_abc_1000_n435.value_0_0 & yosys_simplec_get_bit_4_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1257 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n437
  state->_abc_1000_n439.value_0_0 = !(yosys_simplec_get_bit_5_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_4_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1259 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n439
  state->_abc_1000_n440.value_0_0 = state->_abc_1000_n435.value_0_0 & (!state->_abc_1000_n439.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1260 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n440
  state->_abc_1000_n442.value_0_0 = state->_abc_1000_n440.value_0_0 & yosys_simplec_get_bit_6_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1262 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n442
  yosys_simplec_set_bit_6_of_32(&state->spec_pc_wdata, state->_abc_1000_n440.value_0_0 ^ yosys_simplec_get_bit_6_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1261 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [6]
  yosys_simplec_set_bit_7_of_32(&state->spec_pc_wdata, state->_abc_1000_n442.value_0_0 ^ yosys_simplec_get_bit_7_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1263 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [7]
  yosys_simplec_set_bit_4_of_32(&state->spec_pc_wdata, state->_abc_1000_n435.value_0_0 ^ yosys_simplec_get_bit_4_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1256 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [4]
  yosys_simplec_set_bit_5_of_32(&state->spec_pc_wdata, state->_abc_1000_n437.value_0_0 ^ yosys_simplec_get_bit_5_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1258 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [5]
  state->_abc_1000_n445.value_0_0 = state->_abc_1000_n444.value_0_0 | state->_abc_1000_n439.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1265 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n445
  state->_abc_1000_n446.value_0_0 = state->_abc_1000_n435.value_0_0 & (!state->_abc_1000_n445.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1266 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n446
  state->_abc_1000_n335_1.value_0_0 = !(yosys_simplec_get_bit_20_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_20_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1155 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n335_1
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n355.value_0_0 = state->_abc_1000_n335_1.value_0_0 | (!state->_abc_1000_n345_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1175 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n355
  state->_abc_1000_n373.value_0_0 = !(state->_abc_1000_n371.value_0_0 | state->_abc_1000_n355.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1193 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n373
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n346.value_0_0 = !(yosys_simplec_get_bit_20_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_20_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1166 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n346
  state->_abc_1000_n353_1.value_0_0 = state->_abc_1000_n345_1.value_0_0 & (!state->_abc_1000_n346.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1173 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n353_1
  state->_abc_1000_n354_1.value_0_0 = !(state->_abc_1000_n353_1.value_0_0 | state->_abc_1000_n352.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1174 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n354_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n372.value_0_0 = !((state->_abc_1000_n371.value_0_0 | state->_abc_1000_n354_1.value_0_0) & state->_abc_1000_n370.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1192 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n372
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n407.value_0_0 = state->_abc_1000_n396.value_0_0 | state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1227 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n407
  state->_abc_1000_n408.value_0_0 = state->_abc_1000_n390.value_0_0 & (!state->_abc_1000_n407.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1228 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n408
  state->_abc_1000_n409.value_0_0 = state->_abc_1000_n408.value_0_0 | state->_abc_1000_n406.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1229 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n409
  state->_abc_1000_n410.value_0_0 = !(state->_abc_1000_n407.value_0_0 | state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1230 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n410
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n398.value_0_0 = !state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1218 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n398
  state->_abc_1000_n405.value_0_0 = state->_abc_1000_n399.value_0_0 & (!state->_abc_1000_n396.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1225 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n405
  state->_abc_1000_n406.value_0_0 = state->_abc_1000_n405.value_0_0 | state->_abc_1000_n404.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1226 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n406
  state->_abc_1000_n409.value_0_0 = state->_abc_1000_n408.value_0_0 | state->_abc_1000_n406.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1229 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n409
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n336_1.value_0_0 = state->_abc_1000_n329_1.value_0_0 & (!state->_abc_1000_n330_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1156 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n336_1
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n338_1.value_0_0 = state->_abc_1000_n322.value_0_0 | (!state->_abc_1000_n329_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1158 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n338_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n448.value_0_0 = state->_abc_1000_n446.value_0_0 & yosys_simplec_get_bit_8_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1268 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n448
  yosys_simplec_set_bit_8_of_32(&state->spec_pc_wdata, state->_abc_1000_n446.value_0_0 ^ yosys_simplec_get_bit_8_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1267 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [8]
  state->_abc_1000_n450.value_0_0 = !(yosys_simplec_get_bit_9_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_8_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1270 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n450
  yosys_simplec_set_bit_9_of_32(&state->spec_pc_wdata, state->_abc_1000_n448.value_0_0 ^ yosys_simplec_get_bit_9_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1269 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [9]
  state->_abc_1000_n469.value_0_0 = state->_abc_1000_n446.value_0_0 & (!state->_abc_1000_n468.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1289 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n469
  state->_abc_1000_n457.value_0_0 = state->_abc_1000_n456_1.value_0_0 | (!state->_abc_1000_n446.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1277 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n457
  state->_abc_1000_n462.value_0_0 = state->_abc_1000_n461.value_0_0 | state->_abc_1000_n457.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1282 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n462
  state->_abc_1000_n464.value_0_0 = yosys_simplec_get_bit_14_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n462.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1284 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n464
  yosys_simplec_set_bit_14_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n462.value_0_0 ^ yosys_simplec_get_bit_14_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1283 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [14]
  yosys_simplec_set_bit_15_of_32(&state->spec_pc_wdata, state->_abc_1000_n464.value_0_0 ^ yosys_simplec_get_bit_15_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1285 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [15]
  state->_abc_1000_n459.value_0_0 = yosys_simplec_get_bit_12_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n457.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1279 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n459
  yosys_simplec_set_bit_12_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n457.value_0_0 ^ yosys_simplec_get_bit_12_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1278 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [12]
  yosys_simplec_set_bit_13_of_32(&state->spec_pc_wdata, state->_abc_1000_n459.value_0_0 ^ yosys_simplec_get_bit_13_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1280 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [13]
  state->_abc_1000_n451.value_0_0 = state->_abc_1000_n446.value_0_0 & (!state->_abc_1000_n450.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1271 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n451
  state->_abc_1000_n453_1.value_0_0 = state->_abc_1000_n451.value_0_0 & yosys_simplec_get_bit_10_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1273 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n453_1
  yosys_simplec_set_bit_10_of_32(&state->spec_pc_wdata, state->_abc_1000_n451.value_0_0 ^ yosys_simplec_get_bit_10_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1272 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [10]
  yosys_simplec_set_bit_11_of_32(&state->spec_pc_wdata, state->_abc_1000_n453_1.value_0_0 ^ yosys_simplec_get_bit_11_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1274 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [11]
  state->_abc_1000_n456_1.value_0_0 = state->_abc_1000_n455.value_0_0 | state->_abc_1000_n450.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1276 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n456_1
  state->_abc_1000_n457.value_0_0 = state->_abc_1000_n456_1.value_0_0 | (!state->_abc_1000_n446.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1277 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n457
  state->_abc_1000_n462.value_0_0 = state->_abc_1000_n461.value_0_0 | state->_abc_1000_n457.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1282 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n462
  state->_abc_1000_n464.value_0_0 = yosys_simplec_get_bit_14_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n462.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1284 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n464
  yosys_simplec_set_bit_14_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n462.value_0_0 ^ yosys_simplec_get_bit_14_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1283 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [14]
  yosys_simplec_set_bit_15_of_32(&state->spec_pc_wdata, state->_abc_1000_n464.value_0_0 ^ yosys_simplec_get_bit_15_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1285 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [15]
  state->_abc_1000_n459.value_0_0 = yosys_simplec_get_bit_12_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n457.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1279 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n459
  yosys_simplec_set_bit_12_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n457.value_0_0 ^ yosys_simplec_get_bit_12_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1278 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [12]
  yosys_simplec_set_bit_13_of_32(&state->spec_pc_wdata, state->_abc_1000_n459.value_0_0 ^ yosys_simplec_get_bit_13_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1280 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [13]
  state->_abc_1000_n468.value_0_0 = state->_abc_1000_n467.value_0_0 | state->_abc_1000_n456_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1288 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n468
  state->_abc_1000_n469.value_0_0 = state->_abc_1000_n446.value_0_0 & (!state->_abc_1000_n468.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1289 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n469
  state->_abc_1000_n337.value_0_0 = !((yosys_simplec_get_bit_19_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_19_of_32(&state->rvfi_rs1_rdata)) | state->_abc_1000_n336_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1157 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n337
  state->_abc_1000_n329_1.value_0_0 = yosys_simplec_get_bit_19_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_19_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1149 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n329_1
  state->_abc_1000_n336_1.value_0_0 = state->_abc_1000_n329_1.value_0_0 & (!state->_abc_1000_n330_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1156 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n336_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n338_1.value_0_0 = state->_abc_1000_n322.value_0_0 | (!state->_abc_1000_n329_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1158 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n338_1
  state->_abc_1000_n337.value_0_0 = !((yosys_simplec_get_bit_19_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_19_of_32(&state->rvfi_rs1_rdata)) | state->_abc_1000_n336_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1157 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n337
  state->_abc_1000_n396.value_0_0 = !(yosys_simplec_get_bit_27_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_27_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1216 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n396
  state->_abc_1000_n407.value_0_0 = state->_abc_1000_n396.value_0_0 | state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1227 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n407
  state->_abc_1000_n408.value_0_0 = state->_abc_1000_n390.value_0_0 & (!state->_abc_1000_n407.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1228 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n408
  state->_abc_1000_n409.value_0_0 = state->_abc_1000_n408.value_0_0 | state->_abc_1000_n406.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1229 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n409
  state->_abc_1000_n410.value_0_0 = !(state->_abc_1000_n407.value_0_0 | state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1230 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n410
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n405.value_0_0 = state->_abc_1000_n399.value_0_0 & (!state->_abc_1000_n396.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1225 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n405
  state->_abc_1000_n406.value_0_0 = state->_abc_1000_n405.value_0_0 | state->_abc_1000_n404.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1226 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n406
  state->_abc_1000_n409.value_0_0 = state->_abc_1000_n408.value_0_0 | state->_abc_1000_n406.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1229 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n409
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n397.value_0_0 = !state->_abc_1000_n396.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1217 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n397
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n404.value_0_0 = yosys_simplec_get_bit_27_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_27_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1224 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n404
  state->_abc_1000_n406.value_0_0 = state->_abc_1000_n405.value_0_0 | state->_abc_1000_n404.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1226 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n406
  state->_abc_1000_n409.value_0_0 = state->_abc_1000_n408.value_0_0 | state->_abc_1000_n406.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1229 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n409
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n253.value_0_0 = state->_abc_1000_n252.value_0_0 ^ state->_abc_1000_n246.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1073 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n253
  state->_abc_1000_n265.value_0_0 = state->_abc_1000_n255.value_0_0 | state->_abc_1000_n246.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1085 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n265
  state->_abc_1000_n266.value_0_0 = state->_abc_1000_n252.value_0_0 & (!state->_abc_1000_n265.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1086 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n266
  state->_abc_1000_n264.value_0_0 = !((state->_abc_1000_n257.value_0_0 | state->_abc_1000_n255.value_0_0) & state->_abc_1000_n263.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1084 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n264
  state->_abc_1000_n258.value_0_0 = !((state->_abc_1000_n256.value_0_0 | state->_abc_1000_n246.value_0_0) & state->_abc_1000_n257.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1078 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n258
  state->_abc_1000_n259.value_0_0 = state->_abc_1000_n258.value_0_0 ^ state->_abc_1000_n255.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1079 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n259
  state->_abc_1000_n282_1.value_0_0 = state->_abc_1000_n279_1.value_0_0 & (!state->_abc_1000_n265.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1102 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n282_1
  state->_abc_1000_n283.value_0_0 = !((state->_abc_1000_n282_1.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n281_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1103 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n283
  state->_abc_1000_n313.value_0_0 = state->_abc_1000_n282_1.value_0_0 & (!state->_abc_1000_n311_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1133 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n313
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n280.value_0_0 = !((state->_abc_1000_n279_1.value_0_0 & state->_abc_1000_n264.value_0_0) | state->_abc_1000_n278_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1100 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n280
  state->_abc_1000_n312_1.value_0_0 = !((state->_abc_1000_n311_1.value_0_0 | state->_abc_1000_n280.value_0_0) & state->_abc_1000_n310.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1132 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n312_1
  state->_abc_1000_n281_1.value_0_0 = !state->_abc_1000_n280.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1101 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n281_1
  state->_abc_1000_n283.value_0_0 = !((state->_abc_1000_n282_1.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n281_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1103 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n283
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n267.value_0_0 = state->_abc_1000_n266.value_0_0 | state->_abc_1000_n264.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1087 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n267
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n268.value_0_0 = state->_abc_1000_n267.value_0_0 ^ state->_abc_1000_n262.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1088 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n268
  state->_abc_1000_n274.value_0_0 = !(state->_abc_1000_n273_1.value_0_0 ^ state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1094 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n274
  state->_abc_1000_n478.value_0_0 = !(yosys_simplec_get_bit_19_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_18_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1298 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n478
  state->_abc_1000_n476_1.value_0_0 = yosys_simplec_get_bit_18_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n474.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1296 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n476_1
  yosys_simplec_set_bit_18_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n474.value_0_0 ^ yosys_simplec_get_bit_18_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1295 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [18]
  yosys_simplec_set_bit_19_of_32(&state->spec_pc_wdata, state->_abc_1000_n476_1.value_0_0 ^ yosys_simplec_get_bit_19_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1297 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [19]
  state->_abc_1000_n367.value_0_0 = !(yosys_simplec_get_bit_24_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_24_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1187 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n367
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n391.value_0_0 = state->_abc_1000_n367.value_0_0 | (!state->_abc_1000_n379.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1211 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n391
  state->_abc_1000_n410.value_0_0 = !(state->_abc_1000_n407.value_0_0 | state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1230 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n410
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n383.value_0_0 = !(yosys_simplec_get_bit_24_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_24_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1203 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n383
  state->_abc_1000_n389.value_0_0 = state->_abc_1000_n379.value_0_0 & (!state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1209 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n389
  state->_abc_1000_n390.value_0_0 = state->_abc_1000_n389.value_0_0 | state->_abc_1000_n388.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1210 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n390
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n408.value_0_0 = state->_abc_1000_n390.value_0_0 & (!state->_abc_1000_n407.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1228 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n408
  state->_abc_1000_n409.value_0_0 = state->_abc_1000_n408.value_0_0 | state->_abc_1000_n406.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1229 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n409
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n284_1.value_0_0 = !(state->_abc_1000_n283.value_0_0 ^ state->_abc_1000_n276_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1104 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n284_1
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n288_1.value_0_0 = !((state->_abc_1000_n283.value_0_0 | state->_abc_1000_n276_1.value_0_0) & state->_abc_1000_n287_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1108 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n288_1
  state->_abc_1000_n295.value_0_0 = state->_abc_1000_n286.value_0_0 | state->_abc_1000_n276_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1115 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n295
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n293_1.value_0_0 = !((state->_abc_1000_n287_1.value_0_0 | state->_abc_1000_n286.value_0_0) & state->_abc_1000_n292.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1113 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n293_1
  state->_abc_1000_n289.value_0_0 = state->_abc_1000_n288_1.value_0_0 ^ state->_abc_1000_n286.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1109 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n289
  state->_abc_1000_n311_1.value_0_0 = state->_abc_1000_n295.value_0_0 | (!state->_abc_1000_n309_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1131 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n311_1
  state->_abc_1000_n313.value_0_0 = state->_abc_1000_n282_1.value_0_0 & (!state->_abc_1000_n311_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1133 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n313
  state->_abc_1000_n312_1.value_0_0 = !((state->_abc_1000_n311_1.value_0_0 | state->_abc_1000_n280.value_0_0) & state->_abc_1000_n310.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1132 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n312_1
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n310.value_0_0 = !((state->_abc_1000_n309_1.value_0_0 & state->_abc_1000_n293_1.value_0_0) | state->_abc_1000_n308_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1130 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n310
  state->_abc_1000_n312_1.value_0_0 = !((state->_abc_1000_n311_1.value_0_0 | state->_abc_1000_n280.value_0_0) & state->_abc_1000_n310.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1132 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n312_1
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n294_1.value_0_0 = !state->_abc_1000_n293_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1114 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n294_1
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n368.value_0_0 = !((state->_abc_1000_n338_1.value_0_0 | state->_abc_1000_n324_1.value_0_0) & state->_abc_1000_n337.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1188 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n368
  state->_abc_1000_n340.value_0_0 = state->_abc_1000_n339_1.value_0_0 & state->_abc_1000_n337.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1160 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n340
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n341_1.value_0_0 = state->_abc_1000_n338_1.value_0_0 | state->_abc_1000_n325.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1161 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n341_1
  state->_abc_1000_n339_1.value_0_0 = state->_abc_1000_n338_1.value_0_0 | state->_abc_1000_n324_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1159 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n339_1
  state->_abc_1000_n340.value_0_0 = state->_abc_1000_n339_1.value_0_0 & state->_abc_1000_n337.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1160 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n340
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n319.value_0_0 = !((state->_abc_1000_n314_1.value_0_0 | state->_abc_1000_n306_1.value_0_0) & state->_abc_1000_n318_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1139 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n319
  state->_abc_1000_n325.value_0_0 = state->_abc_1000_n317_1.value_0_0 | state->_abc_1000_n306_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1145 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n325
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n341_1.value_0_0 = state->_abc_1000_n338_1.value_0_0 | state->_abc_1000_n325.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1161 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n341_1
  state->_abc_1000_n315_1.value_0_0 = !(state->_abc_1000_n314_1.value_0_0 ^ state->_abc_1000_n306_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1135 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n315_1
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n342_1.value_0_0 = !((state->_abc_1000_n341_1.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n340.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1162 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n342_1
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n375.value_0_0 = state->_abc_1000_n341_1.value_0_0 | (!state->_abc_1000_n373.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1195 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n375
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n374.value_0_0 = !((state->_abc_1000_n373.value_0_0 & state->_abc_1000_n368.value_0_0) | state->_abc_1000_n372.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1194 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n374
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n323_1.value_0_0 = !(state->_abc_1000_n318_1.value_0_0 | state->_abc_1000_n317_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1143 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n323_1
  state->_abc_1000_n324_1.value_0_0 = !((yosys_simplec_get_bit_17_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_17_of_32(&state->rvfi_rs1_rdata)) | state->_abc_1000_n323_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1144 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n324_1
  state->_abc_1000_n368.value_0_0 = !((state->_abc_1000_n338_1.value_0_0 | state->_abc_1000_n324_1.value_0_0) & state->_abc_1000_n337.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1188 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n368
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n339_1.value_0_0 = state->_abc_1000_n338_1.value_0_0 | state->_abc_1000_n324_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1159 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n339_1
  state->_abc_1000_n340.value_0_0 = state->_abc_1000_n339_1.value_0_0 & state->_abc_1000_n337.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1160 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n340
  state->_abc_1000_n342_1.value_0_0 = !((state->_abc_1000_n341_1.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n340.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1162 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n342_1
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n374.value_0_0 = !((state->_abc_1000_n373.value_0_0 & state->_abc_1000_n368.value_0_0) | state->_abc_1000_n372.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1194 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n374
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n320_1.value_0_0 = state->_abc_1000_n319.value_0_0 ^ state->_abc_1000_n317_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1140 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n320_1
  state->_abc_1000_n370.value_0_0 = !((state->_abc_1000_n363.value_0_0 & state->_abc_1000_n360.value_0_0) | state->_abc_1000_n369.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1190 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n370
  state->_abc_1000_n372.value_0_0 = !((state->_abc_1000_n371.value_0_0 | state->_abc_1000_n354_1.value_0_0) & state->_abc_1000_n370.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1192 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n372
  state->_abc_1000_n374.value_0_0 = !((state->_abc_1000_n373.value_0_0 & state->_abc_1000_n368.value_0_0) | state->_abc_1000_n372.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1194 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n374
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n371.value_0_0 = state->_abc_1000_n351_1.value_0_0 | (!state->_abc_1000_n360.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1191 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n371
  state->_abc_1000_n372.value_0_0 = !((state->_abc_1000_n371.value_0_0 | state->_abc_1000_n354_1.value_0_0) & state->_abc_1000_n370.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1192 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n372
  state->_abc_1000_n373.value_0_0 = !(state->_abc_1000_n371.value_0_0 | state->_abc_1000_n355.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1193 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n373
  state->_abc_1000_n375.value_0_0 = state->_abc_1000_n341_1.value_0_0 | (!state->_abc_1000_n373.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1195 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n375
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n374.value_0_0 = !((state->_abc_1000_n373.value_0_0 & state->_abc_1000_n368.value_0_0) | state->_abc_1000_n372.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1194 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n374
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n361.value_0_0 = !state->_abc_1000_n351_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1181 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n361
  state->_abc_1000_n358.value_0_0 = state->_abc_1000_n357.value_0_0 ^ state->_abc_1000_n351_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1178 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n358
  state->_abc_1000_n364.value_0_0 = !((state->_abc_1000_n357.value_0_0 & state->_abc_1000_n361.value_0_0) | state->_abc_1000_n363.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1184 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n364
  state->_abc_1000_n365.value_0_0 = state->_abc_1000_n364.value_0_0 ^ state->_abc_1000_n360.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1185 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n365
  state->_abc_1000_n369.value_0_0 = yosys_simplec_get_bit_23_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_23_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1189 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n369
  state->_abc_1000_n370.value_0_0 = !((state->_abc_1000_n363.value_0_0 & state->_abc_1000_n360.value_0_0) | state->_abc_1000_n369.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1190 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n370
  state->_abc_1000_n372.value_0_0 = !((state->_abc_1000_n371.value_0_0 | state->_abc_1000_n354_1.value_0_0) & state->_abc_1000_n370.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1192 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n372
  state->_abc_1000_n374.value_0_0 = !((state->_abc_1000_n373.value_0_0 & state->_abc_1000_n368.value_0_0) | state->_abc_1000_n372.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1194 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n374
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n360.value_0_0 = yosys_simplec_get_bit_23_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_23_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1180 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n360
  state->_abc_1000_n370.value_0_0 = !((state->_abc_1000_n363.value_0_0 & state->_abc_1000_n360.value_0_0) | state->_abc_1000_n369.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1190 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n370
  state->_abc_1000_n372.value_0_0 = !((state->_abc_1000_n371.value_0_0 | state->_abc_1000_n354_1.value_0_0) & state->_abc_1000_n370.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1192 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n372
  state->_abc_1000_n374.value_0_0 = !((state->_abc_1000_n373.value_0_0 & state->_abc_1000_n368.value_0_0) | state->_abc_1000_n372.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1194 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n374
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n365.value_0_0 = state->_abc_1000_n364.value_0_0 ^ state->_abc_1000_n360.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1185 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n365
  state->_abc_1000_n371.value_0_0 = state->_abc_1000_n351_1.value_0_0 | (!state->_abc_1000_n360.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1191 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n371
  state->_abc_1000_n372.value_0_0 = !((state->_abc_1000_n371.value_0_0 | state->_abc_1000_n354_1.value_0_0) & state->_abc_1000_n370.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1192 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n372
  state->_abc_1000_n373.value_0_0 = !(state->_abc_1000_n371.value_0_0 | state->_abc_1000_n355.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1193 ($_NOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n373
  state->_abc_1000_n375.value_0_0 = state->_abc_1000_n341_1.value_0_0 | (!state->_abc_1000_n373.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1195 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n375
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n374.value_0_0 = !((state->_abc_1000_n373.value_0_0 & state->_abc_1000_n368.value_0_0) | state->_abc_1000_n372.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1194 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n374
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n235.value_0_0 = state->_abc_1000_n226.value_0_0 | state->_abc_1000_n219.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1055 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n235
  state->_abc_1000_n236.value_0_0 = !((state->_abc_1000_n235.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n234.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1056 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n236
  state->_abc_1000_n243.value_0_0 = !((state->_abc_1000_n236.value_0_0 & state->_abc_1000_n240.value_0_0) | state->_abc_1000_n242.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1063 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n243
  state->_abc_1000_n237.value_0_0 = state->_abc_1000_n236.value_0_0 ^ state->_abc_1000_n231.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1057 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n237
  state->_abc_1000_n244.value_0_0 = !(state->_abc_1000_n243.value_0_0 ^ state->_abc_1000_n239.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1064 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n244
  state->_abc_1000_n233.value_0_0 = !((state->_abc_1000_n227.value_0_0 | state->_abc_1000_n226.value_0_0) & state->_abc_1000_n232.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1053 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n233
  state->_abc_1000_n229.value_0_0 = state->_abc_1000_n228.value_0_0 ^ state->_abc_1000_n226.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1049 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n229
  state->_abc_1000_n251.value_0_0 = state->_abc_1000_n235.value_0_0 | (!state->_abc_1000_n249.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1071 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n251
  state->_abc_1000_n252.value_0_0 = !((state->_abc_1000_n251.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n250.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1072 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n252
  state->_abc_1000_n283.value_0_0 = !((state->_abc_1000_n282_1.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n281_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1103 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n283
  state->_abc_1000_n266.value_0_0 = state->_abc_1000_n252.value_0_0 & (!state->_abc_1000_n265.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1086 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n266
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n256.value_0_0 = !state->_abc_1000_n252.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1076 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n256
  state->_abc_1000_n253.value_0_0 = state->_abc_1000_n252.value_0_0 ^ state->_abc_1000_n246.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1073 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n253
  state->_abc_1000_n258.value_0_0 = !((state->_abc_1000_n256.value_0_0 | state->_abc_1000_n246.value_0_0) & state->_abc_1000_n257.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1078 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n258
  state->_abc_1000_n259.value_0_0 = state->_abc_1000_n258.value_0_0 ^ state->_abc_1000_n255.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1079 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n259
  state->_abc_1000_n267.value_0_0 = state->_abc_1000_n266.value_0_0 | state->_abc_1000_n264.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1087 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n267
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n268.value_0_0 = state->_abc_1000_n267.value_0_0 ^ state->_abc_1000_n262.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1088 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n268
  state->_abc_1000_n274.value_0_0 = !(state->_abc_1000_n273_1.value_0_0 ^ state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1094 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n274
  state->_abc_1000_n284_1.value_0_0 = !(state->_abc_1000_n283.value_0_0 ^ state->_abc_1000_n276_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1104 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n284_1
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n288_1.value_0_0 = !((state->_abc_1000_n283.value_0_0 | state->_abc_1000_n276_1.value_0_0) & state->_abc_1000_n287_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1108 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n288_1
  state->_abc_1000_n289.value_0_0 = state->_abc_1000_n288_1.value_0_0 ^ state->_abc_1000_n286.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1109 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n289
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n319.value_0_0 = !((state->_abc_1000_n314_1.value_0_0 | state->_abc_1000_n306_1.value_0_0) & state->_abc_1000_n318_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1139 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n319
  state->_abc_1000_n315_1.value_0_0 = !(state->_abc_1000_n314_1.value_0_0 ^ state->_abc_1000_n306_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1135 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n315_1
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n342_1.value_0_0 = !((state->_abc_1000_n341_1.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n340.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1162 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n342_1
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n320_1.value_0_0 = state->_abc_1000_n319.value_0_0 ^ state->_abc_1000_n317_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1140 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n320_1
  state->_abc_1000_n358.value_0_0 = state->_abc_1000_n357.value_0_0 ^ state->_abc_1000_n351_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1178 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n358
  state->_abc_1000_n364.value_0_0 = !((state->_abc_1000_n357.value_0_0 & state->_abc_1000_n361.value_0_0) | state->_abc_1000_n363.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1184 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n364
  state->_abc_1000_n365.value_0_0 = state->_abc_1000_n364.value_0_0 ^ state->_abc_1000_n360.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1185 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n365
  state->_abc_1000_n250.value_0_0 = !((state->_abc_1000_n249.value_0_0 & state->_abc_1000_n233.value_0_0) | state->_abc_1000_n248.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1070 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n250
  state->_abc_1000_n252.value_0_0 = !((state->_abc_1000_n251.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n250.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1072 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n252
  state->_abc_1000_n283.value_0_0 = !((state->_abc_1000_n282_1.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n281_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1103 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n283
  state->_abc_1000_n266.value_0_0 = state->_abc_1000_n252.value_0_0 & (!state->_abc_1000_n265.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1086 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n266
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n256.value_0_0 = !state->_abc_1000_n252.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1076 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n256
  state->_abc_1000_n253.value_0_0 = state->_abc_1000_n252.value_0_0 ^ state->_abc_1000_n246.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1073 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n253
  state->_abc_1000_n258.value_0_0 = !((state->_abc_1000_n256.value_0_0 | state->_abc_1000_n246.value_0_0) & state->_abc_1000_n257.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1078 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n258
  state->_abc_1000_n259.value_0_0 = state->_abc_1000_n258.value_0_0 ^ state->_abc_1000_n255.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1079 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n259
  state->_abc_1000_n267.value_0_0 = state->_abc_1000_n266.value_0_0 | state->_abc_1000_n264.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1087 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n267
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n268.value_0_0 = state->_abc_1000_n267.value_0_0 ^ state->_abc_1000_n262.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1088 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n268
  state->_abc_1000_n274.value_0_0 = !(state->_abc_1000_n273_1.value_0_0 ^ state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1094 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n274
  state->_abc_1000_n284_1.value_0_0 = !(state->_abc_1000_n283.value_0_0 ^ state->_abc_1000_n276_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1104 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n284_1
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n288_1.value_0_0 = !((state->_abc_1000_n283.value_0_0 | state->_abc_1000_n276_1.value_0_0) & state->_abc_1000_n287_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1108 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n288_1
  state->_abc_1000_n289.value_0_0 = state->_abc_1000_n288_1.value_0_0 ^ state->_abc_1000_n286.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1109 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n289
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n319.value_0_0 = !((state->_abc_1000_n314_1.value_0_0 | state->_abc_1000_n306_1.value_0_0) & state->_abc_1000_n318_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1139 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n319
  state->_abc_1000_n315_1.value_0_0 = !(state->_abc_1000_n314_1.value_0_0 ^ state->_abc_1000_n306_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1135 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n315_1
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n342_1.value_0_0 = !((state->_abc_1000_n341_1.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n340.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1162 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n342_1
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n320_1.value_0_0 = state->_abc_1000_n319.value_0_0 ^ state->_abc_1000_n317_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1140 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n320_1
  state->_abc_1000_n358.value_0_0 = state->_abc_1000_n357.value_0_0 ^ state->_abc_1000_n351_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1178 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n358
  state->_abc_1000_n364.value_0_0 = !((state->_abc_1000_n357.value_0_0 & state->_abc_1000_n361.value_0_0) | state->_abc_1000_n363.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1184 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n364
  state->_abc_1000_n365.value_0_0 = state->_abc_1000_n364.value_0_0 ^ state->_abc_1000_n360.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1185 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n365
  state->_abc_1000_n234.value_0_0 = !state->_abc_1000_n233.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1054 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n234
  state->_abc_1000_n236.value_0_0 = !((state->_abc_1000_n235.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n234.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1056 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n236
  state->_abc_1000_n243.value_0_0 = !((state->_abc_1000_n236.value_0_0 & state->_abc_1000_n240.value_0_0) | state->_abc_1000_n242.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1063 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n243
  state->_abc_1000_n237.value_0_0 = state->_abc_1000_n236.value_0_0 ^ state->_abc_1000_n231.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1057 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n237
  state->_abc_1000_n244.value_0_0 = !(state->_abc_1000_n243.value_0_0 ^ state->_abc_1000_n239.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1064 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n244
  state->_abc_1000_n219.value_0_0 = !(yosys_simplec_get_bit_4_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_4_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1039 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n219
  state->_abc_1000_n224.value_0_0 = !(state->_abc_1000_n223.value_0_0 ^ state->_abc_1000_n219.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1044 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n224
  state->_abc_1000_n228.value_0_0 = !((state->_abc_1000_n223.value_0_0 | state->_abc_1000_n219.value_0_0) & state->_abc_1000_n227.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1048 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n228
  state->_abc_1000_n235.value_0_0 = state->_abc_1000_n226.value_0_0 | state->_abc_1000_n219.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1055 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n235
  state->_abc_1000_n236.value_0_0 = !((state->_abc_1000_n235.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n234.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1056 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n236
  state->_abc_1000_n243.value_0_0 = !((state->_abc_1000_n236.value_0_0 & state->_abc_1000_n240.value_0_0) | state->_abc_1000_n242.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1063 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n243
  state->_abc_1000_n237.value_0_0 = state->_abc_1000_n236.value_0_0 ^ state->_abc_1000_n231.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1057 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n237
  state->_abc_1000_n244.value_0_0 = !(state->_abc_1000_n243.value_0_0 ^ state->_abc_1000_n239.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1064 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n244
  state->_abc_1000_n229.value_0_0 = state->_abc_1000_n228.value_0_0 ^ state->_abc_1000_n226.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1049 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n229
  state->_abc_1000_n251.value_0_0 = state->_abc_1000_n235.value_0_0 | (!state->_abc_1000_n249.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1071 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n251
  state->_abc_1000_n252.value_0_0 = !((state->_abc_1000_n251.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n250.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1072 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n252
  state->_abc_1000_n283.value_0_0 = !((state->_abc_1000_n282_1.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n281_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1103 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n283
  state->_abc_1000_n266.value_0_0 = state->_abc_1000_n252.value_0_0 & (!state->_abc_1000_n265.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1086 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n266
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n256.value_0_0 = !state->_abc_1000_n252.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1076 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n256
  state->_abc_1000_n253.value_0_0 = state->_abc_1000_n252.value_0_0 ^ state->_abc_1000_n246.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1073 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n253
  state->_abc_1000_n258.value_0_0 = !((state->_abc_1000_n256.value_0_0 | state->_abc_1000_n246.value_0_0) & state->_abc_1000_n257.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1078 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n258
  state->_abc_1000_n259.value_0_0 = state->_abc_1000_n258.value_0_0 ^ state->_abc_1000_n255.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1079 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n259
  state->_abc_1000_n267.value_0_0 = state->_abc_1000_n266.value_0_0 | state->_abc_1000_n264.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1087 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n267
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n268.value_0_0 = state->_abc_1000_n267.value_0_0 ^ state->_abc_1000_n262.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1088 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n268
  state->_abc_1000_n274.value_0_0 = !(state->_abc_1000_n273_1.value_0_0 ^ state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1094 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n274
  state->_abc_1000_n284_1.value_0_0 = !(state->_abc_1000_n283.value_0_0 ^ state->_abc_1000_n276_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1104 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n284_1
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n288_1.value_0_0 = !((state->_abc_1000_n283.value_0_0 | state->_abc_1000_n276_1.value_0_0) & state->_abc_1000_n287_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1108 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n288_1
  state->_abc_1000_n289.value_0_0 = state->_abc_1000_n288_1.value_0_0 ^ state->_abc_1000_n286.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1109 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n289
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n319.value_0_0 = !((state->_abc_1000_n314_1.value_0_0 | state->_abc_1000_n306_1.value_0_0) & state->_abc_1000_n318_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1139 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n319
  state->_abc_1000_n315_1.value_0_0 = !(state->_abc_1000_n314_1.value_0_0 ^ state->_abc_1000_n306_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1135 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n315_1
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n342_1.value_0_0 = !((state->_abc_1000_n341_1.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n340.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1162 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n342_1
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n320_1.value_0_0 = state->_abc_1000_n319.value_0_0 ^ state->_abc_1000_n317_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1140 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n320_1
  state->_abc_1000_n358.value_0_0 = state->_abc_1000_n357.value_0_0 ^ state->_abc_1000_n351_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1178 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n358
  state->_abc_1000_n364.value_0_0 = !((state->_abc_1000_n357.value_0_0 & state->_abc_1000_n361.value_0_0) | state->_abc_1000_n363.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1184 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n364
  state->_abc_1000_n365.value_0_0 = state->_abc_1000_n364.value_0_0 ^ state->_abc_1000_n360.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1185 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n365
  state->_abc_1000_n227.value_0_0 = !(yosys_simplec_get_bit_4_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_4_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1047 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n227
  state->_abc_1000_n228.value_0_0 = !((state->_abc_1000_n223.value_0_0 | state->_abc_1000_n219.value_0_0) & state->_abc_1000_n227.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1048 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n228
  state->_abc_1000_n233.value_0_0 = !((state->_abc_1000_n227.value_0_0 | state->_abc_1000_n226.value_0_0) & state->_abc_1000_n232.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1053 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n233
  state->_abc_1000_n229.value_0_0 = state->_abc_1000_n228.value_0_0 ^ state->_abc_1000_n226.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1049 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n229
  state->_abc_1000_n250.value_0_0 = !((state->_abc_1000_n249.value_0_0 & state->_abc_1000_n233.value_0_0) | state->_abc_1000_n248.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1070 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n250
  state->_abc_1000_n252.value_0_0 = !((state->_abc_1000_n251.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n250.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1072 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n252
  state->_abc_1000_n283.value_0_0 = !((state->_abc_1000_n282_1.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n281_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1103 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n283
  state->_abc_1000_n266.value_0_0 = state->_abc_1000_n252.value_0_0 & (!state->_abc_1000_n265.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1086 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n266
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n256.value_0_0 = !state->_abc_1000_n252.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1076 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n256
  state->_abc_1000_n253.value_0_0 = state->_abc_1000_n252.value_0_0 ^ state->_abc_1000_n246.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1073 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n253
  state->_abc_1000_n258.value_0_0 = !((state->_abc_1000_n256.value_0_0 | state->_abc_1000_n246.value_0_0) & state->_abc_1000_n257.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1078 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n258
  state->_abc_1000_n259.value_0_0 = state->_abc_1000_n258.value_0_0 ^ state->_abc_1000_n255.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1079 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n259
  state->_abc_1000_n267.value_0_0 = state->_abc_1000_n266.value_0_0 | state->_abc_1000_n264.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1087 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n267
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n268.value_0_0 = state->_abc_1000_n267.value_0_0 ^ state->_abc_1000_n262.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1088 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n268
  state->_abc_1000_n274.value_0_0 = !(state->_abc_1000_n273_1.value_0_0 ^ state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1094 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n274
  state->_abc_1000_n284_1.value_0_0 = !(state->_abc_1000_n283.value_0_0 ^ state->_abc_1000_n276_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1104 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n284_1
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n288_1.value_0_0 = !((state->_abc_1000_n283.value_0_0 | state->_abc_1000_n276_1.value_0_0) & state->_abc_1000_n287_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1108 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n288_1
  state->_abc_1000_n289.value_0_0 = state->_abc_1000_n288_1.value_0_0 ^ state->_abc_1000_n286.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1109 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n289
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n319.value_0_0 = !((state->_abc_1000_n314_1.value_0_0 | state->_abc_1000_n306_1.value_0_0) & state->_abc_1000_n318_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1139 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n319
  state->_abc_1000_n315_1.value_0_0 = !(state->_abc_1000_n314_1.value_0_0 ^ state->_abc_1000_n306_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1135 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n315_1
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n342_1.value_0_0 = !((state->_abc_1000_n341_1.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n340.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1162 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n342_1
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n320_1.value_0_0 = state->_abc_1000_n319.value_0_0 ^ state->_abc_1000_n317_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1140 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n320_1
  state->_abc_1000_n358.value_0_0 = state->_abc_1000_n357.value_0_0 ^ state->_abc_1000_n351_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1178 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n358
  state->_abc_1000_n364.value_0_0 = !((state->_abc_1000_n357.value_0_0 & state->_abc_1000_n361.value_0_0) | state->_abc_1000_n363.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1184 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n364
  state->_abc_1000_n365.value_0_0 = state->_abc_1000_n364.value_0_0 ^ state->_abc_1000_n360.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1185 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n365
  state->_abc_1000_n234.value_0_0 = !state->_abc_1000_n233.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1054 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n234
  state->_abc_1000_n236.value_0_0 = !((state->_abc_1000_n235.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n234.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1056 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n236
  state->_abc_1000_n243.value_0_0 = !((state->_abc_1000_n236.value_0_0 & state->_abc_1000_n240.value_0_0) | state->_abc_1000_n242.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1063 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n243
  state->_abc_1000_n237.value_0_0 = state->_abc_1000_n236.value_0_0 ^ state->_abc_1000_n231.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1057 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n237
  state->_abc_1000_n244.value_0_0 = !(state->_abc_1000_n243.value_0_0 ^ state->_abc_1000_n239.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1064 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n244
  yosys_simplec_set_bit_1_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n206.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1027 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [1]
  yosys_simplec_set_bit_2_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n211.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1032 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [2]
  yosys_simplec_set_bit_3_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n217.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1038 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [3]
  yosys_simplec_set_bit_4_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n224.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1045 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [4]
  yosys_simplec_set_bit_12_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n284_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1105 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [12]
  yosys_simplec_set_bit_23_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n365.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1186 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [23]
  yosys_simplec_set_bit_26_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n394.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1215 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [26]
  yosys_simplec_set_bit_19_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n333_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1154 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [19]
  yosys_simplec_set_bit_22_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n358.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1179 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [22]
  yosys_simplec_set_bit_11_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n274.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1095 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [11]
  yosys_simplec_set_bit_18_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n327_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1148 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [18]
  yosys_simplec_set_bit_25_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n385.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1206 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [25]
  yosys_simplec_set_bit_10_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n268.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1089 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [10]
  yosys_simplec_set_bit_21_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n349.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1170 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [21]
  yosys_simplec_set_bit_17_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n320_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1141 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [17]
  yosys_simplec_set_bit_31_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n431.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1252 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [31]
  yosys_simplec_set_bit_24_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n377.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1198 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [24]
  yosys_simplec_set_bit_16_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n315_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1136 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [16]
  yosys_simplec_set_bit_30_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n426.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1247 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [30]
  yosys_simplec_set_bit_20_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n343.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1164 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [20]
  yosys_simplec_set_bit_9_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n259.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1080 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [9]
  yosys_simplec_set_bit_8_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n253.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1074 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [8]
  yosys_simplec_set_bit_29_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n418.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1239 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [29]
  yosys_simplec_set_bit_15_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n304.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1125 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [15]
  yosys_simplec_set_bit_7_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n244.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1065 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [7]
  yosys_simplec_set_bit_28_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n412.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1233 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [28]
  yosys_simplec_set_bit_14_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n297_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1118 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [14]
  yosys_simplec_set_bit_6_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n237.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1058 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [6]
  yosys_simplec_set_bit_27_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n401.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1222 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [27]
  yosys_simplec_set_bit_13_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n289.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1110 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [13]
  yosys_simplec_set_bit_5_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n229.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1050 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [5]
  yosys_simplec_set_bit_0_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n198_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1023 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [0]
  state->_abc_1000_n210.value_0_0 = !((state->_abc_1000_n205_1.value_0_0 | state->_abc_1000_n204_1.value_0_0) & state->_abc_1000_n209.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1030 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n210
  state->_abc_1000_n211.value_0_0 = !(state->_abc_1000_n210.value_0_0 ^ state->_abc_1000_n208.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1031 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n211
  yosys_simplec_set_bit_2_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n211.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1032 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [2]
  state->_abc_1000_n223.value_0_0 = !((state->_abc_1000_n222.value_0_0 & state->_abc_1000_n210.value_0_0) | state->_abc_1000_n221.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1043 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n223
  state->_abc_1000_n224.value_0_0 = !(state->_abc_1000_n223.value_0_0 ^ state->_abc_1000_n219.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1044 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n224
  state->_abc_1000_n252.value_0_0 = !((state->_abc_1000_n251.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n250.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1072 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n252
  state->_abc_1000_n283.value_0_0 = !((state->_abc_1000_n282_1.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n281_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1103 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n283
  state->_abc_1000_n266.value_0_0 = state->_abc_1000_n252.value_0_0 & (!state->_abc_1000_n265.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1086 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n266
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n256.value_0_0 = !state->_abc_1000_n252.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1076 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n256
  state->_abc_1000_n253.value_0_0 = state->_abc_1000_n252.value_0_0 ^ state->_abc_1000_n246.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1073 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n253
  state->_abc_1000_n258.value_0_0 = !((state->_abc_1000_n256.value_0_0 | state->_abc_1000_n246.value_0_0) & state->_abc_1000_n257.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1078 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n258
  state->_abc_1000_n259.value_0_0 = state->_abc_1000_n258.value_0_0 ^ state->_abc_1000_n255.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1079 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n259
  state->_abc_1000_n267.value_0_0 = state->_abc_1000_n266.value_0_0 | state->_abc_1000_n264.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1087 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n267
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n268.value_0_0 = state->_abc_1000_n267.value_0_0 ^ state->_abc_1000_n262.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1088 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n268
  state->_abc_1000_n274.value_0_0 = !(state->_abc_1000_n273_1.value_0_0 ^ state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1094 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n274
  state->_abc_1000_n284_1.value_0_0 = !(state->_abc_1000_n283.value_0_0 ^ state->_abc_1000_n276_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1104 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n284_1
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n288_1.value_0_0 = !((state->_abc_1000_n283.value_0_0 | state->_abc_1000_n276_1.value_0_0) & state->_abc_1000_n287_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1108 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n288_1
  state->_abc_1000_n289.value_0_0 = state->_abc_1000_n288_1.value_0_0 ^ state->_abc_1000_n286.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1109 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n289
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n319.value_0_0 = !((state->_abc_1000_n314_1.value_0_0 | state->_abc_1000_n306_1.value_0_0) & state->_abc_1000_n318_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1139 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n319
  state->_abc_1000_n315_1.value_0_0 = !(state->_abc_1000_n314_1.value_0_0 ^ state->_abc_1000_n306_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1135 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n315_1
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n342_1.value_0_0 = !((state->_abc_1000_n341_1.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n340.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1162 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n342_1
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n320_1.value_0_0 = state->_abc_1000_n319.value_0_0 ^ state->_abc_1000_n317_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1140 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n320_1
  state->_abc_1000_n358.value_0_0 = state->_abc_1000_n357.value_0_0 ^ state->_abc_1000_n351_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1178 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n358
  state->_abc_1000_n364.value_0_0 = !((state->_abc_1000_n357.value_0_0 & state->_abc_1000_n361.value_0_0) | state->_abc_1000_n363.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1184 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n364
  state->_abc_1000_n365.value_0_0 = state->_abc_1000_n364.value_0_0 ^ state->_abc_1000_n360.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1185 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n365
  state->_abc_1000_n236.value_0_0 = !((state->_abc_1000_n235.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n234.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1056 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n236
  state->_abc_1000_n243.value_0_0 = !((state->_abc_1000_n236.value_0_0 & state->_abc_1000_n240.value_0_0) | state->_abc_1000_n242.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1063 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n243
  state->_abc_1000_n237.value_0_0 = state->_abc_1000_n236.value_0_0 ^ state->_abc_1000_n231.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1057 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n237
  state->_abc_1000_n244.value_0_0 = !(state->_abc_1000_n243.value_0_0 ^ state->_abc_1000_n239.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1064 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n244
  state->_abc_1000_n228.value_0_0 = !((state->_abc_1000_n223.value_0_0 | state->_abc_1000_n219.value_0_0) & state->_abc_1000_n227.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1048 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n228
  state->_abc_1000_n229.value_0_0 = state->_abc_1000_n228.value_0_0 ^ state->_abc_1000_n226.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1049 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n229
  yosys_simplec_set_bit_4_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n224.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1045 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [4]
  yosys_simplec_set_bit_12_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n284_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1105 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [12]
  yosys_simplec_set_bit_23_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n365.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1186 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [23]
  yosys_simplec_set_bit_26_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n394.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1215 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [26]
  yosys_simplec_set_bit_19_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n333_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1154 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [19]
  yosys_simplec_set_bit_22_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n358.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1179 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [22]
  yosys_simplec_set_bit_11_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n274.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1095 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [11]
  yosys_simplec_set_bit_18_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n327_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1148 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [18]
  yosys_simplec_set_bit_25_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n385.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1206 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [25]
  yosys_simplec_set_bit_10_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n268.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1089 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [10]
  yosys_simplec_set_bit_21_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n349.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1170 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [21]
  yosys_simplec_set_bit_17_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n320_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1141 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [17]
  yosys_simplec_set_bit_31_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n431.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1252 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [31]
  yosys_simplec_set_bit_24_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n377.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1198 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [24]
  yosys_simplec_set_bit_16_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n315_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1136 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [16]
  yosys_simplec_set_bit_30_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n426.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1247 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [30]
  yosys_simplec_set_bit_20_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n343.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1164 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [20]
  yosys_simplec_set_bit_9_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n259.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1080 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [9]
  yosys_simplec_set_bit_8_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n253.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1074 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [8]
  yosys_simplec_set_bit_29_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n418.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1239 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [29]
  yosys_simplec_set_bit_15_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n304.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1125 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [15]
  yosys_simplec_set_bit_7_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n244.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1065 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [7]
  yosys_simplec_set_bit_28_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n412.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1233 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [28]
  yosys_simplec_set_bit_14_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n297_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1118 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [14]
  yosys_simplec_set_bit_6_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n237.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1058 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [6]
  yosys_simplec_set_bit_27_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n401.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1222 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [27]
  yosys_simplec_set_bit_13_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n289.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1110 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [13]
  yosys_simplec_set_bit_5_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n229.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1050 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [5]
  state->_abc_1000_n216.value_0_0 = !((state->_abc_1000_n210.value_0_0 & state->_abc_1000_n208.value_0_0) | state->_abc_1000_n215.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1036 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n216
  state->_abc_1000_n217.value_0_0 = !(state->_abc_1000_n216.value_0_0 ^ state->_abc_1000_n213.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1037 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n217
  yosys_simplec_set_bit_3_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n217.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1038 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [3]
  state->_abc_1000_n206.value_0_0 = !(state->_abc_1000_n205_1.value_0_0 ^ state->_abc_1000_n204_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1026 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n206
  yosys_simplec_set_bit_1_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n206.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1027 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [1]
  state->_abc_1000_n213.value_0_0 = !(yosys_simplec_get_bit_3_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_3_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1033 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n213
  state->_abc_1000_n221.value_0_0 = !((state->_abc_1000_n214.value_0_0 | state->_abc_1000_n213.value_0_0) & state->_abc_1000_n220.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1041 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n221
  state->_abc_1000_n217.value_0_0 = !(state->_abc_1000_n216.value_0_0 ^ state->_abc_1000_n213.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1037 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n217
  yosys_simplec_set_bit_3_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n217.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1038 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [3]
  state->_abc_1000_n222.value_0_0 = state->_abc_1000_n208.value_0_0 & (!state->_abc_1000_n213.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1042 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n222
  state->_abc_1000_n223.value_0_0 = !((state->_abc_1000_n222.value_0_0 & state->_abc_1000_n210.value_0_0) | state->_abc_1000_n221.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1043 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n223
  state->_abc_1000_n224.value_0_0 = !(state->_abc_1000_n223.value_0_0 ^ state->_abc_1000_n219.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1044 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n224
  state->_abc_1000_n252.value_0_0 = !((state->_abc_1000_n251.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n250.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1072 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n252
  state->_abc_1000_n283.value_0_0 = !((state->_abc_1000_n282_1.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n281_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1103 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n283
  state->_abc_1000_n266.value_0_0 = state->_abc_1000_n252.value_0_0 & (!state->_abc_1000_n265.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1086 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n266
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n256.value_0_0 = !state->_abc_1000_n252.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1076 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n256
  state->_abc_1000_n253.value_0_0 = state->_abc_1000_n252.value_0_0 ^ state->_abc_1000_n246.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1073 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n253
  state->_abc_1000_n258.value_0_0 = !((state->_abc_1000_n256.value_0_0 | state->_abc_1000_n246.value_0_0) & state->_abc_1000_n257.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1078 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n258
  state->_abc_1000_n259.value_0_0 = state->_abc_1000_n258.value_0_0 ^ state->_abc_1000_n255.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1079 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n259
  state->_abc_1000_n267.value_0_0 = state->_abc_1000_n266.value_0_0 | state->_abc_1000_n264.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1087 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n267
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n268.value_0_0 = state->_abc_1000_n267.value_0_0 ^ state->_abc_1000_n262.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1088 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n268
  state->_abc_1000_n274.value_0_0 = !(state->_abc_1000_n273_1.value_0_0 ^ state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1094 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n274
  state->_abc_1000_n284_1.value_0_0 = !(state->_abc_1000_n283.value_0_0 ^ state->_abc_1000_n276_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1104 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n284_1
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n288_1.value_0_0 = !((state->_abc_1000_n283.value_0_0 | state->_abc_1000_n276_1.value_0_0) & state->_abc_1000_n287_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1108 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n288_1
  state->_abc_1000_n289.value_0_0 = state->_abc_1000_n288_1.value_0_0 ^ state->_abc_1000_n286.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1109 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n289
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n319.value_0_0 = !((state->_abc_1000_n314_1.value_0_0 | state->_abc_1000_n306_1.value_0_0) & state->_abc_1000_n318_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1139 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n319
  state->_abc_1000_n315_1.value_0_0 = !(state->_abc_1000_n314_1.value_0_0 ^ state->_abc_1000_n306_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1135 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n315_1
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n342_1.value_0_0 = !((state->_abc_1000_n341_1.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n340.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1162 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n342_1
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n320_1.value_0_0 = state->_abc_1000_n319.value_0_0 ^ state->_abc_1000_n317_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1140 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n320_1
  state->_abc_1000_n358.value_0_0 = state->_abc_1000_n357.value_0_0 ^ state->_abc_1000_n351_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1178 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n358
  state->_abc_1000_n364.value_0_0 = !((state->_abc_1000_n357.value_0_0 & state->_abc_1000_n361.value_0_0) | state->_abc_1000_n363.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1184 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n364
  state->_abc_1000_n365.value_0_0 = state->_abc_1000_n364.value_0_0 ^ state->_abc_1000_n360.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1185 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n365
  state->_abc_1000_n236.value_0_0 = !((state->_abc_1000_n235.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n234.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1056 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n236
  state->_abc_1000_n243.value_0_0 = !((state->_abc_1000_n236.value_0_0 & state->_abc_1000_n240.value_0_0) | state->_abc_1000_n242.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1063 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n243
  state->_abc_1000_n237.value_0_0 = state->_abc_1000_n236.value_0_0 ^ state->_abc_1000_n231.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1057 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n237
  state->_abc_1000_n244.value_0_0 = !(state->_abc_1000_n243.value_0_0 ^ state->_abc_1000_n239.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1064 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n244
  state->_abc_1000_n228.value_0_0 = !((state->_abc_1000_n223.value_0_0 | state->_abc_1000_n219.value_0_0) & state->_abc_1000_n227.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1048 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n228
  state->_abc_1000_n229.value_0_0 = state->_abc_1000_n228.value_0_0 ^ state->_abc_1000_n226.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1049 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n229
  yosys_simplec_set_bit_4_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n224.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1045 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [4]
  yosys_simplec_set_bit_12_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n284_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1105 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [12]
  yosys_simplec_set_bit_23_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n365.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1186 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [23]
  yosys_simplec_set_bit_26_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n394.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1215 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [26]
  yosys_simplec_set_bit_19_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n333_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1154 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [19]
  yosys_simplec_set_bit_22_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n358.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1179 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [22]
  yosys_simplec_set_bit_11_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n274.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1095 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [11]
  yosys_simplec_set_bit_18_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n327_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1148 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [18]
  yosys_simplec_set_bit_25_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n385.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1206 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [25]
  yosys_simplec_set_bit_10_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n268.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1089 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [10]
  yosys_simplec_set_bit_21_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n349.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1170 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [21]
  yosys_simplec_set_bit_17_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n320_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1141 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [17]
  yosys_simplec_set_bit_31_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n431.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1252 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [31]
  yosys_simplec_set_bit_24_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n377.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1198 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [24]
  yosys_simplec_set_bit_16_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n315_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1136 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [16]
  yosys_simplec_set_bit_30_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n426.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1247 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [30]
  yosys_simplec_set_bit_20_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n343.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1164 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [20]
  yosys_simplec_set_bit_9_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n259.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1080 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [9]
  yosys_simplec_set_bit_8_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n253.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1074 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [8]
  yosys_simplec_set_bit_29_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n418.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1239 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [29]
  yosys_simplec_set_bit_15_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n304.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1125 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [15]
  yosys_simplec_set_bit_7_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n244.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1065 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [7]
  yosys_simplec_set_bit_28_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n412.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1233 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [28]
  yosys_simplec_set_bit_14_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n297_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1118 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [14]
  yosys_simplec_set_bit_6_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n237.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1058 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [6]
  yosys_simplec_set_bit_27_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n401.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1222 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [27]
  yosys_simplec_set_bit_13_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n289.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1110 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [13]
  yosys_simplec_set_bit_5_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n229.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1050 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [5]
  state->_abc_1000_n220.value_0_0 = !(yosys_simplec_get_bit_3_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_3_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1040 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n220
  state->_abc_1000_n221.value_0_0 = !((state->_abc_1000_n214.value_0_0 | state->_abc_1000_n213.value_0_0) & state->_abc_1000_n220.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1041 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n221
  state->_abc_1000_n223.value_0_0 = !((state->_abc_1000_n222.value_0_0 & state->_abc_1000_n210.value_0_0) | state->_abc_1000_n221.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1043 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n223
  state->_abc_1000_n224.value_0_0 = !(state->_abc_1000_n223.value_0_0 ^ state->_abc_1000_n219.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1044 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n224
  state->_abc_1000_n252.value_0_0 = !((state->_abc_1000_n251.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n250.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1072 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n252
  state->_abc_1000_n283.value_0_0 = !((state->_abc_1000_n282_1.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n281_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1103 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n283
  state->_abc_1000_n266.value_0_0 = state->_abc_1000_n252.value_0_0 & (!state->_abc_1000_n265.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1086 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n266
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n256.value_0_0 = !state->_abc_1000_n252.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1076 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n256
  state->_abc_1000_n253.value_0_0 = state->_abc_1000_n252.value_0_0 ^ state->_abc_1000_n246.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1073 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n253
  state->_abc_1000_n258.value_0_0 = !((state->_abc_1000_n256.value_0_0 | state->_abc_1000_n246.value_0_0) & state->_abc_1000_n257.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1078 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n258
  state->_abc_1000_n259.value_0_0 = state->_abc_1000_n258.value_0_0 ^ state->_abc_1000_n255.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1079 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n259
  state->_abc_1000_n267.value_0_0 = state->_abc_1000_n266.value_0_0 | state->_abc_1000_n264.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1087 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n267
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n268.value_0_0 = state->_abc_1000_n267.value_0_0 ^ state->_abc_1000_n262.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1088 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n268
  state->_abc_1000_n274.value_0_0 = !(state->_abc_1000_n273_1.value_0_0 ^ state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1094 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n274
  state->_abc_1000_n284_1.value_0_0 = !(state->_abc_1000_n283.value_0_0 ^ state->_abc_1000_n276_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1104 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n284_1
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n288_1.value_0_0 = !((state->_abc_1000_n283.value_0_0 | state->_abc_1000_n276_1.value_0_0) & state->_abc_1000_n287_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1108 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n288_1
  state->_abc_1000_n289.value_0_0 = state->_abc_1000_n288_1.value_0_0 ^ state->_abc_1000_n286.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1109 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n289
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n319.value_0_0 = !((state->_abc_1000_n314_1.value_0_0 | state->_abc_1000_n306_1.value_0_0) & state->_abc_1000_n318_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1139 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n319
  state->_abc_1000_n315_1.value_0_0 = !(state->_abc_1000_n314_1.value_0_0 ^ state->_abc_1000_n306_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1135 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n315_1
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n342_1.value_0_0 = !((state->_abc_1000_n341_1.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n340.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1162 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n342_1
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n320_1.value_0_0 = state->_abc_1000_n319.value_0_0 ^ state->_abc_1000_n317_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1140 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n320_1
  state->_abc_1000_n358.value_0_0 = state->_abc_1000_n357.value_0_0 ^ state->_abc_1000_n351_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1178 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n358
  state->_abc_1000_n364.value_0_0 = !((state->_abc_1000_n357.value_0_0 & state->_abc_1000_n361.value_0_0) | state->_abc_1000_n363.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1184 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n364
  state->_abc_1000_n365.value_0_0 = state->_abc_1000_n364.value_0_0 ^ state->_abc_1000_n360.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1185 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n365
  state->_abc_1000_n236.value_0_0 = !((state->_abc_1000_n235.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n234.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1056 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n236
  state->_abc_1000_n243.value_0_0 = !((state->_abc_1000_n236.value_0_0 & state->_abc_1000_n240.value_0_0) | state->_abc_1000_n242.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1063 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n243
  state->_abc_1000_n237.value_0_0 = state->_abc_1000_n236.value_0_0 ^ state->_abc_1000_n231.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1057 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n237
  state->_abc_1000_n244.value_0_0 = !(state->_abc_1000_n243.value_0_0 ^ state->_abc_1000_n239.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1064 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n244
  state->_abc_1000_n228.value_0_0 = !((state->_abc_1000_n223.value_0_0 | state->_abc_1000_n219.value_0_0) & state->_abc_1000_n227.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1048 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n228
  state->_abc_1000_n229.value_0_0 = state->_abc_1000_n228.value_0_0 ^ state->_abc_1000_n226.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1049 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n229
  yosys_simplec_set_bit_4_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n224.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1045 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [4]
  yosys_simplec_set_bit_12_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n284_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1105 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [12]
  yosys_simplec_set_bit_23_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n365.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1186 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [23]
  yosys_simplec_set_bit_26_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n394.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1215 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [26]
  yosys_simplec_set_bit_19_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n333_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1154 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [19]
  yosys_simplec_set_bit_22_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n358.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1179 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [22]
  yosys_simplec_set_bit_11_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n274.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1095 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [11]
  yosys_simplec_set_bit_18_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n327_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1148 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [18]
  yosys_simplec_set_bit_25_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n385.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1206 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [25]
  yosys_simplec_set_bit_10_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n268.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1089 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [10]
  yosys_simplec_set_bit_21_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n349.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1170 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [21]
  yosys_simplec_set_bit_17_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n320_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1141 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [17]
  yosys_simplec_set_bit_31_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n431.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1252 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [31]
  yosys_simplec_set_bit_24_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n377.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1198 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [24]
  yosys_simplec_set_bit_16_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n315_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1136 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [16]
  yosys_simplec_set_bit_30_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n426.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1247 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [30]
  yosys_simplec_set_bit_20_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n343.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1164 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [20]
  yosys_simplec_set_bit_9_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n259.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1080 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [9]
  yosys_simplec_set_bit_8_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n253.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1074 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [8]
  yosys_simplec_set_bit_29_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n418.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1239 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [29]
  yosys_simplec_set_bit_15_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n304.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1125 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [15]
  yosys_simplec_set_bit_7_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n244.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1065 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [7]
  yosys_simplec_set_bit_28_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n412.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1233 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [28]
  yosys_simplec_set_bit_14_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n297_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1118 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [14]
  yosys_simplec_set_bit_6_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n237.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1058 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [6]
  yosys_simplec_set_bit_27_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n401.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1222 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [27]
  yosys_simplec_set_bit_13_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n289.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1110 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [13]
  yosys_simplec_set_bit_5_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n229.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1050 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [5]
  state->_abc_1000_n214.value_0_0 = !(yosys_simplec_get_bit_2_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_2_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1034 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n214
  state->_abc_1000_n215.value_0_0 = !state->_abc_1000_n214.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1035 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n215
  state->_abc_1000_n221.value_0_0 = !((state->_abc_1000_n214.value_0_0 | state->_abc_1000_n213.value_0_0) & state->_abc_1000_n220.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1041 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n221
  state->_abc_1000_n223.value_0_0 = !((state->_abc_1000_n222.value_0_0 & state->_abc_1000_n210.value_0_0) | state->_abc_1000_n221.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1043 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n223
  state->_abc_1000_n224.value_0_0 = !(state->_abc_1000_n223.value_0_0 ^ state->_abc_1000_n219.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1044 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n224
  state->_abc_1000_n252.value_0_0 = !((state->_abc_1000_n251.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n250.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1072 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n252
  state->_abc_1000_n283.value_0_0 = !((state->_abc_1000_n282_1.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n281_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1103 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n283
  state->_abc_1000_n266.value_0_0 = state->_abc_1000_n252.value_0_0 & (!state->_abc_1000_n265.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1086 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n266
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n256.value_0_0 = !state->_abc_1000_n252.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1076 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n256
  state->_abc_1000_n253.value_0_0 = state->_abc_1000_n252.value_0_0 ^ state->_abc_1000_n246.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1073 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n253
  state->_abc_1000_n258.value_0_0 = !((state->_abc_1000_n256.value_0_0 | state->_abc_1000_n246.value_0_0) & state->_abc_1000_n257.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1078 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n258
  state->_abc_1000_n259.value_0_0 = state->_abc_1000_n258.value_0_0 ^ state->_abc_1000_n255.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1079 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n259
  state->_abc_1000_n267.value_0_0 = state->_abc_1000_n266.value_0_0 | state->_abc_1000_n264.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1087 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n267
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n268.value_0_0 = state->_abc_1000_n267.value_0_0 ^ state->_abc_1000_n262.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1088 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n268
  state->_abc_1000_n274.value_0_0 = !(state->_abc_1000_n273_1.value_0_0 ^ state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1094 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n274
  state->_abc_1000_n284_1.value_0_0 = !(state->_abc_1000_n283.value_0_0 ^ state->_abc_1000_n276_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1104 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n284_1
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n288_1.value_0_0 = !((state->_abc_1000_n283.value_0_0 | state->_abc_1000_n276_1.value_0_0) & state->_abc_1000_n287_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1108 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n288_1
  state->_abc_1000_n289.value_0_0 = state->_abc_1000_n288_1.value_0_0 ^ state->_abc_1000_n286.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1109 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n289
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n319.value_0_0 = !((state->_abc_1000_n314_1.value_0_0 | state->_abc_1000_n306_1.value_0_0) & state->_abc_1000_n318_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1139 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n319
  state->_abc_1000_n315_1.value_0_0 = !(state->_abc_1000_n314_1.value_0_0 ^ state->_abc_1000_n306_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1135 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n315_1
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n342_1.value_0_0 = !((state->_abc_1000_n341_1.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n340.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1162 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n342_1
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n320_1.value_0_0 = state->_abc_1000_n319.value_0_0 ^ state->_abc_1000_n317_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1140 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n320_1
  state->_abc_1000_n358.value_0_0 = state->_abc_1000_n357.value_0_0 ^ state->_abc_1000_n351_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1178 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n358
  state->_abc_1000_n364.value_0_0 = !((state->_abc_1000_n357.value_0_0 & state->_abc_1000_n361.value_0_0) | state->_abc_1000_n363.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1184 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n364
  state->_abc_1000_n365.value_0_0 = state->_abc_1000_n364.value_0_0 ^ state->_abc_1000_n360.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1185 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n365
  state->_abc_1000_n236.value_0_0 = !((state->_abc_1000_n235.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n234.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1056 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n236
  state->_abc_1000_n243.value_0_0 = !((state->_abc_1000_n236.value_0_0 & state->_abc_1000_n240.value_0_0) | state->_abc_1000_n242.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1063 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n243
  state->_abc_1000_n237.value_0_0 = state->_abc_1000_n236.value_0_0 ^ state->_abc_1000_n231.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1057 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n237
  state->_abc_1000_n244.value_0_0 = !(state->_abc_1000_n243.value_0_0 ^ state->_abc_1000_n239.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1064 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n244
  state->_abc_1000_n228.value_0_0 = !((state->_abc_1000_n223.value_0_0 | state->_abc_1000_n219.value_0_0) & state->_abc_1000_n227.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1048 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n228
  state->_abc_1000_n229.value_0_0 = state->_abc_1000_n228.value_0_0 ^ state->_abc_1000_n226.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1049 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n229
  yosys_simplec_set_bit_4_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n224.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1045 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [4]
  yosys_simplec_set_bit_12_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n284_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1105 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [12]
  yosys_simplec_set_bit_23_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n365.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1186 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [23]
  yosys_simplec_set_bit_26_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n394.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1215 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [26]
  yosys_simplec_set_bit_19_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n333_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1154 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [19]
  yosys_simplec_set_bit_22_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n358.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1179 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [22]
  yosys_simplec_set_bit_11_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n274.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1095 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [11]
  yosys_simplec_set_bit_18_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n327_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1148 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [18]
  yosys_simplec_set_bit_25_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n385.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1206 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [25]
  yosys_simplec_set_bit_10_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n268.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1089 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [10]
  yosys_simplec_set_bit_21_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n349.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1170 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [21]
  yosys_simplec_set_bit_17_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n320_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1141 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [17]
  yosys_simplec_set_bit_31_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n431.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1252 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [31]
  yosys_simplec_set_bit_24_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n377.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1198 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [24]
  yosys_simplec_set_bit_16_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n315_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1136 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [16]
  yosys_simplec_set_bit_30_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n426.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1247 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [30]
  yosys_simplec_set_bit_20_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n343.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1164 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [20]
  yosys_simplec_set_bit_9_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n259.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1080 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [9]
  yosys_simplec_set_bit_8_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n253.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1074 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [8]
  yosys_simplec_set_bit_29_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n418.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1239 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [29]
  yosys_simplec_set_bit_15_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n304.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1125 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [15]
  yosys_simplec_set_bit_7_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n244.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1065 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [7]
  yosys_simplec_set_bit_28_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n412.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1233 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [28]
  yosys_simplec_set_bit_14_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n297_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1118 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [14]
  yosys_simplec_set_bit_6_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n237.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1058 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [6]
  yosys_simplec_set_bit_27_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n401.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1222 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [27]
  yosys_simplec_set_bit_13_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n289.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1110 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [13]
  yosys_simplec_set_bit_5_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n229.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1050 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [5]
  state->_abc_1000_n216.value_0_0 = !((state->_abc_1000_n210.value_0_0 & state->_abc_1000_n208.value_0_0) | state->_abc_1000_n215.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1036 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n216
  state->_abc_1000_n217.value_0_0 = !(state->_abc_1000_n216.value_0_0 ^ state->_abc_1000_n213.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1037 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n217
  yosys_simplec_set_bit_3_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n217.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1038 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [3]
  state->_abc_1000_n208.value_0_0 = yosys_simplec_get_bit_2_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_2_of_32(&state->rvfi_rs1_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1028 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n208
  state->_abc_1000_n222.value_0_0 = state->_abc_1000_n208.value_0_0 & (!state->_abc_1000_n213.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1042 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n222
  state->_abc_1000_n211.value_0_0 = !(state->_abc_1000_n210.value_0_0 ^ state->_abc_1000_n208.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1031 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n211
  yosys_simplec_set_bit_2_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n211.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1032 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [2]
  state->_abc_1000_n223.value_0_0 = !((state->_abc_1000_n222.value_0_0 & state->_abc_1000_n210.value_0_0) | state->_abc_1000_n221.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1043 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n223
  state->_abc_1000_n224.value_0_0 = !(state->_abc_1000_n223.value_0_0 ^ state->_abc_1000_n219.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1044 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n224
  state->_abc_1000_n252.value_0_0 = !((state->_abc_1000_n251.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n250.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1072 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n252
  state->_abc_1000_n283.value_0_0 = !((state->_abc_1000_n282_1.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n281_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1103 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n283
  state->_abc_1000_n266.value_0_0 = state->_abc_1000_n252.value_0_0 & (!state->_abc_1000_n265.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1086 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n266
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n256.value_0_0 = !state->_abc_1000_n252.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1076 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n256
  state->_abc_1000_n253.value_0_0 = state->_abc_1000_n252.value_0_0 ^ state->_abc_1000_n246.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1073 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n253
  state->_abc_1000_n258.value_0_0 = !((state->_abc_1000_n256.value_0_0 | state->_abc_1000_n246.value_0_0) & state->_abc_1000_n257.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1078 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n258
  state->_abc_1000_n259.value_0_0 = state->_abc_1000_n258.value_0_0 ^ state->_abc_1000_n255.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1079 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n259
  state->_abc_1000_n267.value_0_0 = state->_abc_1000_n266.value_0_0 | state->_abc_1000_n264.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1087 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n267
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n268.value_0_0 = state->_abc_1000_n267.value_0_0 ^ state->_abc_1000_n262.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1088 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n268
  state->_abc_1000_n274.value_0_0 = !(state->_abc_1000_n273_1.value_0_0 ^ state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1094 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n274
  state->_abc_1000_n284_1.value_0_0 = !(state->_abc_1000_n283.value_0_0 ^ state->_abc_1000_n276_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1104 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n284_1
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n288_1.value_0_0 = !((state->_abc_1000_n283.value_0_0 | state->_abc_1000_n276_1.value_0_0) & state->_abc_1000_n287_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1108 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n288_1
  state->_abc_1000_n289.value_0_0 = state->_abc_1000_n288_1.value_0_0 ^ state->_abc_1000_n286.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1109 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n289
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n319.value_0_0 = !((state->_abc_1000_n314_1.value_0_0 | state->_abc_1000_n306_1.value_0_0) & state->_abc_1000_n318_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1139 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n319
  state->_abc_1000_n315_1.value_0_0 = !(state->_abc_1000_n314_1.value_0_0 ^ state->_abc_1000_n306_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1135 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n315_1
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n342_1.value_0_0 = !((state->_abc_1000_n341_1.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n340.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1162 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n342_1
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n320_1.value_0_0 = state->_abc_1000_n319.value_0_0 ^ state->_abc_1000_n317_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1140 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n320_1
  state->_abc_1000_n358.value_0_0 = state->_abc_1000_n357.value_0_0 ^ state->_abc_1000_n351_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1178 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n358
  state->_abc_1000_n364.value_0_0 = !((state->_abc_1000_n357.value_0_0 & state->_abc_1000_n361.value_0_0) | state->_abc_1000_n363.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1184 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n364
  state->_abc_1000_n365.value_0_0 = state->_abc_1000_n364.value_0_0 ^ state->_abc_1000_n360.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1185 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n365
  state->_abc_1000_n236.value_0_0 = !((state->_abc_1000_n235.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n234.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1056 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n236
  state->_abc_1000_n243.value_0_0 = !((state->_abc_1000_n236.value_0_0 & state->_abc_1000_n240.value_0_0) | state->_abc_1000_n242.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1063 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n243
  state->_abc_1000_n237.value_0_0 = state->_abc_1000_n236.value_0_0 ^ state->_abc_1000_n231.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1057 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n237
  state->_abc_1000_n244.value_0_0 = !(state->_abc_1000_n243.value_0_0 ^ state->_abc_1000_n239.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1064 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n244
  state->_abc_1000_n228.value_0_0 = !((state->_abc_1000_n223.value_0_0 | state->_abc_1000_n219.value_0_0) & state->_abc_1000_n227.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1048 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n228
  state->_abc_1000_n229.value_0_0 = state->_abc_1000_n228.value_0_0 ^ state->_abc_1000_n226.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1049 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n229
  yosys_simplec_set_bit_4_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n224.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1045 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [4]
  yosys_simplec_set_bit_12_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n284_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1105 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [12]
  yosys_simplec_set_bit_23_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n365.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1186 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [23]
  yosys_simplec_set_bit_26_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n394.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1215 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [26]
  yosys_simplec_set_bit_19_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n333_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1154 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [19]
  yosys_simplec_set_bit_22_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n358.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1179 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [22]
  yosys_simplec_set_bit_11_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n274.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1095 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [11]
  yosys_simplec_set_bit_18_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n327_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1148 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [18]
  yosys_simplec_set_bit_25_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n385.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1206 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [25]
  yosys_simplec_set_bit_10_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n268.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1089 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [10]
  yosys_simplec_set_bit_21_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n349.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1170 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [21]
  yosys_simplec_set_bit_17_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n320_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1141 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [17]
  yosys_simplec_set_bit_31_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n431.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1252 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [31]
  yosys_simplec_set_bit_24_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n377.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1198 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [24]
  yosys_simplec_set_bit_16_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n315_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1136 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [16]
  yosys_simplec_set_bit_30_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n426.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1247 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [30]
  yosys_simplec_set_bit_20_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n343.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1164 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [20]
  yosys_simplec_set_bit_9_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n259.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1080 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [9]
  yosys_simplec_set_bit_8_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n253.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1074 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [8]
  yosys_simplec_set_bit_29_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n418.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1239 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [29]
  yosys_simplec_set_bit_15_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n304.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1125 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [15]
  yosys_simplec_set_bit_7_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n244.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1065 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [7]
  yosys_simplec_set_bit_28_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n412.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1233 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [28]
  yosys_simplec_set_bit_14_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n297_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1118 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [14]
  yosys_simplec_set_bit_6_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n237.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1058 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [6]
  yosys_simplec_set_bit_27_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n401.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1222 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [27]
  yosys_simplec_set_bit_13_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n289.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1110 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [13]
  yosys_simplec_set_bit_5_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n229.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1050 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [5]
  state->_abc_1000_n216.value_0_0 = !((state->_abc_1000_n210.value_0_0 & state->_abc_1000_n208.value_0_0) | state->_abc_1000_n215.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1036 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n216
  state->_abc_1000_n217.value_0_0 = !(state->_abc_1000_n216.value_0_0 ^ state->_abc_1000_n213.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1037 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n217
  yosys_simplec_set_bit_3_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n217.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1038 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [3]
  state->_abc_1000_n209.value_0_0 = !(yosys_simplec_get_bit_1_of_32(&state->rvfi_rs2_rdata) & yosys_simplec_get_bit_1_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1029 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n209
  state->_abc_1000_n210.value_0_0 = !((state->_abc_1000_n205_1.value_0_0 | state->_abc_1000_n204_1.value_0_0) & state->_abc_1000_n209.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1030 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n210
  state->_abc_1000_n211.value_0_0 = !(state->_abc_1000_n210.value_0_0 ^ state->_abc_1000_n208.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1031 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n211
  yosys_simplec_set_bit_2_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n211.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1032 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [2]
  state->_abc_1000_n223.value_0_0 = !((state->_abc_1000_n222.value_0_0 & state->_abc_1000_n210.value_0_0) | state->_abc_1000_n221.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1043 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n223
  state->_abc_1000_n224.value_0_0 = !(state->_abc_1000_n223.value_0_0 ^ state->_abc_1000_n219.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1044 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n224
  state->_abc_1000_n252.value_0_0 = !((state->_abc_1000_n251.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n250.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1072 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n252
  state->_abc_1000_n283.value_0_0 = !((state->_abc_1000_n282_1.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n281_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1103 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n283
  state->_abc_1000_n266.value_0_0 = state->_abc_1000_n252.value_0_0 & (!state->_abc_1000_n265.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1086 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n266
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n256.value_0_0 = !state->_abc_1000_n252.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1076 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n256
  state->_abc_1000_n253.value_0_0 = state->_abc_1000_n252.value_0_0 ^ state->_abc_1000_n246.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1073 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n253
  state->_abc_1000_n258.value_0_0 = !((state->_abc_1000_n256.value_0_0 | state->_abc_1000_n246.value_0_0) & state->_abc_1000_n257.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1078 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n258
  state->_abc_1000_n259.value_0_0 = state->_abc_1000_n258.value_0_0 ^ state->_abc_1000_n255.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1079 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n259
  state->_abc_1000_n267.value_0_0 = state->_abc_1000_n266.value_0_0 | state->_abc_1000_n264.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1087 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n267
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n268.value_0_0 = state->_abc_1000_n267.value_0_0 ^ state->_abc_1000_n262.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1088 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n268
  state->_abc_1000_n274.value_0_0 = !(state->_abc_1000_n273_1.value_0_0 ^ state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1094 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n274
  state->_abc_1000_n284_1.value_0_0 = !(state->_abc_1000_n283.value_0_0 ^ state->_abc_1000_n276_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1104 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n284_1
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n288_1.value_0_0 = !((state->_abc_1000_n283.value_0_0 | state->_abc_1000_n276_1.value_0_0) & state->_abc_1000_n287_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1108 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n288_1
  state->_abc_1000_n289.value_0_0 = state->_abc_1000_n288_1.value_0_0 ^ state->_abc_1000_n286.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1109 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n289
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n319.value_0_0 = !((state->_abc_1000_n314_1.value_0_0 | state->_abc_1000_n306_1.value_0_0) & state->_abc_1000_n318_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1139 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n319
  state->_abc_1000_n315_1.value_0_0 = !(state->_abc_1000_n314_1.value_0_0 ^ state->_abc_1000_n306_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1135 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n315_1
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n342_1.value_0_0 = !((state->_abc_1000_n341_1.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n340.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1162 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n342_1
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n320_1.value_0_0 = state->_abc_1000_n319.value_0_0 ^ state->_abc_1000_n317_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1140 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n320_1
  state->_abc_1000_n358.value_0_0 = state->_abc_1000_n357.value_0_0 ^ state->_abc_1000_n351_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1178 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n358
  state->_abc_1000_n364.value_0_0 = !((state->_abc_1000_n357.value_0_0 & state->_abc_1000_n361.value_0_0) | state->_abc_1000_n363.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1184 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n364
  state->_abc_1000_n365.value_0_0 = state->_abc_1000_n364.value_0_0 ^ state->_abc_1000_n360.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1185 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n365
  state->_abc_1000_n236.value_0_0 = !((state->_abc_1000_n235.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n234.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1056 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n236
  state->_abc_1000_n243.value_0_0 = !((state->_abc_1000_n236.value_0_0 & state->_abc_1000_n240.value_0_0) | state->_abc_1000_n242.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1063 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n243
  state->_abc_1000_n237.value_0_0 = state->_abc_1000_n236.value_0_0 ^ state->_abc_1000_n231.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1057 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n237
  state->_abc_1000_n244.value_0_0 = !(state->_abc_1000_n243.value_0_0 ^ state->_abc_1000_n239.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1064 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n244
  state->_abc_1000_n228.value_0_0 = !((state->_abc_1000_n223.value_0_0 | state->_abc_1000_n219.value_0_0) & state->_abc_1000_n227.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1048 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n228
  state->_abc_1000_n229.value_0_0 = state->_abc_1000_n228.value_0_0 ^ state->_abc_1000_n226.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1049 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n229
  yosys_simplec_set_bit_4_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n224.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1045 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [4]
  yosys_simplec_set_bit_12_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n284_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1105 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [12]
  yosys_simplec_set_bit_23_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n365.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1186 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [23]
  yosys_simplec_set_bit_26_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n394.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1215 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [26]
  yosys_simplec_set_bit_19_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n333_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1154 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [19]
  yosys_simplec_set_bit_22_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n358.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1179 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [22]
  yosys_simplec_set_bit_11_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n274.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1095 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [11]
  yosys_simplec_set_bit_18_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n327_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1148 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [18]
  yosys_simplec_set_bit_25_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n385.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1206 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [25]
  yosys_simplec_set_bit_10_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n268.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1089 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [10]
  yosys_simplec_set_bit_21_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n349.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1170 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [21]
  yosys_simplec_set_bit_17_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n320_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1141 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [17]
  yosys_simplec_set_bit_31_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n431.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1252 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [31]
  yosys_simplec_set_bit_24_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n377.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1198 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [24]
  yosys_simplec_set_bit_16_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n315_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1136 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [16]
  yosys_simplec_set_bit_30_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n426.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1247 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [30]
  yosys_simplec_set_bit_20_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n343.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1164 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [20]
  yosys_simplec_set_bit_9_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n259.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1080 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [9]
  yosys_simplec_set_bit_8_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n253.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1074 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [8]
  yosys_simplec_set_bit_29_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n418.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1239 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [29]
  yosys_simplec_set_bit_15_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n304.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1125 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [15]
  yosys_simplec_set_bit_7_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n244.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1065 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [7]
  yosys_simplec_set_bit_28_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n412.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1233 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [28]
  yosys_simplec_set_bit_14_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n297_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1118 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [14]
  yosys_simplec_set_bit_6_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n237.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1058 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [6]
  yosys_simplec_set_bit_27_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n401.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1222 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [27]
  yosys_simplec_set_bit_13_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n289.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1110 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [13]
  yosys_simplec_set_bit_5_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n229.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1050 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [5]
  state->_abc_1000_n216.value_0_0 = !((state->_abc_1000_n210.value_0_0 & state->_abc_1000_n208.value_0_0) | state->_abc_1000_n215.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1036 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n216
  state->_abc_1000_n217.value_0_0 = !(state->_abc_1000_n216.value_0_0 ^ state->_abc_1000_n213.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1037 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n217
  yosys_simplec_set_bit_3_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n217.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1038 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [3]
  state->_abc_1000_n204_1.value_0_0 = !(yosys_simplec_get_bit_1_of_32(&state->rvfi_rs2_rdata) ^ yosys_simplec_get_bit_1_of_32(&state->rvfi_rs1_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1024 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n204_1
  state->_abc_1000_n210.value_0_0 = !((state->_abc_1000_n205_1.value_0_0 | state->_abc_1000_n204_1.value_0_0) & state->_abc_1000_n209.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1030 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n210
  state->_abc_1000_n211.value_0_0 = !(state->_abc_1000_n210.value_0_0 ^ state->_abc_1000_n208.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1031 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n211
  yosys_simplec_set_bit_2_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n211.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1032 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [2]
  state->_abc_1000_n223.value_0_0 = !((state->_abc_1000_n222.value_0_0 & state->_abc_1000_n210.value_0_0) | state->_abc_1000_n221.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1043 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n223
  state->_abc_1000_n224.value_0_0 = !(state->_abc_1000_n223.value_0_0 ^ state->_abc_1000_n219.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1044 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n224
  state->_abc_1000_n252.value_0_0 = !((state->_abc_1000_n251.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n250.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1072 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n252
  state->_abc_1000_n283.value_0_0 = !((state->_abc_1000_n282_1.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n281_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1103 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n283
  state->_abc_1000_n266.value_0_0 = state->_abc_1000_n252.value_0_0 & (!state->_abc_1000_n265.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1086 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n266
  state->_abc_1000_n314_1.value_0_0 = !((state->_abc_1000_n313.value_0_0 & state->_abc_1000_n252.value_0_0) | state->_abc_1000_n312_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1134 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n314_1
  state->_abc_1000_n256.value_0_0 = !state->_abc_1000_n252.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1076 ($_NOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n256
  state->_abc_1000_n253.value_0_0 = state->_abc_1000_n252.value_0_0 ^ state->_abc_1000_n246.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1073 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n253
  state->_abc_1000_n258.value_0_0 = !((state->_abc_1000_n256.value_0_0 | state->_abc_1000_n246.value_0_0) & state->_abc_1000_n257.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1078 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n258
  state->_abc_1000_n259.value_0_0 = state->_abc_1000_n258.value_0_0 ^ state->_abc_1000_n255.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1079 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n259
  state->_abc_1000_n267.value_0_0 = state->_abc_1000_n266.value_0_0 | state->_abc_1000_n264.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1087 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n267
  state->_abc_1000_n273_1.value_0_0 = !((state->_abc_1000_n267.value_0_0 & state->_abc_1000_n261.value_0_0) | state->_abc_1000_n272_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1093 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n273_1
  state->_abc_1000_n268.value_0_0 = state->_abc_1000_n267.value_0_0 ^ state->_abc_1000_n262.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1088 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n268
  state->_abc_1000_n274.value_0_0 = !(state->_abc_1000_n273_1.value_0_0 ^ state->_abc_1000_n270_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1094 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n274
  state->_abc_1000_n284_1.value_0_0 = !(state->_abc_1000_n283.value_0_0 ^ state->_abc_1000_n276_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1104 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n284_1
  state->_abc_1000_n296_1.value_0_0 = !((state->_abc_1000_n295.value_0_0 | state->_abc_1000_n283.value_0_0) & state->_abc_1000_n294_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1116 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n296_1
  state->_abc_1000_n303_1.value_0_0 = !((state->_abc_1000_n296_1.value_0_0 & state->_abc_1000_n300_1.value_0_0) | state->_abc_1000_n302_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1123 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n303_1
  state->_abc_1000_n297_1.value_0_0 = state->_abc_1000_n296_1.value_0_0 ^ state->_abc_1000_n291_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1117 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n297_1
  state->_abc_1000_n304.value_0_0 = !(state->_abc_1000_n303_1.value_0_0 ^ state->_abc_1000_n299_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1124 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n304
  state->_abc_1000_n288_1.value_0_0 = !((state->_abc_1000_n283.value_0_0 | state->_abc_1000_n276_1.value_0_0) & state->_abc_1000_n287_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1108 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n288_1
  state->_abc_1000_n289.value_0_0 = state->_abc_1000_n288_1.value_0_0 ^ state->_abc_1000_n286.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1109 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n289
  state->_abc_1000_n326_1.value_0_0 = !((state->_abc_1000_n325.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n324_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1146 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n326_1
  state->_abc_1000_n331.value_0_0 = state->_abc_1000_n326_1.value_0_0 & (!state->_abc_1000_n322.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1151 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n331
  state->_abc_1000_n332_1.value_0_0 = state->_abc_1000_n330_1.value_0_0 & (!state->_abc_1000_n331.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1152 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n332_1
  state->_abc_1000_n333_1.value_0_0 = state->_abc_1000_n332_1.value_0_0 ^ state->_abc_1000_n329_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1153 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n333_1
  state->_abc_1000_n327_1.value_0_0 = state->_abc_1000_n326_1.value_0_0 ^ state->_abc_1000_n322.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1147 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n327_1
  state->_abc_1000_n381.value_0_0 = state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1201 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n381
  state->_abc_1000_n319.value_0_0 = !((state->_abc_1000_n314_1.value_0_0 | state->_abc_1000_n306_1.value_0_0) & state->_abc_1000_n318_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1139 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n319
  state->_abc_1000_n315_1.value_0_0 = !(state->_abc_1000_n314_1.value_0_0 ^ state->_abc_1000_n306_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1135 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n315_1
  state->_abc_1000_n376.value_0_0 = !((state->_abc_1000_n375.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n374.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1196 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n376
  state->_abc_1000_n392.value_0_0 = state->_abc_1000_n376.value_0_0 & (!state->_abc_1000_n391.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1212 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n392
  state->_abc_1000_n393.value_0_0 = state->_abc_1000_n392.value_0_0 | state->_abc_1000_n390.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1213 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n393
  state->_abc_1000_n377.value_0_0 = state->_abc_1000_n376.value_0_0 ^ state->_abc_1000_n367.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1197 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n377
  state->_abc_1000_n411.value_0_0 = !((state->_abc_1000_n410.value_0_0 & state->_abc_1000_n376.value_0_0) | state->_abc_1000_n409.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1231 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n411
  state->_abc_1000_n425.value_0_0 = !((state->_abc_1000_n424.value_0_0 | state->_abc_1000_n411.value_0_0) & state->_abc_1000_n422.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1245 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n425
  state->_abc_1000_n417.value_0_0 = !((state->_abc_1000_n411.value_0_0 | state->_abc_1000_n403.value_0_0) & state->_abc_1000_n416.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1237 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n417
  state->_abc_1000_n412.value_0_0 = !(state->_abc_1000_n411.value_0_0 ^ state->_abc_1000_n403.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1232 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n412
  state->_abc_1000_n418.value_0_0 = state->_abc_1000_n417.value_0_0 ^ state->_abc_1000_n414.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1238 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n418
  state->_abc_1000_n430.value_0_0 = !((state->_abc_1000_n425.value_0_0 & state->_abc_1000_n420.value_0_0) | state->_abc_1000_n429.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1250 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n430
  state->_abc_1000_n426.value_0_0 = !(state->_abc_1000_n425.value_0_0 ^ state->_abc_1000_n420.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1246 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n426
  state->_abc_1000_n431.value_0_0 = state->_abc_1000_n430.value_0_0 ^ state->_abc_1000_n428.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1251 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n431
  state->_abc_1000_n394.value_0_0 = state->_abc_1000_n393.value_0_0 ^ state->_abc_1000_n387.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1214 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n394
  state->_abc_1000_n400.value_0_0 = !((state->_abc_1000_n393.value_0_0 & state->_abc_1000_n398.value_0_0) | state->_abc_1000_n399.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1220 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n400
  state->_abc_1000_n401.value_0_0 = state->_abc_1000_n400.value_0_0 ^ state->_abc_1000_n397.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1221 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n401
  state->_abc_1000_n342_1.value_0_0 = !((state->_abc_1000_n341_1.value_0_0 | state->_abc_1000_n314_1.value_0_0) & state->_abc_1000_n340.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1162 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n342_1
  state->_abc_1000_n356_1.value_0_0 = state->_abc_1000_n355.value_0_0 | (!state->_abc_1000_n342_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1176 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n356_1
  state->_abc_1000_n357.value_0_0 = !(state->_abc_1000_n356_1.value_0_0 & state->_abc_1000_n354_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1177 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n357
  state->_abc_1000_n347_1.value_0_0 = state->_abc_1000_n342_1.value_0_0 & (!state->_abc_1000_n335_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1167 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n347_1
  state->_abc_1000_n343.value_0_0 = state->_abc_1000_n342_1.value_0_0 ^ state->_abc_1000_n335_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1163 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n343
  state->_abc_1000_n348_1.value_0_0 = state->_abc_1000_n346.value_0_0 & (!state->_abc_1000_n347_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1168 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n348_1
  state->_abc_1000_n349.value_0_0 = state->_abc_1000_n348_1.value_0_0 ^ state->_abc_1000_n345_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1169 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n349
  state->_abc_1000_n382.value_0_0 = state->_abc_1000_n381.value_0_0 & state->_abc_1000_n374.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1202 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n382
  state->_abc_1000_n384.value_0_0 = !((state->_abc_1000_n382.value_0_0 | state->_abc_1000_n367.value_0_0) & state->_abc_1000_n383.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1204 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n384
  state->_abc_1000_n385.value_0_0 = state->_abc_1000_n384.value_0_0 ^ state->_abc_1000_n380.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1205 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n385
  state->_abc_1000_n320_1.value_0_0 = state->_abc_1000_n319.value_0_0 ^ state->_abc_1000_n317_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1140 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n320_1
  state->_abc_1000_n358.value_0_0 = state->_abc_1000_n357.value_0_0 ^ state->_abc_1000_n351_1.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1178 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n358
  state->_abc_1000_n364.value_0_0 = !((state->_abc_1000_n357.value_0_0 & state->_abc_1000_n361.value_0_0) | state->_abc_1000_n363.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1184 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n364
  state->_abc_1000_n365.value_0_0 = state->_abc_1000_n364.value_0_0 ^ state->_abc_1000_n360.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1185 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n365
  state->_abc_1000_n236.value_0_0 = !((state->_abc_1000_n235.value_0_0 | state->_abc_1000_n223.value_0_0) & state->_abc_1000_n234.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1056 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n236
  state->_abc_1000_n243.value_0_0 = !((state->_abc_1000_n236.value_0_0 & state->_abc_1000_n240.value_0_0) | state->_abc_1000_n242.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1063 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n243
  state->_abc_1000_n237.value_0_0 = state->_abc_1000_n236.value_0_0 ^ state->_abc_1000_n231.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1057 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n237
  state->_abc_1000_n244.value_0_0 = !(state->_abc_1000_n243.value_0_0 ^ state->_abc_1000_n239.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1064 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n244
  state->_abc_1000_n228.value_0_0 = !((state->_abc_1000_n223.value_0_0 | state->_abc_1000_n219.value_0_0) & state->_abc_1000_n227.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1048 ($_OAI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n228
  state->_abc_1000_n229.value_0_0 = state->_abc_1000_n228.value_0_0 ^ state->_abc_1000_n226.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1049 ($_XOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n229
  yosys_simplec_set_bit_4_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n224.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1045 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [4]
  yosys_simplec_set_bit_12_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n284_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1105 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [12]
  yosys_simplec_set_bit_23_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n365.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1186 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [23]
  yosys_simplec_set_bit_26_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n394.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1215 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [26]
  yosys_simplec_set_bit_19_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n333_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1154 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [19]
  yosys_simplec_set_bit_22_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n358.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1179 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [22]
  yosys_simplec_set_bit_11_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n274.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1095 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [11]
  yosys_simplec_set_bit_18_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n327_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1148 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [18]
  yosys_simplec_set_bit_25_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n385.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1206 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [25]
  yosys_simplec_set_bit_10_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n268.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1089 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [10]
  yosys_simplec_set_bit_21_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n349.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1170 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [21]
  yosys_simplec_set_bit_17_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n320_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1141 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [17]
  yosys_simplec_set_bit_31_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n431.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1252 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [31]
  yosys_simplec_set_bit_24_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n377.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1198 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [24]
  yosys_simplec_set_bit_16_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n315_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1136 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [16]
  yosys_simplec_set_bit_30_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n426.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1247 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [30]
  yosys_simplec_set_bit_20_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n343.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1164 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [20]
  yosys_simplec_set_bit_9_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n259.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1080 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [9]
  yosys_simplec_set_bit_8_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n253.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1074 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [8]
  yosys_simplec_set_bit_29_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n418.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1239 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [29]
  yosys_simplec_set_bit_15_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n304.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1125 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [15]
  yosys_simplec_set_bit_7_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n244.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1065 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [7]
  yosys_simplec_set_bit_28_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n412.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1233 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [28]
  yosys_simplec_set_bit_14_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n297_1.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1118 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [14]
  yosys_simplec_set_bit_6_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n237.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1058 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [6]
  yosys_simplec_set_bit_27_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n401.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1222 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [27]
  yosys_simplec_set_bit_13_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n289.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1110 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [13]
  yosys_simplec_set_bit_5_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n229.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1050 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [5]
  state->_abc_1000_n216.value_0_0 = !((state->_abc_1000_n210.value_0_0 & state->_abc_1000_n208.value_0_0) | state->_abc_1000_n215.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1036 ($_AOI3_)
  // Updated signal in rvfi_insn_add: $abc$1000$n216
  state->_abc_1000_n217.value_0_0 = !(state->_abc_1000_n216.value_0_0 ^ state->_abc_1000_n213.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1037 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n217
  yosys_simplec_set_bit_3_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n217.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1038 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [3]
  state->_abc_1000_n206.value_0_0 = !(state->_abc_1000_n205_1.value_0_0 ^ state->_abc_1000_n204_1.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1026 ($_XNOR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n206
  yosys_simplec_set_bit_1_of_32(&state->spec_rd_wdata, state->_abc_1000_n202_1.value_0_0 & (!state->_abc_1000_n206.value_0_0)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1027 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: \spec_rd_wdata [1]
  state->_abc_1000_n492.value_0_0 = state->_abc_1000_n469.value_0_0 & (!state->_abc_1000_n491.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1312 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n492
  state->_abc_1000_n474.value_0_0 = state->_abc_1000_n473.value_0_0 | (!state->_abc_1000_n469.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1294 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n474
  state->_abc_1000_n476_1.value_0_0 = yosys_simplec_get_bit_18_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n474.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1296 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n476_1
  yosys_simplec_set_bit_18_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n474.value_0_0 ^ yosys_simplec_get_bit_18_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1295 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [18]
  yosys_simplec_set_bit_19_of_32(&state->spec_pc_wdata, state->_abc_1000_n476_1.value_0_0 ^ yosys_simplec_get_bit_19_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1297 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [19]
  state->_abc_1000_n471.value_0_0 = state->_abc_1000_n469.value_0_0 & yosys_simplec_get_bit_16_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1291 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n471
  state->_abc_1000_n473.value_0_0 = !(yosys_simplec_get_bit_17_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_16_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1293 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n473
  state->_abc_1000_n474.value_0_0 = state->_abc_1000_n473.value_0_0 | (!state->_abc_1000_n469.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1294 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n474
  state->_abc_1000_n476_1.value_0_0 = yosys_simplec_get_bit_18_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n474.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1296 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n476_1
  yosys_simplec_set_bit_18_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n474.value_0_0 ^ yosys_simplec_get_bit_18_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1295 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [18]
  yosys_simplec_set_bit_19_of_32(&state->spec_pc_wdata, state->_abc_1000_n476_1.value_0_0 ^ yosys_simplec_get_bit_19_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1297 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [19]
  yosys_simplec_set_bit_16_of_32(&state->spec_pc_wdata, state->_abc_1000_n469.value_0_0 ^ yosys_simplec_get_bit_16_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1290 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [16]
  state->_abc_1000_n480.value_0_0 = state->_abc_1000_n469.value_0_0 & (!state->_abc_1000_n479.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1300 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n480
  yosys_simplec_set_bit_17_of_32(&state->spec_pc_wdata, state->_abc_1000_n471.value_0_0 ^ yosys_simplec_get_bit_17_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1292 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [17]
  state->_abc_1000_n479.value_0_0 = state->_abc_1000_n478.value_0_0 | state->_abc_1000_n473.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1299 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n479
  state->_abc_1000_n480.value_0_0 = state->_abc_1000_n469.value_0_0 & (!state->_abc_1000_n479.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1300 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n480
  state->_abc_1000_n491.value_0_0 = state->_abc_1000_n490.value_0_0 | state->_abc_1000_n479.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1311 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n491
  state->_abc_1000_n492.value_0_0 = state->_abc_1000_n469.value_0_0 & (!state->_abc_1000_n491.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1312 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n492
  state->_abc_1000_n510.value_0_0 = yosys_simplec_get_bit_30_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n508.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1330 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n510
  yosys_simplec_set_bit_30_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n508.value_0_0 ^ yosys_simplec_get_bit_30_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1329 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [30]
  state->_abc_1000_n485.value_0_0 = state->_abc_1000_n484.value_0_0 | (!state->_abc_1000_n480.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1305 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n485
  yosys_simplec_set_bit_20_of_32(&state->spec_pc_wdata, state->_abc_1000_n480.value_0_0 ^ yosys_simplec_get_bit_20_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1301 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [20]
  state->_abc_1000_n484.value_0_0 = !(yosys_simplec_get_bit_21_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_20_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1304 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n484
  state->_abc_1000_n485.value_0_0 = state->_abc_1000_n484.value_0_0 | (!state->_abc_1000_n480.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1305 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n485
  state->_abc_1000_n490.value_0_0 = state->_abc_1000_n489.value_0_0 | state->_abc_1000_n484.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1310 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n490
  state->_abc_1000_n491.value_0_0 = state->_abc_1000_n490.value_0_0 | state->_abc_1000_n479.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1311 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n491
  state->_abc_1000_n492.value_0_0 = state->_abc_1000_n469.value_0_0 & (!state->_abc_1000_n491.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1312 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n492
  state->_abc_1000_n482.value_0_0 = state->_abc_1000_n480.value_0_0 & yosys_simplec_get_bit_20_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1302 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n482
  yosys_simplec_set_bit_21_of_32(&state->spec_pc_wdata, state->_abc_1000_n482.value_0_0 ^ yosys_simplec_get_bit_21_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1303 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [21]
  yosys_simplec_set_bit_29_of_32(&state->spec_pc_wdata, state->_abc_1000_n505.value_0_0 ^ yosys_simplec_get_bit_29_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1326 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [29]
  state->_abc_1000_n507.value_0_0 = yosys_simplec_get_bit_29_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_28_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1327 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n507
  state->_abc_1000_n505.value_0_0 = state->_abc_1000_n503.value_0_0 & yosys_simplec_get_bit_28_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1325 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n505
  yosys_simplec_set_bit_29_of_32(&state->spec_pc_wdata, state->_abc_1000_n505.value_0_0 ^ yosys_simplec_get_bit_29_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1326 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [29]
  yosys_simplec_set_bit_28_of_32(&state->spec_pc_wdata, state->_abc_1000_n503.value_0_0 ^ yosys_simplec_get_bit_28_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1324 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [28]
  state->_abc_1000_n508.value_0_0 = !(state->_abc_1000_n507.value_0_0 & state->_abc_1000_n503.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1328 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n508
  state->_abc_1000_n510.value_0_0 = yosys_simplec_get_bit_30_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n508.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1330 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n510
  yosys_simplec_set_bit_30_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n508.value_0_0 ^ yosys_simplec_get_bit_30_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1329 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [30]
  state->_abc_1000_n489.value_0_0 = !(yosys_simplec_get_bit_23_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_22_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1309 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n489
  state->_abc_1000_n490.value_0_0 = state->_abc_1000_n489.value_0_0 | state->_abc_1000_n484.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1310 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n490
  state->_abc_1000_n491.value_0_0 = state->_abc_1000_n490.value_0_0 | state->_abc_1000_n479.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1311 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n491
  state->_abc_1000_n492.value_0_0 = state->_abc_1000_n469.value_0_0 & (!state->_abc_1000_n491.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1312 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n492
  state->_abc_1000_n487.value_0_0 = yosys_simplec_get_bit_22_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n485.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1307 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n487
  yosys_simplec_set_bit_22_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n485.value_0_0 ^ yosys_simplec_get_bit_22_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1306 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [22]
  yosys_simplec_set_bit_23_of_32(&state->spec_pc_wdata, state->_abc_1000_n487.value_0_0 ^ yosys_simplec_get_bit_23_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1308 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [23]
  yosys_simplec_set_bit_31_of_32(&state->spec_pc_wdata, state->_abc_1000_n510.value_0_0 ^ yosys_simplec_get_bit_31_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1331 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [31]
  state->_abc_1000_n497.value_0_0 = state->_abc_1000_n496.value_0_0 | (!state->_abc_1000_n492.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1317 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n497
  state->_abc_1000_n494.value_0_0 = state->_abc_1000_n492.value_0_0 & yosys_simplec_get_bit_24_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1314 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n494
  state->_abc_1000_n496.value_0_0 = !(yosys_simplec_get_bit_25_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_24_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1316 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n496
  state->_abc_1000_n497.value_0_0 = state->_abc_1000_n496.value_0_0 | (!state->_abc_1000_n492.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1317 ($_ORNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n497
  state->_abc_1000_n502.value_0_0 = state->_abc_1000_n501.value_0_0 | state->_abc_1000_n496.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1322 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n502
  yosys_simplec_set_bit_24_of_32(&state->spec_pc_wdata, state->_abc_1000_n492.value_0_0 ^ yosys_simplec_get_bit_24_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1313 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [24]
  state->_abc_1000_n503.value_0_0 = state->_abc_1000_n492.value_0_0 & (!state->_abc_1000_n502.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1323 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n503
  state->_abc_1000_n505.value_0_0 = state->_abc_1000_n503.value_0_0 & yosys_simplec_get_bit_28_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1325 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n505
  yosys_simplec_set_bit_29_of_32(&state->spec_pc_wdata, state->_abc_1000_n505.value_0_0 ^ yosys_simplec_get_bit_29_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1326 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [29]
  yosys_simplec_set_bit_28_of_32(&state->spec_pc_wdata, state->_abc_1000_n503.value_0_0 ^ yosys_simplec_get_bit_28_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1324 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [28]
  state->_abc_1000_n508.value_0_0 = !(state->_abc_1000_n507.value_0_0 & state->_abc_1000_n503.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1328 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n508
  state->_abc_1000_n510.value_0_0 = yosys_simplec_get_bit_30_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n508.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1330 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n510
  yosys_simplec_set_bit_30_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n508.value_0_0 ^ yosys_simplec_get_bit_30_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1329 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [30]
  yosys_simplec_set_bit_31_of_32(&state->spec_pc_wdata, state->_abc_1000_n510.value_0_0 ^ yosys_simplec_get_bit_31_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1331 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [31]
  yosys_simplec_set_bit_25_of_32(&state->spec_pc_wdata, state->_abc_1000_n494.value_0_0 ^ yosys_simplec_get_bit_25_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1315 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [25]
  yosys_simplec_set_bit_26_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n497.value_0_0 ^ yosys_simplec_get_bit_26_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1318 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [26]
  state->_abc_1000_n501.value_0_0 = !(yosys_simplec_get_bit_27_of_32(&state->rvfi_pc_rdata) & yosys_simplec_get_bit_26_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1321 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n501
  state->_abc_1000_n502.value_0_0 = state->_abc_1000_n501.value_0_0 | state->_abc_1000_n496.value_0_0; // $abc$1000$auto$blifparse.cc:346:parse_blif$1322 ($_OR_)
  // Updated signal in rvfi_insn_add: $abc$1000$n502
  state->_abc_1000_n503.value_0_0 = state->_abc_1000_n492.value_0_0 & (!state->_abc_1000_n502.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1323 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n503
  state->_abc_1000_n505.value_0_0 = state->_abc_1000_n503.value_0_0 & yosys_simplec_get_bit_28_of_32(&state->rvfi_pc_rdata); // $abc$1000$auto$blifparse.cc:346:parse_blif$1325 ($_AND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n505
  yosys_simplec_set_bit_29_of_32(&state->spec_pc_wdata, state->_abc_1000_n505.value_0_0 ^ yosys_simplec_get_bit_29_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1326 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [29]
  yosys_simplec_set_bit_28_of_32(&state->spec_pc_wdata, state->_abc_1000_n503.value_0_0 ^ yosys_simplec_get_bit_28_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1324 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [28]
  state->_abc_1000_n508.value_0_0 = !(state->_abc_1000_n507.value_0_0 & state->_abc_1000_n503.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1328 ($_NAND_)
  // Updated signal in rvfi_insn_add: $abc$1000$n508
  state->_abc_1000_n510.value_0_0 = yosys_simplec_get_bit_30_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n508.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1330 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n510
  yosys_simplec_set_bit_30_of_32(&state->spec_pc_wdata, !(state->_abc_1000_n508.value_0_0 ^ yosys_simplec_get_bit_30_of_32(&state->rvfi_pc_rdata))); // $abc$1000$auto$blifparse.cc:346:parse_blif$1329 ($_XNOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [30]
  yosys_simplec_set_bit_31_of_32(&state->spec_pc_wdata, state->_abc_1000_n510.value_0_0 ^ yosys_simplec_get_bit_31_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1331 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [31]
  state->_abc_1000_n499.value_0_0 = yosys_simplec_get_bit_26_of_32(&state->rvfi_pc_rdata) & (!state->_abc_1000_n497.value_0_0); // $abc$1000$auto$blifparse.cc:346:parse_blif$1319 ($_ANDNOT_)
  // Updated signal in rvfi_insn_add: $abc$1000$n499
  yosys_simplec_set_bit_27_of_32(&state->spec_pc_wdata, state->_abc_1000_n499.value_0_0 ^ yosys_simplec_get_bit_27_of_32(&state->rvfi_pc_rdata)); // $abc$1000$auto$blifparse.cc:346:parse_blif$1320 ($_XOR_)
  // Updated signal in rvfi_insn_add: \spec_pc_wdata [27]
  yosys_simplec_set_bit_0_of_32(&state->spec_pc_wdata, yosys_simplec_get_bit_0_of_32(&state->rvfi_pc_rdata));
  yosys_simplec_set_bit_1_of_32(&state->spec_pc_wdata, yosys_simplec_get_bit_1_of_32(&state->rvfi_pc_rdata));
  yosys_simplec_set_bit_0_of_5(&state->spec_rd_addr, yosys_simplec_get_bit_7_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_1_of_5(&state->spec_rd_addr, yosys_simplec_get_bit_8_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_2_of_5(&state->spec_rd_addr, yosys_simplec_get_bit_9_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_3_of_5(&state->spec_rd_addr, yosys_simplec_get_bit_10_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_4_of_5(&state->spec_rd_addr, yosys_simplec_get_bit_11_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_0_of_5(&state->spec_rs1_addr, yosys_simplec_get_bit_15_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_1_of_5(&state->spec_rs1_addr, yosys_simplec_get_bit_16_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_2_of_5(&state->spec_rs1_addr, yosys_simplec_get_bit_17_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_3_of_5(&state->spec_rs1_addr, yosys_simplec_get_bit_18_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_4_of_5(&state->spec_rs1_addr, yosys_simplec_get_bit_19_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_0_of_5(&state->spec_rs2_addr, yosys_simplec_get_bit_20_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_1_of_5(&state->spec_rs2_addr, yosys_simplec_get_bit_21_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_2_of_5(&state->spec_rs2_addr, yosys_simplec_get_bit_22_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_3_of_5(&state->spec_rs2_addr, yosys_simplec_get_bit_23_of_32(&state->rvfi_insn));
  yosys_simplec_set_bit_4_of_5(&state->spec_rs2_addr, yosys_simplec_get_bit_24_of_32(&state->rvfi_insn));
}

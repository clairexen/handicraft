#ifndef MYCORE_H
#define MYCORE_H

#include "ap_int.h"

#define NUM_FACETS 4

#define RAW_FIXED_TYPE    ap_fixed<32, 24>
#define CFG_FIXED_TYPE    ap_fixed<32,  3>
#define CORDIC_FIXED_TYPE ap_fixed<32,  6>
#define VEC_FIXED_TYPE    ap_fixed<32,  4>

#define CFG_LAMDA_FC    config[0x00]
#define CFG_DELTA_LAMDA config[0x01]
#define CFG_DELTA_PHI   config[0x02]
#define CFG_D           config[0x03]

// 0 <= k < 3
#define CFG_L(k)        config[0x10 | (k)]
#define CFG_F(k)        config[0x14 | (k)]
#define CFG_O(k)        config[0x18 | (k)]

// 0 <= i < NUM_FACETS
#define CFG_LAMDA_0(i)  config[0x1c | (i)]

// 0 <= i < NUM_FACETS, 0 <= k < 3
#define CFG_A(i, k)     config[0x20 | ((i)<<2) | (k)]
#define CFG_B(i, k)     config[0x30 | ((i)<<2) | (k)]
#define CFG_C(i, k)     config[0x40 | ((i)<<2) | (k)]

#define CFG_SIZE 0x50

void mycore(const CFG_FIXED_TYPE config[CFG_SIZE], const RAW_FIXED_TYPE *data_in, VEC_FIXED_TYPE *data_out);

#endif

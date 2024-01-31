
#ifndef LSQF3_H
#define LSQF3_H

#undef FLOAT_TEST
#define DUMP_VALUES

#define MUL_SHIFT_STAGE1 (18-(25-18))
#define MUL_SHIFT_STAGE2 18

#ifndef FLOAT_TEST
#  include <ap_fixed.h>
typedef ap_fixed<18, 15> din_t;
typedef ap_fixed<18, 18> dmat_t;
typedef ap_fixed<25, 25> ddet_t;
typedef ap_fixed<18, 15> dout_t;
#else
typedef double din_t;
typedef double dmat_t;
typedef double ddet_t;
typedef double dout_t;
#endif

#ifndef DUMP_VALUES
#  define DUMP(_v) (_v)
#else
#  include <stdio.h>
#  ifndef FLOAT_TEST
#    define DUMP(_v) ({ typeof(_v) __v = _v; printf("## %12s %12.2e\n", #_v, __v.to_double()); __v; })
#  else
#    define DUMP(_v) ({ printf("## %12s %12.2e\n", #_v, _v); _v; })
#  endif
#endif

void lsqf3(din_t *input, dout_t *output);

#endif

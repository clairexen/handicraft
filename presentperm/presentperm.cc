// Usage:
//   cbmc --trace --function present_check presentperm.cc | tee presentperm.out
//   egrep '^  (value|vl|vh|zl|zh|rl|rh|result_rv..)=[^0]' presentperm.out

#include <stdio.h>
#include <assert.h>

#define RVINTRIN_EMULATE 1
#include "rvintrin.h"

uint64_t present_rv64(uint64_t v)
{
	v = _rv64_zip(v);
	v = _rv64_zip(v);
	return v;
}

uint64_t present_rv32(uint64_t v)
{
	uint32_t vl = v;
	uint32_t vh = v >> 32;

	uint32_t zl = _rv32_zip(vl);
	uint32_t zh = _rv32_zip(vh);

	uint32_t rl = _rv32_zip2(_rv32_pack(zl, zh));
	uint32_t rh = _rv32_zip2(_rv32_packu(zl, zh));

	return (uint64_t(rh) << 32) | rl;
}

extern "C" void present_check(uint64_t value)
{
	// force single-bit counter examples
	if (value & (value-1)) return;

	uint64_t result_rv64 = present_rv64(value);
	uint64_t result_rv32 = present_rv32(value);
	assert(result_rv64 == result_rv32);
}

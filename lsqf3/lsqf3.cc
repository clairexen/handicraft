
#include "lsqf3.h"
#include <math.h>

static ddet_t mul3(dmat_t a, dmat_t b, dmat_t c)
{
#ifdef FLOAT_TEST
	ddet_t ab = DUMP(a*b) / pow(2, MUL_SHIFT_STAGE1);
	return DUMP(ab*c) / pow(2, MUL_SHIFT_STAGE2);
#else
	ddet_t ab = DUMP(a*b) >> MUL_SHIFT_STAGE1;
	return DUMP(ab*c) >> MUL_SHIFT_STAGE2;
#endif
}

static ddet_t det3x3(
		dmat_t v11, dmat_t v12, dmat_t v13,
		dmat_t v21, dmat_t v22, dmat_t v23,
		dmat_t v31, dmat_t v32, dmat_t v33)
{
	ddet_t det =
			+ mul3(v11, v22, v33) + mul3(v12, v23, v31) + mul3(v13, v21, v32)
			- mul3(v31, v22, v13) - mul3(v32, v23, v11) - mul3(v33, v21, v12);
	return det;
}

void lsqf3(din_t *input, dout_t *output)
{
#pragma HLS ALLOCATION instances=sdiv limit=1 operation
#pragma HLS ALLOCATION instances=mul limit=4 operation
#pragma HLS INTERFACE ap_fifo port=input depth=512
#pragma HLS INTERFACE ap_fifo port=output depth=16

	dmat_t d1d1 = 0, d1d2 = 0, d1d3 = 0, d2d2 = 0, d2d3 = 0, d3d3 = 0;
	dmat_t d1r = 0, d2r = 0, d3r = 0;

	int input_idx = 0;
	while (1) {
		din_t d1 = input[input_idx++];
		din_t d2 = input[input_idx++];
		din_t d3 = input[input_idx++];
		din_t dr = input[input_idx++];
		if (d1 == 0 && d2 == 0 && d3 == 0 && dr == 0)
			break;
		d1d1 += DUMP(d1*d1);
		d1d2 += DUMP(d1*d2);
		d1d3 += DUMP(d1*d3);
		d2d2 += DUMP(d2*d2);
		d2d3 += DUMP(d2*d3);
		d3d3 += DUMP(d3*d3);
		d1r += DUMP(d1*dr);
		d2r += DUMP(d2*dr);
		d3r += DUMP(d3*dr);
	}

#ifdef DUMP_VALUES
	DUMP(d1d1);
	DUMP(d1d2);
	DUMP(d1d3);
	DUMP(d2d2);
	DUMP(d2d3);
	DUMP(d3d3);
	DUMP(d1r);
	DUMP(d2r);
	DUMP(d3r);
#endif

	ddet_t detM = det3x3(
			d1d1, d1d2, d1d3,
			d1d2, d2d2, d2d3,
			d1d3, d2d3, d3d3);
	ddet_t det1 = det3x3(
			d1r, d1d2, d1d3,
			d2r, d2d2, d2d3,
			d3r, d2d3, d3d3);
	ddet_t det2 = det3x3(
			d1d1, d1r, d1d3,
			d1d2, d2r, d2d3,
			d1d3, d3r, d3d3);
	ddet_t det3 = det3x3(
			d1d1, d1d2, d1r,
			d1d2, d2d2, d2r,
			d1d3, d2d3, d3r);

	output[0] = DUMP(det1) / detM;
	output[1] = DUMP(det2) / detM;
	output[2] = DUMP(det3) / DUMP(detM);
}

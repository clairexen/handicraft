#ifndef HSLBUGTST2_H
#define HSLBUGTST2_H

#include <ap_fixed.h>
#include <hls_stream.h>

struct inrecord_t {
	ap_uint<8> cnt_ls;
	ap_uint<16> pos;
};

struct outrecord_t {
	ap_uint<8> cnt_ls;
};

void hls_uut(hls::stream<inrecord_t> &pde_data_in,
		hls::stream<inrecord_t> &pdae_data_in,
		hls::stream<outrecord_t> &racalc_jobs);

#endif

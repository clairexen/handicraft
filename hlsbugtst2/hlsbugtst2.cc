
#include "hlsbugtst2.h"
#include <stdio.h>
#include <assert.h>

#define PDE_BACKLOG 5
#define PDAE_BACKLOG 2
#define MAX_DELTA_DIFF (1 << 10)

void hls_uut(hls::stream<inrecord_t> &pde_data_in,
		hls::stream<inrecord_t> &pdae_data_in,
		hls::stream<outrecord_t> &racalc_jobs)
{
#pragma HLS pipeline II=25

// #pragma HLS stream depth=256 variable=pde_data_in
// #pragma HLS stream depth=256 variable=pdae_data_in
// #pragma HLS stream depth=256 variable=racalc_jobs

	static bool next_pde_data_loaded = false;
	static bool next_pdae_data_loaded = false;

	static inrecord_t next_pde_data;
	static inrecord_t next_pdae_data;

	static inrecord_t pde_data[PDE_BACKLOG];
	static inrecord_t pdae_data[PDAE_BACKLOG];

	static ap_uint<32> pde_rrpos[PDE_BACKLOG];
	static ap_uint<32> pdae_rrpos[PDAE_BACKLOG];
	static ap_uint<32> current_ls_rrpos;

	static bool pde_data_valid[PDE_BACKLOG];
	static bool pdae_data_valid[PDAE_BACKLOG];
	static ap_uint<8> init_cnt = 0;

#pragma HLS reset variable=next_pde_data_loaded
#pragma HLS reset variable=next_pdae_data_loaded
#pragma HLS reset variable=init_cnt

	if (init_cnt < PDE_BACKLOG || init_cnt < PDAE_BACKLOG) {
		if (init_cnt < PDE_BACKLOG)
			pde_data_valid[init_cnt] = 0;
		if (init_cnt < PDAE_BACKLOG)
			pdae_data_valid[init_cnt] = 0;
		init_cnt++;
		return;
	}

	if (!next_pde_data_loaded && !pde_data_in.empty()) {
		next_pde_data = pde_data_in.read();
		next_pde_data_loaded = true;
	}

	if (!next_pdae_data_loaded && !pdae_data_in.empty())	{
		next_pdae_data = pdae_data_in.read();
		next_pdae_data_loaded = true;
	}

	if (!next_pde_data_loaded || !next_pdae_data_loaded)
		return;

	ap_uint<8> cnt_ls_diff = next_pde_data.cnt_ls - next_pdae_data.cnt_ls;

	if (cnt_ls_diff > 0 && cnt_ls_diff < 128)
	{
		next_pdae_data_loaded = false;

		for (int i = PDAE_BACKLOG-1; i > 0; i--) {
				pdae_data[i] = pdae_data[i-1];
				pdae_data_valid[i] = pdae_data_valid[i-1];
				pdae_rrpos[i] = pdae_rrpos[i-1];
			}

		pdae_data[0] = next_pdae_data;
		pdae_data_valid[0] = true;

		pdae_rrpos[0] = 0;
		if (pde_data_valid[0] && pde_data_valid[1])
			pdae_rrpos[0] = current_ls_rrpos - pdae_data[0].pos;

		int best_delta_diff = MAX_DELTA_DIFF;
		int best_partner_idx = 0;
		int best_pde_idx = 0;

		for (int i = 1; i < PDAE_BACKLOG; i++)
		{
			if (!pdae_data_valid[i])
				continue;

			for (int k = 1; k < PDE_BACKLOG; k++)
			{
				if (!pde_data_valid[k])
					continue;

				if (pde_data[k-1].cnt_ls != pde_data[k].cnt_ls)
					continue;

				int delta_diff = pde_rrpos[k-1] - pde_rrpos[k];

				if (delta_diff < best_delta_diff) {
					best_delta_diff = delta_diff;
					best_partner_idx = i;
					best_pde_idx = k;
				}
			}
		}

		outrecord_t job = {pde_data[best_pde_idx].cnt_ls};
		racalc_jobs.write(job);
	}
	else
	{
		next_pde_data_loaded = false;

		for (int i = PDE_BACKLOG-1; i > 0; i--) {
				pde_data[i] = pde_data[i-1];
				pde_data_valid[i] = pde_data_valid[i-1];
				pde_rrpos[i] = pde_rrpos[i-1];
			}

		pde_data[0] = next_pde_data;
		pde_data_valid[0] = true;

		if (pde_data_valid[1]) {
			pde_rrpos[0] = pde_rrpos[1];
			current_ls_rrpos = pde_rrpos[0] + pde_data[1].pos;
		} else {
			pde_rrpos[0] = 0;
			current_ls_rrpos = 0;
		}
	}
}

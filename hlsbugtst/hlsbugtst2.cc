#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <ap_fixed.h>
#include <hls_stream.h>

struct racalc_config_pdata {
	int sc;
	int roff;
	int rangescale;
};

class racalc_config_io {
	const racalc_config_pdata &config_pdata;
	const ap_uint<32> *config_reftab;
public:
	racalc_config_io(const racalc_config_pdata &config_pdata, const ap_uint<32> config_reftab[2048]) :
			config_pdata(config_pdata), config_reftab(config_reftab) { }

	inline ap_int<32> sc() const {
		return config_pdata.sc;
	}

	inline ap_int<32> roff() const {
		return config_pdata.roff;
	}

	inline ap_int<16> rangescale() const {
		return config_pdata.rangescale;
	}

	inline ap_int<16> nrange(int idx) const {
		assert(0 <= idx && idx < 64);
		return config_reftab[1984 + idx];
	}
};

struct racalc_in_t {
	ap_uint<1> fg_rp;
	ap_uint<32> cnt_smpl;
	ap_uint<16> del_raw_p12;
};

struct racalc_out_t {
	ap_uint<32> range;
};

struct racalc_job_t {
	racalc_in_t pde1, pdae1;
	racalc_in_t pde2, pdae2;
};

struct racalc_state_t {
	int64_t cnt_smpl_ref_sc_p13;
	int64_t del_ref_sc_p25;
};

void classical_racalc(const racalc_in_t &in, racalc_out_t &out, const racalc_config_io &cfg, racalc_state_t &state)
{
#pragma HLS inline

	int64_t sc_p13 = cfg.sc() >> 3;
	int64_t cnt_smpl_sc_p13 = int64_t(in.cnt_smpl) * sc_p13;
	int64_t del_sc_p25 = int64_t(in.del_raw_p12) * sc_p13;
	if (in.fg_rp) {
		state.cnt_smpl_ref_sc_p13 = cnt_smpl_sc_p13;
		state.del_ref_sc_p25 = del_sc_p25;
	} else {
		cnt_smpl_sc_p13 -= state.cnt_smpl_ref_sc_p13;
		del_sc_p25 -= state.del_ref_sc_p25;
	}
	out.range = (cnt_smpl_sc_p13 - (del_sc_p25 >> 12) + (int64_t(cfg.roff()) << 13)) >> 13;

	int nrange_idx = out.range >> 11;
	if (0 <= nrange_idx && nrange_idx+1 < 64) {
		int phase = out.range & (2048 - 1);
		int corr = cfg.nrange(nrange_idx) * (2048 - phase) + cfg.nrange(nrange_idx+1) * phase;
		out.range = ((int64_t(out.range) << 11) + corr) >> 11;
	}
}

void racalc_mta_stage2(
		const racalc_config_pdata &config_pdata,
		const ap_uint<32> config_reftab_1[2048],
		const ap_uint<32> config_reftab_2[2048],
		hls::stream<racalc_job_t> &racalc_jobs,
		hls::stream<racalc_out_t> &racalc_out)
{
#pragma HLS pipeline II=25

	racalc_config_io cfg_1(config_pdata, config_reftab_1);
	racalc_config_io cfg_2(config_pdata, config_reftab_2);

	if (racalc_jobs.empty())
		return;

	racalc_job_t job = racalc_jobs.read();

	racalc_state_t state1 = { /* zeros */ };
	racalc_state_t state2 = { /* zeros */ };
	racalc_out_t out_data1 = { /* zeros */ };
	racalc_out_t out_data2 = { /* zeros */ };

	classical_racalc(job.pde1, out_data1, cfg_1, state1);
	classical_racalc(job.pdae1, out_data1, cfg_1, state1);

	classical_racalc(job.pde2, out_data2, cfg_2, state2);
	classical_racalc(job.pdae2, out_data2, cfg_2, state2);

	out_data1.range = (out_data1.range + out_data2.range) / 2;

	racalc_out.write(out_data1);
}


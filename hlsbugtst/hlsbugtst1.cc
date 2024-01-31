#include <assert.h>
#include <stdint.h>
#include <ap_fixed.h>
#include <hls_stream.h>

struct axi32word_t {
	axi32word_t(ap_uint<32> data = 0, ap_uint<1> last = 0) : data(data), last(last) { }
	ap_uint<32> data;
	ap_uint<1> last;
};

struct config_pdata_t {
	int sc;
	int roff;
	int rangemin;
	int rangemax;
	int rangescale;
	int reflmin;
	int reflmax;
	int amplmin;
	int amplmax;
	int devmin;
	int devmax;
	int devmin2;
	int devmax2;
	int reflmin2;
	int rangenotchmin;
	int rangenotchmax;
	int bouncespan;
	int fiberbounceatten;
	int fiberbouncerange;
	int fiberbouncespan;
	int longrange_rangemin;
	int longrange_rangescale;
	int longrange_offset;
	int delta_refl;
};

enum flagbits_e {
	FL_RANGEMIN   = 1 << 15,
	FL_RANGEMAX   = 1 << 14,
	FL_REFLMIN    = 1 << 13,
	FL_REFLMAX    = 1 << 12,
	FL_AMPLMIN    = 1 << 11,
	FL_AMPLMAX    = 1 << 10,
	FL_DEVMIN     = 1 <<  9,
	FL_DEVMAX     = 1 <<  8,
	FL_DEVMIN2    = 1 <<  7,
	FL_DEVMAX2    = 1 <<  6,
	FL_RANGENOTCH = 1 <<  5,
	FL_REFLMIN2   = 1 <<  3,
	FL_BOUNCE     = 1 <<  2
};

class config_io {
	const config_pdata_t &config_pdata;
	const ap_uint<32> *config_reftab;
	const ap_uint<32> *config_bouncetab;
public:
	config_io(const config_pdata_t &config_pdata,
			const ap_uint<32> config_reftab[2048], const ap_uint<32> config_bouncetab[2048]) :
			config_pdata(config_pdata), config_reftab(config_reftab), config_bouncetab(config_bouncetab) { }

	inline int sc() const {
		return config_pdata.sc;
	}

	inline int roff() const {
		return config_pdata.roff;
	}

	inline int rangemin() const {
		return config_pdata.rangemin;
	}

	inline int rangemax() const {
		return config_pdata.rangemax;
	}

	inline int rangescale() const {
		return config_pdata.rangescale;
	}

	inline int reflmin() const {
		return config_pdata.reflmin;
	}

	inline int reflmax() const {
		return config_pdata.reflmax;
	}

	inline int amplmin() const {
		return config_pdata.amplmin;
	}

	inline int amplmax() const {
		return config_pdata.amplmax;
	}

	inline int devmin() const {
		return config_pdata.devmin;
	}

	inline int devmax() const {
		return config_pdata.devmax;
	}

	inline int devmin2() const {
		return config_pdata.devmin2;
	}

	inline int devmax2() const {
		return config_pdata.devmax2;
	}

	inline int reflmin2() const {
		return config_pdata.reflmin2;
	}

	inline int rangenotchmin() const {
		return config_pdata.rangenotchmin;
	}

	inline int rangenotchmax() const {
		return config_pdata.rangenotchmax;
	}

	inline int bouncespan() const {
		return config_pdata.bouncespan;
	}

	inline int fiberbounceatten() const {
		return config_pdata.fiberbounceatten;
	}

	inline int fiberbouncerange() const {
		return config_pdata.fiberbouncerange;
	}

	inline int fiberbouncespan() const {
		return config_pdata.fiberbouncespan;
	}

	inline int longrange_rangemin() const {
		return config_pdata.longrange_rangemin;
	}

	inline int longrange_rangescale() const {
		return config_pdata.longrange_rangescale;
	}

	inline int longrange_offset() const {
		return config_pdata.longrange_offset;
	}

	inline int delta_refl() const {
		return config_pdata.delta_refl;
	}

	inline int reftab(int idx) const {
		assert(0 <= idx && idx < 1984);
		return config_reftab[idx];
	}

	inline int nrange(int idx) const {
		assert(0 <= idx && idx < 64);
		return config_reftab[1984 + idx];
	}

	inline int bouncetab(int idx) const {
		assert(0 <= idx && idx < 2048);
		return config_bouncetab[idx];
	}
};

struct indata_t {
	int cnt_ls;
	int fg_rp;
	int ch_id;
	int cnt_smpl;
	int del_raw_p12;
	int ampl;
	int flags;
	int dev;
	int pre_first;
};

struct outdata_t {
	int cnt_ls;
	int fg_rp;
	int range;
	int refl;
	int ampl;
	int flags;
	int setbit29;
	int setbit28;
	int setbit27;
	int dev;
	int pre_first;
};

struct job_t {
	ap_uint<2> mode;
	indata_t pde1, pdae1;
	indata_t pde2, pdae2;
};

struct state_t {
	int64_t cnt_smpl_ref_sc_p13;
	int64_t del_ref_sc_p25;
	int bounce_r0, bounce_A0;
};

static void worker(const indata_t &in, outdata_t &out, const config_io &cfg, state_t &state)
{
#pragma HLS inline

	out.cnt_ls = in.cnt_ls;
	out.fg_rp = in.fg_rp;
	out.ampl = in.ampl;
	out.flags = in.flags;
	out.dev = in.dev;
	out.pre_first = in.pre_first;

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

	bool longrange = cfg.longrange_offset() && out.range > cfg.longrange_rangemin();
	int64_t rangescale = longrange ? cfg.longrange_rangescale() : cfg.rangescale();
	int64_t refl_subidx = ((int64_t)out.range * rangescale) >> 8;
	int64_t refl_idx = (refl_subidx >> 16) + (longrange ? cfg.longrange_offset() : 0);
	refl_subidx -= refl_idx << 16;
	out.refl = in.ampl + cfg.delta_refl();
	int refl_without_refltab = out.refl;
	if (0 <= refl_idx && refl_idx < 1983)
		out.refl -= ((cfg.reftab(refl_idx)*(65536 - refl_subidx) + cfg.reftab(refl_idx+1)*refl_subidx) >> 16);

	if (in.fg_rp) {
		state.bounce_r0 = 0;
		state.bounce_A0 = 0;
	} else if (state.bounce_A0 < out.ampl) {
		state.bounce_r0 = out.range;
		state.bounce_A0 = out.ampl;
	}

	out.setbit29 = 0;
	out.setbit28 = 0;
	out.setbit27 = 0;
	if (!in.fg_rp) {
		if (out.range < cfg.rangemin()) out.setbit29 = 1, out.flags |= FL_RANGEMIN;
		if (out.range > cfg.rangemax()) out.setbit29 = 1, out.flags |= FL_RANGEMAX;
		if (out.refl  < cfg.reflmin() ) out.setbit29 = 1, out.flags |= FL_REFLMIN;
		if (out.refl  > cfg.reflmax() ) out.setbit29 = 1, out.flags |= FL_REFLMAX;
		if (out.ampl  < cfg.amplmin() ) out.setbit29 = 1, out.flags |= FL_AMPLMIN;
		if (out.ampl  > cfg.amplmax() ) out.setbit29 = 1, out.flags |= FL_AMPLMAX;
		if (out.dev   < cfg.devmin()  ) out.setbit29 = 1, out.flags |= FL_DEVMIN;
		if (out.dev   > cfg.devmax()  ) out.setbit29 = 1, out.flags |= FL_DEVMAX;
		if (out.dev   < cfg.devmin2() ) out.setbit28 = 1, out.flags |= FL_DEVMIN2;
		if (out.dev   > cfg.devmax2() ) out.setbit28 = 1, out.flags |= FL_DEVMAX2;
		if (out.range > cfg.rangenotchmin() && out.range < cfg.rangenotchmax())
				out.setbit27 = 1, out.flags |= FL_RANGENOTCH;
		if (cfg.reflmin2() > -32768) {
			if (out.refl < cfg.reflmin2()) out.flags |= FL_REFLMIN2;
			out.refl = refl_without_refltab;
		}
		if ((out.ampl < state.bounce_A0 - cfg.bouncetab((state.bounce_r0 >> 9) < 2048 ? state.bounce_r0 >> 9 : 2047)) &&
				(2*state.bounce_r0 - cfg.bouncespan() < out.range) && (out.range < 2*state.bounce_r0 + cfg.bouncespan()))
			out.setbit29 = 1, out.flags |= FL_BOUNCE;
		if ((out.ampl < state.bounce_A0 - cfg.fiberbounceatten()) &&
				(state.bounce_r0 + 16 * cfg.fiberbouncerange() - cfg.fiberbouncespan() < out.range) &&
				(state.bounce_r0 + 16 * cfg.fiberbouncerange() + cfg.fiberbouncespan() > out.range))
			out.setbit29 = 1, out.flags |= FL_BOUNCE;
	}
}

void hlsbugtst1(
		const config_pdata_t &config_pdata,
		const ap_uint<32> config_reftab_1[2048],
		const ap_uint<32> config_reftab_2[2048],
		const ap_uint<32> config_bouncetab[2048],
		hls::stream<job_t> &jobs,
		hls::stream<outdata_t> &outdata)
{
#pragma HLS pipeline II=25
#pragma HLS latency min=50

	config_io cfg_1(config_pdata, config_reftab_1, config_bouncetab);
	config_io cfg_2(config_pdata, config_reftab_2, config_bouncetab);

	if (jobs.empty())
		return;

	job_t job = jobs.read();

	switch (job.mode)
	{
	case 0:	{
			state_t state = { /* zeros */ };
			outdata_t  out_data;

			worker(job.pde1, out_data, cfg_1, state);
			outdata.write(out_data);
		}
		break;
	case 1: {
			state_t state = { /* zeros */ };
			outdata_t  out_data;

			worker(job.pde1, out_data, cfg_1, state);
			worker(job.pdae1, out_data, cfg_1, state);
			outdata.write(out_data);
		}
		break;
	case 2: {
			state_t state1 = { /* zeros */ };
			state_t state2 = { /* zeros */ };
			outdata_t  out_data1, out_data2;

			worker(job.pde1, out_data1, cfg_1, state1);
			worker(job.pdae1, out_data1, cfg_1, state1);

			worker(job.pde2, out_data2, cfg_2, state2);
			worker(job.pdae2, out_data2, cfg_2, state2);

			out_data1.range = (out_data1.range + out_data2.range) / 2;
			out_data1.refl = (out_data1.refl + out_data2.refl) / 2;
			out_data1.ampl = (out_data1.ampl + out_data2.ampl) / 2;
			out_data1.dev = (out_data1.dev + out_data2.dev) / 2;

			outdata.write(out_data1);
		}
		break;
	}
}



#include <hls_math.h>
#include "mycore.h"

#ifndef __SYNTHESIS__

#include <map>
#include <string>

extern std::map<std::string, std::pair<double, double> > resolution_samples;

static void submit_resolution_sample(const char *key, double value)
{
	double value_log2 = log2(fabs(value) + 1e-20);
	if (resolution_samples.count(key)) {
		std::pair<double, double> &d = resolution_samples[key];
		d.first = std::min(d.first, value_log2);
		d.second = std::max(d.second, value_log2);
	} else {
		std::pair<double, double> &d = resolution_samples[key];
		d.first = d.second = value_log2;
	}
}

#else
#  define submit_resolution_sample(...) do { } while (0)
#endif


static inline void vec3_rotz(VEC_FIXED_TYPE vec_out[3], const VEC_FIXED_TYPE vec_in[3], CORDIC_FIXED_TYPE v_cos, CORDIC_FIXED_TYPE v_sin)
{
	vec_out[0] = vec_in[0]*v_cos - vec_in[1]*v_sin;
	vec_out[1] = vec_in[0]*v_sin + vec_in[1]*v_cos;
	vec_out[2] = vec_in[2];

	submit_resolution_sample("vec_in", vec_in[0]);
	submit_resolution_sample("vec_in", vec_in[1]);
	submit_resolution_sample("vec_in", vec_in[2]);

	submit_resolution_sample("vec_out", vec_out[0]);
	submit_resolution_sample("vec_out", vec_out[1]);
	submit_resolution_sample("vec_out", vec_out[2]);
}

static void my_cordic(CORDIC_FIXED_TYPE &sin_val, CORDIC_FIXED_TYPE &cos_val, CORDIC_FIXED_TYPE alpha)
{
	static const CORDIC_FIXED_TYPE PI = M_PI;

	static const CORDIC_FIXED_TYPE angles[] = {
	    0.78539816339745, 0.46364760900081, 0.24497866312686, 0.12435499454676,
	    0.06241880999596, 0.03123983343027, 0.01562372862048, 0.00781234106010,
	    0.00390623013197, 0.00195312251648, 0.00097656218956, 0.00048828121119,
	    0.00024414062015, 0.00012207031189, 0.00006103515617, 0.00003051757812,
	    0.00001525878906, 0.00000762939453, 0.00000381469727, 0.00000190734863,
	    0.00000095367432, 0.00000047683716, 0.00000023841858, 0.00000011920929
	};

	static const CORDIC_FIXED_TYPE kvalue = 0.60725293500925;

	CORDIC_FIXED_TYPE beta = 0, p = 1;
	CORDIC_FIXED_TYPE v0 = 1, v1 = 0;
	CORDIC_FIXED_TYPE k0 = kvalue, k1 = kvalue;

	// Valid alpha range: [ -2*PI ... +4*PI ]

	submit_resolution_sample("cordic_alpha", alpha);

	if (alpha < -2*PI)
		alpha += 2*PI;

	if (alpha > +2*PI)
		alpha -= 2*PI;

	if (alpha > PI) {
		alpha -= PI;
		k0 *= -1;
		k1 *= -1;
	}

	if (alpha > PI/2) {
		alpha = PI - alpha;
		k0 *= -1;
	}

	submit_resolution_sample("cordic_alpha", alpha);
	submit_resolution_sample("cordic_k0", k0);
	submit_resolution_sample("cordic_k1", k1);

	for (int i = 0; i < sizeof(angles)/sizeof(*angles); i++)
	{
		if (beta < alpha) {
			beta += angles[i];
			CORDIC_FIXED_TYPE new_v0 = v0 - p*v1;
			CORDIC_FIXED_TYPE new_v1 = v1 + p*v0;
			v0 = new_v0, v1 = new_v1;
		} else {
			beta -= angles[i];
			CORDIC_FIXED_TYPE new_v0 = v0 + p*v1;
			CORDIC_FIXED_TYPE new_v1 = v1 - p*v0;
			v0 = new_v0, v1 = new_v1;
		}
		p /= 2;

		submit_resolution_sample("cordic_beta", beta);
		submit_resolution_sample("cordic_v0", beta);
		submit_resolution_sample("cordic_v1", beta);
		submit_resolution_sample("cordic_p", beta);
	}

	sin_val = k0 * v0;
	cos_val = k1 * v1;

	submit_resolution_sample("cordic_sin", sin_val);
	submit_resolution_sample("cordic_cos", cos_val);
}

void mycore(const CFG_FIXED_TYPE config[CFG_SIZE], const RAW_FIXED_TYPE *data_in, VEC_FIXED_TYPE *data_out)
{
#pragma HLS PIPELINE II=90
#pragma HLS ALLOCATION instances=my_cordic limit=1 function
// #pragma HLS INTERFACE ap_ctrl_none port=return

#pragma HLS INTERFACE ap_memory port=config
#pragma HLS RESOURCE variable=config core=ROM_1P

// #pragma HLS INTERFACE ap_fifo depth=2 port=data_in
// #pragma HLS INTERFACE ap_fifo depth=6 port=data_out

#pragma HLS INTERFACE axis depth=2 port=data_in
#pragma HLS INTERFACE axis depth=6 port=data_out

	RAW_FIXED_TYPE lamda_raw = *data_in++;
	RAW_FIXED_TYPE phi_raw = *data_in++;
	RAW_FIXED_TYPE lamda_fc = ap_fixed<64, 32>(CFG_LAMDA_FC) << 20;

	submit_resolution_sample("lamda_raw", lamda_raw);
	submit_resolution_sample("phi_raw", phi_raw);

	ap_uint<3> facet = 0;

	for (int i = 1; i < NUM_FACETS-1; i++) {
		if (lamda_raw > lamda_fc) {
			lamda_raw -= lamda_fc;
			facet = i;
		}
	}

	RAW_FIXED_TYPE lamda_0 = ap_fixed<64, 32>(CFG_LAMDA_0(facet)) << 20;
	CORDIC_FIXED_TYPE lamda = ((lamda_raw - lamda_0) * CFG_DELTA_LAMDA) >> 10;
	CORDIC_FIXED_TYPE phi = (phi_raw * CFG_DELTA_PHI) >> 10;

	submit_resolution_sample("lamda", lamda);
	submit_resolution_sample("phi", phi);

#ifndef __SYNTHESIS__
	std::cout << "facet: " << facet << std::endl;
	std::cout << "lamda_0: " << lamda_0 << std::endl;
	std::cout << "lamda_fc: " << lamda_fc << std::endl;
	std::cout << "lamda_raw: " << lamda_raw << std::endl;
	std::cout << "CFG_LAMDA_0(facet): " << CFG_LAMDA_0(facet) << std::endl;
	std::cout << "CFG_DELTA_LAMDA: " << CFG_DELTA_LAMDA << std::endl;
	std::cout << "CFG_DELTA_PHI: " << CFG_DELTA_PHI << std::endl;
	std::cout << "lamda: " << lamda << std::endl;
	std::cout << "phi: " << phi << std::endl;
#endif

	CORDIC_FIXED_TYPE cos_lamda, sin_lamda;
	my_cordic(cos_lamda, sin_lamda, lamda);

	CORDIC_FIXED_TYPE cos_phi, sin_phi;
	my_cordic(cos_phi, sin_phi, phi);

#ifndef __SYNTHESIS__
	std::cout << "sincos(" << lamda << "): " << sin_lamda << " " << cos_lamda << std::endl;
	std::cout << "sincos(" << phi << "): " << sin_phi << " " << cos_phi << std::endl;
#endif

	VEC_FIXED_TYPE n[3];
	for (int i = 0; i < NUM_FACETS-1; i++) {
		n[i] = CFG_A(facet, i) + cos_lamda * CFG_B(facet, i) + sin_lamda * CFG_C(facet, i);
		submit_resolution_sample("n", n[i]);
	}

#ifndef __SYNTHESIS__
	std::cout << "Facet normal: " << n[0] << " " << n[1] << " " << n[2] << std::endl;
#endif

	VEC_FIXED_TYPE nl = n[0]*CFG_L(0) + n[1]*CFG_L(1) + n[2]*CFG_L(2);
	VEC_FIXED_TYPE nf = n[0]*CFG_F(0) + n[1]*CFG_F(1) + n[2]*CFG_F(2);

	submit_resolution_sample("nl", nl);
	submit_resolution_sample("nf", nf);

	VEC_FIXED_TYPE dv_unrot[3];
	VEC_FIXED_TYPE ov_unrot[3];

	for (int i = 0; i < NUM_FACETS-1; i++) {
		dv_unrot[i] = CFG_L(i) - 2*nl*n[i];
		ov_unrot[i] = CFG_O(i) + 2*(CFG_D - nf)*n[i];
		submit_resolution_sample("dv_unrot", dv_unrot[i]);
		submit_resolution_sample("ov_unrot", ov_unrot[i]);
	}

	VEC_FIXED_TYPE dv[3];
	VEC_FIXED_TYPE ov[3];

	vec3_rotz(dv, dv_unrot, cos_phi, sin_phi);
	vec3_rotz(ov, ov_unrot, cos_phi, sin_phi);

	for (int i = 0; i < 3; i++) {
		submit_resolution_sample("dv", dv[i]);
		*data_out++ = dv[i];
	}

	for (int i = 0; i < 3; i++) {
		submit_resolution_sample("ov", ov[i]);
		*data_out++ = ov[i];
	}
}

#include "hlsbugtst2.h"
#include <fstream>

#undef WRITE_EXPECTED
#define EXPECTED_FILENAME "../../../../expected.txt"

hls::stream<inrecord_t> pde_data("pde_data");
hls::stream<inrecord_t> pdae_data("pdae_data");
hls::stream<outrecord_t> racalc_jobs("racalc_jobs");

void pde_din(ap_uint<8> cnt_ls, ap_uint<16> pos)
{
	inrecord_t data;

	data.cnt_ls = cnt_ls;
	data.pos = pos;

	printf("@@ PDE_IN %d,%d\n", (int)cnt_ls, (int)pos);

	pde_data.write(data);
}

void pdae_din(ap_uint<8> cnt_ls, ap_uint<16> pos)
{
	inrecord_t data;

	data.cnt_ls = cnt_ls;
	data.pos = pos;

	printf("@@ PDAE_IN %d,%d\n", (int)cnt_ls, (int)pos);

	pdae_data.write(data);
}


void create_testdata()
{
#include "testdata.cc"
}

int main()
{
	int found_err = 0;
	create_testdata();

	while (!pde_data.empty() && !pdae_data.empty()) {
		hls_uut(pde_data, pdae_data, racalc_jobs);
	}

#ifdef WRITE_EXPECTED
	std::ofstream f(EXPECTED_FILENAME);
	while (!racalc_jobs.empty()) {
		outrecord_t data = racalc_jobs.read();
		printf("@@ RACALC %d\n", (int)data.cnt_ls);
		f << (int)data.cnt_ls << std::endl;
	}
	f.close();
#else
	std::ifstream f(EXPECTED_FILENAME);
	while (!racalc_jobs.empty()) {
		int expected_value;
		f >> expected_value;
		outrecord_t data = racalc_jobs.read();
		if (expected_value == data.cnt_ls) {
			printf("@@ RACALC %d OK\n", (int)data.cnt_ls);
		} else {
			printf("@@ RACALC %d ERROR (expected %d)\n", (int)data.cnt_ls, expected_value);
			found_err = 1;
		}
	}
	f.close();
#endif

	return found_err;
}

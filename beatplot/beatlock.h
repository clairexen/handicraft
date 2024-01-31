#ifndef BEATLOCK_H
#define BEATLOCK_H

#include <stdint.h>
#include <stdbool.h>

#define BEATLOCK_ADC_HZ 22050
#define BEATLOCK_ENERGY_DIV 220 
#define BEATLOCK_ENERGY_SAMPLES 500
#define BEATLOCK_CONV_WIN  10
#define BEATLOCK_CONV_MIN  30
#define BEATLOCK_CONV_MAX 100
#define BEATLOCK_LOWPASS_RC 1e-3
#define BEATLOCK_ENERGY_LP_RC 3e-2
#define BEATLOCK_DBLFRQ_THRESH 10

struct beatlock_state
{
	float last_sample;
	float last_energy;
	int double_freq_cnt;

	int energy_count, energy_idx;
	float energy_sum, energy_sum2;
	float energy_samples[BEATLOCK_ENERGY_SAMPLES];

	int conv_count, conv_best;
	float conv_data[BEATLOCK_CONV_MAX - BEATLOCK_CONV_MIN + 1];
	float conv_filtered[BEATLOCK_CONV_MAX - BEATLOCK_CONV_MIN + 1];

	float left_energy_sum, right_energy_sum;
	float left_energy, right_energy;
	int phase_counter;
};

void beatlock_init(struct beatlock_state *bs);
bool beatlock_sample(struct beatlock_state *bs, uint8_t sample);

#endif

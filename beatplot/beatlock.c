#include "beatlock.h"
#include <assert.h>
#include <math.h>

void beatlock_init(struct beatlock_state *bs)
{
	bs->last_sample = 0;
	bs->double_freq_cnt = 0;
	bs->energy_count = 0;
	bs->energy_idx = 0;
	bs->energy_sum = 0;
	bs->energy_sum2 = 0;
	for (int i = 0; i < BEATLOCK_ENERGY_SAMPLES; i++)
		bs->energy_samples[i] = 0;
	bs->conv_count = 0;
	bs->conv_best = 0;
	for (int i = 0; i < BEATLOCK_CONV_MAX - BEATLOCK_CONV_MIN + 1; i++)
		bs->conv_data[i] = 0;
	bs->left_energy = 0;
	bs->right_energy = 0;
	bs->left_energy_sum = 0;
	bs->right_energy_sum = 0;
	bs->phase_counter = 0;
}

bool beatlock_sample(struct beatlock_state *bs, uint8_t sample)
{
	// low pass filter
	double dt = 1.0 / 22050;
	double alpha = dt / (BEATLOCK_LOWPASS_RC + dt);
	bs->last_sample = alpha*sample + (1-alpha)*(bs->last_sample);

	// add sample to signal energy buffer
	bs->energy_sum += bs->last_sample;
	bs->energy_sum2 += bs->last_sample*bs->last_sample;
	if (++bs->energy_count < BEATLOCK_ENERGY_DIV)
		return false;

	// caluclate signal energy from buffered sums
	float this_energy = bs->energy_sum2 - bs->energy_sum * bs->energy_sum / BEATLOCK_ENERGY_DIV;
	bs->energy_sum = bs->energy_sum2 = 0;
	bs->energy_count = 0;

	// low-pass filter the energy
	dt = 1.0 / (22050.0 / BEATLOCK_ENERGY_DIV);
	alpha = dt / (BEATLOCK_ENERGY_LP_RC + dt);
	bs->last_energy = alpha*this_energy + (1-alpha)*(bs->last_energy);

	// store filtered energy level
	bs->energy_idx = (bs->energy_idx + 1) % BEATLOCK_ENERGY_SAMPLES;
	bs->energy_samples[bs->energy_idx] = bs->last_energy;

	// calculate the value of the convolution for one phase
	float conv_sum = 0;
	for (int16_t i = 0; i < BEATLOCK_ENERGY_SAMPLES - (BEATLOCK_CONV_MIN + bs->conv_count); i++) {
		int16_t idx1 = bs->energy_idx - (BEATLOCK_CONV_MIN + bs->conv_count + i);
		int16_t idx2 = bs->energy_idx - i;
		if (idx1 < 0)
			idx1 += BEATLOCK_ENERGY_SAMPLES;
		if (idx2 < 0)
			idx2 += BEATLOCK_ENERGY_SAMPLES;
		conv_sum += bs->energy_samples[idx1] * bs->energy_samples[idx2];
	}
	conv_sum /= BEATLOCK_ENERGY_SAMPLES - (BEATLOCK_CONV_MIN + bs->conv_count);

	// update conv_data using moving average 
	bs->conv_data[bs->conv_count] = bs->conv_data[bs->conv_count]*0.9 + conv_sum*0.1;
	bs->conv_count = (bs->conv_count + 1) % (BEATLOCK_CONV_MAX - BEATLOCK_CONV_MIN + 1);

	// calculate conv_filtered and find maximum
	if (bs->conv_count == 0)
	{
		int16_t i = (bs->conv_best + BEATLOCK_CONV_MIN) / 2 - BEATLOCK_CONV_MIN;
		if (i > 0 && bs->conv_data[i] > bs->conv_data[bs->conv_best]*0.75)
			bs->double_freq_cnt++;
		else if (bs->double_freq_cnt > 0)
			bs->double_freq_cnt--;
		if (bs->double_freq_cnt > BEATLOCK_DBLFRQ_THRESH) {
			bs->conv_best = i;
			bs->double_freq_cnt = 0;
		}

		float conv_max = bs->conv_data[0], conv_min = bs->conv_data[0];
		for (int16_t i = 0; i < BEATLOCK_CONV_MAX - BEATLOCK_CONV_MIN + 1; i++) {
			conv_max = conv_max > bs->conv_data[i] ? conv_max : bs->conv_data[i];
			conv_min = conv_min < bs->conv_data[i] ? conv_min : bs->conv_data[i];
		}
		for (int16_t i = 0; i < BEATLOCK_CONV_MAX - BEATLOCK_CONV_MIN + 1; i++) {
			bs->conv_filtered[i] = bs->conv_data[i] - conv_min;
			if (i < bs->conv_best - BEATLOCK_CONV_WIN)
				bs->conv_filtered[i] *= 0.9;
			if (i > bs->conv_best + BEATLOCK_CONV_WIN)
				bs->conv_filtered[i] *= 0.6;
				
		}
		for (int16_t i = 0; i < BEATLOCK_CONV_MAX - BEATLOCK_CONV_MIN + 1; i++)
			if (bs->conv_filtered[i] > bs->conv_filtered[bs->conv_best])
				bs->conv_best = i;
	}

	// trigger beat and correct phase
	bs->phase_counter--;
	if (bs->phase_counter < (BEATLOCK_CONV_MIN + bs->conv_best)/2)
		bs->left_energy_sum += bs->energy_samples[bs->energy_idx];
	else
		bs->right_energy_sum += bs->energy_samples[bs->energy_idx];
	if (bs->phase_counter <= 0) {
		bs->phase_counter += bs->conv_best + BEATLOCK_CONV_MIN;
		if (bs->left_energy_sum > bs->right_energy_sum)
			bs->phase_counter--;
		if (bs->left_energy_sum < bs->right_energy_sum)
			bs->phase_counter++;
		bs->left_energy = bs->left_energy_sum;
		bs->right_energy = bs->right_energy_sum;
		bs->left_energy_sum = 0;
		bs->right_energy_sum = 0;
		return true;
	}

	return false;
}


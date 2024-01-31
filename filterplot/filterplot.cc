// Little filter plotter for validating digital filters by running time-domain tests.
//
//  Copyright (C) 2012  Clifford Wolf <clifford@clifford.at>
//
//  Permission to use, copy, modify, and/or distribute this software for any
//  purpose with or without fee is hereby granted, provided that the above
//  copyright notice and this permission notice appear in all copies.
//
// clang -o filterplot -Wall -Werror -std=c++11 filterplot.cc -lstdc++ && ./filterplot

#include <math.h>
#include <stdio.h>
#include <stdarg.h>

#include <vector>
#include <string>

struct FilterBase
{
	virtual ~FilterBase() { };
	virtual double operator()(double sample) { return sample; }
};

struct FirType1 : FilterBase
{
	std::vector<double> state;
	std::vector<double> coefficients;
	void addCoefficient(double coeff) { coefficients.push_back(coeff); state.push_back(0); }
	virtual double operator()(double sample) {
		state.pop_back();
		state.insert(state.begin(), sample);
		sample = 0;
		for (int i = 0; i < int(state.size()); i++)
			sample += coefficients[i]*state[i];
		return sample;
	}
};

typedef std::pair<double, double> plot_sample_t;
typedef std::pair<std::string, std::vector<plot_sample_t>> plot_trace_t;

std::string stringf(const char *fmt, ...)
{
	std::string string;
	char *str = NULL;
	va_list ap;

	va_start(ap, fmt);
	if (vasprintf(&str, fmt, ap) < 0)
		str = NULL;
	va_end(ap);

	if (str != NULL) {
		string = str;
		free(str);
	}

	return string;
}

void plot(FILE *f, std::vector<plot_trace_t> data, std::string opts = std::string(), std::string prep = std::string())
{
	FILE *gp = f ? f : popen("gnuplot -persist - > /dev/null 2>&1", "w");

	if (!prep.empty())
		fprintf(gp, "%s", prep.c_str());

	fprintf(gp, "plot%s", opts.c_str());
	for (size_t i = 0; i < data.size(); i++)
		fprintf(gp, "%s'-' %s", i ? ", " : "", data[i].first.c_str());
	fprintf(gp, "\n");

	for (size_t i = 0; i < data.size(); i++) {
		for (size_t j = 0; j < data[i].second.size(); j++)
			fprintf(gp, "%g %g\n", data[i].second[j].first, data[i].second[j].second);
		fprintf(gp, "e\n");
	}

	if (f == NULL)
		pclose(gp);
}
void testFilter(FilterBase &filter, double fromFreq, double toFreq)
{
	std::vector<plot_sample_t> plot_samples_ampl;
	std::vector<plot_sample_t> plot_samples_phase;

	// for (double freq = fromFreq; freq <= toFreq; freq *= pow(toFreq/fromFreq, 1e-3))
	for (double freq = fromFreq; freq <= toFreq; freq += (toFreq-fromFreq) * 1e-3)
	{
		double omega = 2 * M_PI * freq;

		for (int i = 0; i < 100/freq; i++)
			filter(sin(i*omega));

		double sum_count = 0;
		double sum_sin = 0, sum_cos = 0;
		for (int i = 100/freq; i < 200/freq; i++, sum_count++) {
			double val_sin = sin(i*omega), val_cos = cos(i*omega);
			double val_out = filter(val_sin);
			sum_sin += val_sin * val_out;
			sum_cos += val_cos * val_out;
		}
		sum_sin /= sum_count;
		sum_cos /= sum_count;

		plot_samples_ampl.push_back(plot_sample_t(freq, sqrt(sum_sin*sum_sin + sum_cos*sum_cos)));
		plot_samples_phase.push_back(plot_sample_t(freq, atan2(sum_cos, sum_sin)));
	}

	std::vector<plot_trace_t> plot_data;
	plot_data.push_back(plot_trace_t("with lines title 'amplitude'", plot_samples_ampl));
	plot_data.push_back(plot_trace_t("with lines title 'phase' axes x1y2", plot_samples_phase));

	// plot(NULL, plot_data, "", "set logscale xy\n");
	plot(NULL, plot_data);
}

int main()
{
	// scipy.signal.fir_filter_design.firwin(9, 1.0/16)
	FirType1 filter;
	filter.addCoefficient(0.01666867);
	filter.addCoefficient(0.04687035);
	filter.addCoefficient(0.12178376);
	filter.addCoefficient(0.19896325);
	filter.addCoefficient(0.23142796);
	filter.addCoefficient(0.19896325);
	filter.addCoefficient(0.12178376);
	filter.addCoefficient(0.04687035);
	filter.addCoefficient(0.01666867);

	testFilter(filter, 1.0/(16*16), 0.9);

	return 0;
}


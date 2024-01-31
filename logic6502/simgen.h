#ifndef SIMGEN_H
#define SIMGEN_H

#include <map>
#include <string>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

static uint32_t xorshift32() {
	static uint32_t x = 1;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return x;
}

struct simgen_node_t {
	int output, num_inputs;
	char type; // A=And O=Or N=Not 0=sw0 1=sw1 S=Sr
	const int inputs[16];
};

static bool simgen_worker(bool *state, const simgen_node_t *nodes, int num_nodes, bool update_sr)
{
	bool did_something = false;

	for (int i = 0; i < num_nodes; i++) {
		const simgen_node_t &node = nodes[i];
		bool new_value;
		if (update_sr != (node.type == 'S'))
			continue;
		if (node.type == 'A') {
			new_value = true;
			for (int j = 0; j < node.num_inputs; j++)
				if (!state[node.inputs[j]])
					new_value = false;
		} else
		if (node.type == 'O') {
			new_value = false;
			for (int j = 0; j < node.num_inputs; j++)
				if (state[node.inputs[j]])
					new_value = true;
		} else
		if (node.type == 'N' && node.num_inputs == 1) {
			new_value = !state[node.inputs[0]];
		} else
		if (node.type == '0' && node.num_inputs == 1) {
			new_value = state[node.inputs[0]] ? false : state[node.output];
		} else
		if (node.type == '1' && node.num_inputs == 1) {
			new_value = state[node.inputs[0]] ? true : state[node.output];
		} else
		if (node.type == 'S' && node.num_inputs == 2) {
			new_value = state[node.inputs[0]] ? false :
			            state[node.inputs[1]] ? true : state[node.output];
		} else
			abort();
		if (new_value != state[node.output]) {
			state[node.output] = new_value;
			did_something = true;
		}
	}

	return did_something;
}

void simgen_dump(bool *state, const char **netnames, int num_signals, const char *filename)
{
	FILE *f = fopen(filename, "w");
	if (f == NULL) {
		fprintf(stderr, "Can't open file %s for writing!\n", filename);
		exit(1);
	}
	for (int i = 0; i < num_signals; i++)
		fprintf(f, "%30s %d\n", netnames[i], state[i]);
	fclose(f);
}

void simgen_load(bool *state, const char **netnames, int num_signals, const char *filename)
{
	std::map<std::string, bool> data;
	FILE *f = fopen(filename, "r");
	if (f == NULL) {
		fprintf(stderr, "Can't open file %s for reading!\n", filename);
		exit(1);
	}
	char buffer[100];
	while (fgets(buffer, 100, f) != NULL)
	{
		char *p = strtok(buffer, " \r\n\t");
		if (p && *p == 0)
			p = strtok(NULL, " \r\n\t");
		char *p2 = strtok(NULL, " \r\n\t");
		if (!p || !*p || !p2 || !*p2) {
			fprintf(stderr, "Syntax error in load file: %s\n", buffer);
			exit(1);
		}
		data[p] = atoi(p2);
	}
	fclose(f);
	for (int i = 0; i < num_signals; i++)
		if (data.count(netnames[i]) > 0) {
			state[i] = data.at(netnames[i]);
			data.erase(netnames[i]);
		}
	// if (data.size() > 0) {
	// 	for (std::map<std::string, bool>::iterator it = data.begin(); it != data.end(); it++)
	// 		fprintf(stderr, "Invalid net name in load file: %s\n", it->first.c_str());
	// 	exit(1);
	// }
}

void simgen_vcd_init(int vcd_time, bool *state, const char **netnames, int num_signals, FILE *f)
{
	fprintf(f, "$timescale 1 ns $end\n");
	fprintf(f, "$var reg 1 s STABLE $end\n");
	fprintf(f, "$var reg 32 m MARK $end\n");
	for (int i = 0; i < num_signals; i++)
		fprintf(f, "$var reg 1 n%d %s $end\n", i, netnames[i]);
	fprintf(f, "$enddefinitions $end\n");
	fprintf(f, "#%d\n", vcd_time++);
	fprintf(f, "$dumpvars\n");
	fprintf(f, "0s\n");
	fprintf(f, "bx m\n");
	for (int i = 0; i < num_signals; i++)
		fprintf(f, "%dn%d\n", state[i], i);
	fprintf(f, "$end\n");
}

void simgen_vcd_step(int &vcd_time, bool *state, int num_signals, FILE *f, bool stable, int mark)
{
	fprintf(f, "#%d\n", vcd_time++);
	for (int i = 0; i < num_signals; i++)
		fprintf(f, "%dn%d\n", state[i], i);
	if (mark >= 0) {
		fprintf(f, "b");
		for (int i = 30; i >= 0; i--)
			fprintf(f, "%c", (mark >> i) & 1 ? '1' : '0');
		fprintf(f, " m\n");
	}
	if (stable) {
		fprintf(f, "#%d\n", vcd_time++);
		fprintf(f, "1s\n");
		fprintf(f, "#%d\n", vcd_time++);
		fprintf(f, "0s\n");
	} else
		if (mark >= 0)
			fprintf(f, "#%d\n", vcd_time++);
	if (mark >= 0)
		fprintf(f, "bx m\n");
}

#endif

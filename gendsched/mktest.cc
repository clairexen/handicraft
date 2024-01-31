#include <stdio.h>
#include <stdint.h>
#include <string>
#include <vector>

struct port_info {
	std::string name;
	int width;
	bool is_signed;
	std::vector<int> cycle_var_map;
};

struct var_info {
	std::string name;
	std::pair<int,int> in_port_cycle;
	std::vector<std::pair<int,int>> out_port_cycle;
};

int num_cycles;
std::vector<port_info> inputs, outputs;
std::vector<var_info> variables;
std::vector<std::pair<int,int>> free_in_port_cycle, free_out_port_cycle;

struct xorshift32
{
	uint32_t state;
	xorshift32(uint32_t state = 314159265) : state(state) { }
	void reset(uint32_t st) {
		state = st ? st : ~0;
		(*this)();
		(*this)();
		(*this)();
	}
	uint32_t operator() () {
		state ^= state << 13;
		state ^= state >> 17;
		state ^= state << 5;
		return state;
	}
	uint32_t operator() (uint32_t limit) {
		return (*this)() % limit;
	}
};

int find_near_out_cycle(xorshift32 &xs, int ref_cycle, int iter)
{
	int idx = xs(free_out_port_cycle.size());
	for (int i = 1; i < iter; i++) {
		int new_idx = xs(free_out_port_cycle.size());
		if ((free_out_port_cycle[idx].second+num_cycles-ref_cycle)%num_cycles > (free_out_port_cycle[new_idx].second+num_cycles-ref_cycle)%num_cycles)
			idx = new_idx;
	}
	return idx;
}

int main(int argc, char **argv)
{
	xorshift32 xs;

	if (argc > 1)
		xs.reset(atoi(argv[1]));

	num_cycles = xs(1000) + 1000;

	int num = xs(6)+5;
	for (int i = 0; i < num; i++) {
		port_info info;
		info.name = std::string("IN_") + std::to_string(i);
		info.width = 6 + xs(11);
		info.is_signed = xs(2);
		info.cycle_var_map.insert(info.cycle_var_map.end(), num_cycles, -1);
		for (int j = 0; j < num_cycles; j++)
			free_in_port_cycle.push_back(std::pair<int,int>(i, j));
		inputs.push_back(info);
	}

	num = xs(6)+5;
	for (int i = 0; i < num; i++) {
		port_info info;
		info.name = std::string("OUT_") + std::to_string(i);
		info.width = 6 + xs(11);
		info.is_signed = xs(2);
		info.cycle_var_map.insert(info.cycle_var_map.end(), num_cycles, -1);
		for (int j = 0; j < num_cycles; j++)
			free_out_port_cycle.push_back(std::pair<int,int>(i, j));
		outputs.push_back(info);
	}

	num = 100*xs(10) + 100;
	for (int i = 0; i < num; i++) {
		var_info info;
		info.name = std::string("VAR_") + std::to_string(i);
		int k = xs(free_in_port_cycle.size());
		info.in_port_cycle = free_in_port_cycle[k];
		free_in_port_cycle[k] = free_in_port_cycle.back();
		free_in_port_cycle.pop_back();
		k = find_near_out_cycle(xs, info.in_port_cycle.second, 10);
		info.out_port_cycle.push_back(free_out_port_cycle[k]);
		free_out_port_cycle[k] = free_out_port_cycle.back();
		free_out_port_cycle.pop_back();
		variables.push_back(info);
	}

	num = (free_out_port_cycle.size()/10) * xs(10);
	for (int i = 0; i < num; i++) {
		int j = xs(variables.size());
		int k = find_near_out_cycle(xs, variables[j].in_port_cycle.second, 10);
		variables[j].out_port_cycle.push_back(free_out_port_cycle[k]);
		free_out_port_cycle[k] = free_out_port_cycle.back();
		free_out_port_cycle.pop_back();
	}

	for (int i = 0; i < variables.size(); i++) {
		var_info &info = variables[i];
		inputs[info.in_port_cycle.first].cycle_var_map[info.in_port_cycle.second] = i;
		for (auto &port_cycle : info.out_port_cycle)
			outputs[port_cycle.first].cycle_var_map[port_cycle.second] = i;
	}

	for (auto &it : inputs)
		printf(".input %s %s %d\n", it.name.c_str(), it.is_signed ? "signed" : "unsigned", it.width);
	for (auto &it : outputs)
		printf(".output %s %s %d\n", it.name.c_str(), it.is_signed ? "signed" : "unsigned", it.width);

	printf("#");
	for (auto &it : inputs)
		printf("\t%s", it.name.c_str());
	for (auto &it : outputs)
		printf("\t%s", it.name.c_str());
	printf("\n");

	for (int i = 0; i < num_cycles; i++) {
		printf("%d", i);
		for (auto &it : inputs)
			printf("\t%s", it.cycle_var_map[i] < 0 ? "-" : variables[it.cycle_var_map[i]].name.c_str());
		for (auto &it : outputs)
			printf("\t%s", it.cycle_var_map[i] < 0 ? "-" : variables[it.cycle_var_map[i]].name.c_str());
		printf("\n");
	}

	printf(".end\n");
	return 0;
}


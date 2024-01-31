#ifndef GENDSCHED_H
#define GENDSCHED_H

#include <stdio.h>
#include <string>
#include <vector>
#include <set>

struct transfer
{
	int in_port, out_port;
	int in_cycle, out_cycle;
};

struct edge
{
	std::set<int> in_ports, out_ports;
	std::vector<transfer> transfers;
};

struct port
{
	std::string name;
	bool is_signed, is_input;
	int width;
};

struct config
{
	int num_cycles;
	std::vector<port> ports;
	std::vector<edge> edges;
};

void parse(config &cfg, FILE *f);

#endif

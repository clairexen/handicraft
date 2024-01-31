#include "gatesynth.h"

aitree::aitree()
{
	parents[0] = NULL;
	parents[1] = NULL;
	is_neg = 0;
}

aitree::aitree(std::string id)
{
	parents[0] = NULL;
	parents[1] = NULL;
	atom_id = id;
	is_neg = 0;
}

aitree::aitree(aitree *p1, aitree *p2)
{
	parents[0] = p1;
	parents[1] = p2;
	is_neg = 0;
}

aitree::~aitree()
{
	if (parents[0])
		delete parents[0];
	if (parents[1])
		delete parents[1];
}

void aitree::invert()
{
	is_neg = !is_neg;
}

std::string aitree::dump_aitree()
{
	if (parents[0])
		return "(" + parents[0]->dump_aitree() + "*" + parents[1]->dump_aitree() + ")" + std::string(is_neg ? "'" : "");
	if (!atom_id.empty())
		return atom_id + std::string(is_neg ? "'" : "");
	return is_neg ? "0" : "1";
}

std::string aitree::dump_mapped(bool inverted)
{
	std::string ret;

	if (mapped_gate_invert[inverted])
		ret += "not(";

	if (mapped_gate[inverted] == NULL)
		ret += atom_id;
	else {
		ret += mapped_gate[inverted]->name;
		for (int i = 0; i < 4; i++)
			if (mapped_parents[inverted][i] != NULL) {
				ret += i > 0 ? "," : "(";
				ret += mapped_parents[inverted][i]->dump_mapped(mapped_parents_invert[inverted][i]);
			}
		ret += ")";
	}

	if (mapped_gate_invert[inverted])
		ret += ")";

	return ret;
}

#define INVALID_COST 1000000

void aitree::map(std::vector<struct gate*> &gates, int invert_cost)
{
	for (int i = 0; i < 2; i++) {
		mapped_cost[i] = 0;
		mapped_gate[i] = 0;
		mapped_gate_invert[i] = false;
		for (int j = 0; j < 4; j++)
			mapped_parents[i][j] = NULL, mapped_parents_invert[i][j] = false;
	}

	if (!parents[0]) {
		mapped_cost[0] = 0;
		mapped_cost[1] = invert_cost;
		mapped_gate_invert[1] = true;
		return;
	}

	parents[0]->map(gates, invert_cost);
	parents[1]->map(gates, invert_cost);

	for (int i = 0; i < 2; i++)
	{
		mapped_cost[i] = INVALID_COST;
		mapped_gate_invert[i] = false;
		for (auto &it : gates) {
			struct aitree *map_parents[4] = { NULL, NULL, NULL, NULL };
			bool map_parents_invert[4] = { false, false, false };
			int cost = map_match(it->pattern, map_parents, map_parents_invert, i) + it->cost;
			if (cost < mapped_cost[i]) {
				for (int j = 0; j < 4; j++) {
					mapped_parents[i][j] = map_parents[j];
					mapped_parents_invert[i][j] = map_parents_invert[j];
				}
				mapped_gate[i] = it;
				mapped_cost[i] = cost;
			}
		}
	}

	for (int i = 0; i < 2; i++) {
		if (mapped_cost[i] > mapped_cost[!i]+invert_cost) {
			for (int j = 0; j < 4; j++) {
				mapped_parents[i][j] = mapped_parents[!i][j];
				mapped_parents_invert[i][j] = mapped_parents_invert[!i][j];
			}
			mapped_gate[i] = mapped_gate[!i];
			mapped_cost[i] = mapped_cost[!i] + invert_cost;
			mapped_gate_invert[i] = true;
		}
		assert(mapped_cost[i] < INVALID_COST);
	}

	// printf("DEBUG: %s -> %s(%d) %s(%d)\n", dump_aitree().c_str(), mapped_gate[0]->name.c_str(), mapped_cost[0], mapped_gate[1]->name.c_str(), mapped_cost[1]);
}

int aitree::map_match(struct aitree *pattern, struct aitree *map_parents[4], bool map_parents_invert[4], int root_neg)
{
	if (pattern->parents[0] == NULL) {
		int idx = pattern->atom_id[0] - 'a';
		map_parents[idx] = this;
		if (pattern->is_neg == is_neg) {
			map_parents_invert[idx] = false;
			return mapped_cost[0];
		} else {
			map_parents_invert[idx] = true;
			return mapped_cost[1];
		}
	}

	if (parents[0] == NULL || pattern->is_neg != (root_neg < 0 ? is_neg : root_neg))
		return INVALID_COST;

	return parents[0]->map_match(pattern->parents[0], map_parents, map_parents_invert) + parents[1]->map_match(pattern->parents[1], map_parents, map_parents_invert);
}


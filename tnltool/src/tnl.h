/*
 *  tnltool - Transistor Netlist Tool
 *
 *  Copyright (C) 2011 Clifford Wolf <clifford@clifford.at> and
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

#ifndef TNLTOOL_TNL_H
#define TNLTOOL_TNL_H

#include <map>
#include <set>
#include <vector>
#include <string>

struct Tnl
{
	int numStates;
	std::map<std::string, int> str2state;
	std::map<int, std::string> state2str;
	std::map<std::pair<int, int>, int> stateCombine;
	std::map<int, int> stateAscend;
	int weakestState;

	struct SwitchType
	{
		std::string id;
		std::set<int> onStates;
		std::set<int> offStates;
	};
	std::map<std::string, SwitchType*> switchTypes;

	struct Switch;
	struct Net
	{
		int num;
		std::set<std::string> ids;
		std::set<Switch*> switchGates;
		std::set<Switch*> switchCC;
		int driveState, biasState;
		bool supply, graph;

		Net() {
			num = -1;
			driveState = -1;
			biasState = -1;
			supply = false;
			graph = false;
		}
	};
	std::set<Net*> nets;
	std::map<std::string, Net*> id2net;
	std::vector<Net*> netByNum;

	struct Switch
	{
		int num;
		std::set<std::string> ids;
		SwitchType *type;
		Net *gate;
		Net *cc[2];
		bool graph;

		Switch() {
			num = -1;
			type = NULL;
			gate = NULL;
			cc[0] = NULL;
			cc[1] = NULL;
			graph = false;
		}
	};
	std::set<Switch*> switches;
	std::map<std::string, Switch*> id2switch;
	std::vector<Switch*> switchByNum;

	std::map<std::string, std::vector<std::string>> options;
	std::map<std::string, bool> options_used;

	int numNets;
	int numSwitches;
	int numSupplies;
	int maxCcPerNet;
	int maxGatesPerNet;

	Tnl() {
		numStates = 0;
		weakestState = -1;
		numNets = -1;
		numSupplies = -1;
		maxCcPerNet = -1;
		maxGatesPerNet = -1;
	}

	~Tnl() {
		for (auto i = switchTypes.begin(); i != switchTypes.end(); i++)
			delete i->second;
		for (auto i = nets.begin(); i != nets.end(); i++)
			delete *i;
		for (auto i = switches.begin(); i != switches.end(); i++)
			delete *i;
	}

	// accessing options (implemented in tnl_parser.y)
	std::string opt(std::string key, std::string def);
	bool checkopt(std::string key);

	// simple checks and analytics (checks.cc)
	void check();
	void mergeRedundant();
	void enumerate();

	// solve network for init state (bias.cc)
	void bias();
	void bias_findGroup(Net *net, std::set<Net*> &group, int &state);

	// helper for network graphs (graph.cc)
	struct GraphInfo {
		std::map<Net*, std::vector<std::string>> netInfo;
		std::map<Switch*, std::vector<std::string>> switchInfo;
	};
	bool graph(GraphInfo *info);

	// backend functionality
	void backend_csim(std::string filename_prefix);
};

#endif

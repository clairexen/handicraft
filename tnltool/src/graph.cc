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

#include "tnl.h"

#include <queue>
#include <utility>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

bool Tnl::graph(Tnl::GraphInfo *info)
{
	for (auto i = nets.begin(); i != nets.end(); i++)
		(*i)->graph = false;
	for (auto i = switches.begin(); i != switches.end(); i++)
		(*i)->graph = false;

	std::map<int, std::string> colorMap;
	options_used["graph-colors"] = true;
	for (auto i = options["graph-colors"].begin(); i != options["graph-colors"].end(); i++)
	{
		std::string stateId, colorId;
		size_t split = i->find(":");
		if (split == std::string::npos) {
			fprintf(stderr, "Can't parse graph color mapping `%s'.\n", i->c_str());
			exit(1);
		}
		stateId = i->substr(0, split);
		colorId = i->substr(split+1);
		if (str2state.count(stateId) == 0) {
			fprintf(stderr, "Can't find state `%s' from color Mapping `%s'.\n", stateId.c_str(), i->c_str());
			exit(1);
		}
		colorMap[str2state[stateId]] = colorId;
	}

	std::map<int, std::string> initindMap;
	options_used["graph-initindicator"] = true;
	for (auto i = options["graph-initindicator"].begin(); i != options["graph-initindicator"].end(); i++)
	{
		std::string stateId, initindId;
		size_t split = i->find(":");
		if (split == std::string::npos) {
			fprintf(stderr, "Can't parse graph initindicator mapping `%s'.\n", i->c_str());
			exit(1);
		}
		stateId = i->substr(0, split);
		initindId = i->substr(split+1);
		if (str2state.count(stateId) == 0) {
			fprintf(stderr, "Can't find state `%s' from initindicator Mapping `%s'.\n", stateId.c_str(), i->c_str());
			exit(1);
		}
		initindMap[str2state[stateId]] = initindId;
	}

	std::set<Net*> stopNets;
	std::set<Net*> globNets;
	options_used["graph-stop"] = true;
	for (auto i = options["graph-stop"].begin(); i != options["graph-stop"].end(); i++)
	{
		std::string id(*i);
		if (id2net.count(id) == 0) {
			fprintf(stderr, "Can't find stop graph net `%s'.\n", id.c_str());
			exit(1);
		}
		stopNets.insert(id2net[id]);
	}
	options_used["graph-global"] = true;
	for (auto i = options["graph-global"].begin(); i != options["graph-global"].end(); i++)
	{
		std::string id(*i);
		if (id2net.count(id) == 0) {
			fprintf(stderr, "Can't find global graph net `%s'.\n", id.c_str());
			exit(1);
		}
		stopNets.insert(id2net[id]);
		globNets.insert(id2net[id]);
	}

	std::set<Net*> highNets;
	options_used["graph-highlight"] = true;
	for (auto i = options["graph-highlight"].begin(); i != options["graph-highlight"].end(); i++)
	{
		std::string id(*i);
		if (id2net.count(id) == 0) {
			fprintf(stderr, "Can't find highlighted graph net `%s'.\n", id.c_str());
			exit(1);
		}
		highNets.insert(id2net[id]);
	}

	options_used["graph"] = true;
	for (auto i = options["graph"].begin(); i != options["graph"].end(); i++)
	{
		std::string fromStr(*i), toStr(*i), expandStr("1");
		size_t split = i->find(":");
		if (split != std::string::npos) {
			fromStr = toStr = i->substr(0, split);
			expandStr = i->substr(split+1);
		}
		split = fromStr.find("->");
		if (split != std::string::npos) {
			toStr = fromStr.substr(split+2);
			fromStr = fromStr.substr(0, split);
		}

		if (id2net.count(fromStr) == 0) {
			fprintf(stderr, "Error in processing graph node `%s': Can't find net `%s'.\n", i->c_str(), fromStr.c_str());
			exit(1);
		}
		if (id2net.count(toStr) == 0) {
			fprintf(stderr, "Error in processing graph node `%s': Can't find net `%s'.\n", i->c_str(), toStr.c_str());
			exit(1);
		}
		if (atoi(expandStr.c_str()) <= 0) {
			fprintf(stderr, "Error in processing graph node `%s': Can't process expand info `%s'.\n", i->c_str(), expandStr.c_str());
			exit(1);
		}

		Net *from = id2net[fromStr], *to = id2net[toStr];
		int expand = atoi(expandStr.c_str());
		
		fprintf(stderr, "Finding path from `%s' to `%s' with exapand %d.\n", fromStr.c_str(), toStr.c_str(), atoi(expandStr.c_str()));

		std::map<Net*, int> astarMap;
		std::priority_queue<std::pair<int, Net*>> astarJobs;

		int maxPriority = -1;
		astarJobs.push(std::make_pair(0, from));

		while (astarJobs.size() != 0)
		{
			auto job = astarJobs.top();
			astarJobs.pop();

			int priority = abs(job.first);
			Net *net = job.second;

			if (maxPriority >= 0 && priority > maxPriority)
				break;

			if (astarMap.count(net) != 0)
				continue;

			astarMap[net] = priority;

			if (net == to)
				maxPriority = priority + expand;

			if (from == to)
				net->graph = true;

			if (stopNets.count(net) != 0)
				continue;

			for (auto j = net->switchGates.begin(); j != net->switchGates.end(); j++) {
				astarJobs.push(std::make_pair(job.first - 1, (*j)->cc[0]));
				astarJobs.push(std::make_pair(job.first - 1, (*j)->cc[1]));
			}
			for (auto j = net->switchCC.begin(); j != net->switchCC.end(); j++) {
				if ((*j)->cc[0] != net)
					astarJobs.push(std::make_pair(job.first - 1, (*j)->cc[0]));
				if ((*j)->cc[1] != net)
					astarJobs.push(std::make_pair(job.first - 1, (*j)->cc[1]));
				if (from == to)
					astarJobs.push(std::make_pair(job.first - 1, (*j)->gate));
			}
		}

		if (from != to)
		{
			while (astarJobs.size() != 0)
				astarJobs.pop();

			astarJobs.push(std::make_pair(0, to));

			while (astarJobs.size() != 0)
			{
				auto job = astarJobs.top();
				astarJobs.pop();

				int priority = abs(job.first);
				Net *net = job.second;

				if (astarMap.count(net) == 0)
					continue;

				net->graph = true;

				if (stopNets.count(net) == 0 || net == to)
				{
					std::set<Net*> neigh;
					for (auto j = net->switchCC.begin(); j != net->switchCC.end(); j++) {
						if ((*j)->cc[0] != net)
							neigh.insert((*j)->cc[0]);
						if ((*j)->cc[1] != net)
							neigh.insert((*j)->cc[1]);
						neigh.insert((*j)->gate);
					}

					for (auto j = neigh.begin(); j != neigh.end(); j++) {
						int pri = priority + (astarMap[*j] + 1) - astarMap[net];
						if (pri >= expand)
							continue;
						astarJobs.push(std::make_pair(-pri, *j));
					}
				}

				astarMap.erase(net);
			}
		}
	}

	int netCount = 0;
	for (auto i = nets.begin(); i != nets.end(); i++)
	{
		Net *net = *i;
		if (net->graph == false)
			continue;

		netCount++;
		if (stopNets.count(net) != 0)
			continue;

		for (auto j = net->switchGates.begin(); j != net->switchGates.end(); j++)
			(*j)->graph = true;
		for (auto j = net->switchCC.begin(); j != net->switchCC.end(); j++)
			(*j)->graph = true;
	}

	int switchCount = 0;
	for (auto i = switches.begin(); i != switches.end(); i++)
	{
		Switch *sw = *i;
		if (sw->graph == false)
			continue;
		switchCount++;
	}

	if (netCount != 0 || switchCount != 0)
		fprintf(stderr, "Graph contains %d nets and %d switches.\n", netCount, switchCount);

	if (opt("graph-dump", "").compare("") != 0) {
		std::string fn = opt("graph-dump", "");
		FILE *f = fopen(fn.c_str(), "w");
		for (auto i = nets.begin(); i != nets.end(); i++) {
			Net *net = *i;
			if (net->graph == false)
				continue;
			fprintf(f, "N");
			for (auto j = net->ids.begin(); j != net->ids.end(); j++)
				fprintf(f, " \"%s\"", j->c_str());
			fprintf(f, ": %s%s\n", state2str[net->driveState].c_str(),
					net->supply ? " supply" : "");
		}
		for (auto i = switches.begin(); i != switches.end(); i++)
		{
			Switch *sw = *i;
			if (sw->graph == false)
				continue;
			fprintf(f, "T");
			for (auto j = sw->ids.begin(); j != sw->ids.end(); j++)
				fprintf(f, " \"%s\"", j->c_str());
			fprintf(f, ": %s \"%s\" \"%s\" \"%s\"\n", sw->type->id.c_str(),
					sw->gate->ids.begin()->c_str(),
					sw->cc[0]->ids.begin()->c_str(),
					sw->cc[1]->ids.begin()->c_str());
		}
		fclose(f);
	}

	if (info == NULL)
		return netCount != 0 || switchCount != 0;

	char buffer[1024];
	std::string str;

	info->netInfo.clear();
	for (auto i = nets.begin(); i != nets.end(); i++)
	{
		Net *net = *i;
		if (net->graph == false)
			continue;

		auto &gi = info->netInfo[net];

		for (int state = 0; state < numStates; state++)
		{
			str = "";

			if (globNets.count(net) == 0)
			{
				snprintf(buffer, sizeof(buffer), "n%d [ label=\"", net->num);
				str += buffer;
				for (auto j = net->ids.begin(); j != net->ids.end(); j++) {
					if (j != net->ids.begin())
						str += "\\n";
					str += *j;
				}
				if (initindMap.count(net->driveState))
					str += initindMap[net->driveState];
				str += "\"";
				if (colorMap.count(state)) {
					snprintf(buffer, sizeof(buffer), ", color=\"%s\"", colorMap[state].c_str());
					str += buffer;
				}
				if (highNets.count(net) != 0)
					str += ", fillcolor=yellow, style=filled";
				snprintf(buffer, sizeof(buffer), ", shape=\"ellipse\" ];\n");
				str += buffer;
			}

			std::queue<std::pair<bool, Switch*>> swList;
			for (auto j = net->switchGates.begin(); j != net->switchGates.end(); j++)
				if ((*j)->graph)
					swList.push(std::make_pair(true, *j));
			for (auto j = net->switchCC.begin(); j != net->switchCC.end(); j++)
				if ((*j)->graph)
					swList.push(std::make_pair(false, *j));

			while (swList.size() != 0)
			{
				bool onGate = swList.front().first;
				Switch *sw = swList.front().second;
				swList.pop();

				std::string netName;
				snprintf(buffer, sizeof(buffer), "n%d", net->num);
				netName = buffer;

				if (globNets.count(net) != 0)
				{
					snprintf(buffer, sizeof(buffer), "n%d_t%d [ label=\"", net->num, sw->num);
					str += buffer;
					for (auto j = net->ids.begin(); j != net->ids.end(); j++) {
						if (j != net->ids.begin())
							str += "\\n";
						str += *j;
					}
					if (initindMap.count(net->driveState))
						str += initindMap[net->driveState];
					str += "\"";
					if (colorMap.count(state)) {
						snprintf(buffer, sizeof(buffer), ", color=\"%s\"", colorMap[state].c_str());
						str += buffer;
					}
					if (highNets.count(net) != 0)
						str += ", fillcolor=yellow, style=filled";
					snprintf(buffer, sizeof(buffer), ", shape=\"octagon\" ];\n");
					str += buffer;

					snprintf(buffer, sizeof(buffer), "n%d_t%d", net->num, sw->num);
					netName = buffer;
				}

				if (onGate)
					snprintf(buffer, sizeof(buffer), "%s -> t%d [ color=\"%s\" ];\n", netName.c_str(), sw->num, colorMap[state].c_str());
				else
					snprintf(buffer, sizeof(buffer), "t%d -> %s [ color=\"%s\" ];\n", sw->num, netName.c_str(), colorMap[state].c_str());
				str += buffer;
			}

			gi.push_back(str);
		}
	}

	info->switchInfo.clear();
	for (auto i = switches.begin(); i != switches.end(); i++)
	{
		Switch *sw = *i;
		if (sw->graph == false)
			continue;

		auto &gi = info->switchInfo[sw];

		snprintf(buffer, sizeof(buffer), "t%d [ label=\"", sw->num);
		str = buffer;
		for (auto j = sw->ids.begin(); j != sw->ids.end(); j++) {
			if (j != sw->ids.begin())
				str += "\\n";
			str += *j;
		}
		snprintf(buffer, sizeof(buffer), "\", shape=rectangle");
		str += buffer;

		if (sw->gate->graph && sw->cc[0]->graph && sw->cc[1]->graph)
		{
			gi.push_back(str + ", fillcolor=gray, style=filled ];\n");
			gi.push_back(str + " ];\n");
		}
		else
		{
			gi.push_back(str + ", fillcolor=gold4, style=filled ];\n");
			gi.push_back(str + ", fillcolor=gold2, style=filled ];\n");
		}
	}

	return netCount != 0 || switchCount != 0;
}


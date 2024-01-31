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

void Tnl::bias()
{
	// sort non-supply nets by `least' netname
	// (this guarantees a reproduceable behavior)
	std::map<std::string, Net*> alphaNets;
	for (auto i = nets.begin(); i != nets.end(); i++)
	{
		Net *net = *i;
		if (net->supply == false)
			alphaNets[*net->ids.begin()] = net;
		net->biasState = net->driveState;
	}

	std::string xsubi_opt = opt("bias-seed", "c01d:1337:c0fe");
	unsigned short xsubi[3] = { 0, 0, 0 };
	if (sscanf(xsubi_opt.c_str(), "%hx:%hx:%hx", &xsubi[0], &xsubi[1], &xsubi[2]) != 3) {
		fprintf(stderr, "Can't parse bias seed `%s'.\n", xsubi_opt.c_str());
		exit(1);
	}

	fprintf(stderr, "Calculating net bias (%d nets, seed=%04x:%04x:%04x).\n",
			int(alphaNets.size()), xsubi[0], xsubi[1], xsubi[2]);

	std::set<Net*> dirtyNets;
	std::priority_queue<std::pair<int, Net*>> dirtyNetsQueue;
	for (auto i = alphaNets.begin(); i != alphaNets.end(); i++) {
		Net *net = i->second;
		dirtyNets.insert(net);
		dirtyNetsQueue.push(std::make_pair(nrand48(xsubi), net));
	}

	int netUpdateCount = 0;
	while (dirtyNetsQueue.empty() == false)
	{
		// process the net with the `least' net name
		// (using the less<std::string> ordering)
		Net *net = dirtyNetsQueue.top().second;
		dirtyNets.erase(net);
		dirtyNetsQueue.pop();

		// calculate group state
		std::set<Net*> group;
		int state = net->driveState;
		bias_findGroup(net, group, state);

		// update group state
		for (auto i = group.begin(); i != group.end(); i++)
		{
			Net *n1 = *i;

			if (n1->biasState == state)
				continue;

			n1->biasState = state;
			for (auto j = n1->switchGates.begin(); j != n1->switchGates.end(); j++)
			for (int k = 0; k < 2; k++)
			{
				Net *n2 = (*j)->cc[k];
				if (n2->supply == false && dirtyNets.count(n2) == 0) {
					dirtyNets.insert(n2);
					dirtyNetsQueue.push(std::make_pair(nrand48(xsubi), n2));
				}
			}
		}

		// check for endless loop
		if (netUpdateCount++ > int(nets.size()*100)) {
			fprintf(stderr, "Can't find stable solution for bias. Giving up.\n");
			exit(1);
		}
	}

	// check bias state for correctness (should never find an error)
	for (auto i = nets.begin(); i != nets.end(); i++)
	{
		Net *net = *i;
		std::set<Net*> group;
		int state = net->driveState;
		bias_findGroup(net, group, state);

		if (state != net->biasState) {
			fprintf(stderr, "Error in bias solution: net `%s' has state %d but should have state %d!\n",
					net->ids.begin()->c_str(), net->biasState, state);
			exit(1);
		}
	}

	fprintf(stderr, "Found stable solution for bias after %d net updates.\n", netUpdateCount);
}

void Tnl::bias_findGroup(Net *net, std::set<Net*> &group, int &state)
{
	if (group.count(net) != 0)
		return;

	state = stateCombine[std::make_pair(state, net->driveState)];

	if (net->supply)
		return;

	group.insert(net);

	for (auto i = net->switchCC.begin(); i != net->switchCC.end(); i++)
	{
		Switch *sw = *i;

		if (sw->type->onStates.count(sw->gate->biasState) == 0)
			continue;

		bias_findGroup(sw->cc[0], group, state);
		bias_findGroup(sw->cc[1], group, state);
	}
}


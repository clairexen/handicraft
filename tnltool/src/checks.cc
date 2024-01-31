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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

void Tnl::check()
{
	// combine table and ascend table must be filled
	for (auto i = stateCombine.begin(); i != stateCombine.end(); i++)
		if (i->second == -1) {
			fprintf(stderr, "No entry in combine table for %s*%s.\n",
					state2str[i->first.first].c_str(),
					state2str[i->first.second].c_str());
			exit(1);
		}
	for (auto i = stateAscend.begin(); i != stateAscend.end(); i++)
		if (i->second == -1) {
			fprintf(stderr, "No entry in ascend table for %s.\n",
					state2str[i->first].c_str());
			exit(1);
		}

	// algebraic properties of combine table:
	//    - associative: (A*B)*C = A*(B*C)
	//    - commutative: A*B = B*A
	//    - stable: A*A = A
	//
	// instead of checking for properties of the combine table
	// (e.g. symmetric for the commutative property) we just test
	// for the algebraic properties by trying all relevant
	// combinations..

	// associative property
	for (int a=0; a<numStates; a++)
	for (int b=0; b<numStates; b++)
	for (int c=0; c<numStates; c++)
	{
		int ab = stateCombine[std::pair<int, int>(a, b)];
		int ab_c = stateCombine[std::pair<int, int>(ab, c)];

		int bc = stateCombine[std::pair<int, int>(b, c)];
		int a_bc = stateCombine[std::pair<int, int>(a, bc)];

		if (ab_c != a_bc) {
			fprintf(stderr, "Combine table is not associative: (%s*%s)*%s -> %s, %s*(%s*%s) -> %s.\n",
					state2str[a].c_str(), state2str[b].c_str(),
					state2str[c].c_str(), state2str[ab_c].c_str(),
					state2str[a].c_str(), state2str[b].c_str(),
					state2str[c].c_str(), state2str[a_bc].c_str());
			exit(1);
		}
	}

	// commutative and stable
	for (int a=0; a<numStates; a++)
	for (int b=0; b<numStates; b++)
	{
		int ab = stateCombine[std::pair<int, int>(a, b)];
		int ba = stateCombine[std::pair<int, int>(b, a)];

		if (ab != ba) {
			fprintf(stderr, "Combine table is not commutative: %s*%s -> %s, %s*%s -> %s.\n",
					state2str[a].c_str(), state2str[b].c_str(), state2str[ab].c_str(),
					state2str[b].c_str(), state2str[a].c_str(), state2str[ba].c_str());
			exit(1);
		}

		if (a == b && a != ab) {
			fprintf(stderr, "Combine table is not stable: %s*%s -> %s.\n",
					state2str[a].c_str(), state2str[b].c_str(), state2str[ab].c_str());
			exit(1);
		}
	}

	// the ascend table must stabelize in one step.
	// (maybe we find usecases for different ascend tables.
	// then we might remove this check..)
	for (int a=0; a<numStates; a++)
	{
		int b = stateAscend[a];
		int c = stateAscend[b];

		if (b != c) {
			fprintf(stderr, "Combine table is not stable: %s -> %s -> %s\n",
					state2str[a].c_str(), state2str[b].c_str(), state2str[c].c_str());
			exit(1);
		}
	}

	// the state table must contain a dedicated "weakest state"
	weakestState = -1;
	for (int a=0; a<numStates; a++)
	{
		for (int b=0; b<numStates; b++) {
			int ab = stateCombine[std::pair<int, int>(a, b)];
			if (ab != b)
				goto next_a;
		}
		weakestState = a;
		break;
next_a:;
	}
	if (weakestState < 0) {
		fprintf(stderr, "Combine table does not contain a weakest state.\n");
		exit(1);
	}

	// all switch types must be defined as ON of OFF for each state
	for (auto i = switchTypes.begin(); i != switchTypes.end(); i++)
	for (int j = 0; j < numStates; j++) {
		SwitchType *st = i->second;
		if ((st->onStates.count(j) == 0 && st->offStates.count(j) == 0) ||
				(st->onStates.count(j) != 0 && st->offStates.count(j) != 0)) {
			fprintf(stderr, "Transistor type `%s' has no defined behavior for state `%s'.\n",
					i->first.c_str(), state2str[j].c_str());
			exit(1);
		}
	}

	fprintf(stderr, "TNL data passed basic checks.\n");
}

void Tnl::mergeRedundant()
{
	// FIXME:
	// remove duplicate switches
	// remove switches with gates on supply pins (and merge nets, if needed)
	// repeat (basically this is const folding for netlists..)
}

void Tnl::enumerate()
{
	numNets = 0;
	numSupplies = 0;
	netByNum.clear();
	for (auto i = nets.begin(); i != nets.end(); i++) {
		Net *n = *i;
		netByNum.push_back(n);
		n->num = numNets++;
		if (int(n->switchGates.size()) > maxGatesPerNet)
			maxGatesPerNet = n->switchGates.size();
		if (int(n->switchCC.size()) > maxCcPerNet)
			maxCcPerNet = n->switchCC.size();
		if (n->supply) {
			Net *tmp = netByNum[numSupplies];
			netByNum[numSupplies] = n;
			netByNum[n->num] = tmp;
			tmp->num = n->num;
			n->num = numSupplies;
			numSupplies++;
		}
	}

	numSwitches = 0;
	for (auto i = switches.begin(); i != switches.end(); i++) {
		Switch *s = *i;
		switchByNum.push_back(s);
		s->num = numSwitches++;
	}

	fprintf(stderr, "# of nets: %d\n", int(nets.size()));
	fprintf(stderr, "# of transistors: %d\n", int(switches.size()));
	fprintf(stderr, "max # of channels per net: %d\n", maxCcPerNet);
	fprintf(stderr, "max # of gates per net: %d\n", maxGatesPerNet);
}


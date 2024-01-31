/*
 *  Simple NMOS Simulator (for playing with the 6502 nmos netlist)
 *
 *  Copyright (C) 2011  Clifford Wolf <clifford@clifford.at>
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

#include "nmossim.h"
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

NmosSim::NmosSim()
{
	changedDisplayed = 0;
	displayPageCounter = 0;
	displayFolderCounter = 0;
	transitionCounter = 0;
}

NmosSim::~NmosSim()
{
	for (auto i = allNets.begin(); i != allNets.end(); i++) {
		for (auto j = (*i)->labels.begin(); j != (*i)->labels.end(); j++)
			free(*j);
		delete *i;
	}
	for (auto i = allTrans.begin(); i != allTrans.end(); i++) {
		for (auto j = (*i)->labels.begin(); j != (*i)->labels.end(); j++)
			free(*j);
		delete *i;
	}
}

NmosSim::Net *NmosSim::addNet(bool inPin, bool pullUp, const char *label)
{
	Net *net = new Net;
	net->global = false;
	net->display = false;
	net->highlight = false;
	net->queued = true;
	net->visited = false;
	net->inPin = inPin;
	net->pullUp = pullUp;
	net->state = false;
	allNets.insert(net);
	evalNets.push_back(net);
	addLabel(net, label);
	return net;
}

NmosSim::Trans *NmosSim::addTrans(Net *gate, Net *c1, Net *c2, const char *label)
{
	Trans *trans = new Trans;
	trans->display = false;
	trans->highlight = false;
	trans->gate = gate;
	trans->cc[0] = c1;
	trans->cc[1] = c2;
	gate->transGates.push_back(trans);
	c1->transCC.push_back(trans);
	c2->transCC.push_back(trans);
	allTrans.insert(trans);
	addLabel(trans, label);
	return trans;
}

void NmosSim::deleteDupTransistors()
{
	std::map<std::pair<Net*,std::pair<Net*,Net*>>, std::vector<Trans*>> delTransMap;

	for (auto i = allTrans.begin(); i != allTrans.end(); i++)
	{
		Trans *t = *i;

		Net *tc1 = t->cc[0], *tc2 = t->cc[1];

		if ((long int)tc1 > (long int)tc2) {
			Net *tmp = tc1;
			tc1 = tc2, tc2 = tmp;
		}

		delTransMap[std::pair<Net*,std::pair<Net*,Net*>>(t->gate, std::pair<Net*,Net*>(tc1, tc2))].push_back(t);
	}

	for (auto i = delTransMap.begin(); i != delTransMap.end(); i++)
	{
		i->second.pop_back();
		for (auto j = i->second.begin(); j != i->second.end(); j++)
		{
			Trans *t = *j;

			for (auto j = t->gate->transGates.begin(); j != t->gate->transGates.end(); j++)
				if (*j == t) {
					*j = t->gate->transGates.back();
					t->gate->transGates.pop_back();
					break;
				}

			for (int k = 0; k < 2; k++)
				for (auto j = t->cc[k]->transCC.begin(); j != t->cc[k]->transCC.end(); j++)
					if (*j == t) {
						*j = t->cc[k]->transCC.back();
						t->cc[k]->transCC.pop_back();
						break;
					}

			allTrans.erase(t);
			for (auto j = t->labels.begin(); j != t->labels.end(); j++)
				free(*j);
			delete t;
		}
	}
}

void NmosSim::setGlobal(Net *net)
{
	net->global = true;
}

void NmosSim::addLabel(Net *net, const char *label)
{
	char *txt;
	if (label == NULL)
		asprintf(&txt, "_%p_", net);
	else
		txt = strdup(label);

	assert(mapNets.count(txt) == 0);
	net->labels.push_back(txt);
	mapNets[txt] = net;
}

void NmosSim::addLabel(Trans *trans, const char *label)
{
	char *txt;
	if (label == NULL)
		asprintf(&txt, "_%p_", trans);
	else
		txt = strdup(label);

	assert(mapTrans.count(txt) == 0);
	trans->labels.push_back(txt);
	mapTrans[txt] = trans;
}

void NmosSim::eval()
{
	if (displayPageCounter > 0 && changedDisplayed)
		displayPage();

	while (evalNets.size() != 0)
	{
		std::map<Net*, bool> updList;

		while (evalNets.size() != 0)
		{
			Net *net = evalNets.front();
			evalNets.pop_front();
			net->queued = false;
			evalNet(net, updList);
		}

		for (auto i = updList.begin(); i != updList.end(); i++) {
			i->first->visited = false;
			updateNet(i->first, i->second, false);
		}

		if (displayPageCounter > 0 && changedDisplayed)
			displayPage();

		transitionCounter++;
	}
}

void NmosSim::evalNet(Net *net, std::map<Net*, bool> &updList)
{
	evalFoundCharge  = false;
	evalFoundPullUp  = false;
	evalFoundPinHigh = false;
	evalFoundPinLow  = false;
	evalGroup.clear();
	evalWorker(net);

	bool newState = evalFoundCharge;
	if (evalFoundPullUp)
		newState = true;
	if (evalFoundPinHigh)
		newState = true;
	if (evalFoundPinLow)
		newState = false;

	for (auto i = evalGroup.begin(); i != evalGroup.end(); i++) {
		Net *net = *i;
		updList[net] = newState;
	}
}

void NmosSim::evalWorker(Net *net)
{
	if (net->visited)
		return;
	if (net->inPin) {
		if (net->state)
			evalFoundPinHigh = true;
		else
			evalFoundPinLow = true;
		return;
	}
	if (net->pullUp)
		evalFoundPullUp = true;
	if (net->state)
		evalFoundCharge = true;
	evalGroup.push_back(net);
	net->visited = true;
	for (auto i = net->transCC.begin(); i != net->transCC.end(); i++) {
		Trans *trans = *i;
		if (trans->gate->state) {
			evalWorker(trans->cc[0]);
			evalWorker(trans->cc[1]);
		}
	}
}

NmosSim::Net *NmosSim::findNet(const char *label)
{
	if (mapNets.count(label) == 0)
		return NULL;
	return mapNets[label];
}

NmosSim::Trans *NmosSim::findTrans(const char *label)
{
	if (mapTrans.count(label) == 0)
		return NULL;
	return mapTrans[label];
}

void NmosSim::updateNet(Net *net, bool state, bool setInPin)
{
	if (setInPin)
		net->inPin = true;
	if (net->state == state)
		return;
	net->state = state;
	if (net->onChange)
		net->onChange();
	if (net->display)
		changedDisplayed = true;
	for (auto i = net->transGates.begin(); i != net->transGates.end(); i++) {
		Trans *trans = *i;
		for (int j = 0; j < 2; j++) {
			if (trans->cc[j]->queued)
				continue;
			trans->cc[j]->queued = true;
			evalNets.push_back(trans->cc[j]);
		}
	}
}

void NmosSim::releaseNet(Net *net)
{
	net->inPin = false;
	if (net->queued)
		return;
	net->queued = true;
	evalNets.push_back(net);
}

bool NmosSim::checkNet(const char *label, bool expect)
{
	Net *net = findNet(label);
	if (net)
		return checkNet(net, expect);
	printf("Can't lookup net `%s' (expected to be %s)!\n", label, expect ? "HIGH" : "LOW");
	return false;
}

bool NmosSim::checkNet(Net *net, bool expect)
{
	if (net->state != expect) {
		printf("Net `%s' is expected to be %s but is in fact %s!%s\n", net->labels.back(),
				expect ? "HIGH" : "LOW", net->state ? "HIGH" : "LOW",
				net->inPin ? "   (THIS NET IS AN INPUT PIN)" : "");
		updateNet(net, expect, false);
		return false;
	}
	return true;
}

bool NmosSim::checkStateDump(const char *filename, const char *id)
{
	FILE *f = fopen(filename, "r");
	if (f == NULL) {
		printf("Can't open statedump file `%s': %s\n", filename, strerror(errno));
		return false;
	}

	int magic_ptr = 0;
	char magic[1024];
	snprintf(magic, 1024, ":STATEDUMP:%s:", id);

	int errcount = 0;
	for (int ch = fgetc(f); ch > 0; ch = fgetc(f))
	{
		if (magic[magic_ptr] == ch)
			magic_ptr++;
		else
		if (magic[0] == ch)
			magic_ptr = 1;
		else
			magic_ptr = 0;

		while (!magic[magic_ptr])
		{
			int a = -1, b = -1;

			fscanf(f, "%d,%d,", &a, &b);
			if (a < 0 || b < 0)
				goto closeAndExit;

			char netname[1024];
			snprintf(netname, 1024, "n%d", a);
			errcount += checkNet(netname, b == 1) ? 0 : 1;
		}
	}

	errcount++;
	printf("No dump `%s' found in statedump file `%s'.\n", id, filename);

	if (0) {
closeAndExit:
		printf("Compared state with dump `%s' in statedump file `%s' and found %d errors.%s\n",
				id, filename, errcount, errcount > 0 ? "  (re-evaluating and re-checking..)" : "");
		if (errcount > 0) {
			eval();
			checkStateDump(filename, id);
		}
	}
	fclose(f);

	return errcount == 0;
}


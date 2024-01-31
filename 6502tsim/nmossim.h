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

#ifndef NMOSSIM_H
#define NMOSSIM_H

#include <set>
#include <map>
#include <list>
#include <queue>
#include <vector>
#include <functional>

#include <string.h>
#include <errno.h>

struct NmosSim
{
	struct Net;
	struct Trans;

	struct Net {
		int astarValue;
		bool global, display, highlight;
		bool queued, visited;
		bool inPin, pullUp, state;
		std::vector<char*> labels;
		std::vector<Trans*> transGates;
		std::vector<Trans*> transCC;
		std::function<void()> onChange;
	};

	struct Trans {
		int astarValue;
		bool display, highlight;
		std::vector<char*> labels;
		Net *gate;
		Net *cc[2];
	};

	std::set<Net*> allNets;
	std::set<Trans*> allTrans;

	struct MyCompareString {
		bool operator()(const char *a, const char *b) const {
			return strcmp(a, b) < 0;
		}
	};
	std::map<const char*, Net*, MyCompareString> mapNets;
	std::map<const char*, Trans*, MyCompareString> mapTrans;

	std::list<Net*> evalNets;

	bool evalFoundCharge;
	bool evalFoundPullUp;
	bool evalFoundPinHigh;
	bool evalFoundPinLow;
	std::vector<Net*> evalGroup;
	int transitionCounter;

	NmosSim();
	~NmosSim();

	Net *addNet(bool inPin, bool pullUp, const char *label = NULL);
	Trans *addTrans(Net *gate, Net *c1, Net *c2, const char *label = NULL);
	void deleteDupTransistors();

	void setGlobal(Net *net);
	void addLabel(Net *net, const char *label);
	void addLabel(Trans *trans, const char *label);

	void eval();
	void evalNet(Net *net, std::map<Net*, bool> &updList);
	void evalWorker(Net *net);

	Net *findNet(const char *label);
	Trans *findTrans(const char *label);
	void updateNet(Net *net, bool state, bool setInPin = true);
	void releaseNet(Net *net);

	bool checkNet(const char *label, bool expect);
	bool checkNet(Net *net, bool expect);
	bool checkStateDump(const char *filename, const char *id);

	// graphviz based display functions (see display.cc)
	bool changedDisplayed;
	int displayPageCounter;
	int displayFolderCounter;
	struct AstarJob {
		Net *net;
		Trans *trans;
		int astarValue;
		AstarJob(Net *n, int v) : net(n), trans(), astarValue(v) {};
		AstarJob(Trans *t, int v) : net(), trans(t), astarValue(v) {};
	};
	struct AstarJobCompare {
		bool operator()(const AstarJob &a, const AstarJob &b) {
			return a.astarValue > b.astarValue;
		}
	};
	typedef std::priority_queue<AstarJob, std::vector<AstarJob>, AstarJobCompare> AstarJobQueue;
	void clearDisplay();
	void addDisplay(const char *label, int level = 0, bool highlight = true);
	void addDisplay(Net *net, int level = 0, bool highlight = true);
	void addDisplay(Trans *trans, int level = 0, bool highlight = true);
	void addDisplay(const char *label1, const char *label2, bool silentFail = false);
	void addDisplayAstarFwd(Net *net, int value, AstarJobQueue &queue);
	void addDisplayAstarFwd(Trans *trans, int value, AstarJobQueue &queue);
	void addDisplayAstarBwd(Net *net);
	void addDisplayAstarBwd(Trans *trans);
	void displayPage();
	void displayFolder();
};

#endif

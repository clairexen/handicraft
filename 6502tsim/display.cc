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
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#define CREATE_PNG_AND_GIF 0
#define CREATE_SINGLE_PSDOC 1

void NmosSim::clearDisplay()
{
	for (auto i = allNets.begin(); i != allNets.end(); i++)
		(*i)->display = false, (*i)->highlight = false;
	for (auto i = allTrans.begin(); i != allTrans.end(); i++)
		(*i)->display = false, (*i)->highlight = false;
	changedDisplayed = false;
}

void NmosSim::addDisplay(const char *label, int level, bool highlight)
{
	addDisplay(findNet(label), level, highlight);
	addDisplay(findTrans(label), level, highlight);
}

void NmosSim::addDisplay(Net *net, int level, bool highlight)
{
	if (!net || net->global || net->visited)
		return;

	net->display = true;
	if (highlight)
		net->highlight = true;

	if (level > 0) {
		net->visited = true;
		for (auto i = net->transGates.begin(); i != net->transGates.end(); i++)
			addDisplay(*i, level-1, false);
		for (auto i = net->transCC.begin(); i != net->transCC.end(); i++)
			addDisplay(*i, level-1, false);
		net->visited = false;
	}
}

void NmosSim::addDisplay(Trans *trans, int level, bool highlight)
{
	if (!trans)
		return;

	trans->display = true;
	if (highlight)
		trans->highlight = true;
	addDisplay(trans->gate, level, false);
	addDisplay(trans->cc[0], level, false);
	addDisplay(trans->cc[1], level, false);
}

void NmosSim::addDisplay(const char *label1, const char *label2, bool silentFail)
{
	if (label2 == NULL && label1 != NULL) {
		addDisplay(label2, label1);
		return;
	}

	if (label1 == NULL) {
		int lastNum = -1;
		while (1) {
			int num = 0;
			for (auto i = allNets.begin(); i != allNets.end(); i++) {
				if ((*i)->display == false || (*i)->global)
					continue;
				addDisplay((*i)->labels.front(), label2, true);
				num++;
			}
			for (auto i = allTrans.begin(); i != allTrans.end(); i++) {
				if ((*i)->display == false)
					continue;
				addDisplay((*i)->labels.front(), label2, true);
				num++;
			}
			if (num == lastNum)
				break;
			lastNum = num;
		}
		return;
	}

	for (auto i = allNets.begin(); i != allNets.end(); i++)
		(*i)->astarValue = -1;
	for (auto i = allTrans.begin(); i != allTrans.end(); i++)
		(*i)->astarValue = -1;

	AstarJobQueue astarQueue;
	Net *net = findNet(label1);
	Trans *trans = findTrans(label1);

	if (net)
		astarQueue.push(AstarJob(net, 0));
	if (trans)
		astarQueue.push(AstarJob(net, 0));

	net = findNet(label2);
	trans = findTrans(label2);

	while (astarQueue.size() != 0) {
		AstarJob job = astarQueue.top();
		astarQueue.pop();
		if (job.net)
			addDisplayAstarFwd(job.net, job.astarValue, astarQueue);
		if (job.trans)
			addDisplayAstarFwd(job.trans, job.astarValue, astarQueue);
		if (net && net->astarValue >= 0)
			goto astarFinished;
		if (trans && trans->astarValue >= 0)
			goto astarFinished;
	}

	if (!silentFail) {
		printf("A* from `%s' to `%s' failed!\n", label1, label2);
		return;
	}

astarFinished:
	if (net)
		addDisplayAstarBwd(net);
	if (trans)
		addDisplayAstarBwd(trans);
}

void NmosSim::addDisplayAstarFwd(Net *net, int value, AstarJobQueue &queue)
{
	if (net->astarValue >= 0) {
		assert(net->astarValue <= value);
		return;
	}

	net->astarValue = value;

	int newValue = value;
	newValue += net->transGates.size();
	newValue += net->transCC.size();

	if (!net->global || value == 0)
	{
		for (auto i = net->transGates.begin(); i != net->transGates.end(); i++)
			queue.push(AstarJob(*i, newValue));

		for (auto i = net->transCC.begin(); i != net->transCC.end(); i++)
			queue.push(AstarJob(*i, newValue));
	}
}

void NmosSim::addDisplayAstarFwd(Trans *trans, int value, AstarJobQueue &queue)
{
	if (trans->astarValue >= 0) {
		assert(trans->astarValue <= value);
		return;
	}

	trans->astarValue = value;
	// queue.push(AstarJob(trans->gate, value + 1));
	queue.push(AstarJob(trans->cc[0], value + 1));
	queue.push(AstarJob(trans->cc[1], value + 1));
}

void NmosSim::addDisplayAstarBwd(Net *net)
{
	net->display = true;

	int bestValue = net->astarValue;
	Trans *bestTrans = NULL;

	for (auto i = net->transGates.begin(); i != net->transGates.end(); i++) {
		if ((*i)->astarValue >= 0 && (*i)->astarValue < bestValue)
			bestValue = (*i)->astarValue, bestTrans = *i;
	}

	for (auto i = net->transCC.begin(); i != net->transCC.end(); i++) {
		if ((*i)->astarValue >= 0 && (*i)->astarValue < bestValue)
			bestValue = (*i)->astarValue, bestTrans = *i;
	}

	if (bestTrans)
		addDisplayAstarBwd(bestTrans);
}

void NmosSim::addDisplayAstarBwd(Trans *trans)
{
	trans->display = true;

	int bestValue = trans->astarValue;
	Net *bestNet = NULL;

	if (trans->gate->astarValue >= 0 && trans->gate->astarValue < bestValue)
		bestValue = trans->gate->astarValue, bestNet = trans->gate;

	if (trans->cc[0]->astarValue >= 0 && trans->cc[0]->astarValue < bestValue)
		bestValue = trans->cc[0]->astarValue, bestNet = trans->cc[0];

	if (trans->cc[1]->astarValue >= 0 && trans->cc[1]->astarValue < bestValue)
		bestValue = trans->cc[1]->astarValue, bestNet = trans->cc[1];

	if (bestNet)
		addDisplayAstarBwd(bestNet);
}

void NmosSim::displayPage()
{
	if (displayPageCounter == 0 && displayFolderCounter == 0)
		if (system("rm -f nmossim_disp[0-9][0-9][0-9].* nmossim_disp.dot") != 0) { }

#if CREATE_SINGLE_PSDOC
	FILE *f = fopen("nmossim_disp.dot", displayPageCounter == 0 ? "w" : "a");
#else
	FILE *f = fopen("nmossim_disp.dot", "w");
#endif
	fprintf(f, "strict digraph nmossim%d {\n", transitionCounter);
	fprintf(f, "\trankdir = \"LR\";\n");
	fprintf(f, "\tlabel = \"%d\";\n", transitionCounter);
	for (auto i = allNets.begin(); i != allNets.end(); i++)
	{
		Net *n = *i;
		if (!n->display || n->global)
			continue;
		fprintf(f, "\t%s [label=\"", n->labels.front());
		for (auto j = n->labels.begin(); j != n->labels.end(); j++)
			fprintf(f, "%s%s", j == n->labels.begin() ? "" : "\\n", *j);
		if (n->inPin)
			fprintf(f, " =%d", n->state);
		if (n->pullUp)
			fprintf(f, " +");
		fprintf(f, "\", color=%s, %sshape=ellipse];\n", n->state ? "red" : "blue",
				n->highlight ? "fillcolor=yellow, style=filled, " : "");

		std::list<Trans*> tGG, tCC;
		for (auto j = (*i)->transGates.begin(); j != (*i)->transGates.end(); j++)
			if ((*j)->display == false)
				tGG.push_back(*j);
		for (auto j = (*i)->transCC.begin(); j != (*i)->transCC.end(); j++)
			if ((*j)->display == false)
				tCC.push_back(*j);

		if (tGG.size() != 0) {
			fprintf(f, "\t%s_gg [label=\"", n->labels.front());
			if (tGG.size() >= 10)
				fprintf(f, "(%d transistors)", int(tGG.size()));
			else
				for (auto j = tGG.begin(); j != tGG.end(); j++) {
					if (j != tGG.begin())
						fprintf(f, "\\n");
					for (auto k = (*j)->labels.begin(); k != (*j)->labels.end(); k++)
						fprintf(f, "%s%s", k == (*j)->labels.begin() ? "" : ", ", *k);
				}
			fprintf(f, "\", shape=%s];\n", tGG.size() >= 10 ? "hexagon" : "rectangle");
			fprintf(f, "\t%s -> %s_gg [color=%s];\n",
					n->labels.front(), n->labels.front(),
					n->state ? "red" : "blue");
		}

		if (tCC.size() != 0) {
			fprintf(f, "\t%s_cc [label=\"", n->labels.front());
			if (tCC.size() >= 10)
				fprintf(f, "(%d transistors)", int(tCC.size()));
			else
				for (auto j = tCC.begin(); j != tCC.end(); j++) {
					if (j != tCC.begin())
						fprintf(f, "\\n");
					for (auto k = (*j)->labels.begin(); k != (*j)->labels.end(); k++)
						fprintf(f, "%s%s", k == (*j)->labels.begin() ? "" : ", ", *k);
				}
			fprintf(f, "\", shape=%s];\n", tCC.size() >= 10 ? "hexagon" : "rectangle");
			fprintf(f, "\t%s_cc -> %s [color=%s];\n",
					n->labels.front(), n->labels.front(),
					n->state ? "red" : "blue");
		}
	}
	for (auto i = allTrans.begin(); i != allTrans.end(); i++)
	{
		Trans *t = *i;
		if (!t->display)
			continue;
		fprintf(f, "\t%s [ label=\"", t->labels.front());
		for (auto j = t->labels.begin(); j != t->labels.end(); j++)
			fprintf(f, "%s%s", j == t->labels.begin() ? "" : "\\n", *j);
		fprintf(f, "\", %sshape=rectangle ];\n",
				t->highlight ? "fillcolor=yellow, style=filled, " :
				t->gate->state ? "" : "fillcolor=gray, style=filled, ");

		if (t->gate->global) {
			fprintf(f, "\t%s__%s [label=\"%s%s\", color=%s, shape=octagon];\n",
					t->labels.front(), t->gate->labels.front(), t->gate->labels.back(),
					t->gate->pullUp ? " +" : "", t->gate->state ? "red" : "blue");
			fprintf(f, "\t%s__%s -> %s [color=%s];\n",
					t->labels.front(), t->gate->labels.front(), t->labels.front(),
					t->gate->state ? "red" : "blue");
		} else
			fprintf(f, "\t%s -> %s [color=%s];\n",
					t->gate->labels.front(), t->labels.front(),
					t->gate->state ? "red" : "blue");

		for (int j = 0; j < 2; j++) {
			if (t->cc[j]->global) {
				fprintf(f, "\t%s__%s [label=\"%s%s\", color=%s, shape=octagon];\n",
						t->labels.front(), t->cc[j]->labels.front(), t->cc[j]->labels.back(),
						t->cc[j]->pullUp ? " +" : "", t->cc[j]->state ? "red" : "blue");
				fprintf(f, "\t%s -> %s__%s [color=%s];\n",
						t->labels.front(), t->labels.front(), t->cc[j]->labels.front(),
						t->cc[j]->state ? "red" : "blue");
			} else
				fprintf(f, "\t%s -> %s [color=%s ];\n",
						t->labels.front(), t->cc[j]->labels.front(),
						t->cc[j]->state ? "red" : "blue");
		}
	}
	fprintf(f, "}\n");
	fclose(f);

#if !CREATE_SINGLE_PSDOC
	char cmd[1024];
#  if CREATE_PNG_AND_GIF
	printf("Generating `nmossim_disp%03d.png' using graphviz..\n", displayPageCounter);
	snprintf(cmd, 1024, "dot -Tpng nmossim_disp.dot -o nmossim_disp%03d.png", displayPageCounter);
#  else
	printf("Generating `nmossim_disp%03d.ps' using graphviz..\n", displayPageCounter);
	snprintf(cmd, 1024, "dot -Tps nmossim_disp.dot -o nmossim_disp%03d.ps", displayPageCounter);
#  endif
	if (system(cmd) != 0) { exit(1); }
#endif

	displayPageCounter++;
	changedDisplayed = false;
}

void NmosSim::displayFolder()
{
	if (displayPageCounter == 0)
		return;

#if CREATE_PNG_AND_GIF
	int cmd_size = 1024 + displayPageCounter*30, cmd_len = 0;
	char cmd[cmd_size];

	cmd_len += snprintf(cmd + cmd_len, cmd_size - cmd_len, "convert -delay 50 -loop 0 -scale 640");
	for (int i = 0; i < displayPageCounter; i++)
		cmd_len += snprintf(cmd + cmd_len, cmd_size - cmd_len, " nmossim_disp%03d.png", i);
	cmd_len += snprintf(cmd + cmd_len, cmd_size - cmd_len, " nmossim_disp%03d.gif", displayFolderCounter++);
	printf(">> %s\n", cmd);
	if (system(cmd) != 0) { exit(1); }

	for (int i = 0; i < displayPageCounter; i++) {
		snprintf(cmd, cmd_size, "nmossim_disp%03d.png", i);
		remove(cmd);
	}
	displayPageCounter = 0;
#elif CREATE_SINGLE_PSDOC
	char cmd[1024];
	printf("Generating `nmossim_disp%03d.ps' using graphviz..\n", displayFolderCounter);
	snprintf(cmd, 1024, "dot -Tps nmossim_disp.dot -o nmossim_disp%03d.ps", displayFolderCounter++);
	if (system(cmd) != 0) { exit(1); }
	displayPageCounter = 0;
#endif
}


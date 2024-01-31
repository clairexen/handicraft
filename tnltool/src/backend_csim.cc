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
#include <assert.h>
#include <fnmatch.h>

namespace {
	const char *escapeC(const char *in)
	{
		static std::string str;
		str = "";
		for (int i = 0; in[i]; i++) {
			if (in[i] == '"')
				str += "\\\"";
			else if (in[i] == '\\')
				str += "\\\\";
			else if (in[i] == '\n')
				str += "\\n";
			else
				str += in[i];
		}
		return str.c_str();
	}
}

void Tnl::backend_csim(std::string filename_prefix)
{
	std::string cfn = filename_prefix + ".tab.c";
	std::string hfn = filename_prefix + ".tab.h";

	GraphInfo graphInfo;
	bool graphEnable = graph(&graphInfo);

	FILE *cf = fopen(cfn.c_str(), "w");

	if (cf == NULL) {
		fprintf(stderr, "Can't open output file `%s': %s\n", cfn.c_str(), strerror(errno));
		exit(1);
	}

	FILE *hf = fopen(hfn.c_str(), "w");

	if (hf == NULL) {
		fprintf(stderr, "Can't open output file `%s': %s\n", hfn.c_str(), strerror(errno));
		exit(1);
	}

	std::string prefix = filename_prefix;
	std::string prefixUC;
	for (auto i = prefix.begin(); i != prefix.end(); i++) {
		if (*i >= 'a' && *i <= 'z')
			prefixUC.push_back(*i - 'a' + 'A');
		else
			prefixUC.push_back(*i);
	}
	const char *pf = prefix.c_str();
	const char *pfUC = prefixUC.c_str();

	std::string hdrProtDef("TNL_");
	for (auto i = filename_prefix.begin(); i != filename_prefix.end(); i++) {
		if (*i >= 'a' && *i <= 'z')
			hdrProtDef.push_back(*i - 'a' + 'A');
		else
			hdrProtDef.push_back(*i);
	}
	hdrProtDef.append("_TAB_H");
	fprintf(hf, "#ifndef %s\n#define %s\n\n", hdrProtDef.c_str(), hdrProtDef.c_str());

	fprintf(hf, "#include <stdio.h>\n");

	for (auto i = nets.begin(); i != nets.end(); i++) {
		for (auto j = (*i)->ids.begin(); j != (*i)->ids.end(); j++) {
			if (j->find_first_not_of("abcdefghijklmnopqrstuvwxyz_ABCDEFGHIJKLMNOPQRSTUVWXYZ"
					"0123456789") != std::string::npos)
				continue;
			fprintf(hf, "#define %s_PIN_%s %d\n", pfUC, j->c_str(), (*i)->num);
		}
	}
	fprintf(hf, "\n");

	for (int i = 0; i < numStates; i++) {
		fprintf(hf, "#define %s_STATE_%s %d\n", pfUC, state2str[i].c_str(), i);
	}
	fprintf(hf, "\n");

	fprintf(hf, "void %s_init();\n", pf);
	fprintf(hf, "void %s_eval();\n", pf);
	fprintf(hf, "int %s_get(int net);\n", pf);
	fprintf(hf, "void %s_set(int net, int state);\n", pf);
	fprintf(hf, "\n");

	if (checkopt("vcd")) {
		fprintf(hf, "#define %s_VCD 1\n", pfUC);
		fprintf(hf, "extern void %s_vcd(FILE *f);\n", pf);
		fprintf(hf, "\n");
	}

	if (graphEnable) {
		fprintf(hf, "#define %s_DOT 1\n", pfUC);
		fprintf(hf, "extern int %s_dotChanged;\n", pf);
		fprintf(hf, "extern void %s_dotprint(FILE *f);\n", pf);
		fprintf(hf, "extern void %s_dot(FILE *f);\n", pf);
		fprintf(hf, "\n");
	}

	fprintf(hf, "extern int %s_stopped;\n", pf);
	fprintf(hf, "extern int %s_transitionCounter;\n", pf);
	fprintf(hf, "extern int %s_netevalCounter;\n", pf);
	fprintf(hf, "\n");

	fprintf(hf, "#endif\n");

	fclose(hf);
	fprintf(stderr, "Created simulation file `%s'.\n", hfn.c_str());

	fprintf(cf, "#include \"%s\"\n", hfn.c_str());
	fprintf(cf, "#include <assert.h>\n");

	if (opt("csim-rand-jitter", "") != "") {
		fprintf(cf, "#include <stdlib.h>\n");
		fprintf(cf, "static unsigned short %s_xsubi[3];\n", pf);
	}

	fprintf(cf, "int %s_stopped;\n", pf);
	fprintf(cf, "int %s_transitionCounter;\n", pf);
	fprintf(cf, "int %s_netevalCounter;\n", pf);

	fprintf(cf, "static int evalQueueTop1;\n");
	fprintf(cf, "static int evalQueueTop2;\n");
	fprintf(cf, "static int evalQueue1[%d];\n", 3*numNets);
	fprintf(cf, "static int evalQueue2[%d];\n", 3*numNets / 2);
	fprintf(cf, "static unsigned char queueState[%d];\n", numNets);
	fprintf(cf, "static int queuePtr[%d];\n", numNets);
	fprintf(cf, "static unsigned char driveState[%d];\n", numNets);
	fprintf(cf, "static unsigned char finalState[%d];\n", numNets);
	fprintf(cf, "static unsigned char evalStage[%d];\n", numNets);

	fprintf(cf, "static const unsigned char initDriveState[] = {\n");
	for (int i = 0; i < numNets; i++)
		fprintf(cf, "%d%s", netByNum[i]->driveState,
				i == numNets-1 ? "\n" : i % 20 == 19 ? ",\n" : ", ");
	fprintf(cf, "};\n");

	fprintf(cf, "static const unsigned char initBiasState[] = {\n");
	for (int i = 0; i < numNets; i++)
		fprintf(cf, "%d%s", netByNum[i]->biasState,
				i == numNets-1 ? "\n" : i % 20 == 19 ? ",\n" : ", ");
	fprintf(cf, "};\n");

	std::map<Switch*, int> swOnOffMap;
	std::map<std::pair<Net*,Net*>, int> nl_cache;

	fprintf(cf, "static const int neighList[] = { // IDs of neighbour nets\n");
	int totalNeighLinks = 0;
	std::vector<int> neighPtrData;
	for (int i = 0; i < numNets; i++) {
		Net *n = netByNum[i];
		assert(n->supply == (i < numSupplies));
		neighPtrData.push_back(totalNeighLinks);
		if (i < numSupplies || n->switchCC.size() == 0)
			continue;
		for (auto j = n->switchCC.begin(); j != n->switchCC.end(); j++) {
			Net *other = (*j)->cc[0] == n ? (*j)->cc[1] : (*j)->cc[0];
			fprintf(cf, "%d, ", other->num);
			nl_cache[std::pair<Net*,Net*>(n, other)] = totalNeighLinks;
			swOnOffMap[*j] = totalNeighLinks;
			totalNeighLinks++;
		}
		fprintf(cf, "// %d\n", n->num);
	}
	fprintf(cf, "};\n");
	assert(int(neighPtrData.size()) == numNets);

	fprintf(cf, "static const int neighPtr[] = { // Start indices in neighList per net\n");
	for (int i = 0; i < numNets; i++)
		fprintf(cf, "%d,%c", neighPtrData[i], i % 25 == 24 ? '\n' : ' ');
	fprintf(cf, "%d\n};\n", totalNeighLinks);
	fprintf(cf, "static unsigned char neighOnOffState[%d];\n", totalNeighLinks);

	for (auto i = switchTypes.begin(); i != switchTypes.end(); i++) {
		const char *stp = i->first.c_str();
		SwitchType *st = i->second;
		int totalGatesLinks = 0;
		std::vector<int> switchPtrData;
		std::vector<int> switchCcStateIdx;
		fprintf(cf, "static const int sw_%s_CC_Net[] = { // IDs of nets to schedule\n", stp);
		for (int j = 0; j < numNets; j++) {
			Net *n = netByNum[j];
			switchPtrData.push_back(totalGatesLinks);
			for (auto k = n->switchGates.begin(); k != n->switchGates.end(); k++) {
				Switch *s = *k;
				if (s->type != st)
					continue;
				for (int l = 0; l < 2; l++) {
					Net *n1 = s->cc[l];
					Net *n2 = s->cc[(l+1)%2];
					if (n1->supply)
						continue;
					fprintf(cf, "%d, ", n1->num);
					switchCcStateIdx.push_back(nl_cache[std::pair<Net*,Net*>(n1, n2)]);
					totalGatesLinks++;
				}
			}
			if (totalGatesLinks != switchPtrData.back())
				fprintf(cf, "// %d\n", n->num);
		}
		fprintf(cf, "};\n");
		fprintf(cf, "static const int sw_%s_CC_State[] = { // Index in neighOnOffState\n", stp);
		for (int i = 0; i < int(switchCcStateIdx.size()); i++)
			fprintf(cf, "%d,%c", switchCcStateIdx[i], i % 25 == 24 ? '\n' : ' ');
		fprintf(cf, "};\n");
		fprintf(cf, "static const int sw_%s_Ptr[] = { // Start indices CC arrays\n", stp);
		for (int i = 0; i < numNets; i++)
			fprintf(cf, "%d,%c", switchPtrData[i], i % 25 == 24 ? '\n' : ' ');
		fprintf(cf, "%d\n};\n", totalGatesLinks);
		fprintf(cf, "static const int sw_%s_OnOff[] = {", stp);
		for (int i = 0; i < numStates; i++) {
			if (st->onStates.count(i) != 0)
				fprintf(cf, "1,");
			else
				fprintf(cf, "0,");
		}
		fprintf(cf, "};\n");
	}

	fprintf(cf, "static const int stateAscend[] = {");
	for (int i = 0; i < numStates; i++) {
		fprintf(cf, "%d,", stateAscend[i]);
	}
	fprintf(cf, "};\n");

	fprintf(cf, "static const int stateCombine[%d][%d] = {\n", numStates, numStates);
	for (int i = 0; i < numStates; i++) {
		fprintf(cf, "  { ");
		for (int j = 0; j < numStates; j++)
			fprintf(cf, "%d%s", stateCombine[std::pair<int,int>(i, j)], j == numStates-1 ? "" : ", ");
		fprintf(cf, " }%s\n", i == numStates-1 ? "" : ",");
	}
	fprintf(cf, "};\n");

	if (opt("vcd", "none").compare("none") == 0) {
		/* nothing to do */
	} else
	if (opt("vcd", "none").compare("fourstate") == 0)
	{
		fprintf(cf, "static const char vcd_state[] = {");
		for (int i = 0; i < numStates; i++) {
			static const char fourstates[4] = { '1', '0', 'x', 'z' };
			int isstate[4] = { 0, 0, 0, 0 };
			for (int k = 0; k < 4; k++) {
				char optname[6] = { 'v', 'c', 'd', '-', fourstates[k], 0 };
				options_used[optname] = true;
				for (auto j = options[optname].begin(); j != options[optname].end(); j++) {
					if (str2state.count(*j) == 0) {
						fprintf(stderr, "Unkown state name `%s' in option `%s'.\n", j->c_str(), optname);
						exit(1);
					}
					if (state2str[i].compare(*j) == 0)
						isstate[k]++;
				}
			}
			if (isstate[0] + isstate[1] + isstate[2] + isstate[3] != 1) {
				fprintf(stderr, "State `%s' assigned to none or multiple vcd states.\n", state2str[i].c_str());
				exit(1);
			}
			fprintf(cf, " '%c'%s", isstate[0]*fourstates[0] + isstate[1]*fourstates[1] + isstate[2]*fourstates[2] +
					isstate[3]*fourstates[3], i < numStates-1 ? ", " : " };\n");
		}

		fprintf(cf, "static int vcd_init = 0;\n");
		fprintf(cf, "static FILE *vcdf = NULL;\n");
		fprintf(cf, "void %s_vcd(FILE *f) {\n", pf);
		fprintf(cf, "  if (vcdf != NULL) {\n");
		fprintf(cf, "    fprintf(vcdf, \"#%%d\\n\", %s_transitionCounter);\n", pf);
		fprintf(cf, "  }\n");
		fprintf(cf, "  vcdf = f;\n");
		fprintf(cf, "  if (vcdf != NULL) {\n");
		fprintf(cf, "    fprintf(vcdf, \"$comment Created by tnltool / %s $end\\n\");\n", cfn.c_str());
		std::set<Net*> vcd_nets;
		for (auto i = id2net.begin(); i != id2net.end(); i++) {
			std::string netlabel = i->first;
			Net *n = i->second;
			options_used["vcd-signals"] = true;
			for (auto j = options["vcd-signals"].begin(); j != options["vcd-signals"].end(); j++) {
				std::string pattern = *j;
				if (fnmatch(pattern.c_str(), netlabel.c_str(), 0) == 0) {
					vcd_nets.insert(n);
					fprintf(cf, "    fprintf(vcdf, \"$var reg 1 n%d %s $end\\n\");\n", n->num, netlabel.c_str());
				}
			}
		}
		fprintf(cf, "    fprintf(vcdf, \"$enddefinitions\\n\");\n");
		fprintf(cf, "    vcd_init = 1;\n");
		fprintf(cf, "  }\n");
		fprintf(cf, "}\n");

		fprintf(cf, "static unsigned char vcd_map[] = {\n");
		for (int i = 0; i < numNets; i++)
			fprintf(cf, "%c%s", vcd_nets.count(netByNum[i]) ? '1' : '0',
					i == numNets-1 ? "\n" : i % 40 == 39 ? ",\n" : ", ");
		fprintf(cf, "};\n");

	}
	else {
		fprintf(stderr, "Unkown vcd mode: `%s'.\n", opt("vcd", "none").c_str());
		exit(1);
	}

	if (graphEnable)
	{
		fprintf(cf, "int %s_dotChanged;\n", pf);
		fprintf(cf, "static FILE *dot_file = NULL;\n");

		fprintf(cf, "void %s_dotprint(FILE *f) {\n", pf);

		if (checkopt("graph-steps")) {
			for (auto i = options["graph-steps"].begin(); i != options["graph-steps"].end(); i++) {
				size_t split = i->find(":");
				if (split == std::string::npos) {
					fprintf(cf, "      if (%s_transitionCounter == %s)\n", pf, i->c_str());
					fprintf(cf, "        goto dotprint_ok;\n");
				} else {
					std::string lower = i->substr(0, split), upper = i->substr(split+1);
					fprintf(cf, "      if (%s <= %s_transitionCounter && %s_transitionCounter <= %s)\n",
							lower.c_str(), pf, pf, upper.c_str());
					fprintf(cf, "        goto dotprint_ok;\n");
				}
			}
			fprintf(cf, "  return;\n");
			fprintf(cf, "dotprint_ok:\n");
		}
		fprintf(cf, "  fprintf(f, \"strict digraph %s_%%d {\\n\", %s_transitionCounter);\n", pf, pf);
		fprintf(cf, "  fprintf(f, \"rankdir = \\\"LR\\\"; label = \\\"%s -- %%d\\\";\\n\", %s_transitionCounter);\n", pf, pf);
		for (auto i = graphInfo.netInfo.begin(); i != graphInfo.netInfo.end(); i++) {
			Net *net = i->first;
			auto &gi = i->second;
			for (int k = 0; k < numStates; k++) {
				fprintf(cf, "  if (finalState[%d] == %d || (finalState[%d] == %d && driveState[%d] == %d))\n",
						net->num, k, net->num, numStates, net->num, k);
				fprintf(cf, "    fputs(\"%s\", f);\n", escapeC(gi[k].c_str()));
			}
		}
		for (auto i = graphInfo.switchInfo.begin(); i != graphInfo.switchInfo.end(); i++) {
			Switch *sw = i->first;
			auto &gi = i->second;
			for (int k = 0; k < 2; k++) {
				fprintf(cf, "  if (neighOnOffState[%d] == %d)\n", swOnOffMap[sw], k);
				fprintf(cf, "    fputs(\"%s\", f);\n", escapeC(gi[k].c_str()));
			}
		}
		fprintf(cf, "  fprintf(f, \"}\\n\");\n");
		fprintf(cf, "  %s_dotChanged = 0;\n", pf);
		fprintf(cf, "}\n");

		fprintf(cf, "static const char dot_netMap[] = {\n");
		for (int i = 0; i < numNets; i++)
			fprintf(cf, "%c%s", netByNum[i]->graph ? '1' : '0',
					i == numNets-1 ? "\n" : i % 40 == 39 ? ",\n" : ", ");
		fprintf(cf, "};\n");

		fprintf(cf, "void %s_dot(FILE *f) {\n", pf);
		fprintf(cf, "  dot_file = f;\n");
		fprintf(cf, "  if (dot_file)\n");
		fprintf(cf, "    %s_dotprint(dot_file);\n", pf);
		fprintf(cf, "  %s_dotChanged = 1;\n", pf);
		fprintf(cf, "}\n");
	}

	fprintf(cf, "void %s_init() {\n", pf);
	fprintf(cf, "  int i, net, onoff, endIdx, qPtr;\n");
	fprintf(cf, "  %s_stopped = 0;\n", pf);
	fprintf(cf, "  %s_transitionCounter = 0;\n", pf);
	fprintf(cf, "  %s_netevalCounter = 0;\n", pf);
	fprintf(cf, "  for (i = 0; i < %d; i++)\n", totalNeighLinks);
	fprintf(cf, "    neighOnOffState[i] = 0;\n");
	fprintf(cf, "  for (net = 0; net < %d; net++) {\n", numNets);
	fprintf(cf, "    driveState[net] = initDriveState[net];\n");
	fprintf(cf, "    finalState[net] = initBiasState[net];\n");
	fprintf(cf, "    evalStage[net] = 1;\n");
	fprintf(cf, "    assert(evalQueueTop1 < (int)(0.9*sizeof(evalQueue1)/sizeof(*evalQueue1)));\n");
	fprintf(cf, "    evalQueue1[evalQueueTop1++] = net;\n");
	for (auto i = switchTypes.begin(); i != switchTypes.end(); i++) {
		const char *stp = i->first.c_str();
		fprintf(cf, "    onoff = sw_%s_OnOff[finalState[net]];\n", stp);
		fprintf(cf, "    endIdx = sw_%s_Ptr[net+1];\n", stp);
		fprintf(cf, "    for (i = sw_%s_Ptr[net]; i < endIdx; i++) {\n", stp);
		fprintf(cf, "      qPtr = sw_%s_CC_State[i];\n", stp);
		fprintf(cf, "      neighOnOffState[qPtr] = onoff;\n");
		fprintf(cf, "    }\n");
	}
	fprintf(cf, "  }\n");
	if (opt("csim-rand-jitter", "") != "") {
		unsigned short xsubi[3] = { 0, 0, 0 };
		std::string optstr = opt("csim-rand-jitter", "");
		if (sscanf(optstr.c_str(), "%hx:%hx:%hx", &xsubi[0], &xsubi[1], &xsubi[2]) != 3) {
			fprintf(stderr, "Can't parse csim-rand-jitter seed `%s'.\n", optstr.c_str());
			exit(1);
		}
		fprintf(cf, "  %s_xsubi[0] = 0x%04x;\n", pf, xsubi[0]);
		fprintf(cf, "  %s_xsubi[1] = 0x%04x;\n", pf, xsubi[1]);
		fprintf(cf, "  %s_xsubi[2] = 0x%04x;\n", pf, xsubi[2]);
	}
	fprintf(cf, "}\n");

	fprintf(cf, "void %s_eval() {\n", pf);
	fprintf(cf, "  int i, net, neigh, endIdx, qPtr, onoff;\n");
	fprintf(cf, "  int evalQueueHead = 0;\n");
	fprintf(cf, "  unsigned char state;\n");

	if (opt("vcd", "none").compare("fourstate") == 0) {
		fprintf(cf, "    if (vcdf != NULL && vcd_init) {\n");
		fprintf(cf, "      fprintf(vcdf, \"#%%d $dumpall\", %s_transitionCounter);\n", pf);
		fprintf(cf, "      for (i = 0; i < %d; i++)\n", numNets);
		fprintf(cf, "        if (vcd_map[i])\n");
		fprintf(cf, "          fprintf(vcdf, \" %%cn%%d\", vcd_state[finalState[i]], i);\n");
		fprintf(cf, "      fprintf(vcdf, \" $end\\n\");\n");
		fprintf(cf, "      vcd_init = 0;\n");
		fprintf(cf, "    }\n");
	}

	fprintf(cf, "  while (evalQueueTop1 || evalQueueTop2) {\n");
	fprintf(cf, "    %s_transitionCounter++;\n", pf);

	if (opt("csim-max-transitions", "") != "") {
		fprintf(cf, "    if (%s_transitionCounter >= %s) {\n", pf, opt("csim-max-transitions", "").c_str());
		fprintf(cf, "      if (%s_transitionCounter == %s)\n", pf, opt("csim-max-transitions", "").c_str());
		fprintf(cf, "        fprintf(stderr, \"Max transitions reached. Stopping simulation of `%s'.\\n\");\n", pf);
		fprintf(cf, "      %s_stopped = 1;\n", pf);
		fprintf(cf, "      return;\n");
		fprintf(cf, "    }\n");
	}

	if (opt("csim-rand-jitter", "") != "") {
		fprintf(cf, "    if (evalQueueTop1 > 1 && nrand48(%s_xsubi) %% 2 == 0) {\n", pf);
		fprintf(cf, "      evalQueueHead = nrand48(%s_xsubi) %% evalQueueTop1;\n", pf);
		fprintf(cf, "      for (i = evalQueueTop1; i > evalQueueHead; i--) {\n");
		fprintf(cf, "        net = nrand48(%s_xsubi) %% i;\n", pf);
		fprintf(cf, "        neigh = evalQueue1[i-1];\n");
		fprintf(cf, "        evalQueue1[i-1] = evalQueue1[net];\n");
		fprintf(cf, "        evalQueue1[net] = neigh;\n");
		fprintf(cf, "      }\n");
		fprintf(cf, "    } else\n");
		fprintf(cf, "      evalQueueHead = 0;\n");
	}

	fprintf(cf, "    for (i = evalQueueHead; i < evalQueueTop1; i++) {\n");
	fprintf(cf, "      net = evalQueue1[i];\n");
	fprintf(cf, "      queuePtr[net] = net;\n");
	fprintf(cf, "      queueState[net] = %d;\n", weakestState);
	fprintf(cf, "    }\n");

	fprintf(cf, "    while (evalQueueTop1 > evalQueueHead) {\n");
	fprintf(cf, "      net = evalQueue1[--evalQueueTop1];\n");
	fprintf(cf, "      if (evalStage[net] != 1)\n");
	fprintf(cf, "        continue;\n");
	fprintf(cf, "      evalStage[net] = 2;\n");
	fprintf(cf, "      assert(evalQueueTop1 < (int)(0.9*sizeof(evalQueue2)/sizeof(*evalQueue2)));\n");
	fprintf(cf, "      evalQueue2[evalQueueTop2++] = net;\n");
	fprintf(cf, "      qPtr = queuePtr[net];\n"); 
	fprintf(cf, "      state = queueState[qPtr];\n");
	fprintf(cf, "      state = stateCombine[state][driveState[net]];\n");
	fprintf(cf, "      state = stateCombine[state][stateAscend[finalState[net]]];\n");
	fprintf(cf, "      endIdx = neighPtr[net+1];\n");
	fprintf(cf, "      for (i = neighPtr[net]; i < endIdx; i++) {\n");
	fprintf(cf, "        if (!neighOnOffState[i])\n");
	fprintf(cf, "          continue;\n");
	fprintf(cf, "        neigh = neighList[i];\n");
	fprintf(cf, "        if (neigh < %d) {\n", numSupplies);
	fprintf(cf, "          state = stateCombine[state][driveState[neigh]];\n");
	fprintf(cf, "          continue;\n");
	fprintf(cf, "        }\n");
	fprintf(cf, "        if (evalStage[neigh] == 2)\n");
	fprintf(cf, "          continue;\n");
	fprintf(cf, "        assert(evalQueueTop1 < (int)(0.9*sizeof(evalQueue1)/sizeof(*evalQueue1)));\n");
	fprintf(cf, "        evalQueue1[evalQueueTop1++] = neigh;\n");
	fprintf(cf, "        queuePtr[neigh] = qPtr;\n");
	fprintf(cf, "        evalStage[neigh] = 1;\n");
	fprintf(cf, "      }\n");
	fprintf(cf, "      queueState[qPtr] = state;\n");
	fprintf(cf, "    }\n");

	fprintf(cf, "    while (evalQueueTop2) {\n");
	fprintf(cf, "      %s_netevalCounter++;\n", pf);
	fprintf(cf, "      net = evalQueue2[--evalQueueTop2];\n");
	fprintf(cf, "      if (evalStage[net] == 2)\n");
	fprintf(cf, "        evalStage[net] = 0;\n");
	fprintf(cf, "      state = net < %d ? driveState[net] : queueState[queuePtr[net]];\n", numSupplies);
	fprintf(cf, "      if (state == finalState[net])\n");
	fprintf(cf, "        continue;\n");
	fprintf(cf, "      finalState[net] = state;\n");
	if (opt("vcd", "none").compare("fourstate") == 0) {
		fprintf(cf, "      if (vcd_map[net])\n");
		fprintf(cf, "        fprintf(vcdf, \"#%%d %%cn%%d\\n\", %s_transitionCounter, vcd_state[state], net);\n", pf);
	}
	if (graphEnable) {
		fprintf(cf, "      if (dot_netMap[net])\n");
		fprintf(cf, "        %s_dotChanged = 1;\n", pf);
	}
	for (auto i = switchTypes.begin(); i != switchTypes.end(); i++) {
		const char *stp = i->first.c_str();
		fprintf(cf, "      onoff = sw_%s_OnOff[state];\n", stp);
		fprintf(cf, "      endIdx = sw_%s_Ptr[net+1];\n", stp);
		fprintf(cf, "      for (i = sw_%s_Ptr[net]; i < endIdx; i++) {\n", stp);
		fprintf(cf, "        qPtr = sw_%s_CC_State[i];\n", stp);
		fprintf(cf, "        if (onoff == neighOnOffState[qPtr])\n");
		fprintf(cf, "          continue;\n");
		fprintf(cf, "        neighOnOffState[qPtr] = onoff;\n");
		fprintf(cf, "        neigh = sw_%s_CC_Net[i];\n", stp);
		fprintf(cf, "        if (evalStage[neigh] == 1)\n");
		fprintf(cf, "          continue;\n");
		fprintf(cf, "        assert(evalQueueTop1 < (int)(0.9*sizeof(evalQueue1)/sizeof(*evalQueue1)));\n");
		fprintf(cf, "        evalQueue1[evalQueueTop1++] = neigh;\n");
		fprintf(cf, "        evalStage[neigh] = 1;\n");
		fprintf(cf, "      }\n");
	}
	fprintf(cf, "    }\n");

	if (graphEnable) {
		fprintf(cf, "    if (dot_file != NULL && %s_dotChanged) {\n", pf);
		fprintf(cf, "      %s_dotprint(dot_file);\n", pf);
		fprintf(cf, "    }\n");
	}

	fprintf(cf, "  }\n");
	fprintf(cf, "}\n");

	fprintf(cf, "int %s_get(int net) {\n", pf);
	fprintf(cf, "  assert(0 <= net && net < %d);\n", numNets);
	fprintf(cf, "  return finalState[net];\n");
	fprintf(cf, "}\n");

	fprintf(cf, "void %s_set(int net, int state) {\n", pf);
	fprintf(cf, "  assert(%d <= net && net < %d);\n", numSupplies, numNets);
	fprintf(cf, "  assert(0 <= state && state < %d);\n", numStates);
	fprintf(cf, "  driveState[net] = state;\n");
	fprintf(cf, "  if (evalStage[net] == 0) {\n");
	fprintf(cf, "    assert(evalQueueTop1 < (int)(0.9*sizeof(evalQueue1)/sizeof(*evalQueue1)));\n");
	fprintf(cf, "    evalQueue1[evalQueueTop1++] = net;\n");
	fprintf(cf, "    evalStage[net] = 1;\n");
	fprintf(cf, "  }\n");
	fprintf(cf, "}\n");

	fclose(cf);
	fprintf(stderr, "Created simulation file `%s'.\n", cfn.c_str());
}


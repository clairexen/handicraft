
#include "kernel/register.h"
#include "kernel/sigtools.h"
#include "kernel/celltypes.h"
#include "kernel/log.h"

#include <algorithm>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

struct SimGenerator
{
	RTLIL::Design *design;
	RTLIL::Module *module;
	CellTypes ct;
	SigMap sigmap;

	std::vector<RTLIL::Wire*> idx2wire;
	std::map<RTLIL::Wire*, int> wire2idx;

	struct node_t {
		std::vector<int> inputs;
		int unsorted_users, sorted_users;
		char type;
	};

	std::vector<node_t> idx2node;
	std::vector<int> sortedNodes;

	SimGenerator(RTLIL::Design *design, RTLIL::Module *module) : design(design), module(module), sigmap(module)
	{
		ct.setup_internals();
		ct.setup_stdcells();
		ct.setup_design(design);

		for (auto it : module->wires) {
			RTLIL::Wire *wire = it.second;
			if (wire->width != 1)
				log_error("Multi-bit wire %s not supported!\n", RTLIL::id2cstr(wire->name));
			wire2idx[wire] = idx2wire.size();
			idx2wire.push_back(wire);
			log("Mapping signal %s to state variable %d.\n", log_signal(wire), wire2idx[wire]);
		}

		idx2node.resize(idx2wire.size());

		for (auto it : module->cells) {
			RTLIL::Cell *cell = it.second;
			if (cell->type == "$and") {
				int a_idx = wire2idx.at(cell->connections.at("\\A").chunks.at(0).wire);
				int b_idx = wire2idx.at(cell->connections.at("\\B").chunks.at(0).wire);
				int y_idx = wire2idx.at(cell->connections.at("\\Y").chunks.at(0).wire);
				if (idx2node.at(y_idx).type != 0)
					goto conflict;
				idx2node.at(y_idx).inputs.push_back(a_idx);
				idx2node.at(y_idx).inputs.push_back(b_idx);
				idx2node.at(y_idx).type = 'A';
			} else
			if (cell->type == "$or") {
				int a_idx = wire2idx.at(cell->connections.at("\\A").chunks.at(0).wire);
				int b_idx = wire2idx.at(cell->connections.at("\\B").chunks.at(0).wire);
				int y_idx = wire2idx.at(cell->connections.at("\\Y").chunks.at(0).wire);
				if (idx2node.at(y_idx).type != 0)
					goto conflict;
				idx2node.at(y_idx).inputs.push_back(a_idx);
				idx2node.at(y_idx).inputs.push_back(b_idx);
				idx2node.at(y_idx).type = 'O';
			} else
			if (cell->type == "$not") {
				int a_idx = wire2idx.at(cell->connections.at("\\A").chunks.at(0).wire);
				int y_idx = wire2idx.at(cell->connections.at("\\Y").chunks.at(0).wire);
				if (idx2node.at(y_idx).type != 0)
					goto conflict;
				idx2node.at(y_idx).inputs.push_back(a_idx);
				idx2node.at(y_idx).type = 'N';
			} else
			if (cell->type == "$reduce_or") {
				int y_idx = wire2idx.at(cell->connections.at("\\Y").chunks.at(0).wire);
				if (idx2node.at(y_idx).type != 0)
					goto conflict;
				RTLIL::SigSpec sig = cell->connections.at("\\A");
				sig.expand();
				for (size_t i = 0; i < sig.chunks.size(); i++)
					idx2node.at(y_idx).inputs.push_back(wire2idx.at(sig.chunks[i].wire));
				idx2node.at(y_idx).type = 'O';
			} else
			if (cell->type == "\\SW0") {
				int gate_idx = wire2idx.at(cell->connections.at("\\gate").chunks.at(0).wire);
				int cc_idx = wire2idx.at(cell->connections.at("\\cc").chunks.at(0).wire);
				if (idx2node.at(cc_idx).type == '1') {
					idx2node.at(cc_idx).inputs.insert(idx2node.at(cc_idx).inputs.begin(), gate_idx);
					idx2node.at(cc_idx).type = 'S';
				} else {
					if (idx2node.at(cc_idx).type != 0)
						goto conflict;
					idx2node.at(cc_idx).inputs.push_back(gate_idx);
					idx2node.at(cc_idx).type = '0';
				}
			} else
			if (cell->type == "\\SW1") {
				int gate_idx = wire2idx.at(cell->connections.at("\\gate").chunks.at(0).wire);
				int cc_idx = wire2idx.at(cell->connections.at("\\cc").chunks.at(0).wire);
				if (idx2node.at(cc_idx).type == '0') {
					idx2node.at(cc_idx).inputs.push_back(gate_idx);
					idx2node.at(cc_idx).type = 'S';
				} else {
					if (idx2node.at(cc_idx).type != 0)
						goto conflict;
					idx2node.at(cc_idx).inputs.push_back(gate_idx);
					idx2node.at(cc_idx).type = '1';
				}
			} else
			if (cell->type == "\\SR") {
				int r_idx = wire2idx.at(cell->connections.at("\\R").chunks.at(0).wire);
				int s_idx = wire2idx.at(cell->connections.at("\\S").chunks.at(0).wire);
				int q_idx = wire2idx.at(cell->connections.at("\\Q").chunks.at(0).wire);
				if (idx2node.at(q_idx).type != 0)
					goto conflict;
				idx2node.at(q_idx).inputs.push_back(r_idx);
				idx2node.at(q_idx).inputs.push_back(s_idx);
				idx2node.at(q_idx).type = 'S';
			} else
			if (cell->type == "\\PULLUP") {
				/* ignore PULLUP cells */
			} else
				log_error("Cell type %s is not supported!\n", RTLIL::id2cstr(cell->type));
			if (0)
		conflict:
				log_error("Found conflicting drivers at cell %s (%s).\n", RTLIL::id2cstr(cell->name), RTLIL::id2cstr(cell->type));
		}

		for (auto conn : module->connections) {
			if (conn.first.width != 1)
				log_error("Only single bit connections are supported!\n");
			int l_idx = wire2idx.at(conn.first.chunks.at(0).wire);
			int r_idx = wire2idx.at(conn.second.chunks.at(0).wire);
			if (idx2node.at(l_idx).type != 0)
				log_error("Found conflicting drivers at connection for variable %d.\n", l_idx);
			idx2node.at(l_idx).inputs.push_back(r_idx);
			idx2node.at(l_idx).type = 'O';
		}

		// cheap and inefficient weak topological sorting

		int max_num_inputs = 0;
		std::vector<int> unsortedNodes;
		for (size_t i = 0; i < idx2node.size(); i++) {
			for (auto idx : idx2node[i].inputs)
				idx2node.at(idx).unsorted_users++;
			if (idx2node[i].type != 0)
				unsortedNodes.push_back(i);
			max_num_inputs = std::max(max_num_inputs, int(idx2node[i].inputs.size()));
		}
		log("Largest input port found in design: %d bits.\n", max_num_inputs);

		while (unsortedNodes.size() > 0)
		{
			size_t best_i = 0;
			for (size_t i = 0; i < unsortedNodes.size(); i++) {
				node_t &best_node = idx2node[unsortedNodes[best_i]];
				node_t &this_node = idx2node[unsortedNodes[i]];
				if (this_node.unsorted_users > best_node.unsorted_users)
					continue;
				if (this_node.unsorted_users < best_node.unsorted_users || this_node.sorted_users > best_node.sorted_users)
					best_i = i;
			}

			for (int idx : idx2node[unsortedNodes[best_i]].inputs) {
				idx2node[idx].unsorted_users--;
				idx2node[idx].sorted_users++;
			}

			sortedNodes.push_back(unsortedNodes[best_i]);
			unsortedNodes[best_i] = unsortedNodes.back();
			unsortedNodes.pop_back();
		}
	}

	void write(std::string include_guard, std::string h_filename, std::string cc_filename)
	{
		FILE *f = fopen(h_filename.c_str(), "wt");
		if (f == NULL)
			log_error("Can't open simulator *.h file for writing!\n");

		fprintf(f, "#ifndef %s\n", include_guard.c_str());
		fprintf(f, "#define %s\n", include_guard.c_str());
		fprintf(f, "#include <stdio.h>\n");

		fprintf(f, "struct sim_%s\n", RTLIL::id2cstr(module->name));
		fprintf(f, "{\n");

		fprintf(f, "\t// inputs: set to -1 for z-state\n");
		for (auto it : module->wires) {
			RTLIL::Wire *wire = it.second;
			if (wire->port_input)
				fprintf(f, "\tint input_%s;\n", RTLIL::id2cstr(wire->name));
		}
		fprintf(f, "\n");

		fprintf(f, "\t// outputs: set by update()\n");
		for (auto it : module->wires) {
			RTLIL::Wire *wire = it.second;
			if (wire->port_output)
				fprintf(f, "\tbool output_%s;\n", RTLIL::id2cstr(wire->name));
		}
		fprintf(f, "\n");

		fprintf(f, "\t// internal state\n");
		fprintf(f, "\tstatic const int num_states = %d;\n", int(idx2wire.size()));
		fprintf(f, "\tbool state[%d];\n", int(idx2wire.size()));
		fprintf(f, "\tstatic const char *netnames[%d];\n", int(idx2wire.size()));
		fprintf(f, "\n");
		fprintf(f, "\t// vcd engine\n");
		fprintf(f, "\tint vcd_time;\n");
		fprintf(f, "\tFILE *vcd_file;\n");
		fprintf(f, "\tvoid vcd_init(FILE *f);\n");
		fprintf(f, "\tvoid vcd_step(FILE *f, int mark = -1);\n");
		fprintf(f, "\n");
		fprintf(f, "\tsim_%s();\n", RTLIL::id2cstr(module->name));
		fprintf(f, "\tbool step(bool shuffle = false);\n");
		fprintf(f, "\tvoid dump(const char *filename);\n");
		fprintf(f, "\tvoid load(const char *filename);\n");
		fprintf(f, "\tvoid update();\n");
		fprintf(f, "\tbool try_update();\n");
		fprintf(f, "\tvoid init();\n");
		fprintf(f, "};\n");

		fprintf(f, "#endif\n");

		fclose(f);

		f = fopen(cc_filename.c_str(), "wt");
		if (f == NULL)
			log_error("Can't open simulator *.cc file for writing!\n");

		fprintf(f, "#include \"%s\"\n", h_filename.c_str());
		fprintf(f, "#include \"simgen.h\"\n");

		fprintf(f, "const char *sim_%s::netnames[%d] = {\n", RTLIL::id2cstr(module->name), int(idx2wire.size()));
		for (int i = 0; i < int(idx2wire.size()); i++) {
			RTLIL::Wire *wire = idx2wire[i];
			fprintf(f, "\t\"%s\"%-*s// %5d\n", RTLIL::id2cstr(wire->name), int(30 - wire->name.size()), i+1 < int(idx2wire.size()) ? "," : "", i);
		}
		fprintf(f, "};\n");

		fprintf(f, "sim_%s::sim_%s() {\n", RTLIL::id2cstr(module->name), RTLIL::id2cstr(module->name));
		fprintf(f, "\tfor (int i = 0; i < %d; i++)\n", int(module->wires.size()));
		fprintf(f, "\t\tstate[i] = false;\n");
		for (auto it : module->wires) {
			RTLIL::Wire *wire = it.second;
			if (wire->port_input)
				fprintf(f, "\tinput_%s = -1;\n", RTLIL::id2cstr(wire->name));
			if (wire->port_output)
				fprintf(f, "\toutput_%s = false;\n", RTLIL::id2cstr(wire->name));
		}
		fprintf(f, "\tvcd_time = 0;\n");
		fprintf(f, "\tvcd_file = NULL;\n");
		fprintf(f, "}\n");
		fprintf(f, "\n");

		fprintf(f, "void sim_%s::vcd_init(FILE *f) {\n", RTLIL::id2cstr(module->name));
		fprintf(f, "\tsimgen_vcd_init(vcd_time, state, netnames, %d, f);\n", int(idx2wire.size()));
		fprintf(f, "}\n");

		fprintf(f, "void sim_%s::vcd_step(FILE *f, int mark) {\n", RTLIL::id2cstr(module->name));
		fprintf(f, "\tsimgen_vcd_step(vcd_time, state, %d, f, true, mark);\n", int(idx2wire.size()));
		fprintf(f, "}\n");

		fprintf(f, "bool sim_%s::step(bool shuffle) {\n", RTLIL::id2cstr(module->name));
		fprintf(f, "\tbool newState[%d];\n", int(module->wires.size()));
		fprintf(f, "\tfor (int i = 0; i < %d; i++)\n", int(module->wires.size()));
		fprintf(f, "\t\tnewState[i] = state[i];\n");

		for (auto it : module->wires) {
			RTLIL::Wire *wire = it.second;
			if (wire->port_input)
				fprintf(f, "\tif (input_%s >= 0) newState[%d] = input_%s;\n",
						RTLIL::id2cstr(wire->name), wire2idx.at(wire), RTLIL::id2cstr(wire->name));
		}

		fprintf(f, "\tstatic const simgen_node_t nodes[%d] = {\n", int(sortedNodes.size()));
		for (int i = sortedNodes.size()-1; i >= 0; i--) {
			node_t &node = idx2node[sortedNodes[i]];
			fprintf(f, "\t\t\t{ %5d, %2d, '%c', {", sortedNodes[i], int(node.inputs.size()), node.type);
			for (size_t j = 0; j < node.inputs.size(); j++)
				fprintf(f, "%s%5d", j ? ", " : " ", node.inputs[j]);
			fprintf(f, " } }%s\n", i > 0 ? "," : "");
		}
		fprintf(f, "\t};\n");

		fprintf(f, "\tbool did_something = false;\n");
		fprintf(f, "\tif (simgen_worker(newState, nodes, %d, false))\n", int(sortedNodes.size()));
		fprintf(f, "\t\tdid_something = true;\n");
		fprintf(f, "\telse\n");
		fprintf(f, "\tif (simgen_worker(newState, nodes, %d, true))\n", int(sortedNodes.size()));
		fprintf(f, "\t\tdid_something = true;\n");

		fprintf(f, "\tif (vcd_file)\n");
		fprintf(f, "\t\tsimgen_vcd_step(vcd_time, newState, %d, vcd_file, !did_something, -1);\n", int(idx2wire.size()));

		// list="11877:1->0 11878:1->0 11879:1->0 11880:1->0 11883:1->0 11903:1->0" logfile="simgen.log"
		// echo $( for id in $( echo "$list" | sed -r 's,:\S+\s*, ,g'; ); do grep "Mapping.*variable $id." $logfile; done | sort -u | cut -f3 -d' '; )

		fprintf(f, "\tif (did_something) {\n");
		fprintf(f, "\t\t// printf(\"changed:\");\n");
		fprintf(f, "\t\t// for (int i = 0; i < %d; i++)\n", int(module->wires.size()));
		fprintf(f, "\t\t// \tif (state[i] != newState[i])\n");
		fprintf(f, "\t\t// \t\tprintf(\" %%d:%%d->%%d\", i, state[i], newState[i]);\n");
		fprintf(f, "\t\t// printf(\"\\n\");\n");
		fprintf(f, "\t\tfor (int i = 0; i < %d; i++)\n", int(module->wires.size()));
		fprintf(f, "\t\t\tif (!shuffle || xorshift32() %% 2 == 0)\n");
		fprintf(f, "\t\t\t\tstate[i] = newState[i];\n");
		fprintf(f, "\t\treturn true;\n");
		fprintf(f, "\t}\n");

		for (auto it : module->wires) {
			RTLIL::Wire *wire = it.second;
			if (wire->port_output)
				fprintf(f, "\toutput_%s = newState[%d];\n", RTLIL::id2cstr(wire->name), wire2idx.at(wire));
		}

		fprintf(f, "\treturn false;\n");
		fprintf(f, "}\n");

		// gawk 'ARGIND == 1 { d[$1] = $2; } ARGIND == 2 && $1 in d && d[$1] != $2 { print "select -add " $1; }' cmpnets.dat1 cmpnets.dat2

		fprintf(f, "void sim_%s::dump(const char *filename) {\n", RTLIL::id2cstr(module->name));
		fprintf(f, "\tsimgen_dump(state, netnames, %d, filename);\n", int(idx2wire.size()));
		fprintf(f, "}\n");

		fprintf(f, "void sim_%s::load(const char *filename) {\n", RTLIL::id2cstr(module->name));
		fprintf(f, "\tsimgen_load(state, netnames, %d, filename);\n", int(idx2wire.size()));
		fprintf(f, "}\n");

		fprintf(f, "void sim_%s::update() {\n", RTLIL::id2cstr(module->name));
		fprintf(f, "\twhile (step()) { }\n");
		fprintf(f, "}\n");

		fprintf(f, "bool sim_%s::try_update() {\n", RTLIL::id2cstr(module->name));
		fprintf(f, "\tfor (int i = 0; step(); i++)\n");
		fprintf(f, "\t\tif (i > 100) return false;\n");
		fprintf(f, "\treturn true;\n");
		fprintf(f, "}\n");

		fprintf(f, "void sim_%s::init() {\n", RTLIL::id2cstr(module->name));
		fprintf(f, "\twhile (step()) step(true);\n");
		fprintf(f, "}\n");

		fclose(f);
	}
};

struct SimgenPass : public Pass {
	SimgenPass() : Pass("simgen") { }
	virtual void execute(std::vector<std::string> args, RTLIL::Design *design)
	{
		log_header("Executing SIMGEN pass (convert set-reset-logic to C++).\n");
		extra_args(args, 1, design);

		for (auto &it : design->modules)
			if (design->selected_whole_module(it.first) && !it.second->attributes.count("\\placeholder")) {
				SimGenerator simgen(design, it.second);
				simgen.write("SIM_" + RTLIL::unescape_id(it.first), "sim_" + RTLIL::unescape_id(it.first) + ".h", "sim_" + RTLIL::unescape_id(it.first) + ".cc");
			}
	}
} SimgenPass;


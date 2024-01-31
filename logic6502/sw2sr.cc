
#include "kernel/register.h"
#include "kernel/sigtools.h"
#include "kernel/celltypes.h"
#include "kernel/log.h"
#include <algorithm>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

using RTLIL::id2cstr;

static int clusterCounter, cellCounter;

struct Sw2srSolver
{
	RTLIL::Design *design;
	RTLIL::Module *module;
	CellTypes ct;
	SigMap sigmap;

	std::map<RTLIL::SigSpec, std::set<RTLIL::Cell*>> wire2cc, wire2gates, wire2pullup;
	std::map<RTLIL::Cell*, std::set<RTLIL::SigSpec>> switch2cc;
	std::map<RTLIL::Cell*, RTLIL::SigSpec> switch2gate, pullup2wire;
	std::set<RTLIL::Cell*> switchOff, switchOn;
	std::set<RTLIL::SigSpec> workQueue;
	SigPool inputs, outputs;

	std::set<RTLIL::SigSpec> findExprStack;
	std::map<std::string, RTLIL::SigSpec> exprCache;

	RTLIL::SigSpec sigbit(const RTLIL::SigSpec &sig) {
		assert(sig.width == 1);
		return sigmap(sig);
	}

	std::string hashCell(RTLIL::Cell *cell)
	{
		if (cell->type == "\\SW" || cell->type == "\\SW0" || cell->type == "\\SW1")
		{
			RTLIL::SigSpec gate = sigbit(cell->connections.at("\\gate"));
			RTLIL::SigSpec cc1, cc2;
			
			if (cell->type == "\\SW") {
				cc1 = sigbit(cell->connections.at("\\cc1"));
				cc2 = sigbit(cell->connections.at("\\cc2"));
			} else {
				cc1 = sigbit(cell->connections.at("\\cc"));
			}

			if (cc1 == cc2)
				return "DELETE";

			if (cc1 < cc2) {
				RTLIL::SigSpec tmp = cc1;
				cc1 = cc2, cc2 = tmp;
			}

			return stringf("%s %s %s %s", cell->type.c_str(), log_signal(gate), log_signal(cc1), log_signal(cc2));
		}

		if (cell->type == "\\PULLUP")
		{
			RTLIL::SigSpec wire = sigbit(cell->connections.at("\\y"));
			return stringf("%s %s", cell->type.c_str(), log_signal(wire));
		}

		return std::string();
	}

	void registerCell(RTLIL::Cell *cell)
	{
		if (cell->type == "\\SW" || cell->type == "\\SW0" || cell->type == "\\SW1")
		{
			RTLIL::SigSpec gate = sigbit(cell->connections.at("\\gate"));

			wire2gates[gate].insert(cell);
			switch2gate[cell] = gate;
			outputs.add(gate);

			if (cell->type == "\\SW") {
				RTLIL::SigSpec cc1 = sigbit(cell->connections.at("\\cc1"));
				RTLIL::SigSpec cc2 = sigbit(cell->connections.at("\\cc2"));
				wire2cc[cc1].insert(cell), switch2cc[cell].insert(cc1), workQueue.insert(cc1);
				wire2cc[cc2].insert(cell), switch2cc[cell].insert(cc2), workQueue.insert(cc2);
			} else {
				RTLIL::SigSpec cc = sigbit(cell->connections.at("\\cc"));
				wire2cc[cc].insert(cell), switch2cc[cell].insert(cc), workQueue.insert(cc);
			}

			if (cell->type == "\\SW0")
				switchOff.insert(cell);
			if (cell->type == "\\SW1")
				switchOn.insert(cell);
		}
		else
		if (cell->type == "\\PULLUP")
		{
			RTLIL::SigSpec wire = sigbit(cell->connections.at("\\y"));
			wire2pullup[wire].insert(cell);
			pullup2wire[cell] = wire;
		}
		else
		{
			for (auto &conn : cell->connections)
			{
				if (!ct.cell_known(cell->type) || ct.cell_output(cell->type, conn.first))
					inputs.add(sigmap(conn.second));
				if (!ct.cell_known(cell->type) || ct.cell_input(cell->type, conn.first))
					outputs.add(sigmap(conn.second));
			}
		}
	}

	void deleteCell(RTLIL::Cell *cell)
	{
		assert(cell->type == "\\SW" || cell->type == "\\SW0" || cell->type == "\\SW1" || cell->type == "\\PULLUP");

		if (switch2cc.count(cell) > 0) {
			for (auto &it : switch2cc.at(cell))
				wire2cc[it].erase(cell);
			switch2cc.erase(cell);
		}

		if (switch2gate.count(cell) > 0) {
			wire2gates[switch2gate.at(cell)].erase(cell);
			switch2gate.erase(cell);
		}

		if (pullup2wire.count(cell) > 0) {
			wire2pullup[pullup2wire.at(cell)].erase(cell);
			pullup2wire.erase(cell);
		}

		switchOff.erase(cell);
		switchOn.erase(cell);

		module->cells.erase(cell->name);
		delete cell;
	}

	Sw2srSolver(RTLIL::Design *design, RTLIL::Module *module) : design(design), module(module), sigmap(module)
	{
		ct.setup_internals();
		ct.setup_internals_mem();
		ct.setup_stdcells();
		ct.setup_stdcells_mem();
		ct.setup_design(design);

		for (auto &it : module->wires) {
			if (it.second->port_input)
				inputs.add(sigmap(it.second));
			if (it.second->port_output)
				outputs.add(sigmap(it.second));
		}

		std::set<std::string> cellHash;
		std::vector<RTLIL::Cell*> deleteCells;

		for (auto &it : module->cells) {
			std::string hashString = hashCell(it.second);
			if (hashString != "DELETE" && (hashString.empty() || cellHash.count(hashString) == 0)) {
				cellHash.insert(hashString);
				registerCell(it.second);
			} else
				deleteCells.push_back(it.second);
		}

		log("Removing %d redundant switches.\n", int(deleteCells.size()));
		for (auto &it : deleteCells) {
			module->cells.erase(it->name);
			delete it;
		}
	}

	struct BooleanNode
	{
		RTLIL::SigSpec signal;
		std::set<BooleanNode*> children;
		std::string type, text;

		BooleanNode(const RTLIL::SigSpec &signal)
		{
			this->signal = signal;
		}

		BooleanNode(const std::string &type)
		{
			this->type = type;
		}

		BooleanNode *combine(const std::string &combineType, BooleanNode *other)
		{
			if (type == combineType) {
				children.insert(other);
				return this;
			}

			BooleanNode *node = new BooleanNode(combineType);
			node->children.insert(this);
			node->children.insert(other);
			return node;
		}

		std::string toString()
		{
			if (text.empty()) {
				if (signal.width > 0)
					return RTLIL::unescape_id(signal.chunks.at(0).wire->name);
				if (type == "0" || type == "1")
					return type;
				text = type + "(";
				for (auto &it : children) {
					if (it != *children.begin())
						text += ", ";
					text += it->toString();
				}
				text += ")";
			}
			return text;
		}

		void optimize()
		{
			std::set<BooleanNode*> oldChildren;
			std::set<std::string> childrenCache;
			oldChildren.swap(children);

			for (auto child : oldChildren)
			{
				child->optimize();

				if (child->type == "0") {
					if (type == "AND")
						type = "0";
					delete child;
					continue;
				}

				if (child->type == "1") {
					if (type == "OR")
						type = "1";
					delete child;
					continue;
				}

				std::string str = child->toString();
				if (childrenCache.count(str) > 0) {
					delete child;
					continue;
				}

				childrenCache.insert(str);
				children.insert(child);
			}

			if (type == "AND" && children.size() == 0)
				type = "1";
			if (type == "OR" && children.size() == 0)
				type = "0";

			if ((type == "AND" || type == "OR") && children.size() == 1) {
				BooleanNode *child = *children.begin();
				signal = child->signal;
				type = child->type;
				children.clear();
				children.swap(child->children);
				delete child;
			}
		}

		RTLIL::SigSpec genrtlil(RTLIL::Module *module, std::map<std::string, RTLIL::SigSpec> &cache)
		{
			if (signal.width > 0)
				return signal;

			if (type == "0")
				return RTLIL::SigSpec(0, 1);

			if (type == "1")
				return RTLIL::SigSpec(1, 1);

			std::string hash = toString();
			if (cache.count(hash) == 0)
			{
				RTLIL::Cell *cell = new RTLIL::Cell;
				cell->name = NEW_ID;
				module->add(cell);

				RTLIL::Wire *wire = new RTLIL::Wire;
				wire->name = NEW_ID;
				module->add(wire);

				if (children.size() == 2)
				{
					if (type == "AND")
						cell->type = "$and";
					if (type == "OR")
						cell->type = "$or";
					assert(!cell->type.empty());

					cell->parameters["\\A_SIGNED"] = RTLIL::Const(0);
					cell->parameters["\\B_SIGNED"] = RTLIL::Const(0);
					cell->parameters["\\A_WIDTH"] = RTLIL::Const(1);
					cell->parameters["\\B_WIDTH"] = RTLIL::Const(1);
					cell->parameters["\\Y_WIDTH"] = RTLIL::Const(1);

					cell->connections["\\A"] = (*children.begin())->genrtlil(module, cache);
					cell->connections["\\B"] = (*children.rbegin())->genrtlil(module, cache);
				}
				else
				{
					if (type == "AND")
						cell->type = "$reduce_and";
					if (type == "OR")
						cell->type = "$reduce_or";
					assert(!cell->type.empty());

					cell->parameters["\\A_SIGNED"] = RTLIL::Const(0);
					cell->parameters["\\A_WIDTH"] = RTLIL::Const(children.size());
					cell->parameters["\\Y_WIDTH"] = RTLIL::Const(1);

					for (auto &it : children)
						cell->connections["\\A"].append(it->genrtlil(module, cache));
				}

				cell->connections["\\Y"] = RTLIL::SigSpec(wire);
				cache[hash] = RTLIL::SigSpec(wire);
			}

			return cache.at(hash);
		}

		~BooleanNode()
		{
			for (auto &it : children)
				delete it;
		}
	};

	bool followCc(std::set<RTLIL::SigSpec> &signals, std::set<RTLIL::Cell *> &cells, const RTLIL::SigSpec &rootSignal)
	{
		bool ok = true;

		workQueue.erase(rootSignal);
		signals.insert(rootSignal);

		// no constant drivers (use PULLUP, SW0 and SW1 cells instead)
		if (rootSignal.chunks[0].wire == NULL)
			return false;

		// no inputs wires on pass transistors
		if (inputs.check_any(rootSignal))
			return false;

		for (auto &i1 : wire2pullup[rootSignal])
			cells.insert(i1);

		for (auto &i1 : wire2cc[rootSignal]) {
			for (auto &i2 : switch2cc[i1])
				if (signals.count(i2) == 0)
					if (!followCc(signals, cells, i2))
						ok = false;
			cells.insert(i1);
		}

		return ok;
	}

	BooleanNode *findExpr(const RTLIL::SigSpec &rootSignal, bool polarity)
	{
		BooleanNode *node = new BooleanNode("OR");

		assert(findExprStack.count(rootSignal) == 0);
		findExprStack.insert(rootSignal);

		if (polarity == true && wire2pullup[rootSignal].size() > 0) {
			BooleanNode *n = new BooleanNode("1");
			node->children.insert(n);
		}

		for (auto &i1 : wire2cc[rootSignal]) {
			for (auto &i2 : switch2cc[i1])
				if (findExprStack.count(i2) == 0) {
					BooleanNode *n1 = new BooleanNode(switch2gate[i1]);
					BooleanNode *n2 = findExpr(i2, polarity);
					BooleanNode *n = new BooleanNode("AND");
					n->children.insert(n1);
					n->children.insert(n2);
					node->children.insert(n);
				}
			if (polarity == false && switchOff.count(i1) > 0) {
				BooleanNode *n = new BooleanNode(switch2gate[i1]);
				node->children.insert(n);
			}
			if (polarity == true && switchOn.count(i1) > 0) {
				BooleanNode *n = new BooleanNode(switch2gate[i1]);
				node->children.insert(n);
			}
		}

		findExprStack.erase(rootSignal);
		return node;
	}

	void solve(RTLIL::SigSpec rootSignal)
	{
		std::set<RTLIL::SigSpec> signals;
		std::set<RTLIL::Cell *> cells;

		if (!followCc(signals, cells, rootSignal))
			return;

		clusterCounter++;
		cellCounter += cells.size();

		log("Found cluster with %d cells and %d signals:\n", int(cells.size()), int(signals.size()));

		log("   cells:");
		for (auto &it : cells)
			log(" %s", RTLIL::id2cstr(it->name));
		log("\n");

		log("   signals:");
		for (auto &it : signals)
			log(" %s", log_signal(it));
		log("\n");

		for (auto &signal : signals)
			if (outputs.check_any(signal))
			{
				log("   set-reset logic for signal %s:\n", log_signal(signal));

				BooleanNode *setExpr = findExpr(signal, true);
				BooleanNode *resetExpr = findExpr(signal, false);

				setExpr->optimize();
				resetExpr->optimize();

				log("      S: %s\n", setExpr->toString().c_str());
				log("      R: %s\n", resetExpr->toString().c_str());

				RTLIL::SigSpec setSignal = setExpr->genrtlil(module, exprCache);
				RTLIL::SigSpec resetSignal = resetExpr->genrtlil(module, exprCache);

				if (setSignal.is_fully_const() && setSignal.as_bool() == true)
				{
					RTLIL::Cell *cell = new RTLIL::Cell;
					cell->name = NEW_ID;
					cell->type = "$not";
					cell->parameters["\\A_SIGNED"] = RTLIL::Const(0);
					cell->parameters["\\A_WIDTH"] = RTLIL::Const(1);
					cell->parameters["\\Y_WIDTH"] = RTLIL::Const(1);
					cell->connections["\\A"] = resetSignal;
					cell->connections["\\Y"] = signal;
					module->add(cell);
				}
				else
				{
					RTLIL::Cell *cell = new RTLIL::Cell;
					cell->name = NEW_ID;
			#if 0
					cell->type = "$sr";
					cell->parameters["\\WIDTH"] = RTLIL::Const(1);
			#else
					cell->type = "\\SR";
			#endif
					cell->connections["\\S"] = setSignal;
					cell->connections["\\R"] = resetSignal;
					cell->connections["\\Q"] = signal;
					module->add(cell);
				}

				delete setExpr;
				delete resetExpr;
			}

		for (auto &cell : cells)
			deleteCell(cell);
	}

	void solve()
	{
		while (workQueue.size() > 0) {
			RTLIL::SigSpec sig = *workQueue.begin();
			solve(sig);
		}
	}
};

struct Sw2srPass : public Pass {
	Sw2srPass() : Pass("sw2sr") { }
	virtual void execute(std::vector<std::string> args, RTLIL::Design *design)
	{
		log_header("Executing SW2SR pass (convert switch level netlists to set-reset-logic).\n");
		extra_args(args, 1, design);

		clusterCounter = 0;
		cellCounter = 0;
		for (auto &it : design->modules)
			if (design->selected_whole_module(it.first)) {
				Sw2srSolver solver(design, it.second);
				solver.solve();
			}
		log("Processed %d switch clusters (%d cells).\n", clusterCounter, cellCounter);
	}
} Sw2srPass;


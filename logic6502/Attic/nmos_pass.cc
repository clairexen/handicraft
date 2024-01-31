
#include "kernel/register.h"
#include "kernel/sigtools.h"
#include "kernel/log.h"
#include <algorithm>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

using RTLIL::id2cstr;

static int cellCounter, switchCounter;

struct NmosSolver
{
	RTLIL::Design *design;
	RTLIL::Module *module;
	SigMap sigmap;

	std::map<RTLIL::IdString, RTLIL::SigSpec> switchToGateSignal;
	std::map<RTLIL::IdString, std::set<RTLIL::SigSpec>> switchToPassSignals;
	std::map<RTLIL::SigSpec, std::set<RTLIL::IdString>> signalToPassSwitch;
	std::map<RTLIL::SigSpec, std::set<RTLIL::IdString>> signalToOtherCell;
	std::vector<std::pair<RTLIL::IdString, RTLIL::SigSpec>> pullups;

	NmosSolver(RTLIL::Design *design, RTLIL::Module *module) : design(design), module(module), sigmap(module)
	{
		std::map<RTLIL::SigSpec, int> wireDriverCount;
		std::set<std::string> uniqueCells;
		std::map<std::string, std::vector<RTLIL::Cell*>> deleteCells;

		for (auto &it : module->cells)
		{
			if (it.second->type == "\\SW" || it.second->type == "\\SW0" || it.second->type == "\\SW1")
			{
				RTLIL::SigSpec gate, cc1, cc2;
				gate = sigmap(it.second->connections.at("\\gate"));
				if (it.second->type == "\\SW0") {
					cc1 = sigmap(it.second->connections.at("\\cc"));
					cc2 = RTLIL::SigSpec(0, 1);
				} else
				if (it.second->type == "\\SW1") {
					cc1 = sigmap(it.second->connections.at("\\cc"));
					cc2 = RTLIL::SigSpec(1, 1);
				} else {
					cc1 = sigmap(it.second->connections.at("\\cc1"));
					cc2 = sigmap(it.second->connections.at("\\cc2"));
				}
				assert(gate.width == 1 && cc1.width == 1 && cc2.width == 1);

				gate.optimize();
				cc1.optimize();
				cc2.optimize();

				if (!(cc1 < cc2)) {
					RTLIL::SigSpec tmp = cc1;
					cc1 = cc2, cc2 = tmp;
				}

				std::string uniqueId = stringf("SW %s %s %s", log_signal(gate), log_signal(cc1), log_signal(cc2));
				if (uniqueCells.count(uniqueId) > 0) {
					deleteCells[uniqueId].push_back(it.second);
					continue;
				}
				uniqueCells.insert(uniqueId);

				switchToPassSignals[it.first].insert(cc1);
				switchToPassSignals[it.first].insert(cc2);

				signalToPassSwitch[cc1].insert(it.first);
				signalToPassSwitch[cc2].insert(it.first);

				switchToGateSignal[it.first] = gate;
				signalToOtherCell[gate].insert(it.first);

				wireDriverCount[cc1]++;
				wireDriverCount[cc2]++;
			}
			else
			{
				for (auto &conn : it.second->connections) {
					RTLIL::SigSpec sig = sigmap(conn.second);
					sig.optimize();
					signalToOtherCell[sig].insert(it.first);
				}
			}

			if (it.second->type == "\\PULLUP")
			{
				RTLIL::SigSpec y = sigmap(it.second->connections.at("\\y"));
				assert(y.width == 1);

				y.optimize();
				assert(y.chunks[0].wire != NULL);

				std::string uniqueId = stringf("PULLUP %s", log_signal(y));
				if (uniqueCells.count(uniqueId) > 0) {
					deleteCells[uniqueId].push_back(it.second);
					continue;
				}
				uniqueCells.insert(uniqueId);

				wireDriverCount[y]++;
				pullups.push_back(std::pair<RTLIL::IdString, RTLIL::SigSpec>(it.first, y));
			}
		}

		if (deleteCells.size() > 0)
			log("Deleting duplicate cells:");
		for (auto &i1 : deleteCells) {
			log("  %s:", i1.first.c_str());
			for (auto &i2 : i1.second) {
				log(" %s", i2->name.c_str());
				module->cells.erase(i2->name);
				delete i2;
			}
			log("\n");
		}

		for (auto &it : module->cells)
		{
			if (it.second->type == "\\SW" || it.second->type == "\\SW0" || it.second->type == "\\SW1")
			{
				RTLIL::SigSpec cc1, cc2;
				if (it.second->type == "\\SW0" || it.second->type == "\\SW1") {
					cc1 = sigmap(it.second->connections.at("\\cc"));
					cc2 = RTLIL::SigSpec(RTLIL::State::Sx);
				} else {
					cc1 = sigmap(it.second->connections.at("\\cc1"));
					cc2 = sigmap(it.second->connections.at("\\cc2"));
				}
				assert(cc1.width == 1 && cc2.width == 1);

				cc1.optimize();
				cc2.optimize();

				int sibblings = 0;
				if (cc1.chunks.at(0).wire != NULL)
					sibblings += wireDriverCount.at(cc1) - 1;
				if (cc2.chunks.at(0).wire != NULL)
					sibblings += wireDriverCount.at(cc2) - 1;

				it.second->attributes["\\nmos_sibblings"] = RTLIL::Const(sibblings);
			}

			if (it.second->type == "\\PULLUP")
			{
				RTLIL::SigSpec y = sigmap(it.second->connections.at("\\y"));
				assert(y.width == 1);

				y.optimize();

				int sibblings = 0;
				if (y.chunks.at(0).wire != NULL)
					sibblings += wireDriverCount.at(y) - 1;

				it.second->attributes["\\nmos_sibblings"] = RTLIL::Const(sibblings);
			}
		}
	}

	bool findInputSignals(std::set<RTLIL::SigSpec> &inputSignals, std::set<RTLIL::SigSpec> &internalSignals, std::set<RTLIL::IdString> &switches, RTLIL::SigSpec outputSignal, RTLIL::SigSpec net)
	{
		if (internalSignals.count(net) > 0)
			return true;

		if (net.is_fully_def())
			return net.as_bool() == false;

		if (net != outputSignal && signalToOtherCell.count(net))
			return false;

		internalSignals.insert(net);
		for (auto &i1 : signalToPassSwitch.at(net)) {
			switches.insert(i1);
			if (!switchToGateSignal.at(i1).is_fully_const())
				inputSignals.insert(switchToGateSignal.at(i1));
			for (auto &i2 : switchToPassSignals.at(i1))
				if (!findInputSignals(inputSignals, internalSignals, switches, outputSignal, i2))
					return false;
		}

		return true;
	}

	bool evalGate(std::map<RTLIL::SigSpec, bool> &inputMap, std::set<RTLIL::SigSpec> &processedSignals, RTLIL::SigSpec net)
	{
		if (processedSignals.count(net) > 0)
			return true;

		if (net.is_fully_def())
			return false;

		processedSignals.insert(net);
		for (auto &i1 : signalToPassSwitch.at(net)) {
			RTLIL::SigSpec gate = switchToGateSignal.at(i1);
			if (gate.is_fully_const() ? gate.as_bool() : inputMap.at(gate))
				for (auto &i2 : switchToPassSignals.at(i1))
					if (!evalGate(inputMap, processedSignals, i2))
						return false;
		}

		return true;
	}

	void generateTruthTable(std::set<std::vector<bool>> &truthTable, std::vector<RTLIL::SigSpec> &inputSignals, std::vector<bool> &inputStates, RTLIL::SigSpec outputSignal)
	{
		if (inputStates.size() < inputSignals.size()) {
			inputStates.push_back(false);
			generateTruthTable(truthTable, inputSignals, inputStates, outputSignal);
			inputStates.back() = true;
			generateTruthTable(truthTable, inputSignals, inputStates, outputSignal);
			inputStates.pop_back();
			return;
		}

		std::map<RTLIL::SigSpec, bool> inputMap;
		for (size_t i = 0; i < inputSignals.size(); i++)
			inputMap[inputSignals[i]] = inputStates[i];

		std::set<RTLIL::SigSpec> processedSignals;
		if (evalGate(inputMap, processedSignals, outputSignal))
			truthTable.insert(inputStates);
	}

	bool isNorGate(const std::set<std::vector<bool>> &truthTable)
	{
		if (truthTable.size() != 1)
			return false;
		for (bool it : *truthTable.begin())
			if (it)
				return false;
		return true;
	}

	bool isNandGate(const std::set<std::vector<bool>> &truthTable)
	{
		if (truthTable.size() == 0)
			return false;
		int inputs = truthTable.begin()->size();

		if (truthTable.size() != (1 << inputs) - 1)
			return false;

		std::vector<bool> allOnes(inputs, true);
		return truthTable.count(allOnes) == 0;
	}

	void solve(RTLIL::IdString pullup, RTLIL::SigSpec outputSignal)
	{
		std::set<RTLIL::SigSpec> inputSignals, internalSignals;
		std::vector<RTLIL::SigSpec> orderedInputSignals;
		std::set<RTLIL::IdString> switches;
		std::set<std::vector<bool>> truthTable;
		std::vector<bool> inputStates;

		if (signalToPassSwitch.count(outputSignal) == 0)
			return;
		if (!findInputSignals(inputSignals, internalSignals, switches, outputSignal, outputSignal))
			return;

		for (auto &it : inputSignals)
			orderedInputSignals.push_back(it);

		generateTruthTable(truthTable, orderedInputSignals, inputStates, outputSignal);

		if (inputSignals.size() == 0 || truthTable.size() == 0)
			return;

		log("Found gate %s.%s with %d inputs:\n", RTLIL::id2cstr(module->name), RTLIL::id2cstr(pullup), int(inputSignals.size()));

		log("    signals:");
		for (auto &it : inputSignals)
			log(" %s", log_signal(it));
		log(" -> %s\n", log_signal(outputSignal));

		log("    switches:");
		for (auto &it : switches)
			log(" %s", RTLIL::id2cstr(it));
		log("\n");

		for (auto &i1 : truthTable) {
			log("    active state: ");
			for (auto i2 : i1)
				log("%c", i2 ? '1' : '0');
			log("\n");
		}

		delete module->cells.at(pullup);
		module->cells.erase(pullup);

		for (auto &it : switches) {
			delete module->cells.at(it);
			module->cells.erase(it);
		}

		if (orderedInputSignals.size() <= 9 && isNorGate(truthTable))
		{
			RTLIL::Cell *gate = new RTLIL::Cell;
			gate->name = stringf("$gate%d_%s", cellCounter, RTLIL::id2cstr(pullup));

			switch (orderedInputSignals.size())
			{
			case 1:
				gate->type = "\\gate_not";
				break;
			case 2:
				gate->type = "\\gate_nor2";
				break;
			case 3:
				gate->type = "\\gate_nor3";
				break;
			case 4:
				gate->type = "\\gate_nor4";
				break;
			case 5:
				gate->type = "\\gate_nor5";
				break;
			case 6:
				gate->type = "\\gate_nor6";
				break;
			case 7:
				gate->type = "\\gate_nor7";
				break;
			case 8:
				gate->type = "\\gate_nor8";
				break;
			case 9:
				gate->type = "\\gate_nor9";
				break;
			default:
				abort();
			}

			int portId = 0;
			for (auto &it : orderedInputSignals)
				gate->connections[stringf("\\%c", 'a' + portId++)] = it;

			gate->connections["\\y"] = outputSignal;
			module->add(gate);
		}
		else
		if (orderedInputSignals.size() <= 3 && isNandGate(truthTable))
		{
			RTLIL::Cell *gate = new RTLIL::Cell;
			gate->name = stringf("$gate%d_%s", cellCounter, RTLIL::id2cstr(pullup));

			switch (orderedInputSignals.size())
			{
			case 2:
				gate->type = "\\gate_nand2";
				break;
			case 3:
				gate->type = "\\gate_nand3";
				break;
			default:
				abort();
			}

			int portId = 0;
			for (auto &it : orderedInputSignals)
				gate->connections[stringf("\\%c", 'a' + portId++)] = it;

			gate->connections["\\y"] = outputSignal;
			module->add(gate);
		}
		else
		{
			RTLIL::SigSpec inputSigVect;
			for (auto &it : orderedInputSignals)
				inputSigVect.append(it);

			RTLIL::Wire *allCasesWire = new RTLIL::Wire;
			allCasesWire->name = stringf("$gate%d_cases", cellCounter);
			allCasesWire->width = truthTable.size();
			module->add(allCasesWire);

			int allCasesCount = 0;
			for (auto &i1 : truthTable)
			{
				RTLIL::SigSpec thisValue;
				for (auto i2 : i1)
					thisValue.append(RTLIL::SigSpec(i2 ? 1 : 0, 1));

				RTLIL::Cell *eq_cell = new RTLIL::Cell;
				eq_cell->name = stringf("$gate%d_case%d", cellCounter, allCasesCount);
				eq_cell->type = "$eq";
				eq_cell->parameters["\\A_SIGNED"] = RTLIL::Const(0);
				eq_cell->parameters["\\B_SIGNED"] = RTLIL::Const(0);
				eq_cell->parameters["\\A_WIDTH"] = RTLIL::Const(inputSigVect.width);
				eq_cell->parameters["\\B_WIDTH"] = RTLIL::Const(thisValue.width);
				eq_cell->parameters["\\Y_WIDTH"] = RTLIL::Const(1);
				eq_cell->connections["\\A"] = inputSigVect;
				eq_cell->connections["\\B"] = thisValue;
				eq_cell->connections["\\Y"] = RTLIL::SigSpec(allCasesWire, 1, allCasesCount++);
				module->add(eq_cell);
			}

			RTLIL::Cell *or_cell = new RTLIL::Cell;
			or_cell->name = stringf("$gate%d_%s", cellCounter, RTLIL::id2cstr(pullup));
			or_cell->type = "$reduce_or";
			or_cell->parameters["\\A_SIGNED"] = RTLIL::Const(0);
			or_cell->parameters["\\A_WIDTH"] = RTLIL::Const(allCasesWire->width);
			or_cell->parameters["\\Y_WIDTH"] = RTLIL::Const(1);
			or_cell->connections["\\A"] = RTLIL::SigSpec(allCasesWire);
			or_cell->connections["\\Y"] = outputSignal;
			module->add(or_cell);
		}

		switchCounter += switches.size();
		cellCounter++;
	}

	void solve()
	{
		for (auto &pullup : pullups)
			solve(pullup.first, pullup.second);
	}
};

struct NmosPass : public Pass {
	NmosPass() : Pass("nmos") { }
	virtual void execute(std::vector<std::string> args, RTLIL::Design *design)
	{
		log_header("Executing NMOS pass (extract simple NMOS gates).\n");
		extra_args(args, 1, design);

		cellCounter = 0;
		switchCounter = 0;
		for (auto &it : design->modules)
			if (design->selected_whole_module(it.first)) {
				NmosSolver solver(design, it.second);
				solver.solve();
			}
		log("Processed %d NMOS cells (%d switches).\n", cellCounter, switchCounter);
	}
} NmosPass;


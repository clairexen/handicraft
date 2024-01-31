
#include "mhslib.h"
#include <assert.h>
#include <sys/types.h>
#include <stdio.h>
#include <algorithm>

MultiHotSolver::MultiHotSolver()
{
	verbosityCounter = 0;
}

void MultiHotSolver::addClause(std::vector<int> &literals, int minHot, int maxHot)
{
	clauses.push_back(Clause());
	clauses.back().literals = literals;
	clauses.back().minHot = minHot;
	clauses.back().maxHot = maxHot >= 0 ? maxHot : minHot;
}

bool MultiHotSolver::solve(std::vector<bool> &model, int verbosityDepth, int verbosityDiv, int verbosityLen)
{
	bool success = setupLiterals();
	if (success)
		success = worker(0, verbosityDepth, verbosityDiv, 0, verbosityLen);
	if (success) {
		model.resize(literals.size());
		for (size_t i = 0; i < literals.size(); i++)
			model.at(i) = literals.at(i).state == 1;
	} else
		model.clear();
	return success;
}

void MultiHotSolver::print()
{
	for (auto &c : clauses) {
		printf("%3d %3d:", c.minHot, c.maxHot);
		for (auto l : c.literals)
			printf(" %3d", l);
		printf("\n");
	}
}

bool MultiHotSolver::setupLiterals()
{
	int maxLiteralId = 0;
	for (auto &c : clauses)
	for (auto l : c.literals)
		maxLiteralId = std::max(maxLiteralId, l);
	
	literals.clear();
	literals.resize(maxLiteralId+1);

	for (size_t i = 0; i < clauses.size(); i++) {
		clauses[i].activeHot = 0;
		clauses[i].undefCount = clauses[i].literals.size();
		for (size_t j = 0; j < clauses[i].literals.size(); j++) {
			int l = clauses[i].literals[j];
			literals[l].clauseIdx.push_back(i);
			literals[l].state = 2;
		}
	}

	std::vector<int> dummyJournal;
	for (size_t i = 0; i < clauses.size(); i++)
		if (!optimizeClause(i, dummyJournal))
			return false;
	return true;
}

bool MultiHotSolver::optimizeClause(int clauseIdx, std::vector<int> &journal)
{
	Clause &c = clauses[clauseIdx];

	if (c.activeHot + c.undefCount == c.minHot) {
		for (int l : c.literals)
			if (literals[l].state == 2)
				if (!setLiteral(l, 1, journal, true))
					return false;
	}

	if (c.activeHot == c.maxHot) {
		for (int l : c.literals)
			if (literals[l].state == 2)
				if (!setLiteral(l, 0, journal, true))
					return false;
	}

	if (c.activeHot + c.undefCount < c.minHot)
		return false;

	if (c.activeHot > c.maxHot)
		return false;

	return true;
}

bool MultiHotSolver::setLiteral(int literalIdx, int value, std::vector<int> &journal, bool automatic)
{
	Literal &lit = literals[literalIdx];

	if (lit.state != 2)
		return false;

	lit.state = value;
	lit.automatic = automatic;
	for (size_t i = 0; i < lit.clauseIdx.size(); i++) {
		Clause &c = clauses[lit.clauseIdx[i]];
		if (value)
			c.activeHot++;
		c.undefCount--;
	}
	journal.push_back(literalIdx);

	for (size_t i = 0; i < lit.clauseIdx.size(); i++)
		if (!optimizeClause(lit.clauseIdx[i], journal))
			return false;

	return true;
}

void MultiHotSolver::unsetLiteral(int literalIdx)
{
	assert(literals.at(literalIdx).state < 2);

	for (size_t i = 0; i < literals[literalIdx].clauseIdx.size(); i++) {
		Clause &c = clauses[literals[literalIdx].clauseIdx[i]];
		if (literals[literalIdx].state)
			c.activeHot--;
		c.undefCount++;
	}

	literals[literalIdx].state = 2;
	literals[literalIdx].automatic = false;
}

bool MultiHotSolver::worker(int literalOffset, int verbosityDepth, int verbosityDiv, int verbosityId, int verbosityLen)
{
	while (literalOffset < int(literals.size()) && literals[literalOffset].state < 2)
		literalOffset++;
	if (literalOffset == int(literals.size()))
		return true;

	verbosityCounter++;
	if (literalOffset < verbosityLen && (verbosityDepth == 0 || (verbosityDiv > 0 && verbosityCounter > verbosityDiv)))
	{
		printf("%10d: ", verbosityId);
		for (int i = 0; i < std::min(verbosityLen, int(literals.size())); i++) {
			if (i == literalOffset)
				printf("|<%c>|", verbosityDepth == 0 ? '-' : '*');
			if (literals[i].state == 2)
				printf(".");
			else if (literals[i].automatic)
				printf("%c", literals[i].state ? 'i' : 'o');
			else
				printf("%c", literals[i].state ? '1' : '0');
		}
		printf("%s\n", verbosityDepth == 0 ? "" : " (*)");
		verbosityCounter = 0;
	}

	for (int value = 1; value >= 0; value--)
	{
		std::vector<int> journal;
		bool success = setLiteral(literalOffset, value, journal, false);
		if (success) {
			if (verbosityDepth <= 0)
				success = worker(literalOffset+1, verbosityDepth-1, verbosityDiv, verbosityId, verbosityLen);
			else
				success = worker(literalOffset+1, verbosityDepth-1, verbosityDiv, (verbosityId << 1) | value, verbosityLen);
		}
		if (success)
			return true;
		for (int l : journal)
			unsetLiteral(l);
	}

	return false;
}


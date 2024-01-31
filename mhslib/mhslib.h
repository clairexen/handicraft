#ifndef MHSLIB_H
#define MHSLIB_H

#include <vector>

class MultiHotSolver
{
public:
	MultiHotSolver();
	void addClause(std::vector<int> &literals, int minHot = 1, int maxHot = -1);
	bool solve(std::vector<bool> &model, int verbosityDepth = -1, int verbosityDiv = -1, int verbosityLen = 1024);
	void print();

private:
	struct Clause {
		std::vector<int> literals;
		int activeHot, undefCount, minHot, maxHot;
	};

	struct Literal
	{
		// reference to all users of this literal
		std::vector<int> clauseIdx;

		// current state of the literal (0 = off, 1 = on, 2 = undef)
		int state;

		// set when value forced by other values via optimizeClause
		bool automatic;
	};

	std::vector<Clause> clauses;
	std::vector<Literal> literals;
	int verbosityCounter;

	bool setupLiterals();
	bool optimizeClause(int clauseIdx, std::vector<int> &journal);
	bool setLiteral(int literalIdx, int value, std::vector<int> &journal, bool automatic);
	void unsetLiteral(int literalIdx);
	bool worker(int literalOffset, int verbosityDepth, int verbosityDiv, int verbosityId, int verbosityLen);
};

#endif

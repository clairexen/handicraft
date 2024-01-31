
#include "mhslib.h"
#include <stdio.h>
#include <assert.h>
#include <set>

struct SudokuSolver
{
	int N;
	MultiHotSolver mhs;
	std::vector<bool> model;
	std::set<int> hints;

	int getLiteral(int row, int col, int digit)
	{
		return digit + N*N*col + N*N*N*N*row;
	}

	void setHint(int row, int col, int digit)
	{
		int lit = getLiteral(row, col, digit-1);
		std::vector<int> literals;
		literals.push_back(lit);
		mhs.addClause(literals);
		hints.insert(lit);
	}

	bool solve()
	{
		// mhs.print();
		return mhs.solve(model);
	}

	void print()
	{
		const char *digits = N == 4 ? "0123456789ABCDEF" : "123456789";

		if (model.size() == 0) {
			printf("No model found!\n");
			return;
		}

		printf("\n");
		for (int i = 0; i < N*N; i++)
		{
			printf("   ");
			if (i > 0 && i % N == 0) {
				for (int j = 0; j < N*N; j++) {
					if (j > 0 && j % N == 0)
						printf("-+-");
					printf("---");
				}
				printf("\n   ");
			}
			for (int j = 0; j < N*N; j++)
			{
				if (j > 0 && j % N == 0)
					printf(" | ");
				int digit = 0;
				bool hint = false;
				for (int k = 0; k < N*N; k++) {
					int lit = getLiteral(i, j, k);
					if (model[lit])
						digit = k;
					if (hints.count(lit) > 0)
						hint = true;
				}
				if (hint)
					printf("[%c]%s", digits[digit], j+1 < N*N ? "" : "\n");
				else
					printf(" %c%c", digits[digit], j+1 < N*N ? ' ' : '\n');
			}
		}
		printf("\n");
	}

	SudokuSolver(int N = 3) : N(N)
	{
		assert(0 < N && N <= 4);

		// only one digit per row/col
		for (int i = 0; i < N*N; i++)
		for (int j = 0; j < N*N; j++) {
			std::vector<int> literals;
			for (int k = 0; k < N*N; k++)
				literals.push_back(getLiteral(i, j, k));
			mhs.addClause(literals);
		}

		// each digit only once per row
		for (int i = 0; i < N*N; i++)
		for (int j = 0; j < N*N; j++) {
			std::vector<int> literals;
			for (int k = 0; k < N*N; k++)
				literals.push_back(getLiteral(j, k, i));
			mhs.addClause(literals);
		}

		// each digit only once per column
		for (int i = 0; i < N*N; i++)
		for (int j = 0; j < N*N; j++) {
			std::vector<int> literals;
			for (int k = 0; k < N*N; k++)
				literals.push_back(getLiteral(k, j, i));
			mhs.addClause(literals);
		}

		// each digit only once per box
		for (int d = 0; d < N*N; d++)
		for (int i = 0; i < N*N; i += N)
		for (int j = 0; j < N*N; j += N) {
			std::vector<int> literals;
			for (int ki = 0; ki < N; ki++)
			for (int kj = 0; kj < N; kj++)
				literals.push_back(getLiteral(i+ki, j+kj, d));
			mhs.addClause(literals);
		}
	}
};

int main()
{
	SudokuSolver sudoku(3);

	sudoku.setHint(0, 1, 1);
	sudoku.setHint(0, 2, 8);
	sudoku.setHint(0, 6, 7);
	sudoku.setHint(1, 3, 3);
	sudoku.setHint(1, 6, 2);
	sudoku.setHint(2, 1, 7);
	sudoku.setHint(3, 4, 7);
	sudoku.setHint(3, 5, 1);
	sudoku.setHint(4, 0, 6);
	sudoku.setHint(4, 7, 4);
	sudoku.setHint(5, 0, 3);
	sudoku.setHint(6, 0, 4);
	sudoku.setHint(6, 3, 5);
	sudoku.setHint(6, 8, 3);
	sudoku.setHint(7, 1, 2);
	sudoku.setHint(7, 4, 8);
	sudoku.setHint(8, 7, 6);

	sudoku.solve();
	sudoku.print();
	return 0;
}


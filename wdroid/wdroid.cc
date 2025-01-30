#include "wdroid.hh"
#include <cstdio>

template struct WordleDroidEngine<4, 3, 10000>; // WordleDroidEngine4
template struct WordleDroidEngine<5, 3, 20000>; // WordleDroidEngine5
template struct WordleDroidEngine<6, 4, 40000>; // WordleDroidEngine6

int main(int argc, const char **argv)
{
	WordleDroidGlobalState state;
	return state.main(argc, argv);
}

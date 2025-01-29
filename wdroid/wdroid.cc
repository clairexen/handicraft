#include "wdroid.hh"
#include <cstdio>

template struct WordleDroidEngine<4, 3, 10000>; // WordleDroidEngine4
template struct WordleDroidEngine<5, 3, 20000>; // WordleDroidEngine5
template struct WordleDroidEngine<6, 4, 40000>; // WordleDroidEngine6

int main(int argc, const char **argv)
{
	WordleDroidEngine4 engine4;
	engine4.loadDefaultDict();
	printf("[WordleDroidEngine4] %5d>\n", engine4.numWords-1);

	WordleDroidEngine5 engine5;
	engine5.loadDefaultDict();
	printf("[WordleDroidEngine5] %5d>\n", engine5.numWords-1);

	WordleDroidEngine6 engine6;
	engine6.loadDefaultDict();
	printf("[WordleDroidEngine6] %5d>\n", engine6.numWords-1);

	return 0;
}

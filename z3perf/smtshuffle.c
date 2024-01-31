#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

uint32_t rndstate;
int chmap[256];

char *keywords[] = {
	"_", "set-logic", "set-info", "QF_AUFBV",
	"declare-sort", "declare-fun", "define-fun",
	"assert", "check-sat", "push", "pop", "exit",
	"Bool", "BitVec", "Array", "true", "false",
	"not", "and", "or", "xor", "ite", "distinct",
	"select", "store", "extract", "concat",
	"bvnot", "bvneg", "bvand", "bvor", "bvxor",
	"bvadd", "bvsub", "bvmul", "bvudiv", "bvurem",
	"bvsdiv", "bvsrem", "bvshl", "bvlshr", "bvashr",
	"bvult", "bvule", "bvuge", "bvugt",
	"bvslt", "bvsle", "bvsge", "bvsgt",
	NULL
};

bool idchar(int c)
{
	if (c == '-') return true;
	if (c == '_') return true;
	if (c == '#') return true;
	if ('0' <= c && c <= '9') return true;
	if ('a' <= c && c <= 'z') return true;
	if ('A' <= c && c <= 'Z') return true;
	return false;
}

bool permchar(int c)
{
	if ('a' <= c && c <= 'z') return true;
	if ('A' <= c && c <= 'Z') return true;
	return false;
}

int rnd()
{
	while (1)
	{
		// xorshift32
		rndstate ^= rndstate << 13;
		rndstate ^= rndstate >> 17;
		rndstate ^= rndstate << 5;

		unsigned char c = rndstate;
		if (permchar(c)) return c;
	}
}


int main(int argc, char **argv)
{
	if (argc != 2 || argv[1][0] == '-' || argv[1][0] == 0)
		goto help;
	
	rndstate = atoi(argv[1]);

	if (rndstate == 0) {
help:
		printf("Usage: %s seed < input.smt2 > output.smt2\n", argv[0]);
		return 1;
	}

	for (int i = 0; i < 256; i++)
		chmap[i] = i;

	for (int i = 0; i < 256; i++)
	{
		int c = rnd();
		if (permchar(i) && i != c) {
			chmap[i] ^= chmap[c];
			chmap[c] ^= chmap[i];
			chmap[i] ^= chmap[c];
		}
	}

	char buffer[64*1024];

	while (fgets(buffer, sizeof(buffer), stdin))
	{
		int cursor = 0;
		int idstart = 0;

		while (buffer[cursor] != 0)
		{
			if (idchar(buffer[cursor])) {
				cursor++;
				continue;
			}

			int idlen = cursor - idstart;

			if (idlen == 0 || buffer[idstart] == '#' || buffer[idstart] == ':')
				goto skip_id;

			if (0 <= buffer[idstart] && buffer[idstart] <= '9')
				goto skip_id;

			for (int i = 0; keywords[i]; i++)
				if ((int)strlen(keywords[i]) == idlen && !strncmp(buffer+idstart, keywords[i], idlen))
					goto skip_id;

			while (idstart < cursor) {
				int c = (unsigned char)buffer[idstart];
				buffer[idstart] = chmap[c];
				idstart++;
			}

		skip_id:
			idstart = ++cursor;
		}

		fputs(buffer, stdout);
	}

	return 0;
}


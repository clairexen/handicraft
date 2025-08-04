#include <vector>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

typedef char YYCTYPE;
std::vector<char> buffer;
std::vector<uint16_t> output;
FILE *outfp = NULL;

void pushtok(uint16_t i) {
	if (!outfp)
		printf(" %d", i);
	output.push_back(i);
}

/*!include:re2c "pattern.re" */

void encode_buffer() {
	const char *YYCURSOR = buffer.data();
	const char *YYLIMIT = buffer.data() + buffer.size();
	const char *YYMARKER, *c;

	if (0) {
reset:
		if (YYCURSOR == YYLIMIT)
			return;
		if (!outfp)
			printf("\nIds:");
		YYCURSOR++;
	}

	while (1) {
		/*!re2c
			re2c:eof = 0;
			re2c:yyfill:enable = 0;

			@c [\a\b\t\n\033\040-\176] { pushtok(*c); continue; }

			!use:pat_keywords;

			[\000] { pushtok(0); goto reset; }
			$ { pushtok(0); goto reset; }
			* { printf("\n"); fflush(stdout); fprintf(stderr, "ENCODER ERROR\n"); exit(1); }
		*/
	}
}

int main(int argc, const char **argv) {
	buffer.reserve(1024*1024);
	while (1) {
		long s = buffer.size(), n = 4096;
		buffer.resize(s+n);
		ssize_t rc = read(0, buffer.data() + s, n);
		if (rc < 0) {
			perror("fastenc read()");
			exit(1);
		}
		buffer.resize(s + rc);
		if (rc == 0) break;
	}
	if (!outfp)
		printf("Ids:");
	encode_buffer();
	if (!outfp)
		printf("\n");
	return 0;
}

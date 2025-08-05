#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include <vector>
#include <string>
#include <string_view>

using namespace std::literals;

typedef char YYCTYPE;
std::vector<char> buffer;
std::vector<uint16_t> output;
FILE *outfp = NULL;
bool flag_O;

/*!re2c
	re2c:eof = 0;
	re2c:yyfill:enable = 0;
*/

/*!include:re2c "pattern.re" */
#include "pattern.cc"

void pushtok(uint16_t i) {
	if (!i && !output.empty() && !output.back())
		return;
	if (!outfp) {
		printf(" %s", toknames[i]);
	}
	output.push_back(i);
}

void encode_buffer() {
	const char *YYCURSOR = buffer.data();
	const char *YYLIMIT = buffer.data() + buffer.size();
	const char *YYMARKER, *c;
	bool str1, str2, str3;

	if (0) {
reset:
		pushtok(0);
		if (YYCURSOR == YYLIMIT || YYCURSOR+1 == YYLIMIT)
			return;
		if (!outfp)
			printf("\nTokens:");
		if (!*YYCURSOR)
			YYCURSOR++;
	}

	str1 = str2 = str3 = false;

main_re:
	for (;;) {
		/*!re2c
			!use:pat_keywords;

			["] { goto str_1; }
			[`][`][\n] { goto str_2; }
			['][']['][\n] { goto str_3; }

			[\t\n ]* { continue; }
			[\000] { goto reset; }
			$ { goto reset; }
			* { goto error; }
		*/
	}

	if (0)
str_3:		str3 = true;
	pushtok(TOK_STR_B);

	if (0)
str_2:		str2 = true;
	pushtok(TOK_STR_B);

	if (0)
str_1:		str1 = true;
	pushtok(TOK_STR_B);

str_re:
	for (;;) {
		if (str1 && *YYCURSOR == '"') {
			pushtok(TOK_STR_E);
			YYCURSOR += 1;
			str1 = false;
			goto next;
		}
		if (str2 && YYCURSOR[0] == '\n' && YYCURSOR[1] == '`' && YYCURSOR[2] == '`' &&
				(YYCURSOR[3] == 0 || YYCURSOR[3] == '\n')) {
			pushtok(TOK_STR_E);
			pushtok(TOK_STR_E);
			YYCURSOR += 3;
			str2 = false;
			goto next;
		}
		if (str3 && YYCURSOR[0] == '\n' && YYCURSOR[1] == '\'' && YYCURSOR[2] == '\'' &&
				YYCURSOR[3] == '\'' && (YYCURSOR[4] == 0 || YYCURSOR[4] == '\n')) {
			pushtok(TOK_STR_E);
			pushtok(TOK_STR_E);
			pushtok(TOK_STR_E);
			YYCURSOR += 4;
			str3 = false;
			goto next;
		}
		if (!str3 && YYCURSOR[0] == '\'') {
			const char *bak = YYCURSOR;
			/*!re2c
				!use:pat_quoted_keywords;

				[\000] { goto reset; }
				$ { goto reset; }
				* { goto try_normal_string_stuff; }
			*/
		try_normal_string_stuff:
			YYCURSOR = bak;
		}
		/*!re2c
			@c [\a\b\t\n\033\040-\176] { pushtok(*c); continue; }

			!use:pat_strtoks;

			[\000] { goto reset; }
			$ { goto reset; }
			* { goto error; }
		*/
	}

next:
	if (str1 || str2 || str3) goto str_re;
	goto main_re;

error:
	printf("\n");
	fflush(stdout);
	fprintf(stderr, "ENCODER ERROR AT '%c'\n", *YYCURSOR);
	exit(1);
}

int main(int argc, const char **argv) {
	const char *arg = NULL;
	for (int i=1; i < argc; i++) {
		std::string_view a = argv[1];
		if (a == "-O"sv) {
			flag_O = 1;
			continue;
		}
		if (!arg) {
			arg = argv[i];
			continue;
		}
		fprintf(stderr, "ARGS ERROR\n");
		exit(1);
	}
	buffer.reserve(16*1024*1024);
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
	if (arg)
		outfp = fopen(arg, "ab");
	if (!buffer.empty() && buffer.back())
		buffer.push_back(0);
	if (!outfp)
		printf("Tokens:");
	encode_buffer();
	if (outfp && fwrite(output.data(), 2*output.size(), 1, outfp) != 1) {
		perror("fastenc write()");
		exit(1);
	}
	if (!outfp)
		printf("\n");
	return 0;
}

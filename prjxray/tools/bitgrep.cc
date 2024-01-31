#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <vector>
#include <string>
#include <map>
#include <set>

bool verbose = false;

std::vector<std::string> split_tokens(const std::string &text, const char *sep)
{
	std::vector<std::string> tokens;
	std::string current_token;
	for (char c:text) {
		if (strchr(sep, c)) {
			if (!current_token.empty()) {
				tokens.push_back(current_token);
				current_token.clear();
			}
		} else
			current_token += c;
	}
	if (!current_token.empty()) {
		tokens.push_back(current_token);
		current_token.clear();
	}
	return tokens;
}

int main(int argc, char **argv)
{
	int opt;
	while ((opt = getopt(argc, argv, "vnsdo:q")) != -1)
		switch (opt)
		{
		case 'v':
			verbose = true;
			break;
		default:
			goto help;
		}

	if (optind+2 >= argc) {
help:
		printf("\n");
		printf("Usage: %s [options] frame-id bit-id filename [...]\n", argv[0]);
		printf("\n");
		printf("  -v\n");
		printf("    verbose status output\n");
		printf("\n");
		return 1;
	}

	uint32_t needle_frameid = strtol(argv[optind++], nullptr, 0);
	std::string needle_bitid = argv[optind++];

	while (optind < argc)
	{
		FILE *f;
		std::string fn = argv[optind++];
		char buffer[1024];
		
		if (verbose)
			printf("Reading '%s'.\n", fn.c_str());
		f = fopen(fn.c_str(), "r");

		if (f == nullptr) {
			printf("Can't open input file '%s'.\n", fn.c_str());
			return 1;
		}

		uint32_t frameid = ~0U;

		while (fgets(buffer, sizeof(buffer), f))
		{
			auto tokens = split_tokens(buffer, " \n");

			if (tokens.empty())
				continue;

			if (tokens[0][0] == '.')
			{
				frameid = ~0U;

				if (tokens[0] == ".frame")
					frameid = strtol(tokens[1].c_str(), nullptr, 0);

				continue;
			}

			if (frameid == needle_frameid) {
				for (auto token : tokens)
					if (token == needle_bitid)
						printf("%s\n", fn.c_str());
			}
		}

		fclose(f);
	}

	return 0;
}


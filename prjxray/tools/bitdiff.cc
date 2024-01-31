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
bool framedecode = false;

uint32_t selbits(uint32_t value, int msb, int lsb)
{
	return (value >> lsb) & ((1 << (msb-lsb+1)) - 1);
}

class frameid
{
	uint32_t value;

public:
	frameid(uint32_t v) : value(v) { }

	uint32_t get_value() const {
		return value;
	}

	int get_type() const {
		return selbits(value, 25, 23);
	}

	int get_topflag() const {
		return selbits(value, 22, 22);
	}

	int get_rowaddr() const {
		return selbits(value, 21, 17);
	}

	int get_coladdr() const {
		return selbits(value, 16, 7);
	}

	int get_minor() const {
		return selbits(value, 6, 0);
	}
};

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
	while ((opt = getopt(argc, argv, "vd")) != -1)
		switch (opt)
		{
		case 'v':
			verbose = true;
			break;
		case 'd':
			framedecode = true;
			break;
		default:
			goto help;
		}

	if (0) {
help:
		printf("\n");
		printf("Usage: %s [options] filename [...]\n", argv[0]);
		printf("\n");
		printf("  -v\n");
		printf("    verbose status output\n");
		printf("\n");
		printf("  -d\n");
		printf("    decode frame addr in [type, top, row, col, minor]\n");
		printf("\n");
		return 1;
	}

	std::map<std::pair<uint32_t, std::string>, std::set<std::string>> database;
	int num_files = 0;

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
		num_files++;

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

			if (frameid != ~0U) {
				for (auto token : tokens)
					database[std::pair<uint32_t, std::string>(frameid, token)].insert(fn);
			}
		}

		fclose(f);
	}

	for (auto &it : database)
	{
		if (int(it.second.size()) == num_files)
			continue;

		if (framedecode)
		{
			frameid fid(it.first.first);
			printf("0x%08x[%d,%d,%d,%d,%d] %s", int(it.first.first), fid.get_type(), fid.get_topflag(),
					fid.get_rowaddr(), fid.get_coladdr(), fid.get_minor(), it.first.second.c_str());
		}
		else
		{
			printf("0x%08x %s", int(it.first.first), it.first.second.c_str());
		}

		for (auto &fn : it.second)
			printf(" %s", fn.c_str());
		printf("\n");
	}

	return 0;
}


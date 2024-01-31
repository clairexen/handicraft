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
bool mode_n = false;
bool mode_s = false;
bool mode_d = false;

std::set<std::string> str_cache;

// create a unique char pointer for each string
// (this uses the fact that elements in an std::set<> are never moved in memory)
const char *str(const std::string &s)
{
	auto it = str_cache.find(s);

	if (it == str_cache.end())
		it = str_cache.insert(s).first;

	return it->c_str();
}

std::set<const char*> all_prefixes;
std::map<const char*, std::set<const char*>> tag_to_prefix;
std::map<std::pair<uint32_t, const char*>, std::set<const char*>> bit_to_prefix;
std::map<std::set<const char*>, std::set<std::pair<uint32_t, const char*>>> prefix_to_bit;

std::vector<const char*> split_tokens(const std::string &text, const char *sep)
{
	std::vector<const char*> tokens;
	std::string current_token;
	for (char c:text) {
		if (strchr(sep, c)) {
			if (!current_token.empty()) {
				tokens.push_back(str(current_token));
				current_token.clear();
			}
		} else
			current_token += c;
	}
	if (!current_token.empty()) {
		tokens.push_back(str(current_token));
		current_token.clear();
	}
	return tokens;
}

int main(int argc, char **argv)
{
	FILE *outf = stdout;

	int opt;
	while ((opt = getopt(argc, argv, "vnsdo:q")) != -1)
		switch (opt)
		{
		case 'v':
			verbose = true;
			break;
		case 'n':
			mode_n = true;
			break;
		case 's':
			mode_s = true;
			break;
		case 'd':
			mode_d = true;
			break;
		case 'o':
			outf = fopen(optarg, "w");
			if (0)
		case 'q':
				outf = fopen("/dev/null", "w");
			if (outf == nullptr) {
				printf("Can't open output file '%s'.\n", optarg);
				return 1;
			}
			break;
		default:
			goto help;
		}

	if (0) {
help:
		printf("\n");
		printf("Usage: %s [options] filename[.tags] [...]\n", argv[0]);
		printf("\n");
		printf("  -n\n");
		printf("    do not print errors about multiple candidates\n");
		printf("\n");
		printf("  -s\n");
		printf("    print bit frequency statistics\n");
		printf("\n");
		printf("  -o <output_file>\n");
		printf("    write bit assignments to this output file\n");
		printf("\n");
		printf("  -d\n");
		printf("    print additional information about unmatched tags\n");
		printf("\n");
		printf("  -v\n");
		printf("    verbose status output\n");
		printf("\n");
		printf("  -q\n");
		printf("    write output to /dev/null\n");
		printf("\n");
		return 1;
	}

	// Read input files

	while (optind < argc)
	{
		std::string prefix = argv[optind++];

		if (prefix.size() > 5 && prefix.substr(prefix.size()-5) == ".tags")
			prefix = prefix.substr(0, prefix.size()-5);

		const char *prefix_ptr = str(prefix);
		all_prefixes.insert(prefix_ptr);

		FILE *f;
		std::string fn;
		char buffer[1024];
		
		fn = prefix + ".tags";
		if (verbose) {
			printf("Reading '%s'.\n", fn.c_str());
			fflush(stdout);
		}
		f = fopen(fn.c_str(), "r");

		if (f == nullptr) {
			printf("Can't open input file '%s'.\n", fn.c_str());
			return 1;
		}

		while (fgets(buffer, sizeof(buffer), f))
			for (auto token : split_tokens(buffer, " \n"))
				tag_to_prefix[token].insert(prefix_ptr);

		fclose(f);

		fn = prefix + ".asc";
		if (verbose) {
			printf("Reading '%s'.\n", fn.c_str());
			fflush(stdout);
		}
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

				if (!strcmp(tokens[0], ".frame"))
					frameid = strtol(tokens[1], nullptr, 0);

				continue;
			}

			if (frameid != ~0U) {
				for (auto token : tokens)
					bit_to_prefix[std::make_pair(frameid, token)].insert(prefix_ptr);
			}
		}

		fclose(f);
	}

	if (mode_s)
	{
		std::vector<std::tuple<int, uint32_t, std::string>> histdata;

		for (auto &it : bit_to_prefix)
			histdata.push_back(std::make_tuple(int(it.second.size()), it.first.first, it.first.second));

		std::sort(histdata.begin(), histdata.end());

		for (auto &it : histdata)
			if (std::get<0>(it) != int(all_prefixes.size()))
				printf("%4d 0x%08x %s\n", std::get<0>(it), std::get<1>(it), std::get<2>(it).c_str());

		return 0;
	}

	// Pivot bit_to_prefix -> prefix_to_bit

	if (verbose) {
		printf("Pivoting indices..\n");
		fflush(stdout);
	}

	for (auto &it : bit_to_prefix)
		prefix_to_bit[it.second].insert(it.first);

	// Output

	int errors_count = 0;
	int silent_errors = 0;

	std::set<std::pair<uint32_t, const char*>> matched_bits;
	std::set<const char*> unmatched_tags;

	if (verbose) {
		printf("Processing..\n");
		fflush(stdout);
	}

	for (auto &it : tag_to_prefix)
	{
		if (prefix_to_bit[it.second].size() == 1) {
			auto &bit = *prefix_to_bit[it.second].begin();
			fprintf(outf, "0x%08x %s %s\n", bit.first, bit.second, it.first);
			if (mode_d)
				matched_bits.insert(bit);
			continue;
		}

		if (mode_n && prefix_to_bit[it.second].size() > 1) {
			errors_count++, silent_errors++;
			continue;
		}

		printf("Can't resolve tag '%s': %d candiates.\n", it.first, int(prefix_to_bit[it.second].size()));
		for (auto &bit : prefix_to_bit[it.second])
			printf("  candidate: 0x%08x %s\n", bit.first, bit.second);
		fflush(stdout);

		if (mode_d)
			unmatched_tags.insert(it.first);
		errors_count++;
	}

	if (mode_d)
	{
		int unmatched_tags_count = 0;
		int unmatched_bits_count = 0;

		for (auto &tag : unmatched_tags) {
			unmatched_tags_count++;
			printf("Unmatched tag: %s\n", tag);
			printf("  prefix:");
			for (auto &pf : tag_to_prefix.at(tag))
				printf(" %s", pf);
			printf("\n");
			fflush(stdout);
		}

		for (auto &it : bit_to_prefix) {
			if (matched_bits.count(it.first))
				continue;
			if (it.second == all_prefixes)
				continue;
			unmatched_bits_count++;
			printf("Unmatched bit: 0x%08x %s\n", it.first.first, it.first.second);
			printf("  prefix:");
			for (auto &pf : it.second)
				printf(" %s", pf);
			printf("\n");
			fflush(stdout);
		}

		printf("Found %d unmatched tags and %d unmatched bits.\n", unmatched_tags_count, unmatched_bits_count);
	}

	if (errors_count) {
		if (silent_errors)
			printf("Encountered %d matching errors (%d silenced).\n", errors_count, silent_errors);
		else
			printf("Encountered %d matching errors.\n", errors_count);
		return 1;
	}

	return 0;
}


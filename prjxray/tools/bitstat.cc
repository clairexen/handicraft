#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <vector>
#include <string>
#include <map>
#include <set>

bool verbose = false;
bool mode_m = false;

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

// input data

std::set<const char*> all_prefixes;
std::map<const char*, std::set<const char*>> tag_to_prefix;
std::map<std::pair<uint32_t, const char*>, std::set<const char*>> bit_to_prefix;

// work data

std::map<std::set<const char*>, int> all_tagsets;
std::map<const char*, std::set<const char*>> prefix_to_tagset;
std::map<const char*, int> prefix_to_tagset_indices;
std::vector<int> tagset_prefix_count;

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
	int span_limit = 10;
	std::set<const char*> tags;

	int opt;
	while ((opt = getopt(argc, argv, "vo:ql:t:")) != -1)
		switch (opt)
		{
		case 'v':
			verbose = true;
			break;
		case 'q':
			optarg = (char*)"/dev/null";
		case 'o':
			outf = fopen(optarg, "w");
			if (outf == nullptr) {
				printf("Can't open output file '%s'.\n", optarg);
				return 1;
			}
			break;
		case 'l':
			span_limit = atoi(optarg);
			break;
		case 't':
			tags.insert(str(optarg));
			break;
		default:
			goto help;
		}

	if (0) {
help:
		printf("\n");
		printf("Usage: %s [options] filename[.tags] [...]\n", argv[0]);
		printf("\n");
		printf("  -l <limit>\n");
		printf("    minimum difference between max and min frequency\n");
		printf("\n");
		printf("  -t <tag>\n");
		printf("    only use specified tags (may be used more than once)\n");
		printf("\n");
		printf("  -o <output_file>\n");
		printf("    write bit assignments to this output file\n");
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
				if (tags.empty() || tags.count(token))
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

	{
		if (verbose) {
			printf("Purging static tags..\n");
			fflush(stdout);
		}

		std::set<const char*> delete_tags;

		for (auto &it : tag_to_prefix)
			if (it.second.size() == all_prefixes.size())
				delete_tags.insert(it.first);

		for (auto &it : delete_tags)
			tag_to_prefix.erase(it);

		if (verbose) {
			printf("  purged %d tags, kept %d tags.\n", int(delete_tags.size()), int(tag_to_prefix.size()));
			fflush(stdout);
		}
	}

	{
		if (verbose) {
			printf("Purging static bits..\n");
			fflush(stdout);
		}

		std::set<std::pair<uint32_t, const char*>> delete_bits;

		for (auto &it : bit_to_prefix)
			if (it.second.size() == all_prefixes.size())
				delete_bits.insert(it.first);

		for (auto &it : delete_bits)
			bit_to_prefix.erase(it);

		if (verbose) {
			printf("  purged %d bits, kept %d bits.\n", int(delete_bits.size()), int(bit_to_prefix.size()));
			fflush(stdout);
		}
	}

	{
		if (verbose) {
			printf("Finding tagsets..\n");
			fflush(stdout);
		}

		for (auto pf : all_prefixes)
			prefix_to_tagset[pf].clear();

		for (auto &it : tag_to_prefix)
		for (auto pf : it.second)
			prefix_to_tagset[pf].insert(it.first);

		for (auto &it : prefix_to_tagset)
			all_tagsets[it.second] = 0;

		int index_cnt = 0;
		for (auto &it : all_tagsets)
			it.second = index_cnt++;
		tagset_prefix_count.resize(index_cnt);

		for (auto &it : prefix_to_tagset)
		{
			int index = all_tagsets.at(it.second);
			prefix_to_tagset_indices[it.first] = index;
			tagset_prefix_count.at(index)++;
		}

		for (auto &it : all_tagsets)
		{
			int index = it.second;

			printf("Tagset #%d (%d files):\n", index, tagset_prefix_count.at(index));
			for (auto tag : it.first)
				printf("  %s\n", tag);

			if (it.first.empty())
				printf("  <empty>\n");
		}
	}

	if (verbose) {
		printf("Checking bits..\n");
		fflush(stdout);
	}

	for (auto &it : bit_to_prefix)
	{
		const auto &bit = it.first;
		const auto &prefixes = it.second;

		std::vector<int> tagset_counters(all_tagsets.size());

		for (auto pf : prefixes)
		{
			int index = prefix_to_tagset_indices.at(pf);
			tagset_counters[index] += 100;
		}

		int min_counter = 100;
		int max_counter = 0;

		for (int i = 0; i < int(all_tagsets.size()); i++) {
			tagset_counters.at(i) /= tagset_prefix_count.at(i);
			min_counter = std::min(min_counter, tagset_counters.at(i));
			max_counter = std::max(max_counter, tagset_counters.at(i));
		}

		int span = max_counter - min_counter;

		if (span >= span_limit) {
			printf("0x%08x %s", bit.first, bit.second);
			for (int i = 0; i < int(all_tagsets.size()); i++)
				printf(" %3d", tagset_counters.at(i));
			printf("\n");
		}
	}

	return 0;
}


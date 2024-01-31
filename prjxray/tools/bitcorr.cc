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

int bitmap_size = 0;
std::vector<std::pair<uint32_t, const char*>> index_to_bit;
std::map<std::pair<uint32_t, const char*>, int> bit_to_index;
std::map<const char*, std::vector<bool>> prefix_to_bitmap;

std::map<const char*, std::set<const char*>> tag_to_crosstags;
std::map<const char*, std::set<const char*>> crosstag_to_tags;

std::map<const char*, std::set<std::pair<uint32_t, const char*>>> tag_to_bitset;
std::map<std::set<std::pair<uint32_t, const char*>>, std::set<const char*>> bitset_to_tags;

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

void bitmap_and(std::vector<bool> &dst, const std::vector<bool> &src)
{
	for (size_t i = 0; i < src.size(); i++)
		if (!src.at(i))
			dst.at(i) = false;
}

void bitmap_or(std::vector<bool> &dst, const std::vector<bool> &src)
{
	for (size_t i = 0; i < src.size(); i++)
		if (src.at(i))
			dst.at(i) = true;
}

int main(int argc, char **argv)
{
	FILE *outf = stdout;

	int opt;
	while ((opt = getopt(argc, argv, "mvo:q")) != -1)
		switch (opt)
		{
		case 'm':
			mode_m = true;
			break;
		case 'v':
			verbose = true;
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
		printf("  -m\n");
		printf("    mask bits based on tile Y coordinate\n");
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
			printf("Creating bitmaps..\n");
			fflush(stdout);
		}

		for (auto &it : bit_to_prefix) {
			auto &bit = it.first;
			index_to_bit.push_back(bit);
			bit_to_index[bit] = bitmap_size++;
		}

		assert(bitmap_size == int(bit_to_prefix.size()));
		assert(bitmap_size == int(index_to_bit.size()));
		assert(bitmap_size == int(bit_to_index.size()));

		for (auto &it : all_prefixes)
			prefix_to_bitmap[it] = std::vector<bool>(bitmap_size);

		for (auto &it : bit_to_prefix) {
			int bitidx = bit_to_index.at(it.first);
			auto &prefixes = it.second;
			for (auto &pf : prefixes)
				prefix_to_bitmap.at(pf).at(bitidx) = true;
		}
	}

	{
		if (verbose) {
			printf("Creating crosstags..\n");
			fflush(stdout);
		}

		for (auto &it : tag_to_prefix)
		{
			const char *tag = it.first;
			const char *crosstag1 = tag, *crosstag2 = tag;

			int slash_pos = -1, ge_pos = -1;

			for (int i = 0; tag[i]; i++) {
				if (tag[i] == '/')
					slash_pos = i;
				if (tag[i] == '>')
					ge_pos = i;
			}

			if (slash_pos >= 0 && ge_pos >= 0)
			{
				std::string str1, str2;

				assert(slash_pos < ge_pos);

				for (int i = 0; tag[i]; i++) {
					if (i <= slash_pos) {
						str1.push_back(tag[i]);
						str2.push_back(tag[i]);
					} else if (i < ge_pos) {
						str1.push_back(tag[i]);
					} else if (i == ge_pos) {
						str1.push_back(tag[i]);
						str1.push_back('*');
						str2.push_back('*');
						str2.push_back(tag[i]);
					} else {
						str2.push_back(tag[i]);
					}
				}

				crosstag1 = str(str1);
				crosstag2 = str(str2);
			}

			tag_to_crosstags[tag].insert(crosstag1);
			tag_to_crosstags[tag].insert(crosstag2);

			crosstag_to_tags[crosstag1].insert(tag);
			crosstag_to_tags[crosstag2].insert(tag);
		}

		if (verbose) {
			printf("  created %d crosstags for %d tags.\n", int(tag_to_crosstags.size()), int(crosstag_to_tags.size()));
			fflush(stdout);
		}
	}

	for (auto &it : tag_to_crosstags)
	{
		const char *tag = it.first;
		std::set<const char*> related_tags;
		char bit_prefix_a[3] = {0, 0, 0};
		char bit_prefix_b[3] = {0, 0, 0};

		if (verbose) {
			printf("Solving correlation problem for '%s':\n", tag);
			fflush(stdout);
		}

		if (mode_m)
		{
			int y_coord = -1;

			for (int i = 0; tag[i]; i++)
				if (tag[i] == 'Y' && '0' <= tag[i+1] && tag[i+1] <= '9') {
					y_coord = atoi(tag+i+1);
					break;
				}

			if (y_coord < 0) {
				if (verbose) {
					printf("  failed to extract Y coordinate -> skip tag.\n");
					fflush(stdout);
				}
				continue;
			}

			int word_offset = 2 * (y_coord % 50);

			if (word_offset >= 50)
				word_offset++;

			bit_prefix_a[0] = 'A' + (word_offset / 26);
			bit_prefix_a[1] = 'B' + (word_offset % 26);

			bit_prefix_b[0] = 'A' + ((word_offset+1) / 26);
			bit_prefix_b[1] = 'B' + ((word_offset+1) % 26);

			if (verbose) {
				printf("  Y coordinate is %d -> offset %d, words %s and %s.\n", y_coord, word_offset, bit_prefix_a, bit_prefix_b);
				fflush(stdout);
			}
		}

		for (auto &it2 : it.second) {
			if (verbose)
				printf("  crosstag: %s\n", it2);
			for (auto &it3 : crosstag_to_tags.at(it2))
				related_tags.insert(it3);
		}

		if (verbose) {
			for (auto &tag : related_tags)
				printf("  related tag: %s\n", tag);
			fflush(stdout);
		}

		std::set<const char*> &prefixes_set = tag_to_prefix.at(tag);
		std::set<const char*> prefixes_unset = all_prefixes;
		int unused_count = -1;

		for (auto &tag : related_tags)
			for (auto &prefix : tag_to_prefix.at(tag))
				if (prefixes_unset.count(prefix)) {
					prefixes_unset.erase(prefix);
					unused_count += 1;
				}

		int set_count = prefixes_set.size(), unset_count = prefixes_unset.size();
		
		if (verbose) {
			printf("  number of examples: %d set, %d unset, %d unused\n",
					set_count, unset_count, unused_count);
			fflush(stdout);
		}

		if (set_count < 10 || unset_count < 10) {
			if (verbose) {
				printf("  insufficient number of examples -> skip tag.\n");
				fflush(stdout);
			}
			continue;
		}

		std::vector<bool> bitmap_set(bitmap_size, true);
		std::vector<bool> bitmap_unset(bitmap_size, false);

		for (auto &prefix : prefixes_set)
			bitmap_and(bitmap_set, prefix_to_bitmap.at(prefix));

		for (auto &prefix : prefixes_unset)
			bitmap_or(bitmap_unset, prefix_to_bitmap.at(prefix));

		std::set<std::pair<uint32_t, const char*>> found_bits;

		for (int i = 0; i < bitmap_size; i++)
		{
			if (!bitmap_set.at(i) || bitmap_unset.at(i))
				continue;

			auto &bit = index_to_bit.at(i);

			if (mode_m && strncmp(bit.second, bit_prefix_a, 2) && strncmp(bit.second, bit_prefix_b, 2))
				continue;

			found_bits.insert(bit);
		}

		if (found_bits.size() > 10) {
			if (verbose) {
				printf("  found %d bits -> ignore results.\n", int(found_bits.size()));
				fflush(stdout);
			}
		} else if (found_bits.empty()) {
			if (verbose) {
				printf("  found no matching bits.\n");
				fflush(stdout);
			}
		} else {
			if (verbose) {
				for (auto &bit : found_bits)
					printf("  0x%08x %s\n", bit.first, bit.second);
				fflush(stdout);
			}

			for (auto &bit : found_bits)
				fprintf(outf, "0x%08x %s %s|%d\n", bit.first, bit.second, tag, int(found_bits.size()));

			tag_to_bitset[tag] = found_bits;
		}
	}

	for (auto &it : tag_to_bitset)
		bitset_to_tags[it.second].insert(it.first);

	for (auto &it : bitset_to_tags)
	{
		if (it.second.size() == 1)
			continue;

		printf("Multiple tags for same pattern:\n");

		for (auto tag : it.second)
			printf("  %s\n", tag);

		for (auto &bit : it.first)
			printf("  0x%08x %s\n", bit.first, bit.second);
	}

	return 0;
}


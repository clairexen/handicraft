
#include <set>
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include <sys/dir.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct observation {
	std::vector<bool> bits;
	std::map<std::string, int> field_size, field_value;
};

void help()
{
	printf("Usage: extractor [-d bin_val_dir]\n");
}

#define check(_cond) do { if (_cond) break; fprintf(stderr, "Check '%s' failed at %s:%d.\n", #_cond, __FILE__, __LINE__); exit(1); } while (0)

int main(int argc, char **argv)
{
	std::vector<observation> database;

	char *dirname = nullptr;
	int opt;

	while ((opt = getopt(argc, argv, "d:")) != -1)
	{
		switch (opt)
		{
		case 'd':
			dirname = optarg;
			break;
		default:
			help();
		}
	}

	if (!dirname || optind < argc)
		help();

	struct dirent **namelist;
	int n = scandir(dirname, &namelist, NULL, alphasort);
	check(n >= 0);

	std::set<std::string> bin_val_basenames;
	while (n--) {
		std::string s = std::string(dirname) + "/" + namelist[n]->d_name;
		if (s.size() > 4 && (s.substr(s.size()-4) == ".bin" || s.substr(s.size()-4) == ".val"))
			bin_val_basenames.insert(s.substr(0, s.size()-4));
		free(namelist[n]);
	}
	free(namelist);

	for (auto bn : bin_val_basenames)
	{
		printf("# Reading %s.bin and %s.val..\n", bn.c_str(), bn.c_str());

		observation obs;
		std::ifstream binfile(bn + ".bin");
		std::ifstream valfile(bn + ".val");

		while (1)
		{
			int c = binfile.get();
			if (c == EOF) break;

			for (int i = 0; i < 8; i++)
				obs.bits.push_back((c & (1 << i)) != 0);
		}

		std::string line;
		while (std::getline(valfile, line))
		{
			std::istringstream iss(line);
			std::string varname, varsize_s, varval_s;
			check(iss >> varname >> varsize_s >> varval_s);

			int varsize = strtol(varsize_s.c_str(), nullptr, 0);
			int varval = strtol(varval_s.c_str(), nullptr, 0);

			obs.field_size[varname] = varsize;
			obs.field_value[varname] = varval;
		}

		if (!database.empty()) {
			check(database.front().field_size == obs.field_size);
			check(database.front().bits.size() == obs.bits.size());
		}

		database.push_back(obs);
	}

	std::map<std::vector<bool>, std::set<int>> bitpos_pattern;

	for (int i = 0; i < database.front().bits.size(); i++)
	{
		std::vector<bool> pattern;
		for (auto &obs : database)
			pattern.push_back(obs.bits.at(i));
		bitpos_pattern[pattern].insert(i);
	}

	int counter = 0;

	for (auto &fs : database.front().field_size)
	{
		std::vector<int> bit_positions;

		for (int i = 0; i < fs.second; i++) {
			std::vector<bool> pattern;
			for (auto &obs : database)
				pattern.push_back((obs.field_value.at(fs.first) & (1 << i)) != 0);
			if (!bitpos_pattern.count(pattern) || !bitpos_pattern.at(pattern).size()) {
				fprintf(stderr, "No match found for %s [%d].\n", fs.first.c_str(), i);
				exit(1);
			}
			if (bitpos_pattern.at(pattern).size() > 1) {
				fprintf(stderr, "Multiple matches found for %s [%d]:", fs.first.c_str(), i);
				for (int k : bitpos_pattern.at(pattern))
					fprintf(stderr, " %d", k);
				fprintf(stderr, "\n");
				exit(1);
			}
			bit_positions.push_back(*bitpos_pattern.at(pattern).begin());
			counter++;
		}

		printf("declare %s", fs.first.c_str());
		for (int k : bit_positions)
			printf(" %6d", k);
		printf("\n");
	}

	printf("# Successfully extracted %d bit positions.\n", counter);
	return 0;
}


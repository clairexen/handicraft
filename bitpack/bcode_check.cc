/*
 *  Check bit sequences with interesting properties..
 *
 *  Copyright (C) 2012  RIEGL Research ForschungsGmbH
 *  Copyright (C) 2012  Clifford Wolf <clifford@clifford.at>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

// ./bcode_xor > bcode.txt
// ./bcode_check $( grep ^Code bcode.txt  | grep -v dup | cut -f2 -d' '; )

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <vector>

struct code_s {
	int len, data[4096];
	bool operator==(const struct code_s &other) const {
		if (len != other.len)
			return false;
		for (int i = 0; i < len; i++)
			if (data[i] != other.data[i])
				return false;
		return true;
	}
	void operator+=(const struct code_s &other) {
		assert(len == other.len);
		for (int i = 0; i < len; i++)
			data[i] += other.data[i];
	}
	void shift() {
		int buf = data[0];
		for (int i = 0; i < len-1; i++)
			data[i] = data[i+1];
		data[len-1] = buf;
	}
	struct code_s absdiff() const {
		struct code_s result;
		for (int i = 0; i < len; i++)
			result.data[i] = abs(data[i] - data[(i+1) % len]);
		result.len = len;
		return result;
	}
	const char *str(int digits = 1) const {
		static char buf[4096 * 10];
		char *p = buf;
		for (int i = 0; i < len; i++)
			p += snprintf(p, sizeof(buf) - (p-buf), "%s%*d", i ? " " : "", digits, data[i]);
		return buf;
	}
};

bool check_code(struct code_s code, bool verbose)
{
	std::vector<struct code_s> code_shifted;

	if (verbose)
		printf("Test: %s\n", code.str(2));

	struct code_s buf = code;
	for (int i = 0; i < code.len; i++) {
		code_shifted.push_back(buf);
		buf.shift();
	}

	for (int i = 0; i < code.len; i++)
		buf.data[i] = 0;

	bool is_ok = true;
	for (int i = 0; i < code.len; i++)
	{
		buf += code_shifted[i];

		if (verbose)
			printf("%4d: %s |", i, buf.str(2));

		struct code_s buf2 = buf.absdiff();

		if (i == code.len-1)
		{
			for (int i = 0; i < code.len; i++)
				if (buf2.data[i] != 0)
					is_ok = false;
			if (verbose)
				printf(" %s %s\n", buf2.str(), is_ok ? "OK" : "** ERROR **");
		}
		else
		{
			int sh_idx = -1;
			for (int j = 0; j < code.len && sh_idx < 0; j++)
				if (buf2 == code_shifted[j])
					sh_idx = j;
			if (sh_idx < 0)
				is_ok = false;
			if (verbose)
				printf(" %s [%d]\n", buf2.str(), sh_idx);
		}

		if (!verbose && !is_ok)
			return false;
	}

	return is_ok;
}

void search_code(struct code_s &code, int degree, int position, int mask)
{
	if (position == code.len) {
		if (mask == 0 && check_code(code, false))
			check_code(code, true);
		return;
	}

	for (int i = 0; i < degree; i++) {
		code.data[position] = i;
		search_code(code, degree, position+1, mask & ~(1 << i));
	}
}

int main(int argc, char **argv)
{
	int passed_count = 0, failed_count = 0;
	if (argc < 2) {
		fprintf(stderr, "Usage: %s <code>\n", argv[0]);
		return 1;
	}

	for (int i = 1; i < argc; i++)
	{
		if (i > 1)
			printf("\n");

		if (argv[i][0] == '-')
		{
			int degree = -1, length = -1, len_to = -1;
			sscanf(argv[i], "-%d:%d:%d", &degree, &length, &len_to);

			do {
				printf("Searching for degree %d codes with length %d:\n", degree, length);

				struct code_s code;
				code.len = length;
				for (int i = 0; i < code.len; i++)
					code.data[i] = 0;

				search_code(code, degree, 0, ~(~0 << degree));
			} while (length++ < len_to);
		}
		else
		{
			struct code_s code;
			for (code.len = 0; argv[i][code.len]; code.len++)
				code.data[code.len] = argv[i][code.len] - '0';
			if (check_code(code, true))
				passed_count++;
			else
				failed_count++;
		}
	}

	if (passed_count || failed_count)
		printf("\nResults: %d codes passed, %d codes failed.\n", passed_count, failed_count);

	return 0;
}


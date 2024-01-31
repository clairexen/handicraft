/*
 *  Generate 7, 15 and 31 bit sequences with interesting properties..
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

#include <stdbool.h>
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <vector>

// #define LEN 31
// #define DEBUG
// #define SHOWDUPS

std::vector<uint32_t> found_codes;

const char *code2str(uint32_t code, int len)
{
	static char buf[33];
	for (int i = 0; i < len; i++)
		buf[i] = (code & (1 << (len-i-1))) != 0 ? '1' : '0';
	buf[len] = 0;
	return buf;
}

bool test_code(uint32_t code, int len)
{
	assert(len < 32);
	uint32_t code_shifted[31];
	uint32_t buf = code;

#ifdef DEBUG
	printf("\nCode: %s  (length=%d)\n", code2str(buf, len), len);
#endif

	for (int i = 0; i < len; i++) {
		if (i == 0)
			code_shifted[0] = code;
		else
			code_shifted[i] = ((code_shifted[i-1] << 1) & ~(1 << len)) | ((code_shifted[i-1] & (1 << (len-1))) ? 1 : 0);
#ifdef DEBUG
		printf("      %s [%d]\n", code2str(code_shifted[i], len), i);
#endif
	}

	for (int i = 1; i < len; i++) {
		int sh = -1;
		buf ^= code_shifted[i];
		for (int j = 0; j < len && sh < 0; j++)
			if (buf == code_shifted[j])
				sh = j;
#ifdef DEBUG
		printf("%4d: %s [%d]\n", i, code2str(buf, len), sh);
#endif
		if (sh == -1 && i != len-1)
			return false;
	}

	if (buf != 0)
		return false;

#ifndef DEBUG
	bool code_is_dup = false;
	for (int i = 0; i < len; i++)
	for (size_t j = 0; j < found_codes.size(); j++) {
		if (found_codes[j] == code_shifted[i])
			code_is_dup = true;
	}
	if (code_is_dup) {
#  ifdef SHOWDUPS
		printf("\nCode: %s (dup)\n", code2str(code_shifted[0], len));
#  endif
	} else {
		printf("\nCode: %s (length=%d)\n", code2str(code_shifted[0], len), len);
		for (int i = 1; i < len; i++)
			printf("      %s [%d]\n", code2str(code_shifted[i], len), i);
	}
	fflush(stdout);
#endif
	found_codes.push_back(code);

	uint32_t mcode = 0;
	for (int i = 0; i < len; i++)
		if ((code & (1 << i)) != 0)
			mcode |= 1 << (len-i-1);
	found_codes.push_back(mcode);

	return true;
}

int main()
{
#ifdef DEBUG
	test_code(0b110, 3);
	test_code(0b1110010, 7);
	test_code(0b1110100, 7);
	test_code(0b111010110010001, 15);
	test_code(0b111000100110101, 15);
	printf("\n");
#else
#  ifdef LEN
	int percent = 0;
	fprintf(stderr, "[%2d%%]", percent++);
	for (uint32_t i = 1; i < (1u << LEN); i++) {
		if (i % 1000000 == 0)
			fputc('.', stderr);
		if (i % ((1u << LEN)/100) == 0)
			fprintf(stderr, "\n[%2d%%]", percent++);
		test_code(i, LEN);
	}
	fprintf(stderr, "\nREADY.\n");
#  else
	for (int k = 3; k < 32; k++) {
		int percent = 0;
		time_t last_progress = time(NULL);
		found_codes.clear();
		fprintf(stderr, "[%2d:%3d%%]", k, percent++);
		for (uint32_t i = 1; i < (1u << k); i++) {
			if (i % 1000000 == 0)
				fputc('.', stderr);
			if (k > 20 && i % ((1u << k)/100) == 0) {
				time_t now = time(NULL);
				if (now != last_progress)
					fprintf(stderr, "\n[%d: %3d%%]", k, percent);
				last_progress = now;
				percent++;
			}
			test_code(i, k);
		}
		fprintf(stderr, "\n");
	}
	fprintf(stderr, "READY.\n");
#  endif
#endif
	return 0;
}


/*
 *  ascii2rdf - Convert ASCII to RIGOL DG1000 RDF files
 *
 *  Copyright (C) 2013  Clifford Wolf <clifford@clifford.at>
 *  
 *  Permission to use, copy, modify, and/or distribute this software for any
 *  purpose with or without fee is hereby granted, provided that the above
 *  copyright notice and this permission notice appear in all copies.
 *  
 *  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 *  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 *  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 *  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 *  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 *  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 */

#include <stdint.h>
#include <stdio.h>

int main()
{
	int rc = 0;
	int wordcount = 0, value;
	int set_extra_flags = 0;
continue_after_special:
	for (; scanf("%d", &value) == 1; wordcount++) {
		if (value < 0 || 16383 < value) {
			fprintf(stderr, "Warning: word %d has value %d, valid range is [0..16383].\n", wordcount, value);
			value = value < 0 ? 0 : 16383;
			rc = 1;
		}
		value |= set_extra_flags;
		set_extra_flags = 0;
		unsigned char buffer[2] = { value, value >> 8 };
		if (fwrite(buffer, 2, 1, stdout) != 1) {
			fprintf(stderr, "Warning: i/o error while writing word %d.\n", wordcount);
			rc = 1;
		}
	}
	while (1) {
		int ch = getc(stdin);
		if (ch < 0)
			break;
		if (ch == ' ' || ch == '\t')
			continue;
		if (ch == '=') {
			set_extra_flags |= 0x8000;
			goto continue_after_special;
		}
		if (ch == '*') {
			set_extra_flags |= 0x4000;
			goto continue_after_special;
		}
		if (ch != '#')
			break;
		while (1) {
			ch = getc(stdin);
			if (ch < 0)
				break;
			if (ch == '\n')
				goto continue_after_special;
		}
	}
	if (wordcount != 4096) {
		fprintf(stderr, "Warning: expected 4096 words (8192 bytes), got %d words.\n", wordcount);
		rc = 1;
	}
	return rc;
}


/*
 *  rdf2ascii - Convert RIGOL DG1000 RDF to ASCII files
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
#include <string.h>
#include <stdio.h>

int main(int argc, char **argv)
{
	int wordcount, verbose = 0;
	unsigned char buffer[2];
	if (argc == 2 && !strcmp(argv[1], "-v"))
		verbose = 1;
	for (wordcount = 0; fread(buffer, 2, 1, stdin) == 1; wordcount++) {
		unsigned int value = (buffer[1] << 8) | buffer[0];
		if (verbose)
			printf("%s%s%d\n", (value & 0x8000) != 0 ? "=" : "", (value & 0x4000) != 0 ? "*" : "", value & 0x3fff);
		else
			printf("%d\n", value & 0x3fff);
	}
	if (wordcount != 4096) {
		fprintf(stderr, "Warning: expected 4096 words (8192 bytes), got %d words.\n", wordcount);
		return 1;
	}
	return 0;
}


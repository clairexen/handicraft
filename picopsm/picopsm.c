/*
 *  PicoPsm - A small programmable state machine
 *
 *  Copyright (C) 2014  Clifford Wolf <clifford@clifford.at>
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

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <stdbool.h>

#include "picopsm_asm.h"

int main(int argc, char **argv)
{
	char *src_filename = NULL;
	char *dst_filename = NULL;
	bool asm_mode = false;
	FILE *f;

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-asm")) {
			asm_mode = true;
			continue;
		}
		if (argv[i][0] != '-' && src_filename == NULL) {
			src_filename = argv[i];
			continue;
		}
		if (argv[i][0] != '-' && dst_filename == NULL) {
			dst_filename = argv[i];
			continue;
		}
		fprintf(stderr, "Usage: %s [-asm] [src_filename [dst_filename]]\n", argv[0]);
		return 1;
	}


	// ------------------------------------------------------
	// Reading source file

	f = src_filename ? fopen(src_filename, "r") : stdin;
	if (f == NULL) {
		fprintf(stderr, "Failed to open source file `%s': %s\n",
				src_filename ? src_filename : "(stdin)", strerror(errno));
		return 1;
	}

	int source_len = 0, source_reserved = 64*1024;
	char *source_text = malloc(source_reserved);

	while (1)
	{
		if (source_reserved < source_len + 4096 + 1) {
			source_reserved *= 2;
			source_text = realloc(source_text, source_reserved);
		}

		int rc = fread(source_text + source_len, 1, 4096, f);
		if (rc <= 0) break;
		source_len += rc;
	}

	source_text[source_len++] = 0;
	source_text = realloc(source_text, source_len);

	if (src_filename)
		fclose(f);


	// ------------------------------------------------------
	// Running tools

	struct picopsm_asm_job_t *asm_job = picopsm_asm_init();

	if (asm_mode)
	{
		asm_job->source_text = source_text;
		if (!picopsm_asm_run(asm_job))
			return 1;
	}
	else
	{
		fprintf(stderr, "Only -asm mode is implemented atm.\n");
		return 1;
	}


	// ------------------------------------------------------
	// Writing destination file

	f = dst_filename ? fopen(dst_filename, "w") : stdout;
	if (f == NULL) {
		fprintf(stderr, "Failed to open destination file `%s': %s\n",
				dst_filename ? dst_filename : "(stdout)", strerror(errno));
		return 1;
	}

	for (int i = 0; i < asm_job->program_len; i++) {
		fprintf(f, "mem['h%04x] = 'h%02x; // ", asm_job->program_offset + i, asm_job->program[i].value);
		if (asm_job->program[i].symbol && asm_job->program[i].constval)
			fprintf(f, "%s+%d", asm_job->program[i].symbol, asm_job->program[i].constval);
		else if (asm_job->program[i].symbol)
			fprintf(f, "%s", asm_job->program[i].symbol);
		else
			fprintf(f, "%d", asm_job->program[i].constval);
		fprintf(f, "%s\n", asm_job->program[i].hi_byte ? " (hi-byte)" : "");
	}

	if (dst_filename)
		fclose(f);


	picopsm_asm_free(asm_job);
	return 0;
}


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
#include <assert.h>
#include <string.h>
#include "picopsm_asm.h"

static void add_symbol(struct picopsm_asm_job_t *job, char *symbol, uint16_t offset)
{
	job->symbols = realloc(job->symbols, sizeof(*job->symbols) * (job->symbols_num+1));
	job->symbols[job->symbols_num].symbol = symbol;
	job->symbols[job->symbols_num].offset = offset;
	job->symbols_num++;
}

static void add_byte(struct picopsm_asm_job_t *job, uint16_t constval, char *symbol, bool hi_byte)
{
	job->program = realloc(job->program, sizeof(*job->program) * (job->program_len+1));
	job->program[job->program_len].symbol = symbol;
	job->program[job->program_len].hi_byte = hi_byte;
	job->program[job->program_len].constval = constval;
	job->program[job->program_len].value = 0;
	job->program_len++;
	job->cursor_prog++;
}

static void handle_arg(struct picopsm_asm_job_t *job, char *arg, bool lo_hi)
{
	uint16_t constval = 0;
	char *symbol = NULL;

	while (*arg == '$') {
		job->program[job->program_len-1].constval++;
		arg++;
	}

	if (*arg < '0' || *arg > '9') {
		symbol = arg;
		while (*arg && *arg != '+')
			arg++;
		if (*arg == '+')
			*(arg++) = 0;
	}

	if (*arg >= '0' && *arg <= '9')
		constval = atoi(arg);

	add_byte(job, constval, symbol, false);

	if (lo_hi)
		add_byte(job, constval, symbol, true);
}

static struct picopsm_asm_symbol_t *find_symbol(struct picopsm_asm_job_t *job, char *symbol)
{
	for (int i = 0; i < job->symbols_num; i++)
		if (!strcmp(job->symbols[i].symbol, symbol))
			return job->symbols + i;
	return NULL;
}

static char *next_token(struct picopsm_asm_job_t *job)
{
	if (job->last_delim == '\r' || job->last_delim == '\n' || job->last_delim == 0)
		return NULL;

	while (job->source_text[job->source_pos] == ' ' || job->source_text[job->source_pos] == '\t')
		job->source_pos++;

	char *p = job->source_text + job->source_pos;

	while (job->source_text[job->source_pos] != ' ' && job->source_text[job->source_pos] != '\t' &&
			job->source_text[job->source_pos] != '\r' && job->source_text[job->source_pos] != '\n' &&
			job->source_text[job->source_pos] != 0)
		job->source_pos++;

	job->last_delim = job->source_text[job->source_pos];
	if (job->last_delim)
		job->source_text[job->source_pos++] = 0;
	return *p ? p : NULL;
}

static bool next_line(struct picopsm_asm_job_t *job)
{
	if (job->last_delim == 0)
		return false;

	assert(job->last_delim == '\r' || job->last_delim == '\n');
	job->last_delim = ' ';
	job->line_count++;
	return true;
}

bool picopsm_asm_run(struct picopsm_asm_job_t *job)
{
	while (1)
	{
		char *tok = next_token(job);

		if (!tok || *tok == '#') {
			while (tok)
				tok = next_token(job);
			if (!next_line(job))
				break;
			continue;
		}

		if (!strcmp(tok, "reg"))
		{
			tok = next_token(job);
			if (find_symbol(job, tok)) {
				fprintf(stderr, "[ASM] Re-declaration of symbol `%s' in line %d.\n", tok, job->line_count);
				return false;
			}
			add_symbol(job, tok, job->cursor_data);

			tok = next_token(job);
			job->cursor_data += tok ? atoi(tok) : 1;

			tok = next_token(job);
		}
		else
		if (!strcmp(tok, "ld") || !strcmp(tok, "add") || !strcmp(tok, "sub") ||
				!strcmp(tok, "addc") || !strcmp(tok, "subc") ||
				!strcmp(tok, "and") || !strcmp(tok, "xor"))
		{
			if (!strcmp(tok, "ld"  )) add_byte(job, 0b00110000, NULL, false);
			if (!strcmp(tok, "add" )) add_byte(job, 0b00000000, NULL, false);
			if (!strcmp(tok, "sub" )) add_byte(job, 0b00001000, NULL, false);
			if (!strcmp(tok, "addc")) add_byte(job, 0b00000100, NULL, false);
			if (!strcmp(tok, "subc")) add_byte(job, 0b00001100, NULL, false);
			if (!strcmp(tok, "and" )) add_byte(job, 0b00011000, NULL, false);
			if (!strcmp(tok, "xor" )) add_byte(job, 0b00011100, NULL, false);

			tok = next_token(job);
			handle_arg(job, tok, false);

			tok = next_token(job);
		}
		else
		if (!strcmp(tok, "b") || !strcmp(tok, "bc"))
		{
			if (!strcmp(tok, "b" )) add_byte(job, 0b00111000, NULL, false);
			if (!strcmp(tok, "bc")) add_byte(job, 0b00111100, NULL, false);

			tok = next_token(job);
			handle_arg(job, tok, true);

			tok = next_token(job);
		}
		else
		if (!strcmp(tok, "st"))
		{
			add_byte(job, 0b00110100, NULL, false);

			tok = next_token(job);
			if (*tok != '$') {
				fprintf(stderr, "[ASM] Argument `%s' to st opcode does not start with `$' in line %d.\n", tok, job->line_count);
				return false;
			}
			handle_arg(job, tok+1, false);

			tok = next_token(job);
		}
		else
		if (tok[strlen(tok)-1] == ':')
		{
			tok[strlen(tok)-1] = 0;
			if (find_symbol(job, tok)) {
				fprintf(stderr, "[ASM] Re-declaration of symbol `%s' in line %d.\n", tok, job->line_count);
				return false;
			}
			add_symbol(job, tok, job->cursor_prog);

			tok = next_token(job);
		}

		if (tok) {
			fprintf(stderr, "[ASM] Unexpected token `%s' in line %d.\n", tok, job->line_count);
			return false;
		}

		next_line(job);
	}

	for (int i = 0; i < job->program_len; i++) {
		uint16_t v = job->program[i].constval;
		if (job->program[i].symbol) {
			struct picopsm_asm_symbol_t *sym = find_symbol(job, job->program[i].symbol);
			if (sym == NULL) {
				fprintf(stderr, "[ASM] Can't resolve symbol `%s'.\n", job->program[i].symbol);
				return false;
			}
			v += sym->offset;
		}
		if (job->program[i].hi_byte)
			v = v >> 8;
		job->program[i].value = v;
	}

	return true;
}

struct picopsm_asm_job_t *picopsm_asm_init()
{
	struct picopsm_asm_job_t *job = malloc(sizeof(struct picopsm_asm_job_t));

	job->source_pos = 0;
	job->source_text = NULL;
	job->last_delim = ' ';
	job->line_count = 1;

	job->program = NULL;
	job->program_len = 0;
	job->program_offset = 256;

	job->symbols = NULL;
	job->symbols_num = 0;

	job->cursor_data = 0;
	job->cursor_prog = 256;

	return job;
}

void picopsm_asm_free(struct picopsm_asm_job_t *job)
{
	free(job->source_text);
	free(job->program);
	free(job->symbols);
}


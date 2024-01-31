#include "gendsched.h"
#include <stdlib.h>
#include <string.h>

void parse(config &cfg, FILE *f)
{
	char buffer[64*1024];

	cfg.num_cycles = 0;
	cfg.ports.clear();
	cfg.edges.clear();

	while (fgets(buffer, sizeof(buffer), f) != NULL)
	{
		char *tok = strtok(buffer, "\t\r\n ");

		if (tok == NULL || *tok == 0 || *tok == '#')
			continue;

		if (!strcmp(tok, ".input") || !strcmp(tok, ".output"))
		{
			char *tok_name = strtok(NULL, "\t\r\n ");
			char *tok_signed = strtok(NULL, "\t\r\n ");
			char *tok_width = strtok(NULL, "\t\r\n ");

			if (strcmp(tok_signed, "signed") && strcmp(tok_signed, "unsigned"))
				goto syntax_error;

			port info;
			info.name = tok_name;
			info.is_signed = !strcmp(tok_signed, "signed");
			info.is_input = !strcmp(tok, ".input");
			info.width = atoi(tok_width);

			if (info.width == 0)
				goto syntax_error;

			cfg.ports.push_back(info);
			continue;
		}

		if (!strcmp(tok, ".end"))
			return;

		char *endp = NULL;
		int cycle = strtol(tok, &endp, 10);
		if (endp == NULL || *endp != 0 || cycle != cfg.num_cycles)
			goto syntax_error;

		// FIXME
	}

syntax_error:
	fprintf(stderr, "Parser error.\n");
	exit(1);
}


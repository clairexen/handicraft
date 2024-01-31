/*
 *  VLNLP - Simple C++ Verilog Netlist Parser/Processor
 *
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

#include "vlnlp.h"
#include <string.h>
#include <assert.h>

using namespace VLNLP;

static void my_setbit(std::vector<uint8_t> &data, int pos)
{
	size_t byte = pos / 8, bit = pos % 8;
	if (byte < data.size())
		data[byte] |= 1 << bit;
}

static int my_div_by_two(std::vector<uint8_t> &digits, int base)
{
	int carry = 0;
	for (size_t i = 0; i < digits.size(); i++) {
		digits[i] += carry * base;
		carry = digits[i] % 2;
		digits[i] /= 2;
	}
	return carry;
}

static void my_strtobin(std::vector<uint8_t> &data, const char *str, int len_in_bits, int base)
{
	data.clear();
	data.resize(len_in_bits);

	std::vector<uint8_t> digits;
	while (*str) {
		if ('0' <= *str && *str <= '9')
			digits.push_back(*str - '0');
		else if ('a' <= *str && *str <= 'f')
			digits.push_back(*str - 'a');
		else if ('A' <= *str && *str <= 'F')
			digits.push_back(*str - 'A');
		else
			digits.push_back(0);
		str++;
	}

	for (int i = 0; i < len_in_bits; i++)
	{
		if (my_div_by_two(digits, base))
			my_setbit(data, i);
	}
}

Value::Value(const char *str)
{
	char *endptr;
	original_str = str;

	if (str == NULL || *str == 0)
		goto silent_error;

	// Strings
	if (*str == '"') {
		int len = strlen(str) - 2;
		len_in_bits = 8 * len;
		data.resize(len);
		for (int i = 0; i < len; i++)
			data[i] = str[len - i];
		goto gen_int_from_data;
	}

	integer_value = strtol(str, &endptr, 10);

	// Just a 32bit decimal number
	if (*endptr == 0) {
		len_in_bits = 32;
		data.push_back(integer_value >>  0);
		data.push_back(integer_value >>  8);
		data.push_back(integer_value >> 16);
		data.push_back(integer_value >> 27);
		return;
	}

	// The "<bits>'[bodh]<digits>" syntax
	if (*endptr == '\'')
	{
		len_in_bits = integer_value;
		switch (*(endptr+1))
		{
		case 'b':
			my_strtobin(data, endptr+2, len_in_bits, 2);
			break;
		case 'o':
			my_strtobin(data, endptr+2, len_in_bits, 8);
			break;
		case 'd':
			my_strtobin(data, endptr+2, len_in_bits, 10);
			break;
		case 'h':
			my_strtobin(data, endptr+2, len_in_bits, 16);
			break;
		default:
			goto error;
		}
gen_int_from_data:
		integer_value = 0;
		if (data.size() >= 1)
			integer_value |= data[0] <<  0;
		if (data.size() >= 2)
			integer_value |= data[1] <<  8;
		if (data.size() >= 3)
			integer_value |= data[2] << 16;
		if (data.size() >= 4)
			integer_value |= data[3] << 27;
		return;
	}

error:
	fprintf(stderr, "WARNING: Value conversion failed: %c%s%c\n",
			str ? '`' : '(', str ? str : "null", str ? '\'' : ')');
silent_error:
	len_in_bits = 0;
	integer_value = -1;
}

Netlist::Netlist()
{
}

Netlist::~Netlist()
{
	for (auto i = modules.begin(); i != modules.end(); i++)
		delete i->second;
}

void Netlist::parse(FILE *f, bool debug)
{
	vlnlp_parser_netlist = this;
	vlnlp_yyrestart(f);
	vlnlp_yydebug = debug ? 1 : 0;
	vlnlp_yyparse();
	vlnlp_yylex_destroy();
	fixup();
}

void Netlist::fixup()
{
	// FIXME
}

void Netlist::dump(FILE *f)
{
	for (auto i = modules.begin(); i != modules.end(); i++)
		i->second->dump(f);
}

Module::Module(Netlist *parent) : parent(parent)
{
}

Module::~Module()
{
	for (auto i = wires.begin(); i != wires.end(); i++)
		delete i->second;
	for (auto i = cells.begin(); i != cells.end(); i++)
		delete i->second;
	for (auto i = assignments.begin(); i != assignments.end(); i++) {
		delete i->first;
		delete i->second;
	}
}

void Module::dump(FILE *f)
{
	fprintf(f, "module %s(", name.c_str());
	for (auto i = ports.begin(); i != ports.end(); i++) {
		if (i != ports.begin())
			fprintf(f, ", ");
		fprintf(f, "%s", (*i)->name.c_str());
	}
	fprintf(f, ");\n");

	for (auto i = wires.begin(); i != wires.end(); i++)
		i->second->dump(f);

	for (auto i = cells.begin(); i != cells.end(); i++)
		i->second->dump(f);

	for (auto i = assignments.begin(); i != assignments.end(); i++) {
		fprintf(f, "assign ");
		i->first->dump(f);
		fprintf(f, " = ");
		i->second->dump(f);
		fprintf(f, ";\n");
	}

	fprintf(f, "endmodule\n");
}

Wire::Wire(Module *parent) : parent(parent)
{
	range_left = 0;
	range_right = 0;
	is_input = false;
	is_output = false;
	stub_decl = false;
}

Wire::~Wire()
{
}

void Wire::dump(FILE *f)
{
	if (is_input && is_output)
		fprintf(f, "inout");
	else if (is_input)
		fprintf(f, "input");
	else if (is_output)
		fprintf(f, "output");
	else
		fprintf(f, "wire");
	if (range_left != 0 || range_right != 0)
		fprintf(f, " [%d:%d]", range_left, range_right);
	fprintf(f, " %s;\n", name.c_str());
}

Cell::Cell(Module *parent) : parent(parent)
{
	type = NULL;
}

Cell::~Cell()
{
	for (auto i = connections.begin(); i != connections.end(); i++)
		delete i->second;
}

void Cell::dump(FILE *f)
{
	fprintf(f, "%s", type_name.c_str());

	if (parameters.size() != 0) {
		fprintf(f, " #(");
		for (auto i = parameters.begin(); i != parameters.end(); i++) {
			if (i != parameters.begin())
				fprintf(f, ",");
			fprintf(f, "\n\t.%s(%s)", i->first.c_str(), i->second.original_str.c_str());
		}
		fprintf(f, "\n)");
	}

	fprintf(f, " %s (", name.c_str());
	for (auto i = connections.begin(); i != connections.end(); i++) {
		if (i != connections.begin())
			fprintf(f, ",");
		fprintf(f, "\n\t.%s(", i->first.c_str());
		i->second->dump(f);
		fprintf(f, ")");
	}
	fprintf(f, "\n);\n");
}

SigSpec::SigSpec(Module *parent) : parent(parent)
{
	total_len = 0;
}

SigSpec::~SigSpec()
{
}

void SigSpec::SigChunk::dump(FILE *f)
{
	if (wire) {
		fprintf(f, "%s", wire->name.c_str());
		if (offset != wire->range_right || len != wire->range_left - wire->range_right + 1) {
			if (len != 1)
				fprintf(f, "[%d:%d]", offset + len-1, offset);
			else
				fprintf(f, "[%d]", offset);
		}
	} else {
		assert(offset == 0 && len == int(value.len_in_bits));
		fprintf(f, "%s", value.original_str.c_str());
	}
}

void SigSpec::dump(FILE *f)
{
	if (chunks.size() == 0)
		return;
	if (chunks.size() == 1) {
		chunks[0].dump(f);
	} else {
		fprintf(f, "{ ");
		for (size_t i = 0; i < chunks.size(); i++) {
			if (i != 0)
				fprintf(f, ", ");
			chunks[1].dump(f);
		}
		fprintf(f, " }");
	}
}


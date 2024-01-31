#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <string>
#include <vector>
#include <map>
#include <set>

int current_op;
std::vector<int> line_numbers;
std::vector<std::vector<std::string>> opcodes;
std::map<std::string, uint16_t> labels;
std::map<uint16_t, std::set<std::string>> rlabels;

std::vector<std::string> split_tokens(const std::string &text, const char *sep)
{
	std::vector<std::string> tokens;
	std::string current_token;
	for (char c : text) {
		if (strchr(sep, c)) {
			if (!current_token.empty()) {
				tokens.push_back(current_token);
				current_token.clear();
			}
		} else
			current_token += c;
	}
	if (!current_token.empty()) {
		tokens.push_back(current_token);
		current_token.clear();
	}
	return tokens;
}

void print_parse_error(const char *errmsg, int idx = -1) __attribute__((noreturn));

void print_parse_error(const char *errmsg, int idx)
{
	auto &tokens = opcodes[current_op];

	fprintf(stderr, "in line %4d:", line_numbers[current_op]);
	for (int i = 0; i < int(tokens.size()); i++)
		fprintf(stderr, " %s", tokens[i].c_str());
	fprintf(stderr, "\n");

	if (0 <= idx && idx < int(tokens.size())) {
		fprintf(stderr, "             ");
		for (int i = 0; i < idx; i++)
			fprintf(stderr, " %*s", int(tokens[i].size()), "");
		fprintf(stderr, " ");
		for (int i = 0; i < int(tokens[idx].size()); i++)
			fprintf(stderr, "^");
		fprintf(stderr, "\n");
	}

	fprintf(stderr, "Parse error: %s\n", errmsg);

	exit(1);
}

int parse_reg(int idx)
{
	auto &tokens = opcodes[current_op];

	if (idx < 0 || idx >= int(tokens.size()))
		print_parse_error("Missing register operand.");

	if (tokens[idx] == "r0") return 0;
	if (tokens[idx] == "r1") return 1;
	if (tokens[idx] == "r2") return 2;
	if (tokens[idx] == "r3") return 3;
	if (tokens[idx] == "r4") return 4;
	if (tokens[idx] == "r5") return 5;
	if (tokens[idx] == "r6") return 6;
	if (tokens[idx] == "r7") return 7;
	if (tokens[idx] == "r8") return 8;
	if (tokens[idx] == "r9") return 9;
	if (tokens[idx] == "r10") return 10;
	if (tokens[idx] == "r11") return 11;
	if (tokens[idx] == "r12") return 12;
	if (tokens[idx] == "r13") return 13;
	if (tokens[idx] == "r14") return 14;
	if (tokens[idx] == "r15") return 15;

	print_parse_error("Can't parse register operand.", idx);
}

uint16_t parse_const(int idx)
{
	auto &tokens = opcodes[current_op];

	if (idx < 0 || idx >= int(tokens.size()))
		print_parse_error("Missing const operand.");

	std::string s;
	for (char ch : tokens[idx])
		if (ch != '_')
			s += ch;

	uint16_t v = 0;

	if (s.substr(0, 2) == "0b") {
		for (char ch : s.substr(2))
			if (ch == '1') v = (v << 1) | 1;
			else if (ch == '0') v = v << 1;
			else goto parse_error;
		return v;
	}

	for (char ch : s)
		if ('0' <= ch && ch <= '9') v = (10 * v) + (ch - '0');
		else goto parse_error;
	return v;

parse_error:
	print_parse_error("Can't parse const operand.", idx);
}

void print_op(uint16_t op)
{
	printf("%04x // %3x:", op, current_op);
	for (auto label : rlabels[current_op])
		printf(" %s:", label.c_str());
	for (auto token : opcodes[current_op])
		printf(" %s", token.c_str());
	printf("\n");
}

int main()
{
	std::map<std::string, int> alu_ops;
	std::string last_op, this_op;

	alu_ops["mv"] = 0;
	alu_ops["add"] = 1;
	alu_ops["sub"] = 2;
	alu_ops["shl"] = 3;
	alu_ops["shr"] = 4;
	alu_ops["sra"] = 5;
	alu_ops["pack"] = 6;
	alu_ops["unpack"] = 7;
	alu_ops["lt"] = 8;
	alu_ops["le"] = 9;
	alu_ops["ne"] = 10;
	alu_ops["eq"] = 11;
	alu_ops["ge"] = 12;
	alu_ops["gt"] = 13;
	alu_ops["slt"] = 14;
	alu_ops["sle"] = 15;
	alu_ops["sge"] = 16;
	alu_ops["sgt"] = 17;
	alu_ops["addc"] = 18;
	alu_ops["subc"] = 19;
	alu_ops["lnot"] = 20;
	alu_ops["bool"] = 21;
	alu_ops["not"] = 22;
	alu_ops["and"] = 23;
	alu_ops["or"] = 24;
	alu_ops["xor"] = 25;
	alu_ops["flip"] = 26;


	// Read assembler code

	char buffer[1024];
	for (int line_nr = 1; fgets(buffer, sizeof(buffer), stdin) != nullptr; line_nr++)
	{
		if (buffer[0] == '#')
			continue;

		std::vector<std::string> tokens = split_tokens(buffer, "#\r\n");

		if (tokens.empty())
			continue;

		tokens = split_tokens(tokens[0], " \t,");

		while (!tokens.empty() && tokens[0].back() == ':') {
			labels[tokens[0].substr(0, tokens[0].size()-1)] = opcodes.size();
			rlabels[opcodes.size()].insert(tokens[0].substr(0, tokens[0].size()-1));
			tokens.erase(tokens.begin());
		}

		if (tokens.empty())
			continue;

		opcodes.push_back(tokens);
		line_numbers.push_back(line_nr);
	}

	// Convert to encoded opcodes and write out

	for (current_op = 0; current_op < int(opcodes.size()); current_op++)
	{
		auto &tokens = opcodes[current_op];
		uint16_t opc = 0;

		last_op = this_op;
		this_op = tokens[0];

		if ((last_op == "seta" || last_op == "setb" || last_op == "ldab" || last_op == "lab") && (this_op == "stab" || this_op == "jab")) {
			std::string errmsg = "Invalid opcode sequence: " + this_op + " must not follow directly " + last_op + ".";
			print_parse_error(errmsg.c_str(), 0);
		}

		if (this_op == "ldi")
		{
			if (tokens.size() != 3)
				goto invalid_args;

			int r1 = parse_reg(1);
			uint8_t val = parse_const(2);

			print_op(0xc000 | ((val & 0xf0) << 4) | (r1 << 4) | (val & 0x0f));
			continue;
		}

		if (this_op.substr(0, 2) == "io")
		{
			if (tokens.size() != 2 && tokens.size() != 3)
				goto invalid_args;

			const char *nptr = this_op.c_str() + 2;
			char *endptr;
			int ep = strtol(nptr, &endptr, 10);
			if (*nptr == 0 || *endptr != 0 || ep < 0 || ep > 31)
				goto invalid_op;

			int r1 = parse_reg(1);
			int r2 = tokens.size() == 3 ? parse_reg(2) : r1;

			print_op(0xa000 | (ep << 8) | (r1 << 4) | r2);
			continue;
		}

		if (this_op == "nop")
		{
			if (tokens.size() != 1)
				goto invalid_args;

			print_op(0x0000);
			continue;
		}

		// Branch instructions

		if (this_op == "b") {
			opc = 0xe000;
			goto op_branch;
		}

		if (this_op == "bz") {
			opc = 0xe800;
			goto op_branch;
		}

		if (this_op == "bnz") {
			opc = 0xf000;
			goto op_branch;
		}
		
		if (0)
		{
	op_branch:
			if (tokens.size() != 2)
				goto invalid_args;

			if (labels.count(tokens[1]) == 0)
				print_parse_error("Unknown label.", 1);

			print_op(opc | labels.at(tokens[1]));
			continue;
		}

		// ALU instructions

		if (alu_ops.count(this_op))
		{
			int r1, r2, r3;
			opc = alu_ops.at(this_op) << 8;

			if (tokens.size() == 3) {
				r1 = r2 = parse_reg(1);
				r3 = parse_reg(2);
			} else
			if (tokens.size() == 4) {
				r1 = parse_reg(1);
				r2 = parse_reg(2);
				r3 = parse_reg(3);
			} else
				goto invalid_args;

			if (r1 < 8 && r2 < 8)
				print_op(opc | ((r2 & 6) << 12) | ((r2 & 1) << 7) | (r1 << 4) | r3);
			else if (r1 == r2)
				print_op(0x8000 | opc | (r1 << 4) | r3);
			else
				print_parse_error("Invalid combination of regs for ALU opcode.", 0);

			continue;
		}

		// Load/Store

		if (this_op == "ld" || this_op == "st" || this_op == "ldab" || this_op == "stab")
		{

			if (tokens.size() != 2 && tokens.size() != 3)
				goto invalid_args;

			int r1 = parse_reg(1);
			int r2 = tokens.size() == 3 ? parse_reg(2) : r1;

			if (this_op == "ld")
				print_op(0xf800 | (r1 << 4) | r2);
			else if (this_op == "st")
				print_op(0xf900 | (r1 << 4) | r2);
			else if (this_op == "ldab")
				print_op(0xfa00 | (r1 << 4) | r2);
			else if (this_op == "stab")
				print_op(0xfb00 | (r1 << 4) | r2);
			else
				abort();
			continue;
		}

		// Register <-> AB transfers

		if (this_op == "seta" || this_op == "setb" || this_op == "geta" || this_op == "getb")
		{

			if (tokens.size() != 2)
				goto invalid_args;

			int r1 = parse_reg(1);

			if (this_op == "seta")
				print_op(0xfc00 | (r1 << 4) | 0);
			else if (this_op == "setb")
				print_op(0xfc00 | (r1 << 4) | 1);
			else if (this_op == "geta")
				print_op(0xfc00 | (r1 << 4) | 2);
			else if (this_op == "getb")
				print_op(0xfc00 | (r1 << 4) | 3);
			else
				abort();
			continue;
		}

		// jump-AB and link-AB instructions

		if (this_op == "jab" || this_op == "lab")
		{
			if (tokens.size() != 1)
				goto invalid_args;

			if (this_op == "jab")
				print_op(0xff00);
			else if (this_op == "lab")
				print_op(0xff01);
			else
				abort();
			continue;
		}

		// wait instruction

		if (this_op == "sync")
		{
			uint16_t val = parse_const(1) & 0xfff;
			print_op(0xd000 | val);
			continue;
		}

	invalid_op:
		print_parse_error("Unknown opcode.", 0);
	invalid_args:
		print_parse_error("Invalid number of arguments.", tokens.size()-1);
	}
}


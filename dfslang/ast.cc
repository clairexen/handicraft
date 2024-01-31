/*
 *  DfsLang -- Data Flow Scheduling Language
 *
 *  Copyright (C) 2013  RIEGL Research ForschungsGmbH
 *  Copyright (C) 2013  Clifford Wolf <clifford@clifford.at>
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

#include "dfslang.h"
#include <assert.h>

AstNode::AstNode(AstNode &other, std::string suffix)
{
	this->value = other.value;
	this->type = other.type;
	this->str = other.str + suffix;

	for (auto &c : other.children)
		this->children.push_back(new AstNode(*c));
}

AstNode::AstNode(std::string type, int value, AstNode *child1, AstNode *child2, AstNode *child3)
{
	this->value = value;
	this->type = type;
	this->str = std::string();

	if (child1 != NULL)
		children.push_back(child1);
	if (child2 != NULL)
		children.push_back(child2);
	if (child3 != NULL)
		children.push_back(child3);
}

AstNode::AstNode(std::string type, std::string str, AstNode *child1, AstNode *child2, AstNode *child3)
{
	this->value = 0x7fffffff;
	this->type = type;
	this->str = str;

	if (child1 != NULL)
		children.push_back(child1);
	if (child2 != NULL)
		children.push_back(child2);
	if (child3 != NULL)
		children.push_back(child3);
}

AstNode::AstNode(std::string type, AstNode *child1, AstNode *child2, AstNode *child3)
{
	this->value = 0x7fffffff;
	this->type = type;
	this->str = std::string();

	if (child1 != NULL)
		children.push_back(child1);
	if (child2 != NULL)
		children.push_back(child2);
	if (child3 != NULL)
		children.push_back(child3);
}

AstNode::~AstNode()
{
	for (auto &c : children)
		delete c;
}

void AstNode::dump_code(std::string indent)
{
	if (type == "port") {
		printf(".%s", str.c_str());
		return;
	}

	if (type == "var") {
		printf("%s", str.c_str());
		for (auto c : children)
			c->dump_code(indent);
		return;
	}

	if (type == "dim") {
		printf("[");
		children.at(0)->dump_code(indent);
		printf("]");
		return;
	}

	if (type == "value") {
		printf("%d", value);
		return;
	}

	if (type == "stmts" || type == "ref") {
		for (auto c : children)
			c->dump_code(indent);
		return;
	}

	if (type.substr(0, 4) == "set_") {
		printf("%s%s ", indent.c_str(), type.c_str());
		children.at(0)->dump_code(indent);
		printf(";\n");
		return;
	}

	if (type == "use") {
		printf("%suse %s as %s;\n", indent.c_str(), children.at(0)->str.c_str(), children.at(1)->str.c_str());
		return;
	}

	if (type == "input" || type == "output" || type == "variable") {
		printf("%s", indent.c_str());
		if (type == "input" || type == "output")
			printf("%s ", type.c_str());
		printf("%c%d ", children.at(0)->value >= 0 ? 'u' : 's', abs(children.at(0)->value));
		for (size_t i = 1; i < children.size(); i++) {
			if (i > 1)
				printf(", ");
			children.at(i)->dump_code(indent);
		}
		printf(";\n");
		return;
	}

	if (type == "core") {
		printf("%s", indent.c_str());
		printf("core %s %d begin\n", str.c_str(), children.front()->value);
		children.back()->dump_code(indent + "  ");
		printf("end\n");
		return;
	}

	if (type == "task") {
		printf("%s", indent.c_str());
		printf("task %s(", str.c_str());
		for (size_t i = 0; i < children.size()-1; i++) {
			if (i > 0)
				printf(", ");
			children[i]->dump_code(indent);
		}
		printf(") begin\n");
		children.back()->dump_code(indent + "  ");
		printf("end\n");
		return;
	}

	if (type == "assign") {
		printf("%s", indent.c_str());
		children.at(0)->dump_code(indent);
		printf(" = ");
		children.at(1)->dump_code(indent);
		printf(";\n");
		return;
	}

	if (type == "call") {
		printf("%s%s(", indent.c_str(), str.c_str());
		for (size_t i = 0; i < children.size(); i++) {
			if (i > 0)
				printf(", ");
			children[i]->dump_code(indent);
		}
		printf(");\n");
		return;
	}

	assert(0);
}

void AstNode::dump(std::string indent)
{
	printf("%s%s", indent.c_str(), type.c_str());
	if (value != 0x7fffffff)
		printf(" %d", value);
	if (!str.empty())
		printf(" '%s'", str.c_str());
	printf("\n");

	for (auto &c : children)
		c->dump(indent + "  ");
}

void AstNode::make_prog(DfsProg &prog)
{
	int output_delay = 0;

	prog.cores.clear();
	prog.regs.clear();
	prog.stamps.clear();

	for (auto c : children)
	{
		if (c->type == "set_delay") {
			if (c->children.at(0)->type != "value") {
				fprintf(stderr, "Non-constant `delay' value!\n");
				exit(1);
			}
			output_delay = c->children.at(0)->value;
		}
		else if (c->type == "set_scheduler") {
			if (c->children.at(0)->type != "ref" || c->children.at(0)->children.at(0)->type != "var") {
				fprintf(stderr, "Invalid argument for 'set scheduler' (expected id)!\n");
				dump();
				exit(1);
			}
			prog.set_scheduler = c->children.at(0)->children.at(0)->str;
		}
		else if (c->type.substr(0, 4) == "set_") {
			if (c->children.at(0)->type == "ref" && c->children.at(0)->children.at(0)->type == "var") {
				prog.options[c->type.substr(4)] = c->children.at(0)->children.at(0)->str;
			} else if (c->children.at(0)->type == "value") {
				prog.options[c->type.substr(4)] = std::to_string(c->children.at(0)->value);
			} else {
				fprintf(stderr, "Invalid expression in set statement: %s\n", c->type.c_str());
				exit(1);
			}
		}

		if (c->type == "input" || c->type == "output" || c->type == "variable")
		{
			DfsReg reg;
			reg.direction = c->type == "input" ? 'i' : c->type == "output" ? 'o' : 'v';
			assert(c->children.at(0)->type == "type");
			reg.type = c->children.at(0)->value;

			for (size_t i = 1; i < c->children.size(); i++)
			{
				assert(c->children.at(i)->type == "var");
				reg.name = c->children.at(i)->str;
				std::vector<std::string> indices;
				indices.push_back("");

				for (auto d : c->children.at(i)->children)
				{
					assert(d->type == "dim");
					if (d->children.at(0)->type != "value") {
						fprintf(stderr, "Non-constant array dimension for variable `%s'!\n", reg.name.c_str());
						exit(1);
					}

					std::vector<std::string> new_indices;
					for (int k = 0; k < d->children.at(0)->value; k++) {
						char buffer[64];
						snprintf(buffer, 64, "[%d]", k);
						for (auto &old_idx : indices)
							new_indices.push_back(old_idx + buffer);
					}
					indices.swap(new_indices);
				}

				for (auto &idx : indices) {
					reg.index = idx;
					prog.regs.push_back(reg);
				}
			}
		}

		if (c->type == "core")
		{
			DfsCore core;

			assert(c->children.at(0)->type == "value");
			core.num_cores = c->children.at(0)->value;
			core.name = c->str;

			assert(c->children.at(1)->type == "stmts");
			for (auto p : c->children.at(1)->children)
			{
				if (p->type != "input" && p->type != "output") {
					fprintf(stderr, "Non-port statement in declaration of core `%s'!\n", core.name.c_str());
					exit(1);
				}

				int port_type = 0;
				for (auto q : p->children) {
					if (q->type == "type")
						port_type = q->value;
					if (q->type == "var") {
						if (q->children.size() != 0) {
							fprintf(stderr, "Unsupported-port type in port `%s' to core `%s'!\n", q->str.c_str(), core.name.c_str());
							exit(1);
						}
						if (core.ports_map.count(q->str)) {
							fprintf(stderr, "Duplicate port `%s' to core `%s'!\n", q->str.c_str(), core.name.c_str());
							exit(1);
						}
						assert(port_type != 0);
						core.ports_map[q->str] = core.ports_types.size();
						core.ports_types.push_back(port_type);
						core.ports_directions.push_back(p->type == "input" ? 'i' : 'o');
						core.ports_names.push_back(q->str);
					}
				}
			}

			prog.cores.push_back(core);
		}
	}

	DfsStamp stamp;
	std::map<std::string, int> cores_map;
	std::map<std::pair<std::string, std::string>, int> regs_map;

	for (auto c : children)
		if (c->type == "use")
		{
			assert(c->children.at(0)->type == "mod");
			assert(c->children.at(1)->type == "var");
			DfsCore *core = NULL;

			for (auto &ci : prog.cores)
				if (ci.name == c->children.at(0)->str)
					core = &ci;

			if (core == NULL) {
				fprintf(stderr, "Lookup of core `%s' failed!\n", c->children.at(0)->str.c_str());
				exit(1);
			}

			cores_map[c->children.at(1)->str] = stamp.cores.size();
			stamp.cores.push_back(core);
		}
	
	for (auto &ri : prog.regs) {
		regs_map[std::pair<std::string, std::string>(ri.name, ri.index)] = stamp.regs.size();
		stamp.regs.push_back(&ri);
	}

	for (auto c : children)
		if (c->type == "assign")
		{
			DfsAction action;
			assert(c->children.size() == 2);
			c->children.at(0)->map_reg(stamp, cores_map, regs_map, action.dst_core, action.dst_port, action.dst_arg);
			c->children.at(1)->map_reg(stamp, cores_map, regs_map, action.src_core, action.src_port, action.src_arg);
			if (action.dst_core < 0 && action.dst_port >= 0 && action.dst_arg >= 0)
				action.dst_arg += output_delay;
			stamp.actions.push_back(action);
		}
	
	prog.output_delay = output_delay;
	prog.stamps.push_back(stamp);
}

void AstNode::map_reg(DfsStamp &stamp, std::map<std::string, int> &cores_map, std::map<std::pair<std::string, std::string>, int> &regs_map, int &reg_core, int &reg_port, int &reg_arg)
{
	reg_core = -1;
	reg_port = -1;
	reg_arg = -1;

	if (type == "value") {
		reg_arg = value;
		return;
	}

	assert(type == "ref");
	assert(children.at(0)->type == "var");

	if (children.size() >= 2 && children.at(1)->type == "port")
	{
		if (cores_map.count(children.at(0)->str) == 0) {
			fprintf(stderr, "Lookup of core `%s' failed!\n", children.at(0)->str.c_str());
			exit(1);
		}

		reg_core = cores_map.at(children.at(0)->str);

		if (stamp.cores.at(reg_core)->ports_map.count(children.at(1)->str) == 0) {
			fprintf(stderr, "Lookup of port `%s.%s' failed!\n", children.at(0)->str.c_str(), children.at(1)->str.c_str());
			exit(1);
		}

		reg_port = stamp.cores.at(reg_core)->ports_map.at(children.at(1)->str);

		if (children.size() < 3 || children.at(2)->type != "dim") {
			fprintf(stderr, "Reference to port `%s.%s' without timing!\n", children.at(0)->str.c_str(), children.at(1)->str.c_str());
			exit(1);
		}

		if (children.at(2)->children.at(0)->type != "value") {
			fprintf(stderr, "Reference to port `%s.%s' with non-const timing!\n", children.at(0)->str.c_str(), children.at(1)->str.c_str());
			exit(1);
		}

		if (children.size() > 3) {
			fprintf(stderr, "Reference to port `%s.%s' with additional dimensions!\n", children.at(0)->str.c_str(), children.at(1)->str.c_str());
			exit(1);
		}

		reg_arg = children.at(2)->children.at(0)->value;
		return;
	}

	std::string reg_name, reg_index;

	for (size_t k = 0; k < children.size(); k++)
	{
		auto c = children[k];

		if (c->type == "var")
			reg_name = c->str;

		if (c->type == "dim") {
			if (c->children.at(0)->type != "value") {
				fprintf(stderr, "Reference to reg `%s' with non-const index!\n", reg_name.c_str());
				exit(1);
			}
			char buffer[64];
			snprintf(buffer, 64, "[%d]", c->children.at(0)->value);
			reg_index = reg_index + buffer;
		}

		if (regs_map.count(std::pair<std::string, std::string>(reg_name, reg_index)) > 0)
		{
			reg_port = regs_map.at(std::pair<std::string, std::string>(reg_name, reg_index));
			DfsReg *reg = stamp.regs.at(reg_port);

			if ((k != children.size()-1 && reg->direction == 'v') || (k != children.size()-2 && reg->direction != 'v')) {
				fprintf(stderr, "Invalid number of dimensions in reference to reg `%s'!\n", reg_name.c_str());
				exit(1);
			}

			if (reg->direction != 'v') {
				assert(children.at(k+1)->type == "dim");
				if (children.at(k+1)->children.at(0)->type != "value") {
					fprintf(stderr, "Reference to reg `%s' with non-const timing!\n", reg_name.c_str());
					exit(1);
				}
				reg_arg = children.at(k+1)->children.at(0)->value;
			}

			return;
		}
	}

	fprintf(stderr, "Can't look up variable `%s%s'!\n", reg_name.c_str(), reg_index.c_str());
	exit(1);
}

void AstNode::optimize()
{
	std::vector<AstNode*> parents;
	do {
		while (optimize_worker(NULL, 0, parents, false)) { }
	} while (optimize_worker(NULL, 0, parents, true));
	optimize_cleanup();
}

void AstNode::optimize_cleanup()
{
	std::vector<AstNode*> new_children;
	for (auto c : children)
		if (c->type == "task") {
			delete c;
		} else {
			new_children.push_back(c);
			c->optimize_cleanup();
		}
	children.swap(new_children);
}

bool AstNode::optimize_worker(AstNode *parent, size_t parent_idx, std::vector<AstNode*> &parents, bool do_call)
{
	bool did_something = false;

	// flatten trees of stmts

	if (type == "stmts")
	{
		std::vector<AstNode*> new_children;
		expand_stmts(new_children);
		if (!children.empty())
			did_something = true;
		delete_children();
		children.swap(new_children);
	}

	// optimize children

	parents.push_back(this);

	bool keep_running = true;
	while (keep_running) {
		keep_running = false;
		for (size_t i = 0; i < children.size(); i++)
			while (children.at(i)->optimize_worker(this, i, parents, do_call))
				did_something = true, keep_running = true;
	}

	parents.pop_back();

	// done if this is a stmts node

	if (type == "stmts")
		return did_something;

	assert(parent != NULL);

	// flatten trees of 'var', 'dim', etc. (parser generates trees, we want flat lists)

	if (type == "var" || type == "dim" || type == "ref" || type == "range" || type == "expr")
	{
		std::vector<AstNode*> new_children;
		int insert_idx = 1;
		for (auto c : children) {
			if (c->type == type) {
				parent->children.insert(parent->children.begin() + parent_idx + (insert_idx++), c);
				did_something = true;
			} else
				new_children.push_back(c);
		}
		children.swap(new_children);
	}

	// unbox expressions

	while (type == "expr" && parent->type != "expr" && children.size() == 1)
	{
		AstNode *child = children.front();
		children.clear();

		value = child->value;
		type = child->type;
		str = child->str;
		children.swap(child->children);

		delete child;
		did_something = true;
	}

	// expand for loops

	if (type == "for")
	{
		std::vector<AstNode*> ranges;
		AstNode *stmts = NULL;

		for (auto c : children) {
			if (c->type == "range") {
				ranges.push_back(c);
			} else if (c->type == "stmts") {
				assert(stmts == NULL);
				stmts = c;
			} else
				assert(0);
		}
		assert(stmts != NULL);
		children.clear();

		for (auto range : ranges)
		{
			std::vector<AstNode*> new_stmts;

			assert(range->children.size() == 3);
			assert(range->children.at(0)->type == "var");
			assert(range->children.at(1)->type == "value");
			assert(range->children.at(2)->type == "value");

			int i_min = std::min(range->children.at(1)->value, range->children.at(2)->value);
			int i_max = std::max(range->children.at(1)->value, range->children.at(2)->value);
			for (int i = i_min; i <= i_max; i++)
				new_stmts.push_back(stmts->specialize(range->children.at(0)->str, i));

			delete stmts;
			stmts = new AstNode("stmts");
			stmts->children.swap(new_stmts);
			delete range;
		}

		type = "stmts";
		children.swap(stmts->children);
		delete stmts;

		did_something = true;
	}

	// flatten calls to tasks

	if (do_call && type == "call")
	{
		AstNode *task = NULL;

		for (auto p : parents) {
			if (p->type != "stmts")
				continue;
			for (auto t : p->children)
				if (t->type == "task" && t->str == str)
					task = t;
		}

		if (task == NULL) {
			fprintf(stderr, "Lookup of task `%s' failed!\n", str.c_str());
			exit(1);
		}

		static int unique_idx = 0;
		char unique_suffix[64];
		snprintf(unique_suffix, 64, "#%d", unique_idx++);

		std::vector<AstNode*> mempool;
		std::map<std::string, AstNode*> rename_map;

		if (task->children.size()-1 != children.size()) {
			fprintf(stderr, "Invalid number of arguments in call to task `%s'!\n", str.c_str());
			exit(1);
		}

		for (size_t i = 0; i < task->children.size()-1; i++) {
			rename_map[task->children.at(i)->str] = children.at(i);
		}

		AstNode *stmts = task->children.back()->instanciate(rename_map, unique_suffix, mempool);

		for (auto n : mempool)
			delete n;

		this->type = stmts->type;
		this->value = stmts->value;
		this->str = stmts->str;
		this->children.swap(stmts->children);
		delete stmts;

		did_something = true;
	}

	// evaluate constant expressions

#define OP(_op_type) if (type == "op(" #_op_type ")" && children.at(0)->type == "value" && children.at(1)->type == "value") \
		{ type = "value"; value = children.at(0)->value _op_type children.at(1)->value; delete_children(); did_something = true; }
	OP(-) OP(+) OP(*) OP(/) OP(%) OP(<) OP(<=) OP(==) OP(!=) OP(>=) OP(>)
#undef OP

	if (type == "neg" && children.at(0)->type == "value") {
		type = "value";
		value = -children.at(0)->value;
		delete_children();
		did_something = true;
	}

	if (type == "ref" && children.at(0)->type == "value")  {
		type = "value";
		value = children.at(0)->value;
		delete_children();
		did_something = true;
	}

	return did_something;
}

AstNode *AstNode::specialize(std::string varname, int value)
{
	if (type == "ref" && children.at(0)->type == "var" && children.at(0)->str == varname)
		return new AstNode("value", value);
	
	AstNode *node = new AstNode(this->type);
	node->value = this->value;
	node->str = this->str;

	for (auto c : children)
		node->children.push_back(c->specialize(varname, value));

	return node;
}

void AstNode::expand_stmts(std::vector<AstNode*> &new_children)
{
	assert(type == "stmts");

	std::vector<AstNode*> rem_children;

	for (auto c : children) {
		if (c->type == "stmts") {
			rem_children.push_back(c);
			c->expand_stmts(new_children);
		} else
			new_children.push_back(c);
	}

	children.swap(rem_children);
}

AstNode *AstNode::instanciate(std::map<std::string, AstNode*> rename_map, std::string rename_suffix, std::vector<AstNode*> &mempool)
{
	if (type == "stmts") {
		for (auto c : children) {
			if (c->type == "variable" || c->type == "use")
				for (auto v : c->children)
					if (v->type == "var") {
						AstNode *n = new AstNode("ref", new AstNode(*v, rename_suffix));
						rename_map[v->str] = n;
						mempool.push_back(n);
					}
		}
	}

	AstNode *node = new AstNode(this->type);
	node->value = this->value;
	node->str = this->str;

	if (node->type == "ref" || node->type == "variable" || node->type == "use")
	{
		for (auto c : children)
			if (c->type == "var" && rename_map.count(c->str) > 0) {
				if (rename_map.at(c->str)->type != "ref")
					node->children.push_back(rename_map.at(c->str)->instanciate(rename_map, rename_suffix, mempool));
				else for (auto n : rename_map.at(c->str)->children)
					node->children.push_back(n->instanciate(rename_map, rename_suffix, mempool));
			} else
				node->children.push_back(c->instanciate(rename_map, rename_suffix, mempool));
	}
	else
	{
		for (auto c : children)
			node->children.push_back(c->instanciate(rename_map, rename_suffix, mempool));
	}

	return node;
}

void AstNode::delete_children()
{
	for (auto c : children)
		delete c;
	children.clear();
}


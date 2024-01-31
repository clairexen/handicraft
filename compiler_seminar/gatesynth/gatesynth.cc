#include "gatesynth.h"

workspace::workspace(const char *charp)
{
	tree = NULL;
	yycharp = charp;
	yyworkspace = this;
	yyparse();
	yylex_destroy();
}

workspace::~workspace()
{
	if (tree)
		delete tree;
	for (auto &it : forest)
		delete it.second;
	for (auto &it : gates) {
		delete it->pattern;
		delete it;
	}
}


void workspace::add_gate(std::string name, int cost, std::string expr)
{
	struct gate *g = new gate;
	struct workspace temp_ws(expr.c_str());
	g->name = name;
	g->pattern = temp_ws.tree;
	g->cost = cost;
	temp_ws.tree = NULL;
	gates.push_back(g);
}

int main(int argc, char **argv)
{
	const char *progname = argv[0];

	if (argc > 1 && !strcmp(argv[1], "-d")) {
		yydebug = 1;
		argv++, argc--;
	}

	if (argc != 2) {
		fprintf(stderr, "Usage: %s { <expr> | - }\n", progname);
		return 1;
	}

	struct workspace ws(!strcmp(argv[1], "-") ? NULL : argv[1]);
	ws.add_gate("nand",  4, "(a*b)'");
	ws.add_gate("nor",   4, "(a+b)'");
	ws.add_gate("aoi",   8, "((a*b)+(c*d))'");
	ws.add_gate("oai",   8, "((a+b)*(c+d))'");
	ws.add_gate("aoi21", 6, "((a*b)+c)'");
	ws.add_gate("oai21", 6, "((a+b)*c)'");
	ws.add_gate("aoi12", 6, "(a+(b*c))'");
	ws.add_gate("oai12", 6, "(a*(b+c))'");

	if (ws.tree) {
		std::string autoname = "y";
		while (ws.inputs.count(autoname) > 0 || ws.forest.count(autoname) > 0)
			autoname = autoname + "_";
		ws.forest[autoname] = ws.tree;
		ws.tree = NULL;
	}

	printf("\n");

	for (auto &it : ws.forest)
		printf("%s = %s;\n", it.first.c_str(), it.second->dump_aitree().c_str());

	printf("\n");
	for (auto &it : ws.forest) {
		it.second->map(ws.gates, 2);
		printf("mapped %s at cost %d.\n", it.first.c_str(), it.second->mapped_cost[it.second->is_neg]);
	}

	printf("\n");

	for (auto &it : ws.forest)
		printf("%s = %s;\n", it.first.c_str(), it.second->dump_mapped().c_str());

	printf("\n");
	return 0;
}


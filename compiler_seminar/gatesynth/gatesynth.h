#ifndef GATESYNTH_H
#define GATESYNTH_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <string>
#include <vector>
#include <map>
#include <set>

struct aitree
{
	struct aitree *parents[2];
	std::string atom_id;
	bool is_neg;

	int mapped_cost[2];
	struct gate *mapped_gate[2];
	bool mapped_gate_invert[2];
	struct aitree *mapped_parents[2][4];
	bool mapped_parents_invert[2][4];

	aitree();
	aitree(std::string id);
	aitree(aitree *p1, aitree *p2);
	~aitree();

	void invert();
	std::string dump_aitree();
	std::string dump_mapped() { return dump_mapped(is_neg); }
	std::string dump_mapped(bool inverted);

	void map(std::vector<struct gate*> &gates, int invert_cost);
	int map_match(struct aitree *pattern, struct aitree *map_parents[4], bool map_parents_invert[4], int root_neg = -1);
};

struct gate
{
	struct aitree *pattern;
	std::string name;
	int cost;
};

struct workspace
{
	struct aitree *tree;
	std::map<std::string, struct aitree*> forest;
	std::set<std::string> inputs;
	std::vector<struct gate*> gates;

	workspace(const char *charp);
	~workspace();

	void add_gate(std::string name, int cost, std::string expr);
};

// non-standard global variables for flex/bison parser
extern const char *yycharp;
extern struct workspace *yyworkspace;

// standard global variables for flex/bison parser
extern int yydebug;
extern int yylex(void);
extern void yyerror(char const *s);
extern void yyrestart(FILE *f);
extern int yyparse(void);
extern int yylex_destroy(void);
extern int yyget_lineno(void);

#endif

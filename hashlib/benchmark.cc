// This is free and unencumbered software released into the public domain.
// 
// Anyone is free to copy, modify, publish, use, compile, sell, or
// distribute this software, either in source code form or as a compiled
// binary, for any purpose, commercial or non-commercial, and by any
// means.

// -------------------------------------------------------
// Written by Clifford Wolf <clifford@clifford.at> in 2014
// -------------------------------------------------------

#include <iostream>
#include <cmath>

#include "hashlib.h"
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>

using namespace hashlib;
using namespace std;

bool balance_active = false;
int balance = 0;

uint32_t xorshift32()
{
	static uint32_t x = 314159265;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return x;
}

int checksum(unsigned int v)
{
	static unsigned int x = 314159265;
	x = x * 33 ^ v;
	return (x & 0xffff) ^ (x >> 16);
}

void error(const char *p)
{
	cerr << "Error: " << p << endl;
	exit(1);
}

template<typename T, typename set_t>
void run_test_set(const vector<T> &database, int N, int M)
{
	set_t uut;

	int p = 0, q = 0;
	bool up_mode = true;
	int swing = std::min(N/10, 1000);
	for (int i = 0; i < M; i++) {
		auto it = uut.find(database[up_mode ? q : p]);
		checksum(it != uut.end() ? q^p : -(q^p));
		if (up_mode) {
			if (it == uut.end())
				uut.insert(database[q]);
			q = q+1 != database.size() ? q+1 : 0;
			if (uut.size() > N + swing) up_mode = false;
		} else {
			if (it != uut.end())
				uut.erase(it);
			p = p+1 != database.size() ? p+1 : 0;
			if (uut.size() < N - swing) up_mode = true;
		}
		if (uut.size() == N)
			balance_active = true;
		if (balance_active)
			balance += up_mode ? +1 : -1;
	}
}

template<typename T, typename map_t>
void run_test_map(const vector<T> &database, int N, int M)
{
	map_t uut;

	int p = 0, q = 0;
	bool up_mode = true;
	int swing = std::min(N/10, 1000);
	for (int i = 0; i < M; i++) {
		auto it = uut.find(database[up_mode ? q : p]);
		checksum(it != uut.end() ? it->second : q ^ p);
		if (up_mode) {
			if (it == uut.end())
				it = uut.insert(std::pair<T, int>(database[q], 0)).first;
			it->second = it->second * 33 ^ q;
			q = q+1 != database.size() ? q+1 : 0;
			if (uut.size() > N + swing) up_mode = false;
		} else {
			if (it != uut.end())
				uut.erase(it);
			p = p+1 != database.size() ? p+1 : 0;
			if (uut.size() < N - swing) up_mode = true;
		}
		if (uut.size() == N)
			balance_active = true;
		if (balance_active)
			balance += up_mode ? +1 : -1;
	}
}

template<typename T>
void run_test(const vector<T> &database, const string &impl, int N, int M)
{
	if (impl == "set") run_test_set<T, set<T>>(database, N, M);
	else if (impl == "map") run_test_map<T, map<T, int>>(database, N, M);
	else if (impl == "unordered_set") run_test_set<T, unordered_set<T>>(database, N, M);
	else if (impl == "unordered_map") run_test_map<T, unordered_map<T, int>>(database, N, M);
	else if (impl == "pool") run_test_set<T, pool<T>>(database, N, M);
	else if (impl == "dict") run_test_map<T, dict<T, int>>(database, N, M);
	else if (impl == "none") return;
	else error("invalid argument");
}

int main(int argc, char **argv)
{
	if (argc != 6)
		error("invalid number of args. Usage: benchmark {set|map|unordered_set|unordered_map|dict|pool|none} {int|string} {dense|sparse} {active_size} {iters}");
	
	string impl = argv[1];
	string type = argv[2];
	string mode = argv[3];
	int N = atoi(argv[4]);
	int M = atoi(argv[5]);

	if (N <= 0 || M <= 0)
		error("invalid argument");

	if (type == "int")
	{
		if (mode != "dense" && mode != "sparse")
			error("invalid argument");

		vector<int> database(5*N);
		bool dense_mode = mode == "dense";
		int max_value = dense_mode ? 2*N : 56*N;

		for (auto &key : database) {
			key = xorshift32() % max_value;
			if (!dense_mode) key = (key*7) % max_value;
		}

		run_test<int>(database, impl, N, M);
	} else
	if (type == "string")
	{
		vector<char> alphabet;

		if (mode == "dense") {
			for (char c = 'a'; c <= 'z'; c++)
				alphabet.push_back(c);
		} else
		if (mode == "sparse") {
			alphabet.push_back('A'); // 0x41
			alphabet.push_back('a'); // 0x61
			alphabet.push_back('E'); // 0x45
			alphabet.push_back('e'); // 0x65
		} else
			error("invalid argument");

		int key_len = log(N) / log(alphabet.size());
		vector<string> database(5*N);

		for (auto &key : database) {
			key.reserve(key_len);
			for (int i = 0; i < key_len; i++)
				key += alphabet.at(xorshift32() % alphabet.size());
		}

		run_test<string>(database, impl, N, M);
	} else
		error("invalid argument");

	cout << checksum(0) << " " << balance << endl;
	return 0;
}


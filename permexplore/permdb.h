// Copyright (C) 2019  Clifford Wolf <clifford@clifford.at>
//
// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice appear in all copies.
//
// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
// WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
// ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
// OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

// Simple C++ header-only library for managing databases of bit permutations
// and how to create them using ROR/GREV/[UN]SHFL instructions.

#define PERMDB_COMPRESSED

#ifndef PERMDB_H
#define PERMDB_H

#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <assert.h>
#include <vector>
#include <tuple>
#include <string>
#include <fstream>

namespace pdb
{
	std::string stringf(const char *fmt, ...)
	{
		std::string s;
		char *p = NULL;
		va_list ap;

		va_start(ap, fmt);
		if (vasprintf(&p, fmt, ap) < 0)
			p = nullptr;
		va_end(ap);

		if (p != nullptr) {
			s = p;
			free(p);
		}

		return s;
	}

	int unperm(int k, int i)
	{
		return ((k + (i & k & ~(k<<1))) & ~k) | (i & ~(k | (k<<1))) | ((i>>1) & k);
	}

	enum op_t : char
	{
		OP_NONE,
		OP_ROR,
		OP_GREV,
		OP_SHFL,
		OP_UNSHFL
	};

	enum flag_t : char
	{
		FLAG_VIA_ROR    = 0x01,
		FLAG_VIA_GREV   = 0x02,
		FLAG_VIA_SHFL   = 0x04,
		FLAG_VIA_UNSHFL = 0x08
	};

	struct info_t
	{
		op_t op;
		char arg;
		char depth;
		char flags;
		long parent;
	};

	template<int N, int SZ>
	struct perm_t
	{
	#ifdef PERMDB_COMPRESSED
		static constexpr int nbits =
			N ==  4 ? 2 :
			N ==  8 ? 3 :
			N == 16 ? 4 :
			N == 32 ? 5 :
			N == 64 ? 6 : -1;

		static constexpr int perword = 32 / nbits;
		static constexpr int nwords = (N+perword-1)/perword;

		uint32_t words[nwords] = {};

		int get(int idx) const
		{
			int w = idx / perword;
			int i = idx % perword;
			int val = (words[w] >> (i*nbits)) & (N-1);
			return val;
		}

		void set(int idx, int val)
		{
			int w = idx / perword;
			int i = idx % perword;
			uint32_t mask = (N-1) << (i*nbits);
			uint32_t data = (val & (N-1)) << (i*nbits);
			words[w] &= ~mask;
			words[w] |= data;
		}

		long hash() const
		{
			uint64_t h = 12345678;
			for (int i = 0; i < nwords; i++) {
				h = ((h << 5) + h) ^ words[i];
				h ^= h << 13;
				h ^= h >> 7;
				h ^= h << 17;
			}
			return h >> 1;
		}

		bool operator==(const perm_t &other) const
		{
			return memcmp(words, other.words, sizeof(words)) == 0;
		}

		bool operator<(const perm_t &other) const
		{
			return memcmp(words, other.words, sizeof(words)) < 0;
		}
	#else
		char data[N] = {};
		int get(int idx) const { return data[idx]; }
		void set(int idx, int val) { data[idx] = val; }

		long hash() const
		{
			uint64_t h = 12345678;
			for (int i = 0; i < N; i++)
				h = ((h << 5) + h) ^ data[i];
			for (int i = 0; i < 2; i++) {
				h ^= h << 13;
				h ^= h >> 7;
				h ^= h << 17;
			}
			return h >> 1;
		}

		bool operator==(const perm_t &other) const
		{
			return memcmp(data, other.data, sizeof(data)) == 0;
		}

		bool operator<(const perm_t &other) const
		{
			return memcmp(data, other.data, sizeof(data)) < 0;
		}
	#endif

		void swap(int i, int j)
		{
			int a = get(i);
			int b = get(j);
			set(i, b);
			set(j, a);
		}

		static perm_t<N,SZ> null()
		{
			perm_t<N,SZ> p;
			for (int i = 0; i < N; i++)
				p.set(i, 0);
			return p;
		}

		static perm_t<N,SZ> identity()
		{
			perm_t<N,SZ> p;
			for (int i = 0; i < N; i++)
				p.set(i, i);
			return p;
		}

		perm_t<N,SZ> ror(int arg) const
		{
			perm_t<N,SZ> p;
			arg &= N-1;
			for (int i = 0; i < N; i++)
				p.set(i, get((i+arg) & (N-1)));
			return p;
		}

		perm_t<N,SZ> grev(int arg) const
		{
			perm_t<N,SZ> p;
			arg &= N-1;
			for (int i = 0; i < N; i++)
				p.set(i, get(i^arg));
			return p;
		}

		perm_t<N,SZ> shfl(int arg) const
		{
			perm_t<N,SZ> p;
			arg &= N/2-1;
			for (int i = 0; i < N; i++) {
				int j = unperm(arg, i);
				p.set(i, get(j));
			}
			return p;
		}

		perm_t<N,SZ> unshfl(int arg) const
		{
			perm_t<N,SZ> p;
			arg &= N/2-1;
			for (int i = 0; i < N; i++) {
				int j = unperm(arg, i);
				p.set(j, get(i));
			}
			return p;
		}

		bool invertible() const
		{
			uint64_t mask = ~uint64_t(0);
			if (N < 64) mask >>= (64-N);
			for (int i = 0; i < N; i++)
				mask &= ~(uint64_t(1) << get(i));
			return mask == 0;
		}

		perm_t<N,SZ> invert() const
		{
			perm_t<N,SZ> p;
			assert(invertible());
			for (int i = 0; i < N; i++)
				p.set(get(i), i);
			return p;
		}

		perm_t<N,SZ> apply(const perm_t<N,SZ> &fun) const
		{
			perm_t<N,SZ> p;
			assert(fun.invertible());
			for (int i = 0; i < N; i++)
				p.set(i, get(fun.get(i)));
			return p;
		}

		perm_t<N,SZ> apply(const info_t &fun) const
		{
			switch (fun.op)
			{
				case OP_NONE:
					return *this;
				case OP_ROR:
					return ror(fun.arg);
				case OP_GREV:
					return grev(fun.arg);
				case OP_SHFL:
					return shfl(fun.arg);
				case OP_UNSHFL:
					return unshfl(fun.arg);
				default:
					assert(0);
			}
		}

		perm_t<N,SZ> rapply(const info_t &fun) const
		{
			switch (fun.op)
			{
				case OP_NONE:
					return *this;
				case OP_ROR:
					return ror((N-1) & (-fun.arg));
				case OP_GREV:
					return grev(fun.arg);
				case OP_SHFL:
					return unshfl(fun.arg);
				case OP_UNSHFL:
					return shfl(fun.arg);
				default:
					assert(0);
			}
		}

		perm_t<N,SZ> func(const perm_t<N,SZ> &identity) const
		{
			perm_t<N,SZ> p;
			int map[N][N];
			int lens[N] = { /* zeros */ };

			for (int i = 0; i < N; i++) {
				int j = identity.get(i);
				map[j][lens[j]++] = i;
			}

			for (int i = 0; i < N; i++) {
				int j = get(i);
				assert(lens[j] > 0);
				j = map[j][--lens[j]];
				p.set(i, j);
			}

			return p;
		}

		std::string str() const
		{
			std::string s;
			if (SZ == 1) {
				for (int i = N-1; i >= 0; i--)
					s += stringf("%3d", get(i));
			} else
			if (N > 8) {
				for (int i = N-1; i >= 0; i--)
					s += stringf("%3d:%-2d", SZ*(get(i)+1)-1, SZ*get(i));
			} else {
				for (int i = N-1; i >= 0; i--)
					s += stringf(" %c", 'A'+(N-1)-get(i));
			}
			return s;
		}
	};

	template<int N, int SZ>
	struct permdb_t
	{
		const int log2bucketsz = 20;

		struct permdb_entry_t
		{
			perm_t<N, SZ> perm;
			info_t info;
			long next;
		};

		long reserve = 0, headptr = 0, hashmask = 0;
		std::vector<std::vector<permdb_entry_t>> buckets;
		std::vector<long> hashtable, queue;

		mutable long statuscnt = 0;

		long size() const { return headptr; }

		void status(const std::string &jobdesc, long jobstate = 0, long jobsize = 0, long maxcnt = 0) const
		{
			if (statuscnt < maxcnt && jobstate != 0 && jobstate != jobsize) {
				statuscnt++;
				return;
			}

			statuscnt = 0;

			float mem_gb = 0;
			std::ifstream statm;
			char fn[1024];
			snprintf(fn, 1024, "/proc/%lld/statm", (long long)getpid());
			statm.open(fn);
			if (statm.is_open()) {
				int sz_total;
				statm >> sz_total;
				mem_gb = sz_total * (getpagesize() / 1024.0 / 1024.0 / 1024.0);
			}

			float mem_database_gb = float(buckets.size()) * float(1 << log2bucketsz) * float(sizeof(permdb_entry_t)) / 1024.0 / 1024.0 / 1024.0;
			float mem_hashtable_gb = float(hashtable.size()) * float(sizeof(long)) / 1024.0 / 1024.0 / 1024.0;
			float mem_queue_gb = float(queue.size()) * float(sizeof(long)) / 1024.0 / 1024.0 / 1024.0;

			if (jobstate != jobsize) {
				float percent = 100*float(jobstate)/float(jobsize);
				printf("%s %6.2f%%  ", jobdesc.c_str(), percent);
			} else {
				printf("%s  -----   ", jobdesc.c_str());
			}

			printf("MEM: %.2f GB  (DB %.2f GB, HASH %.2f GB, QUEUE %.2f GB)  ", mem_gb,
					mem_database_gb, mem_hashtable_gb, mem_queue_gb);
			printf("ENTRIES: %ld\n", headptr);
			fflush(stdout);
		}

		void rehash()
		{
			assert(reserve == 0);
			assert(headptr == long(buckets.size() << log2bucketsz));

			reserve += long(1) << log2bucketsz;
			buckets.resize(buckets.size()+1);
			buckets.back().resize(long(1) << log2bucketsz);

			long newhashmask = buckets.size() << log2bucketsz;
			newhashmask |= newhashmask >> 1;
			newhashmask |= newhashmask >> 2;
			newhashmask |= newhashmask >> 4;
			newhashmask |= newhashmask >> 8;
			newhashmask |= newhashmask >> 16;
			newhashmask |= newhashmask >> 32;

			if (newhashmask == hashmask)
				return;

			time_t start_time = time(NULL);
			hashmask = newhashmask;

			printf("Rehashing. New hashtable size: %ld ...", hashmask+1);
			fflush(stdout);

			hashtable.clear();
			hashtable.resize(hashmask+1, -1);

			for (long i = 0; i < headptr; i++) {
				auto &item = buckets[i >> log2bucketsz][i & ((long(1) << log2bucketsz) - 1)];
				item.next = -1;
			}

			for (long i = 0; i < headptr; i++) {
				auto &item = buckets[i >> log2bucketsz][i & ((long(1) << log2bucketsz) - 1)];
				long h = item.perm.hash() & hashmask;
				item.next = hashtable[h];
				hashtable[h] = i;
			}

			time_t stop_time = time(NULL);

			printf(" rehashing completed after %d seconds.\n", int(stop_time-start_time));
			fflush(stdout);
		}

		const permdb_entry_t &entry(long index) const
		{
			return buckets[index >> log2bucketsz][index & ((long(1) << log2bucketsz) - 1)];
		}

		const perm_t<N,SZ> &perm(long index) const
		{
			return buckets[index >> log2bucketsz][index & ((long(1) << log2bucketsz) - 1)].perm;
		}

		const info_t &info(long index) const
		{
			return buckets[index >> log2bucketsz][index & ((long(1) << log2bucketsz) - 1)].info;
		}

		info_t &info(long index)
		{
			return buckets[index >> log2bucketsz][index & ((long(1) << log2bucketsz) - 1)].info;
		}

		std::string prog(long index, bool verbose = false, perm_t<N, SZ> *permp = nullptr, bool reverse = false) const
		{
			std::string s, t;

			const auto &data = entry(index);

			if (!reverse && data.info.op != OP_NONE)
				s = prog(data.info.parent, verbose, permp, reverse);

			if (reverse)
			{
				if (data.info.op == OP_ROR)
					t = stringf("ROR(%d)", SZ*((N-1) & -data.info.arg));
				if (data.info.op == OP_GREV)
					t = stringf("GREV(%d)", SZ*data.info.arg);
				if (data.info.op == OP_SHFL)
					t = stringf("UNSHFL(%d)", SZ*data.info.arg);
				if (data.info.op == OP_UNSHFL)
					t = stringf("SHFL(%d)", SZ*data.info.arg);

				if (permp != nullptr)
					*permp = permp->rapply(data.info);
			}
			else
			{
				if (data.info.op == OP_ROR)
					t = stringf("ROR(%d)", SZ*data.info.arg);
				if (data.info.op == OP_GREV)
					t = stringf("GREV(%d)", SZ*data.info.arg);
				if (data.info.op == OP_SHFL)
					t = stringf("SHFL(%d)", SZ*data.info.arg);
				if (data.info.op == OP_UNSHFL)
					t = stringf("UNSHFL(%d)", SZ*data.info.arg);

				if (permp != nullptr)
					*permp = permp->apply(data.info);
			}

			if (verbose)
			{
				if (data.info.op == OP_NONE)
					t = "INIT";

				perm_t<N, SZ> p = data.perm;
				if (reverse && data.info.op != OP_NONE)
					p = perm(data.info.parent);

				if (permp != nullptr)
					p = *permp;

				if (data.info.op != OP_NONE || (permp == nullptr && !reverse))
					s += stringf("%-10s %s\n", t.c_str(), p.str().c_str());
			}
			else
			{
				if (s != "")
					s += ",";
				s += t;
			}

			if (reverse && data.info.op != OP_NONE) {
				if (!verbose && info(data.info.parent).op != OP_NONE)
					s += ",";
				s += prog(data.info.parent, verbose, permp, reverse);
			}

			return s;
		}

		std::string rprog(long index, bool verbose = false, perm_t<N, SZ> *permp = nullptr) const
		{
			return prog(index, verbose, permp, true);
		}

		long find(const perm_t<N, SZ> &perm) const
		{
			long h = perm.hash() & hashmask;
			long index = hashtable[h];
			while (1) {
				if (index < 0)
					break;
				auto &e = entry(index);
				if (e.perm == perm)
					break;
				index = e.next;
			}
			return index;
		}

		long insert(const perm_t<N, SZ> &perm, const info_t &info, std::vector<long> *queue)
		{
			if (reserve == 0)
				rehash();

			long index = find(perm);

			if (index >= 0)
			{
				auto &item = buckets[index >> log2bucketsz][index & ((long(1) << log2bucketsz) - 1)];
				if (info.depth < item.info.depth)
					item.info = info;
				else if (info.depth == item.info.depth)
					item.info.flags |= info.flags;
				return index;
			}

			reserve--;
			index = headptr++;

			auto &item = buckets[index >> log2bucketsz][index & ((long(1) << log2bucketsz) - 1)];
			long h = perm.hash() & hashmask;
			item.next = hashtable[h];
			item.perm = perm;
			item.info = info;
			hashtable[h] = index;

			if (queue != nullptr)
				queue->push_back(index);
			return index;
		}

		long insert(const perm_t<N, SZ> &perm, std::vector<long> *queue)
		{
			info_t info;
			info.op = OP_NONE;
			info.arg = 0;
			info.depth = 0;
			info.flags = 0;
			info.parent = -1;
			return insert(perm, info, queue);
		}

		long insert(const perm_t<N, SZ> &perm)
		{
			return insert(perm, &queue);
		}

		long ror(long parent, int arg, std::vector<long> *queue)
		{
			const auto &e = entry(parent);

			info_t info;
			info.op = OP_ROR;
			info.arg = arg & (N-1);
			info.depth = e.info.depth+1;
			info.flags = FLAG_VIA_ROR;
			info.parent = parent;

			const auto n = e.perm.ror(arg);
			return insert(n, info, queue);
		}

		long ror(long parent, int arg)
		{
			return ror(parent, arg, &queue);
		}

		long grev(long parent, int arg, std::vector<long> *queue)
		{
			const auto &e = entry(parent);

			info_t info;
			info.op = OP_GREV;
			info.arg = arg & (N-1);
			info.depth = e.info.depth+1;
			info.flags = FLAG_VIA_GREV;
			info.parent = parent;

			const auto n = e.perm.grev(arg);
			return insert(n, info, queue);
		}

		long grev(long parent, int arg)
		{
			return grev(parent, arg, &queue);
		}

		long shfl(long parent, int arg, std::vector<long> *queue)
		{
			const auto &e = entry(parent);

			info_t info;
			info.op = OP_SHFL;
			info.arg = arg & (N-1);
			info.depth = e.info.depth+1;
			info.flags = FLAG_VIA_SHFL;
			info.parent = parent;

			const auto n = e.perm.shfl(arg);
			return insert(n, info, queue);
		}

		long shfl(long parent, int arg)
		{
			return shfl(parent, arg, &queue);
		}

		long unshfl(long parent, int arg, std::vector<long> *queue)
		{
			const auto &e = entry(parent);

			info_t info;
			info.op = OP_UNSHFL;
			info.arg = arg & (N-1);
			info.depth = e.info.depth+1;
			info.flags = FLAG_VIA_UNSHFL;
			info.parent = parent;

			const auto n = e.perm.unshfl(arg);
			return insert(n, info, queue);
		}

		long unshfl(long parent, int arg)
		{
			return unshfl(parent, arg, &queue);
		}

		void expand(long index, std::vector<long> *queue)
		{
			auto flags = info(index).flags;

			if ((flags & FLAG_VIA_ROR) == 0) {
				for (int i = 0; i < N; i++)
					ror(index, i, queue);
			}

			if ((flags & (FLAG_VIA_GREV|FLAG_VIA_SHFL|FLAG_VIA_UNSHFL)) == 0) {
				for (int i = 0; i < N; i++)
					grev(index, i, queue);
			}

			for (int i = 0; i < N/2; i++) {
				shfl(index, i, queue);
				unshfl(index, i, queue);
			}
		}

		void wave(const std::string &wavedesc)
		{
			std::vector<long> job;
			job.swap(queue);

			for (long cnt = 0; cnt < long(job.size()); cnt++) {
				status(wavedesc, cnt, job.size(), 10000);
				expand(job[cnt], &queue);
			}

			status(wavedesc);
			printf("--------\n");
			fflush(stdout);
		}

		std::pair<int,std::string> find1(const perm_t<N,SZ> &needle, const std::string &jobdesc = "needle") const
		{
			printf("Searching for %s: %s\n", jobdesc.c_str(), needle.str().c_str());
			fflush(stdout);

			long index = find(needle);
			if (index >= 0) {
				printf("Direct match for %s with length %d:\n", jobdesc.c_str(), info(index).depth);
				printf("%s", prog(index, true).c_str());
				fflush(stdout);
				return std::pair<int,std::string>(info(index).depth, prog(index));
			}

			printf("No direct match for %s found.\n", jobdesc.c_str());
			fflush(stdout);
			return std::pair<int,std::string>(-1, "");
		}

		std::pair<int,std::string> find2(const perm_t<N,SZ> &needle, const std::string &jobdesc = "needle") const
		{
			printf("Searching for %s: %s\n", jobdesc.c_str(), needle.str().c_str());
			fflush(stdout);

			long index = find(needle);
			if (index >= 0) {
				printf("Direct match for %s with length %d:\n", jobdesc.c_str(), info(index).depth);
				printf("%s", prog(index, true).c_str());
				fflush(stdout);
				return std::pair<int,std::string>(info(index).depth, prog(index));
			}

			int best_count = 0;
			int best_depth = -1;
			long best_first = -1;
			long best_second = -1;

			std::string scandesc = "scanning for " + jobdesc;

			for (index = 0; index < headptr; index++)
			{
				status(scandesc, index, headptr, 10000000);

				auto new_needle = needle.apply(perm(index).invert());
				long new_index = find(new_needle);

				if (new_index < 0)
					continue;

				const auto &first = entry(new_index);
				const auto &second = entry(index);
				int depth = first.info.depth + second.info.depth;

				if (depth == best_depth)
					best_count++;

				if (best_depth >= 0 && depth >= best_depth)
					continue;

				printf("Indirect match with length %d + %d = %d: %s;%s\n",
						first.info.depth, second.info.depth, depth,
						prog(new_index).c_str(), prog(index).c_str());
				fflush(stdout);

				best_count = 1;
				best_depth = depth;
				best_first = new_index;
				best_second = index;
			}

			status(scandesc);

			if (best_depth >= 0) 
			{
				const auto &first = entry(best_first);
				const auto &second = entry(best_second);
				int depth = first.info.depth + second.info.depth;

				printf("Indirect match for %s with length %d + %d = %d:\n",
						jobdesc.c_str(), first.info.depth, second.info.depth, depth);
				printf("%s", prog(best_first, true).c_str());
				auto p = first.perm;
				printf("%s", prog(best_second, true, &p).c_str());
				assert(p == needle);
				printf("Found a total of %d matches with that length.\n", best_count);
				fflush(stdout);
				return std::pair<int,std::string>(depth, prog(best_first) + ";" + prog(best_second));
			}

			printf("No direct or indirect match for %s found.\n", jobdesc.c_str());
			fflush(stdout);
			return std::pair<int,std::string>(-1, "");
		}
	};

	template<int N, int SZ>
	struct GenericMiner
	{
		std::vector<std::tuple<perm_t<N,SZ>, std::string, int, std::string>> needles;

		void add(const perm_t<N,SZ> &needle)
		{
			needles.emplace_back(needle, pdb::stringf("pattern #%d", int(needles.size())), -1, "");
		}

		void add(const perm_t<N,SZ> &needle, const std::string &desc)
		{
			needles.emplace_back(needle, desc, -1, "");
		}

		void wave(const permdb_t<N,SZ> &database)
		{
			printf("GenericMiner:\n");
			fflush(stdout);
			for (auto &entry : needles)
			{
				const auto &needle = std::get<0>(entry);
				const auto &desc = std::get<1>(entry);
				int &depth = std::get<2>(entry);
				auto &prog = std::get<3>(entry);

				if (depth >= 0)
					continue;

				auto ret = database.find2(needle, desc.c_str());
				if (ret.first >= 0) {
					depth = ret.first;
					prog = ret.second;
				}
			}
			printf("--------\n");
			fflush(stdout);
		}

		void summary()
		{
			printf("GenericMiner:\n");
			fflush(stdout);
			for (auto &entry : needles)
			{
				auto &desc = std::get<1>(entry);
				int depth = std::get<2>(entry);
				auto &prog = std::get<3>(entry);

				if (depth == 0)
					printf("%s: identity\n", desc.c_str());
				else if (depth > 0)
					printf("%s: %s\n", desc.c_str(), prog.c_str());
				else
					printf("%s: ** not found **\n", desc.c_str());
				fflush(stdout);
			}
			printf("--------\n");
			fflush(stdout);
		}
	};

	template<int N, int SZ>
	bool FwdBwdScan(const permdb_t<N,SZ> &fwd, const permdb_t<N,SZ> &bwd)
	{
		printf("FwdBwdScan:\n");

		int best_count = 0;
		int best_depth = -1;
		long best_fwd = -1;
		long best_bwd = -1;

		for (long fwd_index = 0; fwd_index < fwd.headptr; fwd_index++)
		{
			fwd.status("FwdBwd", fwd_index, fwd.headptr, 10000000);

			const auto &perm = fwd.perm(fwd_index);
			long bwd_index = bwd.find(perm);

			if (bwd_index < 0)
				continue;

			int fwd_depth = fwd.info(fwd_index).depth;
			int bwd_depth = bwd.info(bwd_index).depth;
			int depth = fwd_depth + bwd_depth;

			if (depth == best_depth)
				best_count++;

			if (depth >= best_depth && best_depth >= 0)
				continue;

			printf("Match with length %d + %d = %d: %s;%s\n",
					fwd_depth, bwd_depth, depth,
					fwd.prog(fwd_index).c_str(), bwd.rprog(bwd_index).c_str());
			fflush(stdout);

			best_count = 1;
			best_depth = depth;
			best_fwd = fwd_index;
			best_bwd = bwd_index;
		}

		fwd.status("FwdBwd");

		if (best_count)
		{
			printf("Match with length %d + %d = %d:\n", fwd.info(best_fwd).depth, bwd.info(best_bwd).depth, best_depth);
			printf("%s", fwd.prog(best_fwd, true).c_str());
			printf("%s", bwd.rprog(best_bwd, true).c_str());
			printf("Found a total of %d matches with that length.\n", best_count);
			printf("--------\n");
			fflush(stdout);
			return true;
		}

		printf("Found no match.\n");
		printf("--------\n");
		fflush(stdout);
		return false;
	}
}

#endif

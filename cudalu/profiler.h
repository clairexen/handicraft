#ifndef PROFILER_H
#define PROFILER_H

#include <map>
#include <vector>
#include <string>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <pthread.h>
#include <cuda_runtime.h>

// #define PROFILER_VERBOSE

class Profiler
{
private:
	struct pdata_t {
		int onoff;
		double seconds;
		struct timeval tv;
		pdata_t() : onoff(0), seconds(0) { }
	};

	typedef std::map<std::pair<pthread_t, std::string>, pdata_t> profiles_t;
	static profiles_t profiles;

	static pthread_mutex_t mutex;
	static int profiler_depth;

	// this class can not be instanciated (ctor is private and not implemented)
	// it is just a container with public static methods and private static data
	Profiler();

public:
	static bool begin(std::string id) {
		pthread_mutex_lock(&mutex);
		pdata_t *pd = &profiles[std::pair<pthread_t, std::string>(pthread_self(), id)];
		pthread_mutex_unlock(&mutex);
		if (pd->onoff == 0) {
			cudaDeviceSynchronize();
			gettimeofday(&pd->tv, NULL);
		}
		pd->onoff++;
#ifdef PROFILER_VERBOSE
		fprintf(stderr, "%*s`- %s\n", 4*profiler_depth, "", id.c_str());
#endif
		profiler_depth++;
		return true;
	}

	static bool end(std::string id) {
		pthread_mutex_lock(&mutex);
		pdata_t *pd = &profiles[std::pair<pthread_t, std::string>(pthread_self(), id)];
		pthread_mutex_unlock(&mutex);
		if (pd->onoff == 1) {
			struct timeval tv;
			cudaDeviceSynchronize();
			gettimeofday(&tv, NULL);
			pd->seconds += (tv.tv_sec - pd->tv.tv_sec) + 1e-6*(tv.tv_usec - pd->tv.tv_usec);
#ifdef PROFILER_VERBOSE
			fprintf(stderr, "%*s<-- %.4f\n", 4*profiler_depth, "",
					(tv.tv_sec - pd->tv.tv_sec) + 1e-6*(tv.tv_usec - pd->tv.tv_usec));
#endif
		}
		pd->onoff--;
		profiler_depth--;
		return false;
	}

	 static void show(FILE *f = stdout) {
		std::map<std::string, double> data;
		pthread_mutex_lock(&mutex);
		for (profiles_t::iterator it = profiles.begin(); it != profiles.end(); it++) {
			data[it->first.second] += it->second.seconds;
			assert(it->second.onoff == 0);
		}
		pthread_mutex_unlock(&mutex);
		for (std::map<std::string, double>::iterator it = data.begin(); it != data.end(); it++) {
			fprintf(f, "%-20s %10.4f\n", it->first.c_str(), it->second);
		}
	}

	static double get(std::string id) {
		double data = 0;
		pthread_mutex_lock(&mutex);
		for (profiles_t::iterator it = profiles.begin(); it != profiles.end(); it++) {
			if (it->first.second == id)
				data += it->second.seconds;
		}
		pthread_mutex_unlock(&mutex);
		return data;
	}
};

#define Profiler(_id) \
	for (bool _internal_profiler_flag = Profiler::begin(_id); _internal_profiler_flag; _internal_profiler_flag = Profiler::end(_id))

#ifdef PROFILER_SINGULAR_DEFS
Profiler::profiles_t Profiler::profiles;
pthread_mutex_t Profiler::mutex = PTHREAD_MUTEX_INITIALIZER;
int Profiler::profiler_depth = 0;
#endif

#endif /* PROFILER_H */

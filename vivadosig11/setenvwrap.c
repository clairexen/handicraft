// gcc -shared -fPIC setenvwrap.c -o setenvwrap.so -ldl
// LD_PRELOAD=$PWD/setenvwrap.so vivado -mode batch -source synth.tcl

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#undef DEBUG_ENV

int(*orig_setenv)(const char*, const char*, int) = NULL;
char *(*orig_getenv)(const char*) = NULL;

int setenv(const char *name, const char *value, int overwrite)
{
	if (orig_setenv == NULL)
		orig_setenv = dlsym(RTLD_NEXT, "setenv");

	if (value == NULL) {
		fprintf(stderr, "WARNING: Calling setenv() with a NULL value has undefined "
				"behavior! (name=%s, overwrite=%d)\n", name, overwrite);
		return 0;
	}

	int ret = orig_setenv(name, value, overwrite);

#ifdef DEBUG_ENV
	fprintf(stderr, "setenv(\"%s\", \"%s\", %d) = %d\n", name, value, overwrite, ret);
#endif

	return ret;
}

#ifdef DEBUG_ENV
char *getenv(const char *name)
{
	// This variable is read while the dlsym spin lock is locked
	// just return NULL to avoid deadlock in dlsym() below
	if (!strcmp("TCMALLOC_TRANSFER_NUM_OBJ", name))
		return NULL;

	if (orig_getenv == NULL)
		orig_getenv = dlsym(RTLD_NEXT, "getenv");

	char *ret = orig_getenv(name);

	if (ret)
		fprintf(stderr, "getenv(\"%s\") = \"%s\"\n", name, ret);
	else
		fprintf(stderr, "getenv(\"%s\") = NULL\n", name);

	return ret;
}
#endif

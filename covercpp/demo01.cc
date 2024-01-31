#include <stdio.h>
#include <string.h>

struct CoverData
{
	const char *file;
	int line, counter;
}  __attribute__ ((packed));

extern "C" CoverData __start_cover_list[];
extern "C" CoverData __stop_cover_list[];

#define COVER() do { \
  static CoverData __d __attribute__((section("cover_list"), aligned(1))) = { __FILE__, __LINE__, 0 }; \
  __d.counter++; \
} while (0)

int main(int argc, char**)
{
	if (argc <= 1)
		COVER();
	else
		COVER();

	for (CoverData *p = __start_cover_list; p != __stop_cover_list; p++)
		printf("%s:%d %d\n", p->file, p->line, p->counter);

	return 0;
}


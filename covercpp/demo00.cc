#include <stdio.h>

struct Cover
{
	static struct Cover *first_cover;
	struct Cover *next_cover;
	const char *file;
	int line, counter;

	Cover(const char *file_, int line_) {
		next_cover = first_cover;
		first_cover = this;
		file = file_;
		line = line_;
		counter = 0;
	}
};

struct Cover *Cover::first_cover = NULL;

#define COVER() do { static Cover __c(__FILE__, __LINE__); __c.counter++; } while (0)

int main(int argc, char**)
{
	// The code that is not executed will not show up in the stats.
	// See http://stackoverflow.com/questions/55510/

	if (argc <= 1)
		COVER();
	else
		COVER();

	for (auto *p = Cover::first_cover; p; p = p->next_cover)
		printf("%s:%d %d\n", p->file, p->line, p->counter);

	return 0;
}


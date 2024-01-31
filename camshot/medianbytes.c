
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>

unsigned char *data = NULL;
size_t data_len = 0, data_reserved = 0;

static size_t read_file(const char *filename)
{
	fprintf(stderr, "Reading %s.\n", filename);

	if (data_reserved == 0) {
		data = malloc(1024*1024);
		data_reserved = 1024*1024;
	}

	int f = open(filename, O_RDONLY);
	if (f < 0) {
		fprintf(stderr, "Can't open '%s': %s\n", filename, strerror(errno));
		exit(1);
	}

	size_t total = 0;
	while (1) {
		if (data_len + 4096 > data_reserved) {
			data_reserved *= 2;
			data = realloc(data, data_reserved);
		}
		int rc = read(f, data+data_len, 4096);
		if (rc <= 0) break;
		total += rc;
		data_len += rc;
	}

	close(f);
	return total;
}

static int cmpbytes(const void *p1, const void *p2)
{
	return *(unsigned char*)p1 < *(unsigned char*)p2;
}


int main(int argc, char **argv)
{
	size_t record_len = 0;

	for (int i = 1; i < argc; i++) {
		size_t len = read_file(argv[i]);
		if (record_len != 0 && record_len != len) {
			fprintf(stderr, "Length mismatch in '%s'.\n", argv[i]);
			exit(1);
		}
		record_len = len;
	}

	fprintf(stderr, "Sorting..");

	unsigned char p[argc-1];
	for (size_t j = 0; j < record_len; j++)
	{
		if (j % 1024 == 0)
			fprintf(stderr, ".");
		for (int i = 1; i < argc; i++)
			p[i] = data[i*record_len + j];
		qsort(p, argc-1, 1, cmpbytes);
		data[j] = p[argc/2];
	}
	fprintf(stderr, "\n");

	size_t pos = 0;
	while (pos < record_len) {
		int rc = write(1, data+pos, record_len-pos);
		if (rc <= 0) break;
		pos += rc;
	}

	return 0;
}


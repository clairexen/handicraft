#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BINSIZE 32300
#define CHECKSUM_POS 258360

unsigned char *bindata[32];
int bindata_n = 0;

int getbit(int n, int k)
{
	return (bindata[n][k/8] & (1 << (k%8))) != 0;
}

#define check(_cond) do { if (_cond) break; printf("Check `%s' at %s:%d failed.\n", #_cond, __FILE__, __LINE__); exit(1); } while (0)

int main(int argc, char **argv)
{
	int counter = 0;
	int i, j;

	check(argc > 1);

	for (i = 1; i < argc; i++)
	{
		printf("%c: %s\n", 'A'+i-1, argv[i]);
		FILE *f = fopen(argv[i], "r");
		check(f != NULL);
		char *p = argv[i] + strlen(argv[i]);
		bindata[bindata_n] = malloc(BINSIZE);
		fread(bindata[bindata_n++], BINSIZE, 1, f);
		fclose(f);
	}

	check(bindata_n > 0);
	for (i = 0; i < BINSIZE*8; i++)
	{
		if (i == CHECKSUM_POS)
			break;

		int first_v = getbit(0, i);
		int found_diff = 0;

		for (j = 1; j < bindata_n; j++)
			if (first_v != getbit(j, i))
				found_diff = 1;

		if (found_diff)
		{
			if (counter == 0) {
				printf("\n");
				printf("%3s      %6s ", "", "");
				for (j = 0; j < bindata_n; j++)
					printf(" %c", 'A'+j);
				printf("\n");
			}

			printf("%3d: bit %6d:", counter++, i);
			for (j = 0; j < bindata_n; j++)
				printf(" %d", getbit(j, i));
			printf("\n");
		}
	}

	if (counter == 0)
		printf("No differences found.\n");

	return 0;
}


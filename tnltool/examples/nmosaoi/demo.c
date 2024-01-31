#include "nmosaoi.tab.h"
#include <stdio.h>
#include <sys/time.h>
#include <sys/resource.h>

int main()
{
	int i;
	FILE *f = fopen("nmosaoi.vcd", "w");

	nmosaoi_vcd(f);
	nmosaoi_init();
	nmosaoi_eval();

	printf("\n");

	printf(" I1 I2 I3 I4 |OUT\n");
	printf("-------------+---\n");
	for (i = 0; i < 16; i++)
	{
		int in1 = (i&8) != 0;
		int in2 = (i&4) != 0;
		int in3 = (i&2) != 0;
		int in4 = (i&1) != 0;

		nmosaoi_transitionCounter = 10*(nmosaoi_transitionCounter/10) + 9;

		nmosaoi_set(NMOSAOI_PIN_in1, in1 ? NMOSAOI_STATE_pullup : NMOSAOI_STATE_pulldown);
		nmosaoi_set(NMOSAOI_PIN_in2, in2 ? NMOSAOI_STATE_pullup : NMOSAOI_STATE_pulldown);
		nmosaoi_set(NMOSAOI_PIN_in3, in3 ? NMOSAOI_STATE_pullup : NMOSAOI_STATE_pulldown);
		nmosaoi_set(NMOSAOI_PIN_in4, in4 ? NMOSAOI_STATE_pullup : NMOSAOI_STATE_pulldown);
		nmosaoi_eval();

		int out = nmosaoi_get(NMOSAOI_PIN_out) >= NMOSAOI_STATE_charged;
		printf("  %d  %d  %d  %d |  %d\n", in1, in2, in3, in4, out);
	}

	printf("\n");

	nmosaoi_vcd(NULL);
	fclose(f);

	return 0;
}


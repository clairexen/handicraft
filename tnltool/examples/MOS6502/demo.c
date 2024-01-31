#include "model6502.tab.h"
#include <stdio.h>
#include <sys/time.h>
#include <sys/resource.h>

#define PRINTMEMRW 0
#define PRINTCONSOLE 1
#define TOTALCYCLES 1850

unsigned char memory[0x10000] = {
	// xa testprogram.a65 -o - | od -t x1 -An | sed -r 's/^ //; s/\S\S/0x&/g; s/ /, /g; s/$/,/;' | fmt
	0x20, 0x51, 0x00, 0x20, 0x4a, 0x00, 0x4c, 0x03, 0x00, 0xa2, 0x00, 0xbd,
	0x17, 0x00, 0xf0, 0x06, 0x85, 0xff, 0xe8, 0x4c, 0x0b, 0x00, 0x60, 0x48,
	0x65, 0x6c, 0x6c, 0x6f, 0x20, 0x57, 0x6f, 0x72, 0x6c, 0x64, 0x21, 0x20,
	0x2d, 0x2d, 0x20, 0x00, 0xa2, 0x00, 0xbd, 0x36, 0x00, 0xf0, 0x06, 0x85,
	0xff, 0xe8, 0x4c, 0x2a, 0x00, 0x60, 0x54, 0x68, 0x69, 0x73, 0x20, 0x69,
	0x73, 0x20, 0x61, 0x20, 0x74, 0x65, 0x73, 0x74, 0x21, 0x20, 0x2d, 0x2d,
	0x20, 0x00, 0x20, 0x09, 0x00, 0x20, 0x28, 0x00, 0x60, 0xa9, 0x3e, 0x85,
	0xff, 0xa9, 0x20, 0x85, 0xff, 0x60,
};

#if PRINTCONSOLE
int consoleidx = 0;
char consolebuf[1024] = "";
#endif

int AB[] = {
	MODEL6502_PIN_ab0,
	MODEL6502_PIN_ab1,
	MODEL6502_PIN_ab2,
	MODEL6502_PIN_ab3,
	MODEL6502_PIN_ab4,
	MODEL6502_PIN_ab5,
	MODEL6502_PIN_ab6,
	MODEL6502_PIN_ab7,
	MODEL6502_PIN_ab8,
	MODEL6502_PIN_ab9,
	MODEL6502_PIN_ab10,
	MODEL6502_PIN_ab11,
	MODEL6502_PIN_ab12,
	MODEL6502_PIN_ab13,
	MODEL6502_PIN_ab14,
	MODEL6502_PIN_ab15
};
int DB[] = {
	MODEL6502_PIN_db0,
	MODEL6502_PIN_db1,
	MODEL6502_PIN_db2,
	MODEL6502_PIN_db3,
	MODEL6502_PIN_db4,
	MODEL6502_PIN_db5,
	MODEL6502_PIN_db6,
	MODEL6502_PIN_db7
};

void doHalfClock()
{
	int i;
#if defined MODEL6502_VCD || defined MODEL6502_DOT
	model6502_transitionCounter = 100*(model6502_transitionCounter/100) + 99;
#endif
	if (model6502_get(MODEL6502_PIN_clk0) >= MODEL6502_STATE_charged)
	{
		model6502_set(MODEL6502_PIN_clk0, MODEL6502_STATE_pulldown);
		model6502_eval();

		// read from memory
		if (model6502_get(MODEL6502_PIN_rw) >= MODEL6502_STATE_charged)
		{
			int dbData = 0, abData = 0;
			for (i = 0; i < 16; i++)
				if (model6502_get(AB[i]) >= MODEL6502_STATE_charged)
					abData |= 1 << i;
			dbData = memory[abData];
			for (i = 0; i < 8; i++)
				model6502_set(DB[i], (dbData & (1 << i)) != 0 ?
						MODEL6502_STATE_pullup : MODEL6502_STATE_pulldown);
#if PRINTMEMRW
			printf("MEM RD: $%04x -> #$%02x   @%d\n", abData, dbData, model6502_transitionCounter);
#endif
		} else {
			for (i = 0; i < 8; i++)
				model6502_set(DB[i], MODEL6502_STATE_float);
		}
	}
	else
	{
		model6502_set(MODEL6502_PIN_clk0, MODEL6502_STATE_pullup);
		model6502_eval();
		
		// write to memory
		if (model6502_get(MODEL6502_PIN_rw) < MODEL6502_STATE_charged &&
				model6502_get(MODEL6502_PIN_res) >= MODEL6502_STATE_charged) {
			int dbData = 0, abData = 0;
			for (i = 0; i < 16; i++)
				if (model6502_get(AB[i]) >= MODEL6502_STATE_charged)
					abData |= 1 << i;
			for (i = 0; i < 8; i++)
				if (model6502_get(DB[i]) >= MODEL6502_STATE_charged)
					dbData |= 1 << i;
			memory[abData] = dbData;
#if PRINTMEMRW
			printf("MEM WR: $%04x <- #$%02x   @%d\n", abData, dbData, model6502_transitionCounter);
#endif
#if PRINTCONSOLE
			if (abData == 0xff) {
				if (consoleidx == 1023) {
					printf("Resetting console output buffer!\n");
					consoleidx = 0;
				}
				consolebuf[consoleidx] = dbData ?: '.';
				consolebuf[++consoleidx] = 0;
				printf("CONSOLE: %s\n", consolebuf);
			}
#endif
		}
	}
}

int main()
{
	int i;
#ifdef MODEL6502_VCD
	FILE *vcdfile = fopen("model6502.vcd", "w");
#endif
#ifdef MODEL6502_DOT
	FILE *dotfile = fopen("model6502.dot", "w");
#endif

	// initialize
	model6502_init();
#ifdef MODEL6502_VCD
	model6502_vcd(vcdfile);
#endif
#ifdef MODEL6502_DOT
	model6502_dot(dotfile);
#endif
	model6502_set(MODEL6502_PIN_res, MODEL6502_STATE_pulldown);
	model6502_set(MODEL6502_PIN_rw, MODEL6502_STATE_float);
	for (i = 0; i < 8; i++)
		model6502_set(DB[i], MODEL6502_STATE_float);
	for (i = 0; i < 16; i++)
		model6502_set(AB[i], MODEL6502_STATE_float);
	model6502_set(MODEL6502_PIN_sync, MODEL6502_STATE_float);
	model6502_set(MODEL6502_PIN_so, MODEL6502_STATE_pulldown);
	model6502_set(MODEL6502_PIN_clk0, MODEL6502_STATE_pulldown);
	model6502_set(MODEL6502_PIN_clk1out, MODEL6502_STATE_float);
	model6502_set(MODEL6502_PIN_clk2out, MODEL6502_STATE_float);
	model6502_set(MODEL6502_PIN_rdy, MODEL6502_STATE_pullup);
	model6502_set(MODEL6502_PIN_nmi, MODEL6502_STATE_pullup);
	model6502_set(MODEL6502_PIN_irq, MODEL6502_STATE_pullup);
	model6502_eval();

	// run CPU for 8 cycles with RST active
	printf("-- rst active--\n");
	for (i = 0; i < 8; i++)
	{
		doHalfClock();
		doHalfClock();
	}

	// deassert RST line
	printf("-- rst inactive --\n");
	model6502_set(MODEL6502_PIN_res, MODEL6502_STATE_pullup);
	model6502_eval();

	// run CPU for some cycles with RST inactive
	struct rusage ru1, ru2;
	getrusage(RUSAGE_SELF, &ru1);
	if (!model6502_stopped) {
		model6502_transitionCounter = 0;
		model6502_netevalCounter = 0;
		for (i = 0; i < TOTALCYCLES; i++) {
			doHalfClock();
			doHalfClock();
		}
	}
	getrusage(RUSAGE_SELF, &ru2);

#ifdef MODEL6502_VCD
	model6502_vcd(NULL);
	fclose(vcdfile);
#endif
#ifdef MODEL6502_DOT
	model6502_dot(NULL);
	fclose(dotfile);
#endif

	// some timing stats
	double secs = (ru2.ru_utime.tv_sec - ru1.ru_utime.tv_sec) +
			1e-6*(ru2.ru_utime.tv_usec - ru1.ru_utime.tv_usec);
	printf("Executed %d clock cycles (%d half cycles) in %f seconds -> %f kHz.\n",
			TOTALCYCLES, 2*TOTALCYCLES, secs, TOTALCYCLES / secs * 1e-3);

	printf("Avg. #transitions and #netevals per cycle: %f, %f\n",
			model6502_transitionCounter / (double)TOTALCYCLES,
			model6502_netevalCounter / (double)TOTALCYCLES);

	printf("READY.\n");
	return 0;
}


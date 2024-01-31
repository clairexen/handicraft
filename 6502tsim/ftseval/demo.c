/*
 *  ftseval - Fast Transistor Switches Evaluator  (a proof-of-concept)
 *
 *  Copyright (C) 2011  Clifford Wolf <clifford@clifford.at>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

#include "ftseval.h"
#include "model6502.h"
#include <stdio.h>
#include <sys/time.h>
#include <sys/resource.h>

#define PRINTMEMRW 0
#define PRINTCONSOLE 1
#define TOTALCYCLES 1850

unsigned char memory[0x10000] = {
	// see ../visual6502/testprogram.js and ../visual6502/testprogram.a65
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

ftseval_netid_t AB[] = { PIN_ab0, PIN_ab1, PIN_ab2, PIN_ab3, PIN_ab4, PIN_ab5, PIN_ab6, PIN_ab7,
		PIN_ab8, PIN_ab9, PIN_ab10, PIN_ab11, PIN_ab12, PIN_ab13, PIN_ab14, PIN_ab15 };
ftseval_netid_t DB[] = { PIN_db0, PIN_db1, PIN_db2, PIN_db3, PIN_db4, PIN_db5, PIN_db6, PIN_db7 };

void doHalfClock()
{
	int i;
	if (ftseval_get(PIN_clk0) >= FTSEVAL_STATE_CHARGE)
	{
		ftseval_set(PIN_clk0, FTSEVAL_STATE_PULLDOWN);
		ftseval_run();

		// read from memory
		if (ftseval_get(PIN_rw) >= FTSEVAL_STATE_CHARGE)
		{
			int dbData = 0, abData = 0;
			for (i = 0; i < 16; i++)
				if (ftseval_get(AB[i]) >= FTSEVAL_STATE_CHARGE)
					abData |= 1 << i;
			dbData = memory[abData];
			for (i = 0; i < 8; i++)
				ftseval_set(DB[i], (dbData & (1 << i)) != 0 ? FTSEVAL_STATE_PULLUP : FTSEVAL_STATE_PULLDOWN);
#if PRINTMEMRW
			printf("MEM RD: $%04x -> #$%02x   @%d\n", abData, dbData, ftseval_transitionCounter);
#endif
		} else {
			for (i = 0; i < 8; i++)
				ftseval_set(DB[i], FTSEVAL_STATE_FLOAT);
		}
	}
	else
	{
		ftseval_set(PIN_clk0, FTSEVAL_STATE_PULLUP);
		ftseval_run();
		
		// write to memory
		if (ftseval_get(PIN_rw) < FTSEVAL_STATE_CHARGE && ftseval_get(PIN_res) >= FTSEVAL_STATE_CHARGE) {
			int dbData = 0, abData = 0;
			for (i = 0; i < 16; i++)
				if (ftseval_get(AB[i]) >= FTSEVAL_STATE_CHARGE)
					abData |= 1 << i;
			for (i = 0; i < 8; i++)
				if (ftseval_get(DB[i]) >= FTSEVAL_STATE_CHARGE)
					dbData |= 1 << i;
			memory[abData] = dbData;
#if PRINTMEMRW
			printf("MEM WR: $%04x <- #$%02x   @%d\n", abData, dbData, ftseval_transitionCounter);
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

	// initialize simulator
	ftseval_init();
	ftseval_run();

	// initialize all pins
	ftseval_set(PIN_res, FTSEVAL_STATE_PULLDOWN);
	ftseval_set(PIN_rw, FTSEVAL_STATE_FLOAT);
	for (i = 0; i < 8; i++)
		ftseval_set(DB[i], FTSEVAL_STATE_FLOAT);
	for (i = 0; i < 16; i++)
		ftseval_set(AB[i], FTSEVAL_STATE_FLOAT);
	ftseval_set(PIN_sync, FTSEVAL_STATE_FLOAT);
	ftseval_set(PIN_so, FTSEVAL_STATE_PULLDOWN);
	ftseval_set(PIN_clk0, FTSEVAL_STATE_PULLDOWN);
	ftseval_set(PIN_clk1out, FTSEVAL_STATE_FLOAT);
	ftseval_set(PIN_clk2out, FTSEVAL_STATE_FLOAT);
	ftseval_set(PIN_rdy, FTSEVAL_STATE_PULLUP);
	ftseval_set(PIN_nmi, FTSEVAL_STATE_PULLUP);
	ftseval_set(PIN_irq, FTSEVAL_STATE_PULLUP);
	ftseval_run();

	// run CPU for 8 cycles with RST active
	printf("-- rst active--\n");
	for (i = 0; i < 8; i++)
	{
		doHalfClock();
		doHalfClock();
	}

	// deassert RST line
	printf("-- rst inactive --\n");
	ftseval_set(PIN_res, FTSEVAL_STATE_PULLUP);
	ftseval_run();

	// run CPU for some cycles with RST inactive
	struct rusage ru1, ru2;
	getrusage(RUSAGE_SELF, &ru1);
	ftseval_transitionCounter = 0;
	ftseval_netevalCounter = 0;
	for (i = 0; i < TOTALCYCLES; i++) {
		doHalfClock();
		doHalfClock();
	}
	getrusage(RUSAGE_SELF, &ru2);

	// some timing stats
	double secs = (ru2.ru_utime.tv_sec - ru1.ru_utime.tv_sec) + 1e-6*(ru2.ru_utime.tv_usec - ru1.ru_utime.tv_usec);
	printf("Executed %d clock cycles (%d half cycles) in %f seconds -> %f kHz.\n",
			TOTALCYCLES, 2*TOTALCYCLES, secs, TOTALCYCLES / secs * 1e-3);

	printf("Avg. #transitions and #netevals per cycle: %f, %f\n",
			ftseval_transitionCounter / (double)TOTALCYCLES,
			ftseval_netevalCounter / (double)TOTALCYCLES);

	printf("READY.\n");
	return 0;
}


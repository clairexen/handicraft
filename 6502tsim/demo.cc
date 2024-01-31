/*
 *  Simple NMOS Simulator (for playing with the 6502 nmos netlist)
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

#include "nmossim.h"
#include <stdio.h>
#include <sys/time.h>
#include <sys/resource.h>

#define GENCLKSYNGRAPH 1
#define PRINTMEMRW 0
#define TOTALCYCLES 1850

extern void model6502(NmosSim &sim);

NmosSim sim;

NmosSim::Net *pinRES, *pinRW, *pinSYNC, *pinSO, *pinCLK, *pinCLK1, *pinCLK2, *pinRDY, *pinNMI, *pinIRQ;
NmosSim::Net *pinDB0, *pinDB1, *pinDB2, *pinDB3, *pinDB4, *pinDB5, *pinDB6, *pinDB7;
NmosSim::Net *pinAB0, *pinAB1, *pinAB2, *pinAB3, *pinAB4, *pinAB5, *pinAB6, *pinAB7,
		*pinAB8, *pinAB9, *pinAB10, *pinAB11, *pinAB12, *pinAB13, *pinAB14, *pinAB15;
NmosSim::Net *pinVCC, *pinVSS;
NmosSim::Net *DB[8], *AB[16];

unsigned char memory[0x10000] = {
	// see visual6502/testprogram.js and visual6502/testprogram.a65
	0x20, 0x51, 0x00, 0x20, 0x4a, 0x00, 0x4c, 0x03, 0x00, 0xa2, 0x00, 0xbd,
	0x17, 0x00, 0xf0, 0x06, 0x85, 0xff, 0xe8, 0x4c, 0x0b, 0x00, 0x60, 0x48,
	0x65, 0x6c, 0x6c, 0x6f, 0x20, 0x57, 0x6f, 0x72, 0x6c, 0x64, 0x21, 0x20,
	0x2d, 0x2d, 0x20, 0x00, 0xa2, 0x00, 0xbd, 0x36, 0x00, 0xf0, 0x06, 0x85,
	0xff, 0xe8, 0x4c, 0x2a, 0x00, 0x60, 0x54, 0x68, 0x69, 0x73, 0x20, 0x69,
	0x73, 0x20, 0x61, 0x20, 0x74, 0x65, 0x73, 0x74, 0x21, 0x20, 0x2d, 0x2d,
	0x20, 0x00, 0x20, 0x09, 0x00, 0x20, 0x28, 0x00, 0x60, 0xa9, 0x3e, 0x85,
	0xff, 0xa9, 0x20, 0x85, 0xff, 0x60,
};

int consoleidx = 0;
char consolebuf[1024] = "";

void displayStatus()
{
	int dbData = 0, abData = 0;
	for (int i = 0; i < 8; i++)
		if (DB[i]->state)
			dbData |= 1 << i;
	for (int i = 0; i < 16; i++)
		if (AB[i]->state)
			abData |= 1 << i;
	printf("CLK=%d(%d%d) DB=0x%02x AB=0x%04x RW=%d\n", pinCLK->state ? 1 : 0,
			pinCLK1->state ? 1 : 0, pinCLK2->state ? 1 : 0,
			dbData, abData, pinRW->state ? 1 : 0);
}

void doHalfClock()
{
	bool clk = pinCLK->state;

	if (clk == true)
	{
		sim.updateNet(pinCLK, false);
		sim.eval();
		
		// read from memory
		if (pinRW->state == true) {
			int dbData = 0, abData = 0;
			for (int i = 0; i < 16; i++)
				if (AB[i]->state)
					abData |= 1 << i;
			dbData = memory[abData];
			for (int i = 0; i < 8; i++)
				sim.updateNet(DB[i], (dbData & (1 << i)) != 0);
#if PRINTMEMRW
			printf("MEM RD: $%04x -> #$%02x   @%d\n", abData, dbData, sim.transitionCounter);
#endif
		} else {
			for (int i = 0; i < 8; i++)
				sim.releaseNet(DB[i]);
		}
	}
	else
	{
		sim.updateNet(pinCLK, true);
		sim.eval();
		
		// write to memory
		if (pinRW->state == false && pinRES->state == true) {
			int dbData = 0, abData = 0;
			for (int i = 0; i < 16; i++)
				if (AB[i]->state)
					abData |= 1 << i;
			for (int i = 0; i < 8; i++)
				if (DB[i]->state)
					dbData |= 1 << i;
			memory[abData] = dbData;
#if PRINTMEMRW
			printf("MEM WR: $%04x <- #$%02x   @%d\n", abData, dbData, sim.transitionCounter);
#endif
			if (abData == 0xff) {
				if (consoleidx == 1023) {
					printf("Resetting console output buffer!\n");
					consoleidx = 0;
				}
				consolebuf[consoleidx] = dbData ?: '.';
				consolebuf[++consoleidx] = 0;
				printf("CONSOLE: %s\n", consolebuf);
			}
		}
	}
}

int main()
{
	model6502(sim);
	sim.deleteDupTransistors();

	pinRES  = sim.findNet("res");
	pinRW   = sim.findNet("rw");
	pinSYNC = sim.findNet("sync");
	pinSO   = sim.findNet("so");
	pinCLK  = sim.findNet("clk0");
	pinCLK1 = sim.findNet("clk1out");
	pinCLK2 = sim.findNet("clk2out");
	pinRDY  = sim.findNet("rdy");
	pinNMI  = sim.findNet("nmi");
	pinIRQ  = sim.findNet("irq");

	DB[0] = pinDB0  = sim.findNet("db0");
	DB[1] = pinDB1  = sim.findNet("db1");
	DB[2] = pinDB2  = sim.findNet("db2");
	DB[3] = pinDB3  = sim.findNet("db3");
	DB[4] = pinDB4  = sim.findNet("db4");
	DB[5] = pinDB5  = sim.findNet("db5");
	DB[6] = pinDB6  = sim.findNet("db6");
	DB[7] = pinDB7  = sim.findNet("db7");

	AB[ 0] = pinAB0  = sim.findNet("ab0");
	AB[ 1] = pinAB1  = sim.findNet("ab1");
	AB[ 2] = pinAB2  = sim.findNet("ab2");
	AB[ 3] = pinAB3  = sim.findNet("ab3");
	AB[ 4] = pinAB4  = sim.findNet("ab4");
	AB[ 5] = pinAB5  = sim.findNet("ab5");
	AB[ 6] = pinAB6  = sim.findNet("ab6");
	AB[ 7] = pinAB7  = sim.findNet("ab7");
	AB[ 8] = pinAB8  = sim.findNet("ab8");
	AB[ 9] = pinAB9  = sim.findNet("ab9");
	AB[10] = pinAB10 = sim.findNet("ab10");
	AB[11] = pinAB11 = sim.findNet("ab11");
	AB[12] = pinAB12 = sim.findNet("ab12");
	AB[13] = pinAB13 = sim.findNet("ab13");
	AB[14] = pinAB14 = sim.findNet("ab14");
	AB[15] = pinAB15 = sim.findNet("ab15");

	pinVCC  = sim.findNet("vcc");
	pinVSS  = sim.findNet("vss");

	// initialize all pins
	sim.updateNet(pinRES, false);
	sim.releaseNet(pinRW);
	for (int i = 0; i < 8; i++)
		sim.releaseNet(DB[i]);
	for (int i = 0; i < 16; i++)
		sim.releaseNet(AB[i]);
	sim.releaseNet(pinSYNC);
	sim.updateNet(pinSO, false);
	sim.updateNet(pinCLK, false);
	sim.releaseNet(pinCLK1);
	sim.releaseNet(pinCLK2);
	sim.updateNet(pinRDY, true);
	sim.updateNet(pinNMI, true);
	sim.updateNet(pinIRQ, true);
	sim.updateNet(pinVCC, true);
	sim.updateNet(pinVSS, false);
	sim.eval();

#if 0
	// compare our init state with visual6502
	sim.checkStateDump("visual6502/statedumps.txt", "initState");

	// compare state after the first 4 half clocks
	doHalfClock();
	sim.checkStateDump("visual6502/statedumps.txt", "halfClk1");
	doHalfClock();
	sim.checkStateDump("visual6502/statedumps.txt", "halfClk2");
	doHalfClock();
	sim.checkStateDump("visual6502/statedumps.txt", "halfClk3");
	doHalfClock();
	sim.checkStateDump("visual6502/statedumps.txt", "halfClk4");
#endif

#if GENCLKSYNGRAPH
	// create a nice graph of the two-phase clock synth
	sim.clearDisplay();
	sim.setGlobal(pinVCC);
	sim.setGlobal(pinVSS);
	sim.setGlobal(sim.findNet("cp1"));
	sim.setGlobal(sim.findNet("cclk"));
	sim.addDisplay("clk0", 3);
	sim.addDisplay("clk1out", 3);
	sim.addDisplay("clk2out", 3);
	sim.addDisplay("clk0", "clk1out");
	sim.addDisplay("clk0", "clk2out");
	sim.addDisplay("n1105", 3, false);
	sim.addDisplay(NULL, "vcc");
	sim.addDisplay(NULL, "vss");
	sim.displayPage();
#endif

#if 0
	// initialize monitor callbacks
	pinCLK1->onChange = &displayStatus;
	pinCLK2->onChange = &displayStatus;
#endif

	// run CPU for 8 cycles with RST active
	printf("-- rst active--\n");
	for (int i = 0; i < 8; i++)
	{
		doHalfClock();
		doHalfClock();

#if GENCLKSYNGRAPH
		// only generate pics for the first cycle
		if (i == 0)
			sim.displayFolder();
#endif
	}

	// deassert RST line
	printf("-- rst inactive --\n");
	sim.updateNet(pinRES, true);
	sim.eval();

	// run CPU for some cycles with RST inactive
	struct rusage ru1, ru2;
	getrusage(RUSAGE_SELF, &ru1);
	for (int i = 0; i < TOTALCYCLES; i++) {
		doHalfClock();
		doHalfClock();
	}
	getrusage(RUSAGE_SELF, &ru2);

	// some timing stats
	double secs = (ru2.ru_utime.tv_sec - ru1.ru_utime.tv_sec) + 1e-6*(ru2.ru_utime.tv_usec - ru1.ru_utime.tv_usec);
	printf("Executed %d clock cycles (%d half cycles) in %f seconds -> %f kHz.\n",
			TOTALCYCLES, 2*TOTALCYCLES, secs, TOTALCYCLES / secs * 1e-3);

	printf("READY.\n");
	return 0;
}


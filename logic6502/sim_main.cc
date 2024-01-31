#include "sim_MOS6502.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#undef WRITE_VCD
#undef MINIMIZE_RESET_STATE

sim_MOS6502 sim;
int clock_phase;
unsigned char memory[0x10000];
char last_char;

void setDB(uint8_t value)
{
	sim.input_db7 = (value & 0x80) != 0;
	sim.input_db6 = (value & 0x40) != 0;
	sim.input_db5 = (value & 0x20) != 0;
	sim.input_db4 = (value & 0x10) != 0;
	sim.input_db3 = (value & 0x08) != 0;
	sim.input_db2 = (value & 0x04) != 0;
	sim.input_db1 = (value & 0x02) != 0;
	sim.input_db0 = (value & 0x01) != 0;
}

uint8_t getDB()
{
	uint8_t value = 0;
	value |= sim.output_db7 ? 0x80 : 0;
	value |= sim.output_db6 ? 0x40 : 0;
	value |= sim.output_db5 ? 0x20 : 0;
	value |= sim.output_db4 ? 0x10 : 0;
	value |= sim.output_db3 ? 0x08 : 0;
	value |= sim.output_db2 ? 0x04 : 0;
	value |= sim.output_db1 ? 0x02 : 0;
	value |= sim.output_db0 ? 0x01 : 0;
	return value;
}

void releaseDB()
{
	sim.input_db7 = -1;
	sim.input_db6 = -1;
	sim.input_db5 = -1;
	sim.input_db4 = -1;
	sim.input_db3 = -1;
	sim.input_db2 = -1;
	sim.input_db1 = -1;
	sim.input_db0 = -1;
}

uint16_t getAB()
{
	uint16_t addr = 0;
	addr |= sim.output_ab15 ? 0x8000 : 0;
	addr |= sim.output_ab14 ? 0x4000 : 0;
	addr |= sim.output_ab13 ? 0x2000 : 0;
	addr |= sim.output_ab12 ? 0x1000 : 0;
	addr |= sim.output_ab11 ? 0x0800 : 0;
	addr |= sim.output_ab10 ? 0x0400 : 0;
	addr |= sim.output_ab9  ? 0x0200 : 0;
	addr |= sim.output_ab8  ? 0x0100 : 0;
	addr |= sim.output_ab7  ? 0x0080 : 0;
	addr |= sim.output_ab6  ? 0x0040 : 0;
	addr |= sim.output_ab5  ? 0x0020 : 0;
	addr |= sim.output_ab4  ? 0x0010 : 0;
	addr |= sim.output_ab3  ? 0x0008 : 0;
	addr |= sim.output_ab2  ? 0x0004 : 0;
	addr |= sim.output_ab1  ? 0x0002 : 0;
	addr |= sim.output_ab0  ? 0x0001 : 0;
	return addr;
}


void cycle()
{
	sim.vcd_time = ((sim.vcd_time / 100) + 1) * 100;

	sim.input_clk1out = clock_phase == 1;
	sim.input_clk2out = clock_phase == 3;
	clock_phase = (clock_phase + 1) % 4;
	sim.update();

	if (!sim.output_rw)
		releaseDB();

	// printf("clk1=%d, clk2=%d, res=%d, sync=%d, rw=%d, ab=0x%04x, db=0x%02x", sim.input_clk1out, sim.input_clk2out,
	// 		sim.input_res, sim.output_sync, sim.output_rw, getAB(), getDB());

	if (sim.input_clk2out && sim.input_res) {
		if (sim.output_sync && sim.output_rw) {
			// printf("  INSTR: %02X  RD: %04X", memory[getAB()], getAB());
			setDB(memory[getAB()]);
		} else
		if (sim.output_rw) {
			// printf("  DATA:  %02X  RD: %04X", memory[getAB()], getAB());
			setDB(memory[getAB()]);
		} else {
			// printf("  DATA:  %02X  WR: %04X", getDB(), getAB());
			if (getAB() == 0x00f) {
				printf("<%c>", getDB()), fflush(stdout);
				last_char = getDB();
			}
			memory[getAB()] = getDB();
		}
	}

	sim.update();
	// printf("\n");
}

int main()
{
	for (size_t i = 0; i < sizeof(memory); i++)
		memory[i] = 0;

	memory[0x0000] = 0xa9;
	memory[0x0001] = 0x00;
	memory[0x0002] = 0x20;
	memory[0x0003] = 0x10;
	memory[0x0004] = 0x00;
	memory[0x0005] = 0x4c;
	memory[0x0006] = 0x02;
	memory[0x0007] = 0x00;

	memory[0x0008] = 0x00;
	memory[0x0009] = 0x00;
	memory[0x000a] = 0x00;
	memory[0x000b] = 0x00;
	memory[0x000c] = 0x00;
	memory[0x000d] = 0x00;
	memory[0x000e] = 0x00;
	memory[0x000f] = 0x40;

	memory[0x0010] = 0xe8;
	memory[0x0011] = 0x88;
	memory[0x0012] = 0xe6;
	memory[0x0013] = 0x0f;
	memory[0x0014] = 0x38;
	memory[0x0015] = 0x69;
	memory[0x0016] = 0x02;
	memory[0x0017] = 0x60;

	clock_phase = 0;
	releaseDB();

	sim.input_clk1out = false;
	sim.input_clk2out = false;
	sim.input_rdy = true;
	sim.input_irq = true;
	sim.input_nmi = true;
	sim.input_res = false;
	sim.init();

#ifdef WRITE_VCD
	FILE *f = fopen("sim.vcd", "w");
	sim.vcd_init(f);
	sim.vcd_file = f;
#endif

	for (int i = 0; i < 32*4; i++)
		cycle();

	cycle();
	cycle();

#ifdef MINIMIZE_RESET_STATE
	sim_MOS6502 backup_sim = sim;
	int backup_clock_phase = clock_phase;

	unsigned char backup_memory[0x10000];
	memcpy(backup_memory, memory, sizeof(memory));

	printf("reference: ");

	sim.load("reset_state.dat");

	int reset_state[sim_MOS6502::num_states];
	for (int i = 0; i < sim_MOS6502::num_states; i++)
		reset_state[i] = sim.state[i];

	sim.update();
	sim.input_res = true;

	for (int i = 0; i < 128; i++)
		cycle();

	char good_last_char = last_char;
	printf("  ---> known good last char: %c\n", good_last_char);

	int diff_count = 0;
	for (int i = 0; i < sim_MOS6502::num_states; i++)
		if (reset_state[i] == backup_sim.state[i])
			reset_state[i] = -1;
		else
			diff_count++;

	for (int i = 0, count = 0; i < sim_MOS6502::num_states; i++)
	{
		if (reset_state[i] < 0)
			continue;

		printf("test %3d%%: ", 100*count++ / diff_count);

		clock_phase = backup_clock_phase;
		memcpy(memory, backup_memory, sizeof(memory));
		sim = backup_sim;

		for (int j = 0; j < sim_MOS6502::num_states; j++)
			if (reset_state[j] >= 0 && i != j)
				sim.state[j] = reset_state[j];

		last_char = 0;
		if (!sim.try_update()) {
			printf("*update failed*  ---> need reset blop for signal %d (%s) !!\n", i, sim.netnames[i]);
			continue;
		}
		sim.input_res = true;

		for (int j = 0; j < 128; j++)
			cycle();

		if (last_char == good_last_char) {
			printf("  ---> don't need reset blop for signal %d (%s).\n", i, sim.netnames[i]);
			reset_state[i] = -1;
		} else {
			printf("  ---> need reset blop for signal %d (%s) !!\n", i, sim.netnames[i]);
		}
	}

	printf("Needed reset blops:\n");
	for (int i = 0; i < sim_MOS6502::num_states; i++)
		if (reset_state[i] >= 0)
			printf("%5d %-20s %d\n", i, sim.netnames[i], reset_state[i]);

	return 0;
#endif

#if 0
	sim.load("reset_state.dat");
	sim.update();
#endif

	sim.input_res = true;

	for (int i = 0; i < 1024; i++)
		cycle();
	printf("\n");

#ifdef WRITE_VCD
	fclose(f);
	sim.vcd_file = NULL;
#endif

	return 0;
}


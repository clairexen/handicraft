// clang -o  switchsim -std=c++11 switchsim.cc -lstdc++ -O0 -ggdb && ./switchsim

#include <stdio.h>
#include <assert.h>
#include <string.h>

#include <set>
#include <map>
#include <vector>
#include <string>

/*
 *  Simulator data structures
 */

#define Floating     0
#define PulledHigh   1
#define SwitchedHigh 2
#define SwitchedLow  4

struct Group;
struct Segment;
struct Switch;

struct Group
{
	int nextState, oldState;
	std::set<Segment*> segments;
	bool performedUpdate;

	Group(Segment *segment);
	~Group();

	void setState(int newState);
	void merge(Group *other);
	bool update();
};

struct Segment
{
	std::string id;
	int state, defaultState;
	std::set<Switch*> passSwitches, gateSwitches;
	Group *group;

	~Segment();
};

struct Switch
{
	std::string id;
	Segment *gate;
	std::set<Segment*> segments;
	int switchState;
};

struct Design
{
	std::map<std::string, Segment*> segments;
	std::map<std::string, Switch*> switches;

	~Design();

	void addSegment(std::string segmentId, int defaultState);
	void addSwitch(std::string switchId, std::string gateSegmentId, int switchState);
	void connectSwitch(std::string switchId, std::string segmentId);
	void setState(std::string segmentId, int state);
	bool isHigh(std::string segmentId);
	void update();
};


/*
 *  Simulator implementation
 */

Group::Group(Segment *segment)
{
	if (segment->group != NULL) {
		segment->group->segments.erase(segment);
		if (segment->group->segments.size() == 0)
			delete segment->group;
	}
	segment->group = this;
	segments.insert(segment);
	nextState = segment->defaultState;
	oldState = segment->state;
	performedUpdate = false;
}

Group::~Group()
{
	for (auto &it : segments)
		if (it->group == this)
			it->group = NULL;
}

void Group::setState(int newState)
{
	assert(performedUpdate == false);
	nextState |= newState;
}

void Group::merge(Group *other)
{
	assert(performedUpdate == false);
	if (other != this) {
		for (auto &it : other->segments) {
			segments.insert(it);
			it->group = this;
		}
		nextState |= other->nextState;
		oldState |= other->oldState;
		delete other;
	}
}

bool Group::update()
{
	if (performedUpdate)
		return false;

	bool didSomething = false;
	int state = nextState != Floating ? nextState : oldState;

#if 0
	if (state == Floating) {
		printf("floating/uninitialized group: ");
		goto print_segments;
	}
	if ((state & (SwitchedHigh|SwitchedLow)) == (SwitchedHigh|SwitchedLow)) {
		printf("driver conflict on group: ");
		goto print_segments;
	}
	if (0) {
print_segments:
		for (auto &it : segments)
			printf(" %s", it->id.c_str());
		printf("\n");
	}
#endif

	for (auto &it : segments) {
		if (it->state != state)
			didSomething = true;
		it->state = state;
	}
	performedUpdate = true;
	return didSomething;
}

Segment::~Segment() {
	if (group != NULL) {
		group->segments.erase(this);
		if (group->segments.size() == 0)
			delete group;
	}
}

Design::~Design()
{
	for (auto &it : segments)
		delete it.second;
	for (auto &it : switches)
		delete it.second;
}

void Design::addSegment(std::string segmentId, int defaultState)
{
	assert(segments.count(segmentId) == 0);
	segments[segmentId] = new Segment;
	segments[segmentId]->id = segmentId;
	segments[segmentId]->state = defaultState;
	segments[segmentId]->defaultState = defaultState;
	segments[segmentId]->group = NULL;
}

void Design::addSwitch(std::string switchId, std::string gateSegmentId, int switchState)
{
	assert(switches.count(switchId) == 0);

	switches[switchId] = new Switch;
	switches[switchId]->id = switchId;
	switches[switchId]->gate = segments.at(gateSegmentId);
	switches[switchId]->switchState = switchState;
}

void Design::connectSwitch(std::string switchId, std::string segmentId)
{
	switches.at(switchId)->segments.insert(segments.at(segmentId));
}

void Design::setState(std::string segmentId, int state)
{
	segments.at(segmentId)->defaultState = state | (segments.at(segmentId)->defaultState & PulledHigh);
}

bool Design::isHigh(std::string segmentId)
{
	if (segments.at(segmentId)->state == 0 || (segments.at(segmentId)->state & SwitchedLow) != 0)
		return false;
	return true;
}

void Design::update()
{
	bool didSomething = true;
	while (didSomething)
	{
		for (auto &i1 : segments)
			new Group(i1.second);

		for (auto &i1 : switches) {
			Switch *sw = i1.second;
			if (sw->gate->state == 0 || (sw->gate->state & SwitchedLow) != 0)
				continue;
			Group *group = NULL;
			for (auto &i2 : sw->segments)
				if (group == NULL)
					group = i2->group;
				else
					group->merge(i2->group);
			if (group != NULL)
				group->setState(sw->switchState);
		}

		didSomething = false;
		for (auto &it : segments)
			if (it.second->group->update())
				didSomething = true;
	}
}


/*
 *  MOS6502 Simulator
 */

struct MOS6502 : public Design
{
	int cycleCount;
	bool clockState;

	unsigned char memory[0x10000];

	void printNet(std::string netName)
	{
		printf(" %s=%d", netName.c_str(), isHigh(netName) ? 1 : 0);
	}

	void printBus(std::string busName, int upper, int lower, int digits)
	{
		int value = 0;
		for (int bit = upper; bit >= lower; bit--) {
			char buffer[100];
			snprintf(buffer, 100, "%s%d", busName.c_str(), bit);
			value = (value * 2) + (isHigh(buffer) ? 1 : 0);
		}
		printf(" %s[%d:%d]=%0*X", busName.c_str(), upper, lower, digits, value);
	}

	uint16_t getAB()
	{
		char buffer[8];
		uint16_t value = 0;
		for (int bit = 15; bit >= 0; bit--) {
			snprintf(buffer, 8, "ab%d", bit);
			value = (value * 2) + (isHigh(buffer) ? 1 : 0);
		}
		return value;
	}

	uint8_t getDB()
	{
		char buffer[8];
		uint8_t value = 0;
		for (int bit = 7; bit >= 0; bit--) {
			snprintf(buffer, 8, "db%d", bit);
			value = (value * 2) + (isHigh(buffer) ? 1 : 0);
		}
		return value;
	}

	void setDB(uint8_t value)
	{
		char buffer[8];
		for (int bit = 7; bit >= 0; bit--) {
			snprintf(buffer, 8, "db%d", bit);
			setState(buffer, ((value >> bit) & 1) != 0 ? SwitchedHigh : SwitchedLow);
		}
	}

	void releaseDB()
	{
		char buffer[8];
		for (int bit = 7; bit >= 0; bit--) {
			snprintf(buffer, 8, "db%d", bit);
			setState(buffer, Floating);
		}
	}

	void cycle()
	{
		clockState = !clockState;
		setState("clk0", clockState ? SwitchedHigh : SwitchedLow);

		if (!isHigh("rw"))
			releaseDB();

		bool oldClk2Out = isHigh("clk2out");
		update();

		// printf("%3d:", cycleCount++);
		// printNet("clk0");
		// printNet("clk1out");
		// printNet("cp1");
		// printNet("clk2out");
		// printNet("cclk");
		// printNet("res");
		// printNet("sync");
		// printNet("rw");
		// printBus("ab", 15, 0, 4);
		// printBus("db", 7, 0, 2);

		if (!oldClk2Out && isHigh("clk2out") && isHigh("res")) {
			if (isHigh("sync") && isHigh("rw")) {
				// printf("  INSTR: %02X  RD: %04X", memory[getAB()], getAB());
				setDB(memory[getAB()]);
			} else
			if (isHigh("rw")) {
				// printf("  DATA:  %02X  RD: %04X", memory[getAB()], getAB());
				setDB(memory[getAB()]);
			} else {
				// printf("  DATA:  %02X  WR: %04X", getDB(), getAB());
				if (getAB() == 0x00f)
					printf("<%c>", getDB()), fflush(stdout);
				memory[getAB()] = getDB();
			}
		}

		// printf("\n");
	}

	void read_nets_file(std::vector<std::string> &nets, const char *nets_filename)
	{
		if (nets_filename != NULL)
		{
			FILE *f = fopen(nets_filename, "r");
			if (f == NULL) {
				fprintf(stderr, "Can't open file %s for reading!\n", nets_filename);
				exit(1);
			}
			char buffer[100];
			while (fgets(buffer, 100, f) != NULL)
			{
				char *p = strtok(buffer, " \r\n\t");
				if (p && *p == 0)
					p = strtok(NULL, " \r\n\t");
				assert(p && *p);
				nets.push_back(p);
			}
			fclose(f);
		}
		else
		{
			for (auto &it : segments)
				nets.push_back(it.first);
		}
	}

	void dump_verilog_check(const char *verilog_filename, const char *nets_filename = NULL)
	{
		std::vector<std::string> nets;
		read_nets_file(nets, nets_filename);

		FILE *f = fopen(verilog_filename, "w");
		if (f == NULL) {
			fprintf(stderr, "Can't open file %s for writing!\n", verilog_filename);
			exit(1);
		}

		for (auto &net : nets)
		{
			Segment *segment = segments.at(net);
			std::string state = segment->state == Floating ? "Z" : "";
			state += (segment->state & PulledHigh) ? "P" : "";
			state += (segment->state & SwitchedHigh) ? "H" : "";
			state += (segment->state & SwitchedLow) ? "L" : "";
			fprintf(f, "if (uut.\\%s !== 1'bz) $display(\"%%s %20s %d %%b %s\", uut.\\%s === 1'b%d ? \"   \" : \"!!!\", uut.\\%s );\n",
					net.c_str(), net.c_str(), isHigh(net), state.c_str(), net.c_str(), isHigh(net), net.c_str());
		}

		fclose(f);
	}

	void dump_signals(const char *data_filename, const char *nets_filename = NULL)
	{
		std::vector<std::string> nets;
		read_nets_file(nets, nets_filename);

		FILE *f = fopen(data_filename, "w");
		if (f == NULL) {
			fprintf(stderr, "Can't open file %s for writing!\n", data_filename);
			exit(1);
		}

		for (auto &net : nets)
		{
			Segment *segment = segments.at(net);
			std::string state = segment->state == Floating ? "Z" : "";
			state += (segment->state & PulledHigh) ? "P" : "";
			state += (segment->state & SwitchedHigh) ? "H" : "";
			state += (segment->state & SwitchedLow) ? "L" : "";
			fprintf(f, "%-30s %d %s\n", net.c_str(), isHigh(net), state.c_str());
		}

		fclose(f);
	}

	MOS6502()
	{
		clockState = false;
		cycleCount = 0;

		#include "netlist.c"

		for (int i = 0; i < sizeof(segmentsList)/sizeof(*segmentsList); i++)
			addSegment(segmentsList[i].name, segmentsList[i].state);
		for (int i = 0; i < sizeof(switchesList)/sizeof(*switchesList); i++) {
			addSwitch(switchesList[i].name, switchesList[i].gate, switchesList[i].state);
			if (switchesList[i].cc1 != NULL)
				connectSwitch(switchesList[i].name, switchesList[i].cc1);
			if (switchesList[i].cc2 != NULL)
				connectSwitch(switchesList[i].name, switchesList[i].cc2);
		}

		for (int i = 0; i < sizeof(memory); i++)
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

		setState("rdy", SwitchedHigh);
		setState("irq", SwitchedHigh);
		setState("nmi", SwitchedHigh);
		setState("res", SwitchedLow);

		// NOPs everywhere!
		setState("db7", SwitchedHigh);
		setState("db6", SwitchedHigh);
		setState("db5", SwitchedHigh);
		setState("db4", SwitchedLow);
		setState("db3", SwitchedHigh);
		setState("db2", SwitchedLow);
		setState("db1", SwitchedHigh);
		setState("db0", SwitchedLow);

		update();
		cycle();

		for (int i = 1; i < 16; i++)
			cycle(), cycleCount = 0;

		dump_signals("reset_state.dat");

		setState("res", SwitchedHigh);
		update();
	}
};


/*
 *  Main
 */

int main()
{
	MOS6502 *chip = new MOS6502();

	for (int i = 0; i < 128; i++)
		chip->cycle();
	printf("\n");

	delete chip;
	return 0;
}



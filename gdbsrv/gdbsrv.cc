#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>

#include <vector>
#include <string>

using std::vector;
using std::string;


// ----------------------------------------------------------
// Interface to PicoRV32 core

#include "Vpicorv32.h"

Vpicorv32 *simtop;
uint32_t simram[1024*1024];
int cyclecnt = 0;

void cycle()
{
	printf("Running clock cycle %d.\n", cyclecnt++);
	simtop->clk = 0;
	simtop->eval();
	simtop->clk = 1;
	simtop->eval();
}

void setup()
{
	simtop = new Vpicorv32;
	simtop->resetn = 0;
	simtop->mem_ready = 0;
	simtop->debug_active = 0;
	simtop->debug_valid = 0;
	cycle();
	simtop->resetn = 1;
}

void wait_xfer()
{
	simtop->mem_ready = 0;
	cycle();

	while (!simtop->mem_valid)
		cycle();
}

void do_xfer()
{
	if (simtop->mem_valid)
	{
		simtop->mem_ready = 1;
		if (simtop->mem_wstrb) {
			uint32_t data = simram[simtop->mem_addr >> 2];
			if ((simtop->mem_wstrb & 1) != 0) data = (data & 0xffffff00) | (simtop->mem_wdata & 0x000000ff);
			if ((simtop->mem_wstrb & 2) != 0) data = (data & 0xffff00ff) | (simtop->mem_wdata & 0x0000ff00);
			if ((simtop->mem_wstrb & 4) != 0) data = (data & 0xff00ffff) | (simtop->mem_wdata & 0x00ff0000);
			if ((simtop->mem_wstrb & 8) != 0) data = (data & 0x00ffffff) | (simtop->mem_wdata & 0xff000000);
			simram[simtop->mem_addr >> 2] = data;
		} else {
			if (simtop->mem_instr)
				printf("Fetching instruction 0x%08lx from 0x%08lx.",
						(unsigned long)(simram[simtop->mem_addr >> 2]),
						(unsigned long)(simtop->mem_addr));
			simtop->mem_rdata = simram[simtop->mem_addr >> 2];
		}
		cycle();
	}

	simtop->mem_ready = 0;
	cycle();
}

uint32_t get_register_val(int idx)
{
	return idx;
}

void set_register_val(int idx, uint32_t data)
{
}

uint32_t read_memory(uint32_t addr)
{
	return 0;
}

void write_memory(uint32_t addr, uint32_t data)
{
}


// ----------------------------------------------------------
// Low-level gdb server protocol implementation

const int tcp_port = 1234;
int tcp_sockfd;

void netsock_accept()
{
	socklen_t addrlen;
	struct sockaddr_in addr;
	int enable = 1;

	int sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if (sockfd < 0) {
		perror("ERROR socket(AF_INET, SOCK_STREAM)");
		exit(1);
	}

	bzero((char *)&addr, sizeof(addr));
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = INADDR_ANY;
	addr.sin_port = htons(tcp_port);

	if (bind(sockfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
		perror("ERROR on bind()");
		exit(1);
	}

	if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0) {
		perror("ERROR on setsockopt(SO_REUSEADDR)");
		exit(1);
	}

	listen(sockfd, 5);
	addrlen = sizeof(addr);
	tcp_sockfd = accept(sockfd, (struct sockaddr *)&addr, &addrlen);

	if (tcp_sockfd < 0) {
		perror("ERROR on accept()");
		exit(1);
	}

	close(sockfd);
}

uint8_t recv_byte()
{
	uint8_t buf;
	if (read(tcp_sockfd, &buf, 1) != 1) {
		if (errno == 0)
			fprintf(stderr, "ERROR on read(): Connection closed\n");
		else
			perror("ERROR on read()");
		exit(1);
	}
	return buf;
}

void send_byte(uint8_t buf)
{
	if (write(tcp_sockfd, &buf, 1) != 1) {
		perror("ERROR on write()");
		exit(1);
	}
}

string recv_pkt()
{
	string data;
	uint8_t chksum = 0;

	while (true) {
		uint8_t ch = recv_byte();
		if (ch == '$')
			break;
	}

	while (true) {
		uint8_t ch = recv_byte();
		if (ch == '#')
			break;
		data.push_back(ch);
		chksum += ch;
	}

	uint8_t recv_chksum = 0;

	char c2 = recv_byte();
	if ('0' <= c2 && c2 <= '9') recv_chksum += 16 * (c2 - '0');
	if ('a' <= c2 && c2 <= 'f') recv_chksum += 16 * (c2 - 'a' + 10);
	if ('A' <= c2 && c2 <= 'F') recv_chksum += 16 * (c2 - 'A' + 10);

	char c1 = recv_byte();
	if ('0' <= c1 && c1 <= '9') recv_chksum += c1 - '0';
	if ('a' <= c1 && c1 <= 'f') recv_chksum += c1 - 'a' + 10;
	if ('A' <= c1 && c1 <= 'F') recv_chksum += c1 - 'A' + 10;

	printf("-> $");
	for (auto ch : data)
		printf("%c", ch);
	printf("#%c%c\n", c2, c1);

	if (recv_chksum != chksum) {
		fprintf(stderr, "Checksum Error: got=%02x, expected=%02x\n", recv_chksum, chksum);
		send_byte('-');
		return recv_pkt();
	}

	send_byte('+');
	return data;
}

void send_pkt(const string &data)
{
	uint8_t chksum = 0;

	printf("<- $");
	send_byte('$');

	for (auto ch : data) {
		send_byte(ch);
		printf("%c", ch);
		chksum += ch;
	}

	char c2 = "0123456789abcdef"[chksum >> 4];
	char c1 = "0123456789abcdef"[chksum & 15];

	send_byte('#');
	send_byte(c2);
	send_byte(c1);

	printf("#%c%c\n", c2, c1);
}


// ----------------------------------------------------------
// High-level gdb server protocol implementation

string tokenize(string &s)
{
	string r;
	int idx = 0;

	for (; idx < int(s.size()); idx++) {
		auto ch = s[idx];
		if (ch == ' ' || ch == ',' || ch == ';' || ch == ':')
			break;
		r.push_back(ch);
	}

	for (; idx < int(s.size()); idx++) {
		auto ch = s[idx];
		if (ch != ' ' && ch != ',' && ch != ';' && ch != ':')
			break;
	}

	s = s.substr(idx);
	return r;
}

int main()
{
	printf("Setting up simulated PicoRV32..\n");
	setup();

	printf("Waiting for TCP connection at port %d..\n", tcp_port);
	netsock_accept();
	printf("Connection to debugger established.\n");

	while (true)
	{
		string rpkt = recv_pkt();
		string command = tokenize(rpkt);
		printf("Remote command: '%s'\n", command.c_str());

		if (command == "g") {
			string reply;
			for (int i = 0; i <= 32; i++) {
				char buffer[64];
				uint32_t regval = get_register_val(i);
				snprintf(buffer, 64, "%02x%02x%02x%02x",
						(regval >> 0) & 255, (regval >> 8) & 255,
						(regval >> 16) & 255, (regval >> 24) & 255);
				reply += buffer;
			}
			send_pkt(reply);
			continue;
		}

		if (command == "G") {
			send_pkt("OK");
			continue;
		}

		if (command == "m") {
			send_pkt("");
			continue;
		}

		if (command == "M") {
			send_pkt("");
			continue;
		}

		if (command == "c") {
			send_pkt("");
			continue;
		}

		if (command == "s") {
			send_pkt("");
			continue;
		}

		if (command == "P") {
			send_pkt("");
			continue;
		}

		if (command == "?") {
			send_pkt("S05");
			continue;
		}

		if (command == "qC") {
			send_pkt("");
			continue;
		}

		// default: send empty response
		send_pkt("");
	}

	return 0; 
}

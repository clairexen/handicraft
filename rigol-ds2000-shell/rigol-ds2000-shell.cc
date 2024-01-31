/*
 *  Command-Line Interface to RIGOL DS2000 VISA Interface
 *
 *  Copyright (C) 2013  Clifford Wolf <clifford@clifford.at>
 *
 *  Permission to use, copy, modify, and/or distribute this software for any
 *  purpose with or without fee is hereby granted, provided that the above
 *  copyright notice and this permission notice appear in all copies.
 *  
 *  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 *  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 *  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 *  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 *  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 *  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 */

#include <visa.h>

#include <readline/readline.h>
#include <readline/history.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <string>
#include <set>

extern const char *all_commands_raw[];
std::set<std::string> all_cmds;

void add_command(std::string txt)
{
#if 0
	for (size_t i = 0; i < txt.size(); i++)
		if ('a' <= txt[i] && txt[i] <= 'z') {
			size_t k = i;
			while (i < txt.size() && 'a' <= txt[i] && txt[i] <= 'z')
				i++;
			add_command(txt.substr(0, k) + txt.substr(i));
		}
#endif

	for (size_t i = 0; i < txt.size(); i++)
		if ('a' <= txt[i] && txt[i] <= 'z')
			txt[i] -= 'a' - 'A';

	all_cmds.insert(txt);
}

void flush(ViSession &vi)
{
	char buffer[1024];

	while (1) {
		ViUInt32 count;
		if (viRead(vi, reinterpret_cast<ViPBuf>(buffer), sizeof(buffer), &count) != VI_SUCCESS) {
			fprintf(stderr, "Cannot read from resource!\n");
			exit(1);
		}
		if (count <= 0)
			break;
	}
}

void send_cmd(ViSession &vi, std::string cmd)
{
	ViUInt32 count;
	if (viWrite(vi, reinterpret_cast<ViPBuf>(const_cast<char*>(cmd.data())), cmd.size(), &count) != VI_SUCCESS) {
		fprintf(stderr, "Cannot write to resource!\n");
		exit(1);
	}
}

unsigned char *recv_data(ViSession &vi, size_t len)
{
	unsigned char *data = (unsigned char*)malloc(len+1);
	unsigned char *p = data;

	while (p != data+len) {
		ViUInt32 count;
		if (viRead(vi, reinterpret_cast<ViPBuf>(p), data+len-p, &count) != VI_SUCCESS) {
			fprintf(stderr, "Cannot read from resource (after %d bytes of data)!\n", int(p-data));
			exit(1);
		}
		p += count;
	}

	data[len] = 0;
	return data;
}

std::string recv_text(ViSession &vi)
{
	int data_size = 1024, data_pos = 0;
	char *data = (char*)malloc(data_size);
	int timeout = 0;

	while (1) {
		ViUInt32 count;
		if (viRead(vi, reinterpret_cast<ViPBuf>(data+data_pos), data_size-data_pos, &count) != VI_SUCCESS) {
			fprintf(stderr, "Cannot read from resource (after %d bytes of text)!\n", data_pos);
			exit(1);
		}
		if (count == 0) {
			if (++timeout >= 2) {
				printf("TIMEOUT! -- Invalid query or missing query argument?\n");
				break;
			}
		} else
			timeout = 0;
		data_pos += count;
		if (data[data_pos-1] == '\r' || data[data_pos-1] == '\n')
			break;
		if (data_pos > data_size/2) {
			data_size *= 2;
			data = (char*)realloc(data, data_size);
		}
	}

	while (data_pos > 0 && (data[data_pos-1] == '\r' || data[data_pos-1] == '\n' || data[data_pos-1] == 0))
		data_pos--;
	std::string text = std::string(data, data_pos);
	free(data);
	return text;
}

char *readline_cmd_generator(const char *text, int state)
{
	static std::set<std::string>::iterator it;
	static std::string prefix;

	if (!state) {
		it = all_cmds.begin(), prefix = text;
		for (size_t i = 0; i < prefix.size(); i++)
			if ('a' <= prefix[i] && prefix[i] <= 'z')
				prefix[i] -= 'a' - 'A';
	}

	for (; it != all_cmds.end(); it++) {
		std::string pf = it->substr(0, prefix.size());
		if (pf == prefix)
			return strdup((it++)->c_str());
	}

	return NULL;
}

char **readline_completion(const char *text, int start, int)
{
	if (start == 0)
		return rl_completion_matches(text, readline_cmd_generator);
	return NULL;
}

bool run_command(ViSession &vi, char *serial, char *command)
{
	std::string orig_command = command;
	char *cmd = strtok(command, " \t\r\n");

	if (cmd == NULL || cmd[0] == 0 || cmd[0] == '#')
		return true;

	add_history(orig_command.c_str());

	if (cmd[0] == '*' || cmd[0] == ':')
	{
		send_cmd(vi, orig_command);

		for (char *p = cmd; *p; p++)
			if ('a' <= *p && *p <= 'z')
				*p -= 'a' - 'A';

		if (strlen(cmd) > 6 && cmd[strlen(cmd)-6] == ':' && cmd[strlen(cmd)-5] == 'D' && cmd[strlen(cmd)-4] == 'A' &&
				cmd[strlen(cmd)-3] == 'T' && cmd[strlen(cmd)-2] == 'A' && cmd[strlen(cmd)-1] == '?')
		{
			unsigned char *header = recv_data(vi, 11);
			printf("Header: %s\n", header);

			printf("Data:");
			int num_bytes = atoi((char*)header + 2);
			unsigned char *data = recv_data(vi, num_bytes + 1);
			for (int i = 0; i < num_bytes; i++)
				printf(" %d", (unsigned char)data[i]);
			printf("\n\n");

			free(header);
			free(data);
		} else
		if (cmd[strlen(cmd)-1] == '?')
		{
			std::string result = recv_text(vi);
			if (!result.empty())
				printf("%s\n", result.c_str());
			printf("\n");
		}

		return true;
	}

	for (char *p = cmd; *p; p++)
		if ('a' <= *p && *p <= 'z')
			*p -= 'a' - 'A';

	if (!strcmp(cmd, "SLEEP"))
	{
		char *seconds = strtok(NULL, " \t\r\n");
		sleep(atoi(seconds));
		return true;
	}

	if (!strcmp(cmd, "WAIT"))
	{
		std::string status;
		printf("Waiting for trigger..");
		fflush(stdout);
		sleep(1);

		while (1) {
			send_cmd(vi, ":TRIGGER:STATUS?");
			status = recv_text(vi);
			if (status != "WAIT")
				break;
			printf(".");
			fflush(stdout);
		}

		printf(" %s\n\n", status.c_str());
		return true;
	}

	if (!strcmp(cmd, "SAVE_TXT") || !strcmp(cmd, "SAVE_BIN"))
	{
		char *query = strtok(NULL, " \t\r\n");
		char *filename = strtok(NULL, " \t\r\n");

		for (char *p = query; *p; p++)
			if ('a' <= *p && *p <= 'z')
				*p -= 'a' - 'A';

		if (strlen(query) > 6 && query[strlen(query)-6] == ':' && query[strlen(query)-5] == 'D' && query[strlen(query)-4] == 'A' &&
				query[strlen(query)-3] == 'T' && query[strlen(query)-2] == 'A' && query[strlen(query)-1] == '?')
		{
			send_cmd(vi, query);

			unsigned char *header = recv_data(vi, 11);
			printf("Header: %s\n", header);

			int num_bytes = atoi((char*)header + 2);
			unsigned char *data = recv_data(vi, num_bytes + 1);

			FILE *f = fopen(filename, "w");
			if (f == NULL) {
				fprintf(stderr, "Cannot open file '%s' for writing!\n", filename);
				exit(1);
			}
			if (!strcmp(cmd, "SAVE_TXT")) {
				for (int i = 0; i < num_bytes; i++)
					fprintf(f, "%d\n", (unsigned char)data[i]);
			} else
				fwrite(data, num_bytes, 1, f);
			fclose(f);

			printf("Written %d bytes to file (%s).\n", num_bytes, !strcmp(cmd, "SAVE_TXT") ? "ascii" : "binary");

			free(header);
			free(data);
		} else
		if (query[strlen(query)-1] == '?')
		{
			send_cmd(vi, query);
			std::string result = recv_text(vi);

			FILE *f = fopen(filename, "w");
			if (f == NULL) {
				fprintf(stderr, "Cannot open file '%s' for writing!\n", filename);
				exit(1);
			}
			fprintf(f, "%s\n", result.c_str());
			fclose(f);

			printf("Written %d characters to file.\n", int(result.size()));
		} else
			printf("SAVE_* not supported for non-query command '%s'.\n", query);

		printf("\n");
		return true;
	}

	if (!strcmp(cmd, "DOWNLOAD_TXT") || !strcmp(cmd, "DOWNLOAD_BIN"))
	{
		char *filename = strtok(NULL, " \t\r\n");

		send_cmd(vi, ":WAVEFORM:RESET");
		send_cmd(vi, ":WAVEFORM:BEGIN");

		FILE *f = fopen(filename, "w");
		if (f == NULL) {
			fprintf(stderr, "Cannot open file '%s' for writing!\n", filename);
			exit(1);
		}

		int total_bytes = 0;
		printf("Downloading..");
		fflush(stdout);

		while (1)
		{
			send_cmd(vi, ":WAVEFORM:STATUS?");
			std::string status = recv_text(vi);

			send_cmd(vi, ":WAVEFORM:DATA?");
			unsigned char *header = recv_data(vi, 11);

			int num_bytes = atoi((char*)header + 2);
			unsigned char *data = recv_data(vi, num_bytes + 1);

			if (!strcmp(cmd, "DOWNLOAD_TXT")) {
				for (int i = 0; i < num_bytes; i++)
					fprintf(f, "%d\n", (unsigned char)data[i]);
			} else
				fwrite(data, num_bytes, 1, f);

			total_bytes += num_bytes;
			printf(".");
			fflush(stdout);

			free(header);
			free(data);

			if (status[0] == 'I') {
				send_cmd(vi, ":WAVEFORM:END");
				printf("\n");
				break;
			}
		}

		fclose(f);

		printf("Written %d bytes to file (%s).\n\n", total_bytes, !strcmp(cmd, "DOWNLOAD_TXT") ? "ascii" : "binary");
		return true;
	}

	if (!strcmp(cmd, "INSTALL"))
	{
		std::string lic_key;
		char *opt = strtok(NULL, " \t\r\n");

		std::string sh_cmd = std::string("rigol-4in1-keygen ") + serial + " " + opt;
		char buffer[1024];

		FILE *p = popen(sh_cmd.c_str(), "r");
		while (fgets(buffer, 1024, p) != NULL) {
			if (!strncmp(buffer, "your-license-key: ", 18)) {
				lic_key.clear();
				for (int i = 0; i < 4*7+3; i++)
					if (buffer[18+i] != '-')
						lic_key += buffer[18+i];
			}
			printf("*** %s", buffer);
		}
		pclose(p);

		if (lic_key.empty()) {
			fprintf(stderr, "Failed to generate licence key!\n");
			exit(1);
		}

		std::string install_cmd = ":SYSTEM:OPTION:INSTALL " + lic_key;
		printf("> %s\n", install_cmd.c_str());

		send_cmd(vi, install_cmd);
		printf("\n");
		return true;
	}

	if (!strcmp(cmd, "HELP"))
	{
		printf("\n");
		printf("    [:*]...\n");
		printf("        send command to device and print results, e.g. '*IDN?'\n");
		printf("        (see the DS2000 Series Programming Guide)\n");
		printf("\n");
		printf("    SLEEP <n>\n");
		printf("        pause executeion for <n> seconds\n");
		printf("\n");
		printf("    WAIT\n");
		printf("        wait for :TRIGGER:STATUS? to return something else than WAIT\n");
		printf("\n");
		printf("    SAVE_TXT <QUERY> <FILENAME>\n");
		printf("        e.g. 'SAVE_TXT :WAV:DATA? wavedata.txt'\n");
		printf("\n");
		printf("    SAVE_BIN <QUERY> <FILENAME>\n");
		printf("        e.g. 'SAVE_BIN :DISP:DATA? screenshot.bmp'\n");
		printf("\n");
		printf("    DOWNLOAD_TXT <FILENAME>\n");
		printf("    DOWNLOAD_BIN <FILENAME>\n");
		printf("        save large waveforms (use instead of SAVE_* in RAW mode)\n");
		printf("        (see :WAVEFORM:DATA? in Programming Guide)\n");
		printf("\n");
		printf("    INSTALLL <option>\n");
		printf("        generate key and install option, e.g. 'INSTALL DSA9'\n");
		printf("        DSA? for permanent options, VSA? for temporary options\n");
		printf("        this command needs 'rigol-4in1-keygen' in path\n");
		printf("\n");
		printf("    HELP\n");
		printf("        print this help message\n");
		printf("\n");
		printf("    EXIT\n");
		printf("        quit this program\n");
		printf("\n");
		return true;
	}

	if (!strcmp(cmd, "EXIT"))
		return false;

	printf("Unknown command '%s'. Type 'help' for help.\n\n", command);
	return true;
}

int main(int argc, char **argv)
{
	if (argc < 2 || !strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) {
		printf("\n");
		printf("Usage: %s VISA-RESOURCE [commands]\n", argv[0]);
		printf("       %s --scan\n", argv[0]);
		printf("\n");
		printf("E.g.: %s TCPIP::192.168.0.123::INSTR ':CALIBRATE:DATE?'\n", argv[0]);
		printf("\n");
		exit(argc != 2);
	}

	ViSession rmgr;

	if (viOpenDefaultRM(&rmgr) != VI_SUCCESS) {
		fprintf(stderr, "Cannot open default resource manager!\n");
		exit(1);
	}

	if (!strcmp(argv[1], "--scan")) {
		ViFindList fl;
		ViUInt32 count;
		ViChar rsrc[256];
		ViStatus rc = viFindRsrc(rmgr, const_cast<ViChar*>("?*"), &fl, &count, rsrc);
		while (rc == VI_SUCCESS) {
			printf("%s\n", rsrc);
			rc = viFindNext(fl, rsrc);
		}
		viClose(fl);
		viClose(rmgr);
		exit(0);
	}

	ViSession vi;

	if (viOpen(rmgr, argv[1], VI_NO_LOCK, 0, &vi) != VI_SUCCESS) {
		fprintf(stderr, "Cannot open resource %s!\n", argv[1]);
		exit(1);
	}

	flush(vi);
	send_cmd(vi, "*IDN?");
	std::string idn = recv_text(vi);

	char *serial_idn = strdup(idn.c_str());
	char *serial = strtok(serial_idn, ",");
	serial = strtok(NULL, ",");
	serial = strtok(NULL, ",");

	for (int i = 0; all_commands_raw[i]; i++)
		add_command(all_commands_raw[i]);
	add_command("SLEEP");
	add_command("WAIT");
	add_command("SAVE_TXT");
	add_command("SAVE_BIN");
	add_command("DOWNLOAD_TXT");
	add_command("DOWNLOAD_BIN");
	add_command("INSTALL");
	add_command("QUERY");
	add_command("HELP");
	add_command("EXIT");
	
	if (argc == 2)
	{
		printf("\nConnected to %s.\n", idn.c_str());
		printf("Type 'help' for help.\n\n");

		char prompt[1024];
		snprintf(prompt, 1024, "%s> ", serial);
	
		rl_attempted_completion_function = readline_completion;
		rl_basic_word_break_characters = " \t\n";

		char *command = NULL;
		while ((command = readline(prompt)) != NULL)
			if (!run_command(vi, serial, command))
				break;
		if (command == NULL)
			printf("EXIT\n");
	}
	else
	{
		printf("\nConnected to %s.\n\n", idn.c_str());
		for (int i = 2; i < argc; i++) {
			printf("%s> %s\n", serial, argv[i]);
			if (!run_command(vi, serial, argv[i]))
				break;
		}
	}

	viClose(vi);
	viClose(rmgr);

	return 0;
}

/***
  extracted from DS2000_ProgrammingGuide.pdf:

  pdftotext DS2000_ProgrammingGuide_EN.pdf
  sed -r '/^Syntax/,/^[^:]/ !d; /^:/ ! d; s: .*::; /^:(BUS|CHANnel)<n>/ { h; s/<n>/1/; p; g; s/<n>/2/; };
  		/\[/ { h; s/\[.*\]//; p; g; s/\[//; s/\]//; };' DS2000_ProgrammingGuide_EN.txt | \
  		sed 's:.*:"&",:;' | tr '\n' '\t' | expand -t40 | fold -w160; echo NULL
***/
const char *all_commands_raw[] = {
":AUToscale",                           ":CLEar",                               ":RUN",                                 ":SINGle",                              
":STOP",                                ":TFORce",                              ":TLHAlf",                              ":ACQuire:AVERages",                    
":ACQuire:AVERages?",                   ":ACQuire:MDEPth",                      ":ACQuire:MDEPth?",                     ":ACQuire:SRATe?",                      
":ACQuire:TYPE",                        ":ACQuire:TYPE?",                       ":ACQuire:AALias",                      ":ACQuire:AALias?",                     
":BUS1:MODE",                           ":BUS2:MODE",                           ":BUS1:MODE?",                          ":BUS2:MODE?",                          
":BUS1:DISPlay",                        ":BUS2:DISPlay",                        ":BUS1:DISPlay?",                       ":BUS2:DISPlay?",                       
":BUS1:FORMat",                         ":BUS2:FORMat",                         ":BUS1:FORMat?",                        ":BUS2:FORMat?",                        
":BUS1:EVENt",                          ":BUS2:EVENt",                          ":BUS1:EVENt?",                         ":BUS2:EVENt?",                         
":BUS1:EEXPort",                        ":BUS2:EEXPort",                        ":BUS1:PARallel:CLK",                   ":BUS2:PARallel:CLK",                   
":BUS1:PARallel:CLK?",                  ":BUS2:PARallel:CLK?",                  ":BUS1:PARallel:SLOPe",                 ":BUS2:PARallel:SLOPe",                 
":BUS1:PARallel:SLOPe?",                ":BUS2:PARallel:SLOPe?",                ":BUS1:PARallel:BSET",                  ":BUS2:PARallel:BSET",                  
":BUS1:PARallel:BSET?",                 ":BUS2:PARallel:BSET?",                 ":BUS1:PARallel:THReshold",             ":BUS2:PARallel:THReshold",             
":BUS1:PARallel:THReshold?",            ":BUS2:PARallel:THReshold?",            ":BUS1:PARallel:OFFSet",                ":BUS2:PARallel:OFFSet",                
":BUS1:PARallel:OFFSet?",               ":BUS2:PARallel:OFFSet?",               ":BUS1:RS232:TX",                       ":BUS2:RS232:TX",                       
":BUS1:RS232:TX?",                      ":BUS2:RS232:TX?",                      ":BUS1:RS232:RX",                       ":BUS2:RS232:RX",                       
":BUS1:RS232:RX?",                      ":BUS2:RS232:RX?",                      ":BUS1:RS232:POLarity",                 ":BUS2:RS232:POLarity",                 
":BUS1:RS232:POLarity?",                ":BUS2:RS232:POLarity?",                ":BUS1:RS232:ENDian",                   ":BUS2:RS232:ENDian",                   
":BUS1:RS232:ENDian?",                  ":BUS2:RS232:ENDian?",                  ":BUS1:RS232:BAUD",                     ":BUS2:RS232:BAUD",                     
":BUS1:RS232:BAUD?",                    ":BUS2:RS232:BAUD?",                    ":BUS1:RS232:BUSer",                    ":BUS2:RS232:BUSer",                    
":BUS1:RS232:BUSer?",                   ":BUS2:RS232:BUSer?",                   ":BUS1:RS232:DBITs",                    ":BUS2:RS232:DBITs",                    
":BUS1:RS232:DBITs?",                   ":BUS2:RS232:DBITs?",                   ":BUS1:RS232:SBITs",                    ":BUS2:RS232:SBITs",                    
":BUS1:RS232:SBITs?",                   ":BUS2:RS232:SBITs?",                   ":BUS1:RS232:PARity",                   ":BUS2:RS232:PARity",                   
":BUS1:RS232:PARity?",                  ":BUS2:RS232:PARity?",                  ":BUS1:RS232:PACKet",                   ":BUS2:RS232:PACKet",                   
":BUS1:RS232:PACKet?",                  ":BUS2:RS232:PACKet?",                  ":BUS1:RS232:PEND",                     ":BUS2:RS232:PEND",                     
":BUS1:RS232:PEND?",                    ":BUS2:RS232:PEND?",                    ":BUS1:RS232:TTHReshold",               ":BUS2:RS232:TTHReshold",               
":BUS1:RS232:TTHReshold?",              ":BUS2:RS232:TTHReshold?",              ":BUS1:RS232:RTHReshold",               ":BUS2:RS232:RTHReshold",               
":BUS1:RS232:RTHReshold?",              ":BUS2:RS232:RTHReshold?",              ":BUS1:RS232:OFFSet",                   ":BUS2:RS232:OFFSet",                   
":BUS1:RS232:OFFSet?",                  ":BUS2:RS232:OFFSet?",                  ":BUS1:IIC:SCLK:SOURce",                ":BUS2:IIC:SCLK:SOURce",                
":BUS1:IIC:SCLK:SOURce?",               ":BUS2:IIC:SCLK:SOURce?",               ":BUS1:IIC:SCLK:THReshold",             ":BUS2:IIC:SCLK:THReshold",             
":BUS1:IIC:SCLK:THReshold?",            ":BUS2:IIC:SCLK:THReshold?",            ":BUS1:IIC:SDA:SOURce",                 ":BUS2:IIC:SDA:SOURce",                 
":BUS1:IIC:SDA:SOURce?",                ":BUS2:IIC:SDA:SOURce?",                ":BUS1:IIC:SDA:THReshold",              ":BUS2:IIC:SDA:THReshold",              
":BUS1:IIC:SDA:THReshold?",             ":BUS2:IIC:SDA:THReshold?",             ":BUS1:IIC:OFFSet",                     ":BUS2:IIC:OFFSet",                     
":BUS1:IIC:OFFSet?",                    ":BUS2:IIC:OFFSet?",                    ":BUS1:SPI:SCLK:SOURce",                ":BUS2:SPI:SCLK:SOURce",                
":BUS1:SPI:SCLK:SOURce?",               ":BUS2:SPI:SCLK:SOURce?",               ":BUS1:SPI:SCLK:SLOPe",                 ":BUS2:SPI:SCLK:SLOPe",                 
":BUS1:SPI:SCLK:SLOPe?",                ":BUS2:SPI:SCLK:SLOPe?",                ":BUS1:SPI:SCLK:THReshold",             ":BUS2:SPI:SCLK:THReshold",             
":BUS1:SPI:SCLK:THReshold?",            ":BUS2:SPI:SCLK:THReshold?",            ":BUS1:SPI:SDA:SOURce",                 ":BUS2:SPI:SDA:SOURce",                 
":BUS1:SPI:SDA:SOURce?",                ":BUS2:SPI:SDA:SOURce?",                ":BUS1:SPI:SDA:POLarity",               ":BUS2:SPI:SDA:POLarity",               
":BUS1:SPI:SDA:POLarity?",              ":BUS2:SPI:SDA:POLarity?",              ":BUS1:SPI:SDA:THReshold",              ":BUS2:SPI:SDA:THReshold",              
":BUS1:SPI:SDA:THReshold?",             ":BUS2:SPI:SDA:THReshold?",             ":BUS1:SPI:DBITs",                      ":BUS2:SPI:DBITs",                      
":BUS1:SPI:DBITs?",                     ":BUS2:SPI:DBITs?",                     ":BUS1:SPI:ENDian",                     ":BUS2:SPI:ENDian",                     
":BUS1:SPI:ENDian?",                    ":BUS2:SPI:ENDian?",                    ":BUS1:SPI:OFFSet",                     ":BUS2:SPI:OFFSet",                     
":BUS1:SPI:OFFSet?",                    ":BUS2:SPI:OFFSet?",                    ":CALCulate:MODE",                      ":CALCulate:MODE?",                     
":CALCulate:ADD:SA",                    ":CALCulate:ADD:SA?",                   ":CALCulate:ADD:SB",                    ":CALCulate:ADD:SB?",                   
":CALCulate:ADD:INVert",                ":CALCulate:ADD:INVert?",               ":CALCulate:ADD:VSCale",                ":CALCulate:ADD:VSCale?",               
":CALCulate:ADD:VOFFset",               ":CALCulate:ADD:VOFFset?",              ":CALCulate:SUB:SA",                    ":CALCulate:SUB:SA?",                   
":CALCulate:SUB:SB",                    ":CALCulate:SUB:SB?",                   ":CALCulate:SUB:INVert",                ":CALCulate:SUB:INVert?",               
":CALCulate:SUB:VSCale",                ":CALCulate:SUB:VSCale?",               ":CALCulate:SUB:VOFFset",               ":CALCulate:SUB:VOFFset?",              
":CALCulate:MULTiply:SA",               ":CALCulate:MULTiply:SA?",              ":CALCulate:MULTiply:SB",               ":CALCulate:MULTiply:SB?",              
":CALCulate:MULTiply:INVert",           ":CALCulate:MULTiply:INVert?",          ":CALCulate:MULTiply:VSCale",           ":CALCulate:MULTiply:VSCale?",          
":CALCulate:MULTiply:VOFFset",          ":CALCulate:MULTiply:VOFFset?",         ":CALCulate:DIVision:SA",               ":CALCulate:DIVision:SA?",              
":CALCulate:DIVision:SB",               ":CALCulate:DIVision:SB?",              ":CALCulate:DIVision:INVert",           ":CALCulate:DIVision:INVert?",          
":CALCulate:DIVision:VSCale",           ":CALCulate:DIVision:VSCale?",          ":CALCulate:DIVision:VOFFset",          ":CALCulate:DIVision:VOFFset?",         
":CALCulate:FFT:SOURce",                ":CALCulate:FFT:SOURce?",               ":CALCulate:FFT:WINDow",                ":CALCulate:FFT:WINDow?",               
":CALCulate:FFT:SPLit",                 ":CALCulate:FFT:SPLit?",                ":CALCulate:FFT:VSMode",                ":CALCulate:FFT:VSMode?",               
":CALCulate:FFT:VSCale",                ":CALCulate:FFT:VSCale?",               ":CALCulate:FFT:VOFFset",               ":CALCulate:FFT:VOFFset?",              
":CALCulate:FFT:HSCale",                ":CALCulate:FFT:HSCale?",               ":CALCulate:FFT:HOFFset",               ":CALCulate:FFT:HOFFset?",              
":CALCulate:FFT:HSPan",                 ":CALCulate:FFT:HSPan?",                ":CALCulate:FFT:HCENter",               ":CALCulate:FFT:HCENter?",              
":CALCulate:LOGic:SA",                  ":CALCulate:LOGic:SA?",                 ":CALCulate:LOGic:SB",                  ":CALCulate:LOGic:SB?",                 
":CALCulate:LOGic:INVert",              ":CALCulate:LOGic:INVert?",             ":CALCulate:LOGic:VSCale",              ":CALCulate:LOGic:VSCale?",             
":CALCulate:LOGic:VOFFset",             ":CALCulate:LOGic:VOFFset?",            ":CALCulate:LOGic:OPERator",            ":CALCulate:LOGic:OPERator?",           
":CALCulate:LOGic:ATHReshold",          ":CALCulate:LOGic:ATHReshold?",         ":CALCulate:LOGic:BTHReshold",          ":CALCulate:LOGic:BTHReshold?",         
":CALCulate:ADVanced:EXPRession",       ":CALCulate:ADVanced:EXPRession?",      ":CALCulate:ADVanced:INVert",           ":CALCulate:ADVanced:INVert?",          
":CALCulate:ADVanced:VARiable1",        ":CALCulate:ADVanced:VARiable1?",       ":CALCulate:ADVanced:VARiable2",        ":CALCulate:ADVanced:VARiable2?",       
":CALCulate:ADVanced:VSCale",           ":CALCulate:ADVanced:VSCale?",          ":CALCulate:ADVanced:VOFFset",          ":CALCulate:ADVanced:VOFFset?",         
":CALibrate:DATE?",                     ":CALibrate:STARt",                     ":CALibrate:TIME?",                     ":CALibrate:QUIT",                      
":CHANnel1:BWLimit",                    ":CHANnel2:BWLimit",                    ":CHANnel1:BWLimit?",                   ":CHANnel2:BWLimit?",                   
":CHANnel1:COUPling",                   ":CHANnel2:COUPling",                   ":CHANnel1:COUPling?",                  ":CHANnel2:COUPling?",                  
":CHANnel1:DISPlay",                    ":CHANnel2:DISPlay",                    ":CHANnel1:DISPlay?",                   ":CHANnel2:DISPlay?",                   
":CHANnel1:INVert",                     ":CHANnel2:INVert",                     ":CHANnel1:INVert?",                    ":CHANnel2:INVert?",                    
":CHANnel1:OFFSet",                     ":CHANnel2:OFFSet",                     ":CHANnel1:OFFSet?",                    ":CHANnel2:OFFSet?",                    
":CHANnel1:SCALe",                      ":CHANnel2:SCALe",                      ":CHANnel1:SCALe?",                     ":CHANnel2:SCALe?",                     
":CHANnel1:PROBe",                      ":CHANnel2:PROBe",                      ":CHANnel1:PROBe?",                     ":CHANnel2:PROBe?",                     
":CHANnel1:UNITs",                      ":CHANnel2:UNITs",                      ":CHANnel1:UNITs?",                     ":CHANnel2:UNITs?",                     
":CHANnel1:VERNier",                    ":CHANnel2:VERNier",                    ":CHANnel1:VERNier?",                   ":CHANnel2:VERNier?",                   
":CURSor:MODE",                         ":CURSor:MODE?",                        ":CURSor:MANual:TYPE",                  ":CURSor:MANual:TYPE?",                 
":CURSor:MANual:SOURce",                ":CURSor:MANual:SOURce?",               ":CURSor:MANual:TUNit",                 ":CURSor:MANual:TUNit?",                
":CURSor:MANual:VUNit",                 ":CURSor:MANual:VUNit?",                ":CURSor:MANual:CAX",                   ":CURSor:MANual:CAX?",                  
":CURSor:MANual:CBX",                   ":CURSor:MANual:CBX?",                  ":CURSor:MANual:CAY",                   ":CURSor:MANual:CAY?",                  
":CURSor:MANual:CBY",                   ":CURSor:MANual:CBY?",                  ":CURSor:MANual:AXValue?",              ":CURSor:MANual:AYValue?",              
":CURSor:MANual:BXValue?",              ":CURSor:MANual:BYValue?",              ":CURSor:MANual:XDELta?",               ":CURSor:MANual:IXDelta?",              
":CURSor:MANual:YDELta?",               ":CURSor:TRACk:SOURce1",                ":CURSor:TRACk:SOURce1?",               ":CURSor:TRACk:SOURce2",                
":CURSor:TRACk:SOURce2?",               ":CURSor:TRACk:CAX",                    ":CURSor:TRACk:CAX?",                   ":CURSor:TRACk:CBX",                    
":CURSor:TRACk:CBX?",                   ":CURSor:TRACk:CAY?",                   ":CURSor:TRACk:CBY?",                   ":CURSor:TRACk:AXValue?",               
":CURSor:TRACk:AYValue?",               ":CURSor:TRACk:BXValue?",               ":CURSor:TRACk:BYValue?",               ":CURSor:TRACk:XDELta?",                
":CURSor:TRACk:YDELta?",                ":CURSor:TRACk:IXDelta?",               ":DISPlay:CLEar",                       ":DISPlay:TYPE",                        
":DISPlay:TYPE?",                       ":DISPlay:GRADing:TIME",                ":DISPlay:GRADing:TIME?",               ":DISPlay:WBRightness",                 
":DISPlay:WBRightness?",                ":DISPlay:GRID",                        ":DISPlay:GRID?",                       ":DISPlay:GBRightness",                 
":DISPlay:GBRightness?",                ":DISPlay:MPERsistence",                ":DISPlay:MPERsistence?",               ":DISPlay:DATA?",                       
":FUNCtion:WRMode",                     ":FUNCtion:WRMode?",                    ":FUNCtion:WRECord:FEND",               ":FUNCtion:WRECord:FEND?",              
":FUNCtion:WRECord:FMAX?",              ":FUNCtion:WRECord:INTerval",           ":FUNCtion:WRECord:INTerval?",          ":FUNCtion:WRECord:OPERate",            
":FUNCtion:WRECord:OPERate?",           ":FUNCtion:WREPlay:MODE",               ":FUNCtion:WREPlay:MODE?",              ":FUNCtion:WREPlay:INTerval",           
":FUNCtion:WREPlay:INTerval?",          ":FUNCtion:WREPlay:FSTart",             ":FUNCtion:WREPlay:FSTart?",            ":FUNCtion:WREPlay:FCURrent",           
":FUNCtion:WREPlay:FCURrent?",          ":FUNCtion:WREPlay:FEND",               ":FUNCtion:WREPlay:FEND?",              ":FUNCtion:WREPlay:FMAX?",              
":FUNCtion:WREPlay:OPERate",            ":FUNCtion:WREPlay:OPERate?",           ":FUNCtion:WREPlay:TTAG",               ":FUNCtion:WREPlay:TTAG?",              
":FUNCtion:WREPlay:CTAG?",              ":FUNCtion:WANalyze:MODE",              ":FUNCtion:WANalyze:MODE?",             ":FUNCtion:WANalyze:SOURce",            
":FUNCtion:WANalyze:SOURce?",           ":FUNCtion:WANalyze:FCURrent",          ":FUNCtion:WANalyze:FCURrent?",         ":FUNCtion:WANalyze:TDISp",             
":FUNCtion:WANalyze:TDISp?",            ":FUNCtion:WANalyze:SETup:SSTart",      ":FUNCtion:WANalyze:SETup:SSTart?",     ":FUNCtion:WANalyze:SETup:SSENd",       
":FUNCtion:WANalyze:SETup:SSENd?",      ":FUNCtion:WANalyze:SETup:SFRame",      ":FUNCtion:WANalyze:SETup:SFRame?",     ":FUNCtion:WANalyze:SETup:EFRame",      
":FUNCtion:WANalyze:SETup:EFRame?",     ":FUNCtion:WANalyze:SETup:THReshold",   ":FUNCtion:WANalyze:SETup:THReshold?",  ":FUNCtion:WANalyze:SETup:XMASk",       
":FUNCtion:WANalyze:SETup:XMASk?",      ":FUNCtion:WANalyze:SETup:YMASk",       ":FUNCtion:WANalyze:SETup:YMASk?",      ":FUNCtion:WANalyze:STEMplate",         
":FUNCtion:WANalyze:CMASk",             ":FUNCtion:WANalyze:STARt",             ":FUNCtion:WANalyze:PREVious",          ":FUNCtion:WANalyze:NEXT",              
":FUNCtion:WANalyze:EFCount?",          ":FUNCtion:WANalyze:ECURrent",          ":FUNCtion:WANalyze:ECURrent?",         ":FUNCtion:WANalyze:ECDiff?",           
":LAN:DHCP",                            ":LAN:DHCP?",                           ":LAN:AUToip",                          ":LAN:AUToip?",                         
":LAN:GATeway",                         ":LAN:GATeway?",                        ":LAN:DNS",                             ":LAN:DNS?",                            
":LAN:MAC?",                            ":LAN:MANual",                          ":LAN:MANual?",                         ":LAN:INITiate",                        
":LAN:IPADdress",                       ":LAN:IPADdress?",                      ":LAN:SMASk",                           ":LAN:SMASk?",                          
":LAN:STATus?",                         ":LAN:VISA?",                           ":LAN:APPLy",                           ":MASK:ENABle",                         
":MASK:ENABle?",                        ":MASK:SOURce",                         ":MASK:SOURce?",                        ":MASK:OPERate",                        
":MASK:OPERate?",                       ":MASK:MDISplay",                       ":MASK:MDISplay?",                      ":MASK:SOOutput",                       
":MASK:SOOutput?",                      ":MASK:OUTPut",                         ":MASK:OUTPut?",                        ":MASK:X",                              
":MASK:X?",                             ":MASK:Y",                              ":MASK:Y?",                             ":MASK:CREate",                         
":MASK:PASSed?",                        ":MASK:FAILed?",                        ":MASK:TOTal?",                         ":MASK:RESet",                          
":MASK:DATA",                           ":MASK:DATA?",                          ":MEASure:SOURce",                      ":MEASure:SOURce?",                     
":MEASure:COUNter:SOURce",              ":MEASure:COUNter:SOURce?",             ":MEASure:COUNter:VALue?",              ":MEASure:CLEar",                       
":MEASure:RECover",                     ":MEASure:ADISplay",                    ":MEASure:ADISplay?",                   ":MEASure:AMSource",                    
":MEASure:AMSource?",                   ":MEASure:STATistic:DISPlay",           ":MEASure:STATistic:DISPlay?",          ":MEASure:STATistic:MODE",              
":MEASure:STATistic:MODE?",             ":MEASure:STATistic:RESet",             ":MEASure:SETup:TYPE",                  ":MEASure:SETup:TYPE?",                 
":MEASure:SETup:MAX",                   ":MEASure:SETup:MAX?",                  ":MEASure:SETup:MID",                   ":MEASure:SETup:MID?",                  
":MEASure:SETup:MIN",                   ":MEASure:SETup:MIN?",                  ":MEASure:AREA",                        ":MEASure:AREA?",                       
":MEASure:CREGion:CAX",                 ":MEASure:CREGion:CAX?",                ":MEASure:CREGion:CBX",                 ":MEASure:CREGion:CBX?",                
":MEASure:HISTory:DISPlay",             ":MEASure:HISTory:DISPlay?",            ":MEASure:HISTory:DMODe",               ":MEASure:HISTory:DMODe?",              
":MEASure:FDELay?",                     ":MEASure:FDELay:SMAXimum?",            ":MEASure:FDELay:SMINimum?",            ":MEASure:FDELay:SCURrent?",            
":MEASure:FDELay:SAVerage?",            ":MEASure:FDELay:SDEViation?",          ":MEASure:FPHase?",                     ":MEASure:FPHase:SMAXimum?",            
":MEASure:FPHase:SMINimum?",            ":MEASure:FPHase:SCURrent?",            ":MEASure:FPHase:SAVerage?",            ":MEASure:FPHase:SDEViation?",          
":MEASure:FREQuency?",                  ":MEASure:FREQuency:SMAXimum?",         ":MEASure:FREQuency:SMINimum?",         ":MEASure:FREQuency:SCURrent?",         
":MEASure:FREQuency:SAVerage?",         ":MEASure:FREQuency:SDEViation?",       ":MEASure:FTIMe?",                      ":MEASure:FTIMe:SMAXimum?",             
":MEASure:FTIMe:SMINimum?",             ":MEASure:FTIMe:SCURrent?",             ":MEASure:FTIMe:SAVerage?",             ":MEASure:FTIMe:SDEViation?",           
":MEASure:NDUTy?",                      ":MEASure:NDUTy:SMAXimum?",             ":MEASure:NDUTy:SMINimum?",             ":MEASure:NDUTy:SCURrent?",             
":MEASure:NDUTy:SAVerage?",             ":MEASure:NDUTy:SDEViation?",           ":MEASure:NWIDth?",                     ":MEASure:NWIDth:SMAXimum?",            
":MEASure:NWIDth:SMAXimum?",            ":MEASure:NWIDth:SCURrent?",            ":MEASure:NWIDth:SAVerage?",            ":MEASure:NWIDth:SDEViation?",          
":MEASure:OVERshoot?",                  ":MEASure:OVERshoot:SMAXimum?",         ":MEASure:OVERshoot:SMINimum?",         ":MEASure:OVERshoot:SCURrent?",         
":MEASure:OVERshoot:SAVerage?",         ":MEASure:OVERshoot:SDEViation?",       ":MEASure:PDUTy?",                      ":MEASure:PDUTy:SMAXimum?",             
":MEASure:PDUTy:SMINimum?",             ":MEASure:PDUTy:SCURrent?",             ":MEASure:PDUTy:SAVerage?",             ":MEASure:PDUTy:SDEViation?",           
":MEASure:PERiod?",                     ":MEASure:PERiod:SMAXimum?",            ":MEASure:PERiod:SMINimum?",            ":MEASure:PERiod:SCURrent?",            
":MEASure:PERiod:SAVerage?",            ":MEASure:PERiod:SDEViation?",          ":MEASure:PREShoot?",                   ":MEASure:PREShoot:SMAXimum?",          
":MEASure:PREShoot:SMINimum?",          ":MEASure:PREShoot:SCURrent?",          ":MEASure:PREShoot:SAVerage?",          ":MEASure:PREShoot:SDEViation?",        
":MEASure:PWIDth?",                     ":MEASure:PWIDth:SMAXimum?",            ":MEASure:PWIDth:SMINimum?",            ":MEASure:PWIDth:SCURrent?",            
":MEASure:PWIDth:SAVerage?",            ":MEASure:PWIDth:SDEViation?",          ":MEASure:RTIMe?",                      ":MEASure:RTIMe:SMAXimum?",             
":MEASure:RTIMe:SMINimum?",             ":MEASure:RTIMe:SCURrent?",             ":MEASure:RTIMe:SAVerage?",             ":MEASure:RTIMe:SDEViation?",           
":MEASure:RDELay?",                     ":MEASure:RDELay:SMAXimum?",            ":MEASure:RDELay:SMINimum?",            ":MEASure:RDELay:SCURrent?",            
":MEASure:RDELay:SAVerage?",            ":MEASure:RDELay:SDEViation?",          ":MEASure:RPHase?",                     ":MEASure:RPHase:SMAXimum?",            
":MEASure:RPHase:SMINimum?",            ":MEASure:RPHase:SCURrent?",            ":MEASure:RPHase:SAVerage?",            ":MEASure:RPHase:SDEViation?",          
":MEASure:VAMP?",                       ":MEASure:VAMP:SMAXimum?",              ":MEASure:VAMP:SMINimum?",              ":MEASure:VAMP:SCURrent?",              
":MEASure:VAMP:SAVerage?",              ":MEASure:VAMP:SDEViation?",            ":MEASure:VAVG?",                       ":MEASure:VAVG:SMAXimum?",              
":MEASure:VAVG:SMINimum?",              ":MEASure:VAVG:SCURrent?",              ":MEASure:VAVG:SAVerage?",              ":MEASure:VAVG:SDEViation?",            
":MEASure:VBASe?",                      ":MEASure:VBASe:SMAXimum?",             ":MEASure:VBASe:SMINimum?",             ":MEASure:VBASe:SCURrent?",             
":MEASure:VBASe:SAVerage?",             ":MEASure:VBASe:SDEViation?",           ":MEASure:VMAX?",                       ":MEASure:VMAX:SMAXimum?",              
":MEASure:VMAX:SMINimum?",              ":MEASure:VMAX:SCURrent?",              ":MEASure:VMAX:SMAXimum?",              ":MEASure:VMAX:SDEViation?",            
":MEASure:VMIN?",                       ":MEASure:VMIN:SMAXimum?",              ":MEASure:VMIN:SMINimum?",              ":MEASure:VMIN:SCURrent?",              
":MEASure:VMIN:SAVerage?",              ":MEASure:VMIN:SDEViation?",            ":MEASure:VPP?",                        ":MEASure:VPP:SMAXimum?",               
":MEASure:VPP:SMINimum?",               ":MEASure:VPP:SCURrent?",               ":MEASure:VPP:SAVerage?",               ":MEASure:VPP:SDEViation?",             
":MEASure:VRMS?",                       ":MEASure:VRMS:SMAXimum?",              ":MEASure:VRMS:SMINimum?",              ":MEASure:VRMS:SCURrent?",              
":MEASure:VRMS:SAVerage?",              ":MEASure:VRMS:SDEViation?",            ":MEASure:VTOP?",                       ":MEASure:VTOP:SMAXimum?",              
":MEASure:VTOP:SMINimum?",              ":MEASure:VTOP:SCURrent?",              ":MEASure:VTOP:SAVerage?",              ":MEASure:VTOP:SDEViation?",            
":SYSTem:BEEPer",                       ":SYSTem:BEEPer?",                      ":SYSTem:DATE",                         ":SYSTem:DATE?",                        
":SYSTem:TIME",                         ":SYSTem:TIME?",                        ":SYSTem:ERRor?",                       ":SYSTem:ERRor:NEXT?",                  
":SYSTem:EXPand",                       ":SYSTem:EXPand?",                      ":SYSTem:LANGuage",                     ":SYSTem:LANGuage?",                    
":SYSTem:PON",                          ":SYSTem:PON?",                         ":SYSTem:SSAVer:TIME",                  ":SYSTem:SSAVer:TIME?",                 
":SYSTem:SETup",                        ":SYSTem:SETup?",                       ":SYSTem:VERSion?",                     ":SYSTem:AOUTput",                      
":SYSTem:AOUTput?",                     ":SYSTem:RESet",                        ":SYSTem:OPTion:INSTall",               ":SYSTem:OPTion:UNINSTall",             
":SYSTem:UDEVice",                      ":SYSTem:UDEVice?",                     ":SYSTem:GPIB",                         ":SYSTem:GPIB?",                        
":SYSTem:GAMount?",                     ":SYSTem:RAMount?",                     ":TIMebase:DELay:ENABle",               ":TIMebase:DELay:ENABle?",              
":TIMebase:DELay:OFFSet",               ":TIMebase:DELay:OFFSet?",              ":TIMebase:DELay:SCALe",                ":TIMebase:DELay:SCALe?",               
":TIMebase:OFFSet",                     ":TIMebase:MAIN:OFFSet",                ":TIMebase:OFFSet?",                    ":TIMebase:MAIN:OFFSet?",               
":TIMebase:SCALe",                      ":TIMebase:MAIN:SCALe",                 ":TIMebase:SCALe?",                     ":TIMebase:MAIN:SCALe?",                
":TIMebase:MODE",                       ":TIMebase:MODE?",                      ":TIMebase:HREF:MODE",                  ":TIMebase:HREF:MODE?",                 
":TIMebase:HREF:POSition",              ":TIMebase:HREF:POSition?",             ":TIMebase:VERNier",                    ":TIMebase:VERNier?",                   
":TRIGger:MODE",                        ":TRIGger:MODE?",                       ":TRIGger:COUPling",                    ":TRIGger:COUPling?",                   
":TRIGger:STATus?",                     ":TRIGger:SWEep",                       ":TRIGger:SWEep?",                      ":TRIGger:HOLDoff",                     
":TRIGger:HOLDoff?",                    ":TRIGger:NREJect",                     ":TRIGger:NREJect?",                    ":TRIGger:EDGe:SOURce",                 
":TRIGger:EDGe:SOURce?",                ":TRIGger:EDGe:SLOPe",                  ":TRIGger:EDGe:SLOPe?",                 ":TRIGger:EDGe:LEVel",                  
":TRIGger:EDGe:LEVel?",                 ":TRIGger:PULSe:SOURce",                ":TRIGger:PULSe:SOURce?",               ":TRIGger:PULSe:WHEN",                  
":TRIGger:PULSe:WHEN?",                 ":TRIGger:PULSe:UWIDth",                ":TRIGger:PULSe:UWIDth?",               ":TRIGger:PULSe:LWIDth",                
":TRIGger:PULSe:LWIDth?",               ":TRIGger:PULSe:LEVel",                 ":TRIGger:PULSe:LEVel?",                ":TRIGger:RUNT:SOURce",                 
":TRIGger:RUNT:SOURce?",                ":TRIGger:RUNT:POLarity",               ":TRIGger:RUNT:POLarity?",              ":TRIGger:RUNT:WHEN",                   
":TRIGger:RUNT:WHEN?",                  ":TRIGger:RUNT:WLOWer",                 ":TRIGger:RUNT:WLOWer?",                ":TRIGger:RUNT:WUPPer",                 
":TRIGger:RUNT:WUPPer?",                ":TRIGger:RUNT:ALEVel",                 ":TRIGger:RUNT:ALEVel?",                ":TRIGger:RUNT:BLEVel",                 
":TRIGger:RUNT:BLEVel?",                ":TRIGger:WINDows:SOURce",              ":TRIGger:WINDows:SOURce?",             ":TRIGger:WINDows:SLOPe",               
":TRIGger:RUNT:SLOPe?",                 ":TRIGger:WINDows:POSition",            ":TRIGger:RUNT:POSition?",              ":TRIGger:WINDows:TIMe",                
":TRIGger:RUNT:TIMe?",                  ":TRIGger:NEDGe:SOURce",                ":TRIGger:NEDGe:SOURce?",               ":TRIGger:NEDGe:SLOPe",                 
":TRIGger:NEDGe:SLOPe?",                ":TRIGger:NEDGe:IDLE",                  ":TRIGger:NEDGe:IDLE?",                 ":TRIGger:NEDGe:EDGE",                  
":TRIGger:NEDGe:EDGE?",                 ":TRIGger:NEDGe:LEVel",                 ":TRIGger:NEDGe:LEVel?",                ":TRIGger:SLOPe:SOURce",                
":TRIGger:SLOPe:SOURce?",               ":TRIGger:SLOPe:WHEN",                  ":TRIGger:SLOPe:WHEN?",                 ":TRIGger:SLOPe:TUPPer",                
":TRIGger:SLOPe:TUPPer?",               ":TRIGger:SLOPe:TLOWer",                ":TRIGger:SLOPe:TLOWer?",               ":TRIGger:SLOPe:WINDow",                
":TRIGger:SLOPe:WINDow?",               ":TRIGger:SLOPe:ALEVel",                ":TRIGger:SLOPe:ALEVel?",               ":TRIGger:SLOPe:BLEVel",                
":TRIGger:SLOPe:BLEVel?",               ":TRIGger:VIDeo:SOURce",                ":TRIGger:VIDeo:SOURce?",               ":TRIGger:VIDeo:POLarity",              
":TRIGger:VIDeo:POLarity?",             ":TRIGger:VIDeo:MODE",                  ":TRIGger:VIDeo:MODE?",                 ":TRIGger:VIDeo:LINE",                  
":TRIGger:VIDeo:LINE?",                 ":TRIGger:VIDeo:STANdard",              ":TRIGger:VIDeo:STANdard?",             ":TRIGger:VIDeo:LEVel",                 
":TRIGger:VIDeo:LEVel?",                ":TRIGger:PATTern:PATTern",             ":TRIGger:PATTern:PATTern?",            ":TRIGger:PATTern:LEVel",               
":TRIGger:PATTern:LEVel?",              ":TRIGger:DELay:SA",                    ":TRIGger:DELay:SA?",                   ":TRIGger:DELay:SLOPA",                 
":TRIGger:DELay:SLOPA?",                ":TRIGger:DELay:SB",                    ":TRIGger:DELay:SB?",                   ":TRIGger:DELay:SLOPB",                 
":TRIGger:DELay:SLOPB?",                ":TRIGger:DELay:TYPe",                  ":TRIGger:DELay:TYPe?",                 ":TRIGger:DELay:TUPPer",                
":TRIGger:DELay:TUPPer?",               ":TRIGger:DELay:TLOWer",                ":TRIGger:DELay:TLOWer?",               ":TRIGger:TIMeout:SOURce",              
":TRIGger:TIMeout:SOURce?",             ":TRIGger:TIMeout:SLOPe",               ":TRIGger:TIMeout:SLOPe?",              ":TRIGger:TIMeout:TIMe",                
":TRIGger:TIMeout:TIMe?",               ":TRIGger:DURATion:SOURce",             ":TRIGger:DURATion:SOURce?",            ":TRIGger:DURATion:TYPe",               
":TRIGger:DURATion:TYPe?",              ":TRIGger:DURATion:WHEN",               ":TRIGger:DURATion:WHEN?",              ":TRIGger:DURATion:TUPPer",             
":TRIGger:DURATion:TUPPer?",            ":TRIGger:DURATion:TLOWer",             ":TRIGger:DURATion:TLOWer?",            ":TRIGger:SHOLd:DSrc",                  
":TRIGger:SHOLd:DSrc?",                 ":TRIGger:SHOLd:CSrc",                  ":TRIGger:SHOLd:CSrc?",                 ":TRIGger:SHOLd:SLOPe",                 
":TRIGger:SHOLd:SLOPe?",                ":TRIGger:SHOLd:PATTern",               ":TRIGger:SHOLd:PATTern?",              ":TRIGger:SHOLd:TYPe",                  
":TRIGger:SHOLd:TYPe?",                 ":TRIGger:SHOLd:STIMe",                 ":TRIGger:SHOLd:STIMe?",                ":TRIGger:SHOLd:HTIMe",                 
":TRIGger:SHOLd:HTIMe?",                ":TRIGger:RS232:SOURce",                ":TRIGger:RS232:SOURce?",               ":TRIGger:RS232:WHEN",                  
":TRIGger:RS232:WHEN?",                 ":TRIGger:RS232:PARity",                ":TRIGger:RS232:PARity?",               ":TRIGger:RS232:STOP",                  
":TRIGger:RS232:STOP?",                 ":TRIGger:RS232:DATA",                  ":TRIGger:RS232:DATA?",                 ":TRIGger:RS232:WIDTh",                 
":TRIGger:RS232:WIDTh?",                ":TRIGger:RS232:BAUD",                  ":TRIGger:RS232:BAUD?",                 ":TRIGger:RS232:BUSer",                 
":TRIGger:RS232:BUSer?",                ":TRIGger:RS232:LEVel",                 ":TRIGger:RS232:LEVel?",                ":TRIGger:IIC:SCL",                     
":TRIGger:IIC:SCL?",                    ":TRIGger:IIC:SDA",                     ":TRIGger:IIC:SDA?",                    ":TRIGger:IIC:WHEN",                    
":TRIGger:IIC:WHEN?",                   ":TRIGger:IIC:AWIDth",                  ":TRIGger:IIC:AWIDth?",                 ":TRIGger:IIC:ADDRess",                 
":TRIGger:IIC:ADDRess?",                ":TRIGger:IIC:DIRection",               ":TRIGger:IIC:DIRection?",              ":TRIGger:IIC:DATA",                    
":TRIGger:IIC:DATA?",                   ":TRIGger:IIC:CLEVel",                  ":TRIGger:IIC:CLEVel?",                 ":TRIGger:IIC:DLEVel",                  
":TRIGger:IIC:DLEVel?",                 ":TRIGger:SPI:SCL",                     ":TRIGger:SPI:SCL?",                    ":TRIGger:SPI:SDA",                     
":TRIGger:SPI:SDA?",                    ":TRIGger:SPI:WIDTh",                   ":TRIGger:SPI:WIDTh?",                  ":TRIGger:SPI:DATA",                    
":TRIGger:SPI:DATA?",                   ":TRIGger:SPI:TIMeout",                 ":TRIGger:SPI:TIMeout?",                ":TRIGger:SPI:SLOPe",                   
":TRIGger:SPI:SLOPe?",                  ":TRIGger:SPI:CLEVel",                  ":TRIGger:SPI:CLEVel?",                 ":TRIGger:SPI:DLEVel",                  
":TRIGger:SPI:DLEVel?",                 ":TRIGger:USB:DPLus",                   ":TRIGger:USB:DPLus?",                  ":TRIGger:USB:DMINus",                  
":TRIGger:USB:DMINus?",                 ":TRIGger:USB:SPEed",                   ":TRIGger:USB:SPEed?",                  ":TRIGger:USB:WHEN",                    
":TRIGger:USB:WHEN?",                   ":TRIGger:USB:PLEVel",                  ":TRIGger:USB:PLEVel?",                 ":TRIGger:USB:MLEVel",                  
":TRIGger:USB:MLEVel?",                 ":WAVeform:SOURce",                     ":WAVeform:SOURce?",                    ":WAVeform:MODE",                       
":WAVeform:MODE?",                      ":WAVeform:FORMat",                     ":WAVeform:FORMat?",                    ":WAVeform:POINts",                     
":WAVeform:POINts?",                    ":WAVeform:DATA?",                      ":WAVeform:XINCrement?",                ":WAVeform:XORigin?",                   
":WAVeform:XREFerence?",                ":WAVeform:YINCrement?",                ":WAVeform:YORigin?",                   ":WAVeform:YREFerence?",                
":WAVeform:STARt",                      ":WAVeform:STARt?",                     ":WAVeform:STOP",                       ":WAVeform:STOP?",                      
":WAVeform:BEGin",                      ":WAVeform:END",                        ":WAVeform:RESet",                      ":WAVeform:PREamble?",                  
":WAVeform:STATus?",                    NULL
};


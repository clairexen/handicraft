#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <signal.h>
#include <unistd.h>
#include <math.h>

#include "bbclientlib.h"

int got_sigint = 0;
usbDevice_t *dev;

void sigint_handler(int unused)
{
	got_sigint = 1;
}

static int test001()
{
	struct bblab_config_t cfg = {
		.clkdiv = 4,

		.outputmask = 0x0f,
		.outlen = 128,
		.outidx = 0,
		.digioutptr = sizeof(cfg),
		
		.samplelen = 20,
		.sampleidx = 0,
		.digisampleptr = sizeof(cfg)+64,
		.adcsampleptr = { sizeof(cfg)+64+10+0*20, sizeof(cfg)+64+10+1*20, sizeof(cfg)+64+10+2*20, sizeof(cfg)+64+10+3*20 }
	};

	uint8_t digibuf[64] = { /* zeros */ };

	printf("Total mem usage on device: %d bytes\n", (int)(sizeof(cfg)+64+10+4*20));

	for (int i = 0; i < 128; i++) {
		uint8_t bits = i & 0xf;
		if ((i & 1) != 0)
			digibuf[i >> 1] |= bits << 4;
		else
			digibuf[i >> 1] |= bits;
	}

	if (bblabStateGet(dev) != STATE_HALT) {
		bblabStateSet(dev, STATE_SHUTDOWN);
		while (bblabStateGet(dev) != STATE_HALT)
			usleep(1000);
	}

	fprintf(stderr, "Config to device:\n");
	bblabConfigDump(&cfg, "   ");

	bblabMemSet(dev, cfg.digioutptr, &digibuf, sizeof(digibuf));

	bblabConfigConvert(&cfg);
	bblabMemSet(dev, 0, &cfg, sizeof(cfg));

	bblabStateSet(dev, STATE_CONFIG);
	do {
		usleep(10000);
	} while (bblabStateGet(dev) != STATE_RECORD && bblabStateGet(dev) != STATE_STANDBY);

	printf("Running. Press ENTER..");
	fflush(stdout);
	getchar();

	bblabStateSet(dev, STATE_SHUTDOWN);
	do {
		usleep(10000);
	} while (bblabStateGet(dev) != STATE_HALT);

	bblabMemGet(dev, 0, &cfg, sizeof(cfg));
	bblabConfigConvert(&cfg);

	fprintf(stderr, "Config from device:\n");
	bblabConfigDump(&cfg, "   ");

	uint8_t samplebufdigi[10];
	uint8_t samplebufadc1[20];
	uint8_t samplebufadc2[20];
	uint8_t samplebufadc3[20];
	uint8_t samplebufadc4[20];

	bblabMemGet(dev, cfg.digisampleptr, &samplebufdigi, 10);
	bblabMemGet(dev, cfg.adcsampleptr[0], &samplebufadc1, 20);
	bblabMemGet(dev, cfg.adcsampleptr[1], &samplebufadc2, 20);
	bblabMemGet(dev, cfg.adcsampleptr[2], &samplebufadc3, 20);
	bblabMemGet(dev, cfg.adcsampleptr[3], &samplebufadc4, 20);

	{
		int i = cfg.sampleidx, j = 0;
		while (1)
		{
			uint8_t bits = samplebufdigi[i >> 1];
			if ((i & 1) != 0)
				bits = bits >> 4;
			else
				bits = bits & 0x0f;

			printf("%3d: Digi=%d%d%d%d", i, bits & 8 ? 1 : 0, bits & 4 ? 1 : 0, bits & 2 ? 1 : 0, bits & 1 ? 1 : 0);
			printf(samplebufadc1[i] == 0xff ? ", ADC1=***" : ", ADC1=%3d", samplebufadc1[i]);
			printf(samplebufadc2[i] == 0xff ? ", ADC2=***" : ", ADC2=%3d", samplebufadc2[i]);
			printf(samplebufadc3[i] == 0xff ? ", ADC3=***" : ", ADC3=%3d", samplebufadc3[i]);
			printf(samplebufadc4[i] == 0xff ? ", ADC4=***" : ", ADC4=%3d", samplebufadc4[i]);

			for (int k = 0; k < 4; k++) {
				if (j > cfg.collision_tick[k] && j < cfg.collision_tick[k] + cfg.collision_duration[k])
					printf("  **%d**", k+1);
			}

			printf("\n");

			i = i == 0 ? cfg.samplelen-1 : i-1;
			if (i == cfg.sampleidx)
				break;

			j++;
		}
	}

	for (int i = 0; i < 4; i++)
		printf("Collision #%d was %d ticks ago and %d ticks long.\n", i+1, cfg.collision_tick[i], cfg.collision_duration[i]);

	return 0;
}

static int test002()
{
	struct bblab_config_t cfg = {
		.clkdiv = 4,
		.outputmask = 0x01,
		.outlen = 128,
		.outidx = 0,
		.pwmoutptr = sizeof(cfg),
		.digioutptr = sizeof(cfg) + 128,
	};

	uint8_t pwmbuf[128];
	uint8_t digibuf[64];

	printf("Total mem usage on device: %d bytes\n", (int)(sizeof(cfg)+128+64));

	for (int i = 0; i < 128; i++) {
		pwmbuf[i] = 128 + 32 * sin(2*M_PI*i/128.0);
	}

	for (int i = 0; i < 64; i++) {
		digibuf[i] = i >= 16 && i < (16+32) ? 0x11 : 0x00;
	}

	if (bblabStateGet(dev) != STATE_HALT) {
		bblabStateSet(dev, STATE_SHUTDOWN);
		while (bblabStateGet(dev) != STATE_HALT)
			usleep(1000);
	}

	bblabMemSet(dev, cfg.pwmoutptr, &pwmbuf, sizeof(pwmbuf));
	bblabMemSet(dev, cfg.digioutptr, &digibuf, sizeof(digibuf));

	bblabConfigConvert(&cfg);
	bblabMemSet(dev, 0, &cfg, sizeof(cfg));

	bblabStateSet(dev, STATE_CONFIG);
	do {
		usleep(10000);
	} while (bblabStateGet(dev) != STATE_RECORD && bblabStateGet(dev) != STATE_STANDBY);

	printf("Running. Press ENTER..");
	fflush(stdout);
	getchar();

	bblabStateSet(dev, STATE_SHUTDOWN);
	do {
		usleep(10000);
	} while (bblabStateGet(dev) != STATE_HALT);

	return 0;
}

static int test003()
{
	struct bblab_config_t cfg = {
		.clkdiv = 4,

		.outlen = 32,
		.outidx = 0,
		.pwmoutptr = sizeof(cfg),

		.samplelen = 32,
		.sampleidx = 0,
		.adcsampleptr = { sizeof(cfg) + 1*32, sizeof(cfg) + 2*32, sizeof(cfg) + 3*32, sizeof(cfg) + 4*32 }
	};

	uint8_t buffer[328];

	printf("Total mem usage on device: %d bytes\n", (int)(sizeof(cfg) + 5*32));

	for (int i = 0; i < 32; i++)
		buffer[i] = i * 8;

	if (bblabStateGet(dev) != STATE_HALT) {
		bblabStateSet(dev, STATE_SHUTDOWN);
		while (bblabStateGet(dev) != STATE_HALT)
			usleep(1000);
	}

	bblabMemSet(dev, cfg.pwmoutptr, &buffer, sizeof(buffer));

	bblabConfigConvert(&cfg);
	bblabMemSet(dev, 0, &cfg, sizeof(cfg));

	bblabStateSet(dev, STATE_CONFIG);
	do {
		usleep(10000);
	} while (bblabStateGet(dev) != STATE_RECORD && bblabStateGet(dev) != STATE_STANDBY);

	printf("Running. Press ENTER..");
	fflush(stdout);
	getchar();

	bblabStateSet(dev, STATE_SHUTDOWN);
	do {
		usleep(10000);
	} while (bblabStateGet(dev) != STATE_HALT);

	bblabMemGet(dev, 0, &cfg, sizeof(cfg));
	bblabConfigConvert(&cfg);

	for (int i = 0; i < 4; i++) {
		printf("\n+--- ADC %d ------------------------------------------------------+\n", i);
		bblabMemGet(dev, cfg.adcsampleptr[i], buffer, 32);
		for (int j = 0; j < 8; j++) {
			putchar('|');
			for (int k = 0; k < 32; k++) {
				putchar(buffer[k] == 0xff ? '.' : (0xff-buffer[k])/32 == j ? '*' : ' ');
				putchar(buffer[k] == 0xff ? '.' : (0xff-buffer[k])/32 == j ? '*' : ' ');
			}
			putchar('|');
			putchar('\n');
		}
		putchar('|');
		for (int k = 0; k < 32; k+=2)
			printf(" %02x ", buffer[k]);
		putchar('|');
		printf("\n+----------------------------------------------------------------+\n");
	}

	putchar('\n');

	return 0;
}

static int test004()
{
	struct bblab_config_t cfg = {
		.clkdiv = 4,

		.trigger_init_ticks = 200,
		.trigger_mode_channel = TRIGMODE_GT_LE | CHANNEL_ANALOG1,
		.trigger_value = 0x60,
		.posttrig_nticks = 40,

		.outlen = 64,
		.outidx = 0,
		.pwmoutptr = sizeof(cfg),

		.samplelen = 128,
		.sampleidx = 0,
		.adcsampleptr = { sizeof(cfg) + 64, 0, 0, 0 }
	};

	uint8_t buffer[128];

	printf("Total mem usage on device: %d bytes\n", (int)(sizeof(cfg) + 64 + 128));

	for (int i = 0; i < 64; i++)
		buffer[i] = i * 4;

	if (bblabStateGet(dev) != STATE_HALT) {
		bblabStateSet(dev, STATE_SHUTDOWN);
		while (bblabStateGet(dev) != STATE_HALT)
			usleep(1000);
	}

	bblabMemSet(dev, cfg.pwmoutptr, &buffer, 64);

	bblabConfigConvert(&cfg);
	bblabMemSet(dev, 0, &cfg, sizeof(cfg));

	bblabStateSet(dev, STATE_CONFIG);

	printf("\033[H\033[JNOTE: This test requires that the IO Pin 1 is connected to the Analog output!");

	while (1)
	{
		do {
			usleep(100000);
		} while (bblabStateGet(dev) != STATE_STANDBY);

		bblabMemGet(dev, 0, &cfg, sizeof(cfg));
		bblabConfigConvert(&cfg);

		uint8_t buffer_raw[128];
		bblabMemGet(dev, cfg.adcsampleptr[0], buffer_raw, 128);

		for (int i = 0; i < 128; i++)
			buffer[i] = buffer_raw[(i + cfg.sampleidx) % 128];

		printf("\033[H");
		printf("\n\n\n     +--------------------------------------------------------------------------------------------------------------------------------+\n");
		printf("    |");
		for (int k = 0; k < 128; k+=4) {
			int v = buffer[k];
			v = v > buffer[k+1] ? v : buffer[k+1];
			v = v > buffer[k+2] ? v : buffer[k+2];
			v = v > buffer[k+3] ? v : buffer[k+3];
			printf(" %02x ", buffer[k]);
		}
		printf("|\n");
		for (int j = 0; j < 16; j++) {
			printf("    |");
			for (int k = 0; k < 128; k++) {
				putchar(buffer[k] == 0xff ? '.' : (0xff-buffer[k])/16 == j ? '*' : k == 128 - cfg.posttrig_nticks && (j%2) == 0 ? '|' : ' ' );
			}
			printf("|\n");
		}
		printf("    |");
		for (int k = 0; k < 128; k+=4) {
			int v = buffer[k];
			v = v < buffer[k+1] ? v : buffer[k+1];
			v = v < buffer[k+2] ? v : buffer[k+2];
			v = v < buffer[k+3] ? v : buffer[k+3];
			printf(" %02x ", buffer[k]);
		}
		printf("|\n");
		printf("    +--------------------------------------------------------------------------------------------------------------------------------+\n");

		bblabStateSet(dev, STATE_RECORD);
	}

	return 0;
}

int main(int argc, char **argv)
{
	if ((dev = bblabOpen()) == NULL)
		exit(1);

	if (argc == 2 && !strcasecmp(argv[1], "cfgsz")) {
		printf("Config Size: %d\n", (int)(sizeof(struct bblab_config_t)));
		return 0;
	}
		
	if (argc == 2 && !strcasecmp(argv[1], "test001"))
		return test001();
		
	if (argc == 2 && !strcasecmp(argv[1], "test002"))
		return test002();
		
	if (argc == 2 && !strcasecmp(argv[1], "test003"))
		return test003();
		
	if (argc == 2 && !strcasecmp(argv[1], "test004"))
		return test004();
		
	signal(SIGINT, &sigint_handler);

	while (1)
	{
		char line[1024];

		printf("bbshell> ");
		fflush(stdout);

		if (fgets(line, 1024, stdin) == NULL)
			break;

		char *cmd = strtok(line, " \r\n\t");
		if (cmd == NULL)
			continue;

		if (strcasecmp(cmd, "exit") == 0 || strcasecmp(cmd, "quit") == 0) {
			break;
		}

		if (strcasecmp(cmd, "wr") == 0 || strcasecmp(cmd, "write") == 0)
		{
			uint8_t buffer[1024];
			int addr = atoi(strtok(NULL, " \r\n\t") ?: ""), len = 0;

			while (len < 1024) {
				char *p = strtok(NULL, " \r\n\t");
				if (p == NULL)
					break;
				buffer[len++] = atoi(p);
			}
			
			printf("# write addr=0x%x, len=0x%x\n", addr, len);
			bblabMemSet(dev, addr, buffer, len);
			continue;
		}

		if (strcasecmp(cmd, "rd") == 0 || strcasecmp(cmd, "read") == 0)
		{
			uint8_t buffer[1024];
			int addr = atoi(strtok(NULL, " \r\n\t") ?: "");
			int len = atoi(strtok(NULL, " \r\n\t") ?: "");

			printf("# read addr=0x%x, len=0x%x\n", addr, len);
			bblabMemGet(dev, addr, buffer, len);
			for (int i=0; i<len; i++) {
				printf("%s0x%02x", i > 0 ? " " : "", buffer[i]);
			}
			printf("\n");
			continue;
		}

		if (strcasecmp(cmd, "st") == 0 || strcasecmp(cmd, "state") == 0)
		{
			uint8_t new_state = atoi(strtok(NULL, " \r\n\t") ?: "");

			if (new_state != 0) {
				printf("# set state %d\n", new_state);
				bblabStateSet(dev, new_state);
				continue;
			}

			printf("# monitoring state (press Ctrl-C to stop)\n");
			got_sigint = 0;
			uint8_t old_state = bblabStateGet(dev);
			printf("Current engine state: %d\n", old_state);
			while (!got_sigint) {
				new_state = bblabStateGet(dev);
				if (old_state != new_state) {
					printf("New engine state: %d\n", new_state);
					old_state = new_state;
				}
				usleep(100000);
			}
			printf("\n");
			continue;
		}

		if (strcasecmp(cmd, "p") == 0 || strcasecmp(cmd, "parameter") == 0)
		{
			uint8_t para = atoi(strtok(NULL, " \r\n\t") ?: "");

			printf("# parameter %d = %d\n", para, bblabParameter(dev, para));
			continue;
		}
	}
	
	usbhidCloseDevice(dev);
	return 0;
}


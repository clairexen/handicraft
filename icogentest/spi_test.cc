#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <wiringPiSPI.h>
#include <wiringPi.h>

#define SWIPE 1
#define VERBOSITY 1

#define SPI_PIN_CS0 0
#define SPI_PIN_CS1 1

int errcount = 0;
int total_errcount = 0;

void do_spi_csel(uint8_t v)
{
	digitalWrite(SPI_PIN_CS0, !(v & 1) ? LOW : HIGH);
	digitalWrite(SPI_PIN_CS1, !(v & 2) ? LOW : HIGH);

	if (VERBOSITY >= 2)
		printf("SPI: Select EP %d\n", v);
}

void do_spi_xfer(uint8_t v, uint8_t e)
{
	uint8_t r = v;
	wiringPiSPIDataRW(0, &r, 1);

	if (VERBOSITY >= 2 || e != r)
		printf("SPI: XFER I=%d O=%d\n", v, r);

	if (e != r) {
		printf("-> Expected different output: %d\n", e);
		errcount++;
		sleep(1);
	}
}

void setup()
{
	wiringPiSetup();

	pinMode(SPI_PIN_CS0, OUTPUT);
	pinMode(SPI_PIN_CS1, OUTPUT);
	do_spi_csel(0);

	wiringPiSPISetupMode(0, 1000000, 3); // 1 MHz
}

void loop()
{
	printf("-----------------------------------------------\n");

	printf("Sending PWM width config..");
	fflush(stdout);
	do_spi_csel(2);
	do_spi_xfer(0, 0);
	for (int i = 0; i < 32; i++) {
		if (i == 0 && SWIPE)
			do_spi_xfer(0, 0);
		else
			do_spi_xfer(i*8 + 4, 0);
	}
	do_spi_csel(0);
	printf(" DONE.\n");

	printf("Sending PWM phase config..");
	fflush(stdout);
	do_spi_csel(3);
	do_spi_xfer(0, 0);
	for (int i = 0; i < 32; i++)
		do_spi_xfer(i*8 + 4, 0);
	do_spi_csel(0);
	printf(" DONE.\n");

	for (int j = 10; j < 256; j++) {
		if (VERBOSITY < 2) {
			if (j % 10 == 0) {
				if (j != 10) printf("\n");
				printf("## pkt len:");
			}
			printf(" %d", j);
			fflush(stdout);
		}
		do_spi_csel(1);
		do_spi_xfer(0, 143);
		do_spi_xfer(1, 42);
		for (int i = 2; i < j; i++)
			do_spi_xfer(i, i-2);
		do_spi_csel(0);
	}
	if (VERBOSITY < 2)
		printf("\n");

	printf("-> Error count (this run): %d\n", errcount);

	total_errcount += errcount;
	errcount = 0;

	printf("-> Error count (total): %d\n", total_errcount);

	if (VERBOSITY >= 1) {
		printf("Next round in");
		fflush(stdout);
		for (int i = 5; i >= 0; i--)  {
			if (SWIPE) {
				do_spi_csel(2);
				do_spi_xfer(0, 0);
				do_spi_xfer(100 + i*20, 0);
				do_spi_csel(0);
			}
			sleep(1);
			printf(" %d", i);
			fflush(stdout);
		}
		printf(" GO!\n");
	}
}

int main()
{
	setup();
	while (1) loop();
}


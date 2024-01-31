#include <SPI.h>

#define SWIPE 1
#define VERBOSITY 1

#define SPI_PIN_CS0   9
#define SPI_PIN_CS1  10
#define SPI_PIN_MOSI 11
#define SPI_PIN_MISO 12
#define SPI_PIN_SCK  13

int errcount = 0;
int total_errcount = 0;

void do_spi_csel(byte v)
{
	digitalWrite(SPI_PIN_CS0, !(v & 1) ? LOW : HIGH);
	digitalWrite(SPI_PIN_CS1, !(v & 2) ? LOW : HIGH);

	if (VERBOSITY >= 2) {
		Serial.print("SPI: Select EP ");
		Serial.println(v, DEC);
	}
}

void do_spi_xfer(byte v, byte e)
{
	byte r = SPI.transfer(v);

	if (VERBOSITY >= 2 || e != r) {
		Serial.print("SPI: XFER I=");
		Serial.print(v, DEC);
		Serial.print(" O=");
		Serial.println(r, DEC);
	}

	if (e != r) {
		Serial.print("-> Expected different output: ");
		Serial.println(e, DEC);
		errcount++;
		delay(1000);
	}
}

void setup()
{
	Serial.begin(115200);

	SPI.begin();
	// SPI MODE: CPOL=1, CPHA=1
	SPI.setDataMode(SPI_MODE3);
	SPI.setClockDivider(SPI_CLOCK_DIV16);

	pinMode(SPI_PIN_CS0, OUTPUT);
	pinMode(SPI_PIN_CS1, OUTPUT);
	do_spi_csel(0);
}

void loop()
{
	Serial.println("-----------------------------------------------");

	Serial.print("Sending PWM width config..");
	do_spi_csel(2);
	do_spi_xfer(0, 0);
	for (int i = 0; i < 32; i++) {
		if (i == 0 && SWIPE)
			do_spi_xfer(0, 0);
		else
			do_spi_xfer(i*8 + 4, 0);
	}
	do_spi_csel(0);
	Serial.println(" DONE.");

	Serial.print("Sending PWM phase config..");
	do_spi_csel(3);
	do_spi_xfer(0, 0);
	for (int i = 0; i < 32; i++)
		do_spi_xfer(i*8 + 4, 0);
	do_spi_csel(0);
	Serial.println(" DONE.");

	for (int j = 10; j < 256; j++) {
		if (VERBOSITY < 2) {
			if (j % 10 == 0) {
				if (j != 10) Serial.println("");
				Serial.print("## pkt len:");
			}
			Serial.print(" ");
			Serial.print(j, DEC);
		}
		do_spi_csel(1);
		do_spi_xfer(0, 143);
		do_spi_xfer(1, 42);
		for (int i = 2; i < j; i++)
			do_spi_xfer(i, i-2);
		do_spi_csel(0);
	}
	if (VERBOSITY < 2)
		Serial.println("");

	Serial.print("-> Error count (this run): ");
	Serial.println(errcount, DEC);

	total_errcount += errcount;
	errcount = 0;

	Serial.print("-> Error count (total): ");
	Serial.println(total_errcount, DEC);

	if (VERBOSITY >= 1) {
		Serial.print("Next round in");
		for (int i = 5; i >= 0; i--)  {
			if (SWIPE) {
				do_spi_csel(2);
				do_spi_xfer(0, 0);
				do_spi_xfer(100 + i*20, 0);
				do_spi_csel(0);
			}
			delay(1000);
			Serial.print(" ");
			Serial.print(i, DEC);
		}
		Serial.println(" GO!");
	}
}

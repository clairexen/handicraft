#include "lib_console.c"
#include "lib_pinio.c"
#include <util/delay.h>

#define IR_PIN 2
#define MAX_SIGNAL_LEN 128

uint8_t signalData[MAX_SIGNAL_LEN] = { /* zeros */ };
int signalLen;

void readSignal()
{
	while (digitalRead(IR_PIN) == 1) { /* wait */ }

	for (signalLen = 0; signalLen < MAX_SIGNAL_LEN; signalLen++) {
		signalData[signalLen] = 0;
		while (digitalRead(IR_PIN) == signalLen % 2) {
			if (++signalData[signalLen] == 255)
				return;
			_delay_us(10);
		}
	}
}

void printSignal()
{
	int i;
	
	for (i = 0; i < signalLen; i++)
		consolePrintf("%2d ", signalData[i]);
	
	consolePrintf("\n");
}

uint8_t max_hist = 0;
uint8_t hist[255];

void addSamplesToHistogram() 
{
	int i;
	
	// increment histogram slot for each unique sample value
	// use the most significant bit of each histogram slot to 
	// remember whether we already incremented it
	for (i = 0; i < signalLen; i++) {
		if ((hist[signalData[i]] & 0x80) == 0) {
			hist[signalData[i]]++;
			if (hist[signalData[i]] > max_hist)
				max_hist = hist[signalData[i]];
			hist[signalData[i]] |= 0x80;
		}
	}
	
	// clear flags
	for (i = 0; i < signalLen; i++)
		hist[signalData[i]] &= 0x7f; 
}

void printHistogram() 
{
	int i,j;
	
	consolePrint("\n----- HISTOGRAM -----\n");
	for (i = 0; i < 255; i++) {
		if (hist[i] == 0) {
			if (i > 0 && hist[i-1] != 0)
				consolePrint("\n");
			continue;
		}
		consolePrintf("%3d %3d ", i, hist[i]);
		for (j = 0; j < hist[i]; j++)
			consolePrint("=");
		consolePrint("\n");
	}
	
	consolePrint("\n");
}

void clearHistogram()
{
	int i;
	for (i = 0; i < 255; i++)
		hist[i] = 0;
	max_hist = 0;
}


void loop()
{
	// only use signal with correct start pulse
	do {
		readSignal();
	} while (signalLen < 5);
	printSignal();
	
	addSamplesToHistogram();
	consolePrintf("-- %d / 50 --\n", max_hist);

	if (max_hist >= 50) {
		printHistogram();
		clearHistogram();
	}
}

int main()
{
	pinMode(IR_PIN, INPUT);
	consoleInit(9600);
	consolePrint("Waiting for IR...\n");
	
	while(1)
		loop();
		
	return 0;
}

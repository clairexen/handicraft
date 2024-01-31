#include "lib_console.c"
#include "lib_pinio.c"
#include "lib_irsend.c"
#include <util/delay.h>

#define IR_PIN 2
#define LED_PIN 4
#define MAX_SIGNAL_LEN 128

uint8_t signalData[MAX_SIGNAL_LEN+1] = { /* zeros */ };
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

void mapSignalToString()
{
	int i;
	
	for (i = 0; i < signalLen; i++) {
		if (signalData[i] < 20)
			signalData[i] = 'A';
		else if (signalData[i] < 40)
			signalData[i] = 'B';
		else if (signalData[i] < 62)
			signalData[i] = 'C';
		else if (signalData[i] < 100)
			signalData[i] = 'D';
		else
			signalData[i] = 'E';
	}
	signalData[signalLen] = 0;
}

int matchString(char *str)
{
	uint8_t *p = signalData; 
	while (*p || *str)
		if (*p++ != *str++)
			return 0;
	return 1;
}

void loop()
{
	do {
		readSignal();
		mapSignalToString();
	} while (signalLen < 5);
	
	// consolePrintf("%s\n", signalData);

	if (matchString("DEBCBCBABABABABABCBCBCBABABABCBCBCBCBCBABABABABABABABABCBCBCBCBCBCB")) {
		consolePrint("REC\n");
	}

	if (matchString("DEBCBCBABABABABABCBCBCBABABABCBCBCBABABABABCBCBABABCBCBCBCBABABCBCB")) {
		consolePrint("PHOTO\n");
	}

	if (matchString("DEBCBCBABABABABABCBCBCBABABABCBCBCBABABCBCBCBABABABCBCBABABABCBCBCB")) {
		consolePrint("TELE\n");
	}

	if (matchString("DEBCBCBABABABABABCBCBCBABABABCBCBCBCBABCBCBCBABABABABCBABABABCBCBCB")) {
		consolePrint("WIDE\n");
	}

	if (matchString("DEBCBCBABABABABABCBCBCBABABABCBCBCBCBABCBABABCBABABABCBABCBCBABCBCB")) {
		consolePrint("MENU\n");
	}
	
	if (matchString("DEBCBCBABABABABABCBCBCBABABABCBCBCBCBABCBABABCBCBABABCBABCBCBABABCB")) {
		consolePrint("PLAY\n");
	}
	
	if (matchString("DEBCBCBABABABABABCBCBCBABABABCBCBCBABCBCBCBABABCBABCBABABABCBCBABCB")) {
		consolePrint("DISP\n");
	}
	
	if (matchString("DEBCBCBABABABABABCBCBCBABABABCBCBCBABABABABABCBCBABCBCBCBCBCBABABCB")) {
		consolePrint("UP\n");
	}
	
	if (matchString("DEBCBCBABABABABABCBCBCBABABABCBCBCBCBABABABABCBCBABABCBCBCBCBABABCB")) {
		consolePrint("DOWN\n");
	}
	
	if (matchString("DEBCBCBABABABABABCBCBCBABABABCBCBCBCBCBABABABCBCBABABABCBCBCBABABCB")) {
		consolePrint("LEFT\n");
	}
	
	if (matchString("DEBCBCBABABABABABCBCBCBABABABCBCBCBABCBABABABCBCBABCBABCBCBCBABABCB")) {
		consolePrint("RIGHT\n");
	}
	
	if (matchString("DEBCBCBABABABABABCBCBCBABABABCBCBCBABABCBABABCBCBABCBCBABCBCBABABCB")) {
		consolePrint("SET\n");
	}
	
	if (matchString("DEBCBCBABABABABABCBCBCBABABABCBCBCBCBCBCBABCBABABABABABABCBABCBCBCB")) {
		consolePrint("STOP\n");
	}
	
	if (matchString("DEBCBCBABABABABABCBCBCBABABABCBCBCBABABABABABABABCBCBCBCBCBCBCBCBAB")) {
		consolePrint("PAUSE\n");
	}
}

int main()
{
	pinMode(IR_PIN, INPUT);
	pinMode(LED_PIN, OUTPUT);
	
	consoleInit(9600);
	consolePrint("Waiting for IR...\n");		
		
	while(1)
		loop();
		
	return 0;
}

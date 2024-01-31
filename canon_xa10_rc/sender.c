#include "lib_console.c"
#include "lib_pinio.c"
#include "lib_irsend.c"
#include <util/delay.h>

#define IR_PIN 2
#define LED_PIN 4
#define BUTTON_PIN_TELE 15
#define BUTTON_PIN_WIDE 16
#define MAX_SIGNAL_LEN 128

void waitSteps(uint8_t n)
{
	for (uint8_t i = 0; i < n; i++)
		_delay_us(27);
}

void irSend(char *code)
{
	bool on_state = 0;
	while (*code) {
		on_state = !on_state;
		if (on_state)
			irsend_on();
		else
			irsend_off();
		switch (*code) {
		case 'A': waitSteps(15); break;
		case 'B': waitSteps(23); break;
		case 'C': waitSteps(55); break;
		case 'E': waitSteps(165); break;
		case 'D': waitSteps(200); waitSteps(130); break;
		}
		code++;
	}
	irsend_off();
	_delay_ms(1);
}

void loop()
{
	if (digitalRead(BUTTON_PIN_TELE) == 0)
	{
		consolePrint("TELE\n");
		irSend("DEBCBCBABABABABABCBCBCBABABABCBCBCBABABCBCBCBABABABCBCBABABABCBCBCB");
	}
	if (digitalRead(BUTTON_PIN_WIDE) == 0)
	{
		consolePrint("TELE\n");
		irSend("DEBCBCBABABABABABCBCBCBABABABCBCBCBCBABCBCBCBABABABABCBABABABCBCBCB");
	}
}

int main()
{
	pinMode(IR_PIN, INPUT);
	pinMode(LED_PIN, OUTPUT);
	
	pinMode(BUTTON_PIN_TELE, INPUT);
	pinMode(BUTTON_PIN_WIDE, INPUT);
	digitalWrite(BUTTON_PIN_TELE, 1);
	digitalWrite(BUTTON_PIN_WIDE, 1);
	
	consoleInit(9600);

	while(1)
		loop();

	return 0;
}

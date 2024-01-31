#include <stdint.h>
#include <stdbool.h>
#include <avr/io.h>
#include <avr/sleep.h>
#include <util/delay.h>
#include <avr/interrupt.h>
#include <avr/pgmspace.h>

void pwm_cycle(uint8_t value)
{
	uint8_t i;
	for (i = 0; i != 255; i++)
		PORTB = i < value ? 0x31 : 0x30;
}

void pwm_step(uint8_t value)
{
	uint8_t i;
	if ((PINB & 0x02) == 0) {
		PORTB = 0x10;
		for (i = 0; i < 100; i++) { }
	}
	else if ((PINB & 0x04) == 0) {
		PORTB = 0x21;
		for (i = 0; i < 100; i++) { }
	}
	else {
		for (i = 0; i < 10; i++)
			pwm_cycle(value);
	}
}

prog_uint8_t data1[] = {
63, 74, 76, 84, 84, 92, 92, 105, 105, 110, 116, 122, 122, 168, 168,
208, 222, 232, 233, 218, 218, 187, 181, 145, 138, 100, 100, 72, 72,
59, 59, 51, 53, 57, 59, 63, 68, 70, 73, 79, 84, 87, 91, 94, 99, 101,
106, 106, 113, 113, 120, 120, 137, 143, 186, 194, 223, 230, 229, 229,
207, 207, 173, 149, 130, 127, 91, 91, 71, 67, 59, 59, 54, 54, 54, 54,
66, 66, 71, 73, 81, 83, 90, 90, 98, 101, 102, 105, 110, 113, 116, 117,
122, 122, 151, 185, 205, 219, 231, 229, 229, 210, 196, 166, 148, 124,
122, 90, 90, 74, 70, 60, 61, 58, 58, 59, 60, 67, 66, 73, 73, 79, 79, 88,
88, 92, 92, 99, 108, 109, 109, 110, 117, 117, 126, 126, 139, 145, 199,
199, 229, 232, 231, 214, 195, 192, 153, 153, 113, 103, 74, 74, 57, 54,
44, 44, 36, 36, 47, 47, 53, 59, 59, 69, 75, 75, 84, 85, 90, 92, 100,
100, 107, 107, 113, 111, 117, 119, 127, 127, 132, 136, 136, 135, 141,
149, 149, 192, 192, 233, 233, 244, 241, 224, 224, 188, 176, 149, 149,
105, 105, 73, 59, 59, 48, 40, 37, 37, 37, 37, 45, 45, 51, 53, 63, 63,
72, 72, 82, 82, 91, 91, 100, 101, 108, 108, 111, 115, 120, 118, 123,
125, 128, 132, 133, 133, 140, 140, 140, 145, 145, 145, 150, 154, 167,
173, 219, 219, 242, 242, 242, 227, 221, 188, 183, 139, 139, 105, 92, 63,
63, 40, 40, 34, 34, 29, 29, 30, 31, 43, 43, 56, 56, 64, 72, 78, 84, 89,
92, 92, 99, 104, 108, 111, 113, 113, 118, 118, 123, 124, 126, 126, 133,
140, 138, 145, 145, 166, 179, 214, 218, 245, 245, 236, 235, 214, 214,
165, 165, 131, 98, 98, 73, 73, 53, 53, 38, 38, 39, 37, 43, 43, 51, 51,
54, 57, 57, 67, 73, 77, 85, 85, 94, 94, 100, 100, 107, 107, 114, 114,
122, 122, 124, 124, 130, 129, 134, 134, 134, 136, 136, 139, 150, 150,
186, 186, 233, 233, 250, 244, 238, 211, 211, 178, 169, 127, 120, 85,
60, 45, 45, 33, 26, 26, 23, 26, 31, 35, 42, 42, 54, 54, 64, 69, 74,
80, 87, 90, 99, 99, 115, 140, 140, 177, 191, 198, 197, 188, 188, 154,
154, 124, 120, 106, 106, 91, 91, 86, 88, 92, 91, 98, 96, 101, 101, 105,
108, 114, 114, 112, 112, 113, 112, 117, 115, 115, 116, 118, 114, 114,
115, 115, 120, 120, 115, 117, 123, 123, 127, 127, 129, 129, 131, 133,
134, 134, 137, 136, 141, 141, 135, 135, 138, 138, 139, 137, 137, 138,
134, 137, 156, 156, 204, 204, 231, 231, 228, 223, 185, 185, 136, 120,
82, 82, 37, 37, 23, 23, 14, 12, 11, 11, 22, 22, 34, 34, 47, 47, 68, 68,
75, 91, 94, 122, 122, 165, 175, 192, 192, 178, 178, 153, 142, 126, 101,
101, 82, 77, 77, 71, 73, 72, 72, 76, 80, 80, 88, 88, 97, 98, 106, 106,
112, 114, 120, 122, 123, 125, 129, 129, 131, 129, 132, 132, 134, 134,
134, 135, 140, 140, 136, 136, 136, 134, 136, 136, 132, 135, 132, 131,
132, 133, 133, 148, 153, 206, 206, 244, 244, 247, 222, 215, 178, 178,
132, 100, 100, 65, 38, 38, 10, 7, 0, 4, 4, 9, 15, 24, 25, 41, 41, 52,
62, 62, 74, 80, 80, 91, 99, 99, 105, 109, 111, 119, 119, 123, 129, 127,
135, 135, 150, 150, 187, 194, 243, 243, 255, 253, 219, 219, 182, 182,
127, 127, 79, 79, 55, 50, 39, 39, 34, 36, 38, 38, 50, 50, 59, 62, 73,
78, 89, 89, 96, 103, 107, 107, 116, 116, 122, 122, 122, 124, 127, 127,
136, 136, 138, 138, 159, 159, 204, 217, 250, 252, 244, 244, 206, 206,
174, 151, 127, 120, 83, 83, 59, 57, 52, 49, 50, 50
};

prog_uint8_t data2[] = {
255, 255, 247, 247, 247, 247, 247, 240, 240, 240, 240, 232, 232, 232,
232, 232, 232, 225, 225, 225, 225, 225, 225, 217, 217, 217, 217, 210,
210, 210, 210, 210, 210, 202, 202, 202, 202, 195, 202, 195, 195, 150,
195, 187, 187, 187, 187, 187, 187, 180, 180, 187, 180, 180, 172, 180,
180, 172, 172, 172, 172, 172, 172, 165, 165, 165, 165, 165, 165, 157,
165, 157, 157, 157, 157, 157, 157, 150, 150, 150, 150, 150, 150, 150,
142, 150, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 135,
135, 135, 135, 157, 157, 135, 135, 127, 127, 127, 127, 127, 127, 127,
127, 127, 127, 127, 127, 127, 127, 135, 127, 127, 120, 120, 120, 120,
120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
120, 120, 120, 120, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
112, 112, 112, 105, 105, 112, 105, 105, 105, 105, 105, 105, 105, 105,
105, 105, 105, 105, 105, 97, 97, 97, 105, 97, 97, 97, 97, 97, 97, 97,
97, 97, 97, 90, 90, 90, 90, 97, 90, 90, 90, 90, 90, 90, 82, 90, 82, 82,
90, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 75, 82, 75, 75, 75, 75,
75, 75, 75, 75, 75, 75, 75, 75, 67, 75, 67, 67, 75, 67, 67, 67, 67, 67,
67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 60, 60, 67, 67, 60, 67, 60,
60, 60, 60, 60, 67, 60, 52, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
60, 60, 60, 60, 90, 60, 60, 60, 60, 60, 60, 60, 60, 67, 60, 60, 67, 67,
67, 67, 67, 67, 67, 67, 67, 97, 67, 67, 60, 67, 67, 67, 67, 67, 67, 67,
67, 67, 67, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
75, 82, 75, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82,
82, 82, 82, 82, 82, 82, 90, 82, 82, 90, 82, 82, 82, 82, 82, 82, 82, 82,
82, 82, 90, 82, 90, 90, 82, 90, 82, 82, 90, 90, 90, 90, 90, 90, 90, 90,
90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90,
90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90,
90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 97, 90, 90, 105, 90, 90,
90, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 105, 97, 105, 97,
97, 97, 97, 97, 97, 105, 97, 97, 97, 105, 105, 105, 105, 105, 105, 105,
105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105,
105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105,
105, 105, 105, 112, 105, 105, 105, 112, 112, 112, 112, 112, 112, 105,
105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105,
105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 97, 97, 105,
97, 97, 97, 105, 97, 105, 105, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97,
97, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 82, 82, 82, 82, 82, 82,
82, 82, 75, 75, 82, 82, 75, 75, 75, 67, 67, 67, 67, 67, 67, 67, 60, 60,
60, 60, 60, 60, 52, 52, 52, 52, 52, 52, 45, 45, 45, 45, 45, 45, 45, 37,
37, 37, 37, 37, 37, 30, 30, 30, 30, 30, 30, 22, 22, 22, 22, 22, 22, 15,
15, 15, 22, 15, 15, 15, 15, 7, 7, 15, 15, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
7, 7, 7, 7, 7, 7, 7, 7, 7, 0, 7, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0
};

int main()
{
	uint16_t i;
	DDRB = 0x31;

	for (i = 0; i < 3; i++) {
		PORTB = 0x01;
		_delay_ms(50);
		PORTB = 0x00;
		_delay_ms(50);
	}

	while (1) {
		if ((PINB & 0x08) == 0) {
			for (i = 0; i < sizeof(data1); i++)
				pwm_step(pgm_read_byte(data1+i));
		} else {
			for (i = 0; i < sizeof(data2); i++)
				pwm_step(pgm_read_byte(data2+i));
		}
	}
}


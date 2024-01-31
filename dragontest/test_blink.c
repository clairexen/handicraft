
#include <avr/io.h>

#define WAIT() do { for (uint32_t i=0; i<30000; i++) { asm volatile(""); } } while(0)
#define PINON(_PORT, _PIN) do { DDR ## _PORT = 1 << _PIN; PORT ## _PORT = 1 << _PIN; } while (0)
#define PINOFF(_PORT, _PIN) do { DDR ## _PORT = 0; PORT ## _PORT = 0; } while (0)
#define BLINK(_PORT, _PIN) do { PINON(_PORT, _PIN); WAIT(); PINOFF(_PORT, _PIN); } while(0)

int main (void)
{
        while(1) {
		BLINK(B, 0);
		// BLINK(B, 1);
		// BLINK(B, 2);
		// BLINK(B, 3);
		// BLINK(B, 4);
		WAIT();
        }

        return 0;
}


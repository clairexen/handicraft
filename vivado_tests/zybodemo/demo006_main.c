#include <stdio.h>
#include <sleep.h>
#include "platform.h"

int get_sw()
{
	return *(int*)0x50000004;
}

void set_leds(int v)
{
	*(int*)0x50000000 = v;
}

int main()
{
    int counter = 0;

    init_platform();
    printf("Running...\r\n");

    while (1) {
    	usleep(get_sw() * 100000 + 50000);
    	set_leds(counter++);
    }

    return 0;
}

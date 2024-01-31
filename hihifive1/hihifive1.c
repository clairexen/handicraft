#include <stdio.h>
#include "platform.h"

#if 1
void busywait(int ms)
{
	uint32_t time_begin, time_end;
	uint32_t time_target = ms * 261642;

	asm volatile ("csrr %0, mcycle" : "=r"(time_begin));

	do
		asm volatile ("csrr %0, mcycle" : "=r"(time_end));
	while ((time_end - time_begin) < time_target);
}
#else
void busywait(int ms)
{
	while (ms-- > 0) {
		for (int i = 0; i < 131578; i++)
			asm volatile ("");
	}
}
#endif

int main()
{
	printf("Hi, HiFive1!\n");

	GPIO_REG(GPIO_OUTPUT_EN) |= 1 << GREEN_LED_OFFSET;
	GPIO_REG(GPIO_OUTPUT_EN) |= 1 << BLUE_LED_OFFSET;
	GPIO_REG(GPIO_OUTPUT_EN) |= 1 << RED_LED_OFFSET;

	GPIO_REG(GPIO_OUTPUT_VAL) |= 1 << GREEN_LED_OFFSET;
	GPIO_REG(GPIO_OUTPUT_VAL) |= 1 << BLUE_LED_OFFSET;
	GPIO_REG(GPIO_OUTPUT_VAL) |= 1 << RED_LED_OFFSET;

	for (uint32_t i = 0;; i++)
	{
		uint32_t x = i ^ (i>>1);

		if (x & 1)
			GPIO_REG(GPIO_OUTPUT_VAL) &= ~(1 << GREEN_LED_OFFSET);
		else
			GPIO_REG(GPIO_OUTPUT_VAL) |= 1 << GREEN_LED_OFFSET;

		if (x & 2)
			GPIO_REG(GPIO_OUTPUT_VAL) &= ~(1 << BLUE_LED_OFFSET);
		else
			GPIO_REG(GPIO_OUTPUT_VAL) |= 1 << BLUE_LED_OFFSET;

		if (x & 4)
			GPIO_REG(GPIO_OUTPUT_VAL) &= ~(1 << RED_LED_OFFSET);
		else
			GPIO_REG(GPIO_OUTPUT_VAL) |= 1 << RED_LED_OFFSET;

		busywait(500);
	}

	return 0;
}

#include <stdint.h>
#include <stdbool.h>
#include <avr/io.h>

#define INPUT 0
#define OUTPUT 1

void pinMode(uint8_t pin, uint8_t direction)
{
	if (direction) {
		switch(pin)
		{
			case  0: DDRD |= _BV(0); break;
			case  1: DDRD |= _BV(1); break;
			case  2: DDRD |= _BV(2); break;
			case  3: DDRD |= _BV(3); break;
			case  4: DDRD |= _BV(4); break;
			case  5: DDRD |= _BV(5); break;
			case  6: DDRD |= _BV(6); break;
			case  7: DDRD |= _BV(7); break;
			case  8: DDRB |= _BV(0); break;
			case  9: DDRB |= _BV(1); break;
			case 10: DDRB |= _BV(2); break;
			case 11: DDRB |= _BV(3); break;
			case 12: DDRB |= _BV(4); break;
			case 13: DDRB |= _BV(5); break;
			case 14: DDRC |= _BV(0); break;
			case 15: DDRC |= _BV(1); break;
			case 16: DDRC |= _BV(2); break;
			case 17: DDRC |= _BV(3); break;
			case 18: DDRC |= _BV(4); break;
			case 19: DDRC |= _BV(5); break;
		}
	} else {
		switch(pin)
		{
			case  0: DDRD &= ~_BV(0); break;
			case  1: DDRD &= ~_BV(1); break;
			case  2: DDRD &= ~_BV(2); break;
			case  3: DDRD &= ~_BV(3); break;
			case  4: DDRD &= ~_BV(4); break;
			case  5: DDRD &= ~_BV(5); break;
			case  6: DDRD &= ~_BV(6); break;
			case  7: DDRD &= ~_BV(7); break;
			case  8: DDRB &= ~_BV(0); break;
			case  9: DDRB &= ~_BV(1); break;
			case 10: DDRB &= ~_BV(2); break;
			case 11: DDRB &= ~_BV(3); break;
			case 12: DDRB &= ~_BV(4); break;
			case 13: DDRB &= ~_BV(5); break;
			case 14: DDRC &= ~_BV(0); break;
			case 15: DDRC &= ~_BV(1); break;
			case 16: DDRC &= ~_BV(2); break;
			case 17: DDRC &= ~_BV(3); break;
			case 18: DDRC &= ~_BV(4); break;
			case 19: DDRC &= ~_BV(5); break;
		}
	}
}

void digitalWrite(uint8_t pin, uint8_t on)
{
	if (on) {
		switch(pin)
		{
			case  0: PORTD |= _BV(0); break;
			case  1: PORTD |= _BV(1); break;
			case  2: PORTD |= _BV(2); break;
			case  3: PORTD |= _BV(3); break;
			case  4: PORTD |= _BV(4); break;
			case  5: PORTD |= _BV(5); break;
			case  6: PORTD |= _BV(6); break;
			case  7: PORTD |= _BV(7); break;
			case  8: PORTB |= _BV(0); break;
			case  9: PORTB |= _BV(1); break;
			case 10: PORTB |= _BV(2); break;
			case 11: PORTB |= _BV(3); break;
			case 12: PORTB |= _BV(4); break;
			case 13: PORTB |= _BV(5); break;
			case 14: PORTC |= _BV(0); break;
			case 15: PORTC |= _BV(1); break;
			case 16: PORTC |= _BV(2); break;
			case 17: PORTC |= _BV(3); break;
			case 18: PORTC |= _BV(4); break;
			case 19: PORTC |= _BV(5); break;
		}
	} else {
		switch(pin)
		{
			case  0: PORTD &= ~_BV(0); break;
			case  1: PORTD &= ~_BV(1); break;
			case  2: PORTD &= ~_BV(2); break;
			case  3: PORTD &= ~_BV(3); break;
			case  4: PORTD &= ~_BV(4); break;
			case  5: PORTD &= ~_BV(5); break;
			case  6: PORTD &= ~_BV(6); break;
			case  7: PORTD &= ~_BV(7); break;
			case  8: PORTB &= ~_BV(0); break;
			case  9: PORTB &= ~_BV(1); break;
			case 10: PORTB &= ~_BV(2); break;
			case 11: PORTB &= ~_BV(3); break;
			case 12: PORTB &= ~_BV(4); break;
			case 13: PORTB &= ~_BV(5); break;
			case 14: PORTC &= ~_BV(0); break;
			case 15: PORTC &= ~_BV(1); break;
			case 16: PORTC &= ~_BV(2); break;
			case 17: PORTC &= ~_BV(3); break;
			case 18: PORTC &= ~_BV(4); break;
			case 19: PORTC &= ~_BV(5); break;
		}
	}
}

uint8_t digitalRead(uint8_t pin)
{
	switch(pin)
	{
		case  0: return (PIND & _BV(0)) != 0;
		case  1: return (PIND & _BV(1)) != 0;
		case  2: return (PIND & _BV(2)) != 0;
		case  3: return (PIND & _BV(3)) != 0;
		case  4: return (PIND & _BV(4)) != 0;
		case  5: return (PIND & _BV(5)) != 0;
		case  6: return (PIND & _BV(6)) != 0;
		case  7: return (PIND & _BV(7)) != 0;
		case  8: return (PINB & _BV(0)) != 0;
		case  9: return (PINB & _BV(1)) != 0;
		case 10: return (PINB & _BV(2)) != 0;
		case 11: return (PINB & _BV(3)) != 0;
		case 12: return (PINB & _BV(4)) != 0;
		case 13: return (PINB & _BV(5)) != 0;
		case 14: return (PINC & _BV(0)) != 0;
		case 15: return (PINC & _BV(1)) != 0;
		case 16: return (PINC & _BV(2)) != 0;
		case 17: return (PINC & _BV(3)) != 0;
		case 18: return (PINC & _BV(4)) != 0;
		case 19: return (PINC & _BV(5)) != 0;
	}

	return 0;
}

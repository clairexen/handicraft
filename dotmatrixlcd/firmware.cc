
/**
    Copyright (C) 2010  Clifford Wolf <clifford@clifford.at>
    
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    ---------------------------------------------------------------------
    
    Simple example for using a 2x16 dot matrix display in 4-bit mode.

    The RW pin is actually unused. It could be hardwired to the LOW state.
    
    The RS pin and the four DB pins can easily shared with other devices.
    The display does only sample them when the Enable (E) pin has a high-low
    transaction.
    
    Additional Information:
    http://www.sprut.de/electronic/lcd/
    http://www.csd-electronics.de/data/pdf/PRC1602A.pdf
    http://metalab.at/wiki/Bauteilsortiment#Baugruppen
**/

#include <wiring.h>
    
#define bits2byte(_hbits, _lbits) \
   (((((1 ## _hbits) / 1000)&1) << 7) | \
    ((((1 ## _hbits) /  100)&1) << 6) | \
    ((((1 ## _hbits) /   10)&1) << 5) | \
    ((((1 ## _hbits) /    1)&1) << 4) | \
    ((((1 ## _lbits) / 1000)&1) << 3) | \
    ((((1 ## _lbits) /  100)&1) << 2) | \
    ((((1 ## _lbits) /   10)&1) << 1) | \
    ((((1 ## _lbits) /    1)&1) << 0))

#define PIN_RS 10
#define PIN_RW  9
#define PIN_E   8

#define PIN_DB4 7
#define PIN_DB5 6
#define PIN_DB6 5
#define PIN_DB7 4

void dispsend(byte v)
{
  digitalWrite(PIN_RS,  v & bit(5) ? HIGH : LOW);
  digitalWrite(PIN_RW,  v & bit(4) ? HIGH : LOW);
  digitalWrite(PIN_DB7, v & bit(3) ? HIGH : LOW);
  digitalWrite(PIN_DB6, v & bit(2) ? HIGH : LOW);
  digitalWrite(PIN_DB5, v & bit(1) ? HIGH : LOW);
  digitalWrite(PIN_DB4, v & bit(0) ? HIGH : LOW);
  delayMicroseconds(1);
  digitalWrite(PIN_E, HIGH);
  delayMicroseconds(1);
  digitalWrite(PIN_E, LOW);
  delayMicroseconds(1);
}

void dispclr()
{  
  dispsend(bits2byte(00, 0000)); // Set Display/Cursor/Blink
  dispsend(bits2byte(00, 1111));
  delay(1);
  
  dispsend(bits2byte(00, 0000)); // Clear Display
  dispsend(bits2byte(00, 0001));
  delay(1);

  dispsend(bits2byte(00, 0000)); // Entry Mode Set
  dispsend(bits2byte(00, 0110));
  delay(1);
}

void dispdaddr(byte v)
{
  dispsend(bits2byte(00, 1000) + ((v>>4) & 0x0f)); // Set DRAM addr
  dispsend(bits2byte(00, 0000) + (v & 0x0f));
}

void dispwrite(byte v)
{
  dispsend(bits2byte(10, 0000) | ((v >> 4) & 0x0f));
  dispsend(bits2byte(10, 0000) | (v & 0x0f));
}

void dispwritestr(const char *str)
{
  while (*str)
    dispwrite(*(str++));
}

void setup()
{
  pinMode(PIN_RS,  OUTPUT);
  pinMode(PIN_RW,  OUTPUT);
  pinMode(PIN_E,   OUTPUT);
  pinMode(PIN_DB4, OUTPUT);
  pinMode(PIN_DB5, OUTPUT);
  pinMode(PIN_DB6, OUTPUT);
  pinMode(PIN_DB7, OUTPUT);
  digitalWrite(PIN_E, LOW);

  // reset display  
  delay(500);
  dispsend(bits2byte(00, 0011));
  delay(10);
  dispsend(bits2byte(00, 0011));
  delay(10);
  dispsend(bits2byte(00, 0011));
  delay(10);
  
  // initialize display
  dispsend(bits2byte(00, 0010)); // Function set
  dispsend(bits2byte(00, 0010));
  dispsend(bits2byte(00, 1000));
  dispclr();
}

void loop()
{
  dispclr();
  dispdaddr(0);
  dispwritestr("Hello World!"); // Write 1st line
  dispdaddr(40);
  dispwritestr("This is a test.."); // Write 2nd line
  dispdaddr(12);
  
  delay(2000);
  
  dispclr();
  for (int i=0; i<16; i++) {
    dispwrite('a'+i);
    delay(200);
  }
  dispdaddr(0);
  for (int i=0; i<16; i++) {
    dispwrite('A'+i);
    delay(200);
  }
  dispdaddr(40);
  for (int i=0; i<16; i++) {
    dispwrite(128+8+32+i);
    delay(200);
  }

  delay(1000);
}


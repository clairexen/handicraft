// CS_B: pin 10
// MOSI: pin 11
// MISO: pin 12
// SCLK: pin 13

#include <SPI.h>

const int begin_addr = 0x44;
const int end_addr = 0x44 + 16;

void setup()
{
  Serial.begin(9600);
  Serial.println("Hello.");

  SPI.begin();
  SPI.setClockDivider(SPI_CLOCK_DIV16);
  SPI.setDataMode(SPI_MODE0);
  SPI.setBitOrder(MSBFIRST);
  
  pinMode(10, OUTPUT);
  digitalWrite(10, HIGH);

  delay(100);
}

void loop()
{
  static uint8_t itercount = 0;

  Serial.println("Write and read..");
  for (int i = begin_addr; i < end_addr; i += 4) {
    writeRegister(i, addr2val(i, itercount));
  }
  for (int i = begin_addr; i < end_addr; i += 4) {
    uint32_t ref = addr2val(i, itercount);
    uint32_t value = readRegister(i);
    Serial.print("  ");
    printHexByte(i);
    Serial.print(": ");
    printHexWord(ref);
    Serial.print(" ");
    printHexWord(value);
    Serial.println(ref == value ? " ok" : " ERROR");
  }
  for (int i = begin_addr; i < end_addr; i += 4) {
    writeRegister(i, 0);
  }

  delay(1000);
  itercount++;
}

uint32_t addr2val(uint8_t address, uint8_t seed)
{
  uint32_t x = address;
  x = (x << 8) | seed;
  x = (x << 8) | address;
  x = (x << 8) | seed;
  
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  
  return x;
}

void printHexByte(uint8_t value)
{
  Serial.print("0123456789abcdef"[value >> 4]);
  Serial.print("0123456789abcdef"[value & 15]);
}

void printHexWord(uint32_t value)
{
  printHexByte(value >> 24);
  printHexByte(value >> 16);
  printHexByte(value >>  8);
  printHexByte(value >>  0);
}

uint32_t readRegister(uint8_t address)
{
  uint32_t value = 0;

  digitalWrite(10, LOW);
  
  SPI.transfer(address);
  
  value |= uint32_t(SPI.transfer(0x00)) << 0;
  value |= uint32_t(SPI.transfer(0x00)) << 8;
  value |= uint32_t(SPI.transfer(0x00)) << 16;
  value |= uint32_t(SPI.transfer(0x00)) << 24;
  
  digitalWrite(10, HIGH);

  return value;
}

void writeRegister(uint8_t address, uint32_t value)
{
  digitalWrite(10, LOW);

  SPI.transfer(0x80 | address);
  SPI.transfer(value >> 0);
  SPI.transfer(value >> 8);
  SPI.transfer(value >> 16);
  SPI.transfer(value >> 24);
  
  digitalWrite(10, HIGH);
}


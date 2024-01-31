#ifndef LIBVC830_H
#define LIBVC830_H

#include <stdio.h>
#include <hidapi/hidapi.h>

struct VC830
{
	bool initialized;
	unsigned char buf[14];

	char digits_str[6];
	int digits, point, bar;
	float unscaled_value, value;

	bool bit_auto, bit_dc, bit_ac, bit_rel, bit_hold, bit_bpn;
	bool bit_z1, bit_z2, bit_max, bit_min, bit_apo, bit_bat, bit_nano, bit_z3;
	bool bit_micro, bit_milli, bit_kilo, bit_mega;
	bool bit_beep, bit_diode, bit_percent, bit_z4;
	bool bit_volt, bit_amp, bit_ohm, bit_hfe, bit_hz;
	bool bit_farad, bit_celsius, bit_fahrenheit;

	VC830() : initialized(false) { }
	virtual ~VC830() { }

	int update();
	void print(FILE *f);

	static VC830 *openDev(const char *dev);
	virtual int readByte() = 0;
};

struct VC830_HE2325U : VC830
{
	hid_device *handle;
	unsigned char buf[16];
	int buf_pos, buf_len;

	static VC830* openDev(const char *dev);
	virtual int readByte();
	virtual ~VC830_HE2325U();
};

struct VC830_RS232 : VC830
{
	int fd;

	static VC830* openDev(const char *dev);
	virtual int readByte();
	virtual ~VC830_RS232();
};

#endif

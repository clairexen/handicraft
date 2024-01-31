
#include "libvc830.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#undef VC830_PRINT_RAW_DATA

int VC830::update()
{
	if (!initialized) {
		const char *str = "\r\n";
		int state = 0;
		while (str[state] != 0) {
			int byte = readByte();
			state = str[state] == byte ? state+1 : str[0] == byte;
		}
		initialized = true;
	}

	for (int i = 0; i < 14; i++) {
		int byte = readByte();
		if (byte < 0)
			return -1;
		buf[i] = byte;
	}

	if (buf[12] != '\r' || buf[13] != '\n') {
		initialized = false;
		update();
	}

	for (int i = 0; i < 5; i++)
		digits_str[i] = buf[i];
	digits_str[5] = 0;

	digits = atoi(digits_str);
	point = buf[6] - '0';
	bar = (signed char)buf[11];

	switch (point)
	{
	case 0:
		unscaled_value = digits;
		break;
	case 1:
		unscaled_value = digits / 1000.0;
		break;
	case 2:
		unscaled_value = digits / 100.0;
		break;
	case 4:
		unscaled_value = digits / 10.0;
		break;
	default:
		unscaled_value = nanf("");
	}

#define DECODE_BIT(_val, _bit, _flag) _flag = ((_val >> _bit) & 1) != 0;

	DECODE_BIT(buf[7], 5, bit_auto)
	DECODE_BIT(buf[7], 4, bit_dc)
	DECODE_BIT(buf[7], 3, bit_ac)
	DECODE_BIT(buf[7], 2, bit_rel)
	DECODE_BIT(buf[7], 1, bit_hold)
	DECODE_BIT(buf[7], 0, bit_bpn)

	DECODE_BIT(buf[8], 7, bit_z1)
	DECODE_BIT(buf[8], 6, bit_z2)
	DECODE_BIT(buf[8], 5, bit_max)
	DECODE_BIT(buf[8], 4, bit_min)
	DECODE_BIT(buf[8], 3, bit_apo)
	DECODE_BIT(buf[8], 2, bit_bat)
	DECODE_BIT(buf[8], 1, bit_nano)
	DECODE_BIT(buf[8], 0, bit_z3)

	DECODE_BIT(buf[9], 7, bit_micro)
	DECODE_BIT(buf[9], 6, bit_milli)
	DECODE_BIT(buf[9], 5, bit_kilo)
	DECODE_BIT(buf[9], 4, bit_mega)
	DECODE_BIT(buf[9], 3, bit_beep)
	DECODE_BIT(buf[9], 2, bit_diode)
	DECODE_BIT(buf[9], 1, bit_percent)
	DECODE_BIT(buf[9], 0, bit_z4)

	DECODE_BIT(buf[10], 7, bit_volt)
	DECODE_BIT(buf[10], 6, bit_amp)
	DECODE_BIT(buf[10], 5, bit_ohm)
	DECODE_BIT(buf[10], 4, bit_hfe)
	DECODE_BIT(buf[10], 3, bit_hz)
	DECODE_BIT(buf[10], 2, bit_farad)
	DECODE_BIT(buf[10], 1, bit_celsius)
	DECODE_BIT(buf[10], 0, bit_fahrenheit)

#undef DECODE_BIT

	value = unscaled_value;
	if (bit_nano)
		value *= 1e-9;
	if (bit_micro)
		value *= 1e-6;
	if (bit_milli)
		value *= 1e-3;
	if (bit_kilo)
		value *= 1e+3;
	if (bit_mega)
		value *= 1e+6;

	return 0;
}

void VC830::print(FILE *f)
{
	fprintf(f, "%g [ %f ", value, unscaled_value);

#define PRINT_BIT(_txt, _flag) if (bit_ ## _flag) fprintf(f, "%s", _txt);

	PRINT_BIT("n",       nano)
	PRINT_BIT("u",       micro)
	PRINT_BIT("m",       milli)
	PRINT_BIT("k",       kilo)
	PRINT_BIT("M",       mega)

	PRINT_BIT("V",       volt)
	PRINT_BIT("A",       amp)
	PRINT_BIT("Ohm",     ohm)
	PRINT_BIT("hFE",     hfe)
	PRINT_BIT("Hz",      hz)
	PRINT_BIT("F",       farad)
	PRINT_BIT("C(Temp)", celsius)
	PRINT_BIT("F(Temp)", fahrenheit)
	PRINT_BIT("%",       percent)

	PRINT_BIT(" AUTO",   auto)
	PRINT_BIT(" DC",     dc)
	PRINT_BIT(" AC",     ac)
	PRINT_BIT(" REL",    rel)
	PRINT_BIT(" HOLD",   hold)
	PRINT_BIT(" BPN",    bpn)

	PRINT_BIT(" Z1",     z1)
	PRINT_BIT(" Z2",     z1)
	PRINT_BIT(" MAX",    max)
	PRINT_BIT(" MIN",    min)
	PRINT_BIT(" APO",    apo)
	PRINT_BIT(" BAR",    bat)
	PRINT_BIT(" Z3",     z1)

	PRINT_BIT(" BEEP",   beep)
	PRINT_BIT(" DIODE",  beep)
	PRINT_BIT(" Z4",     z4)

#undef PRINT_BIT

	fprintf(f, " | %d ]\n", bar);
}

VC830 *VC830::openDev(const char *dev)
{
	if (dev != NULL && !strncmp(dev, "/dev/tty", 8))
		return VC830_RS232::openDev(dev);
	return VC830_HE2325U::openDev(dev);
}

VC830* VC830_HE2325U::openDev(const char *dev)
{
	unsigned int bps = 2400;
	unsigned char buf[6];
	VC830_HE2325U *that = new VC830_HE2325U;

	if (dev == NULL) {
		// suspend device before open (see see http://www.erste.de/UT61/index.html)
		if (system("bash -ec 'for it in `grep -lx 1a86 /sys/bus/usb/devices/*/idVendor`; do if grep -q e008 ${it/idVendor/idProduct}; then "
				"echo auto > ${it/idVendor/power/control}; echo 0 > ${it/idVendor/power/autosuspend}; fi; done'") != 0)
			goto err;
		that->handle = hid_open(0x1a86, 0xe008, NULL);
	} else
		that->handle = hid_open_path(dev);
	if (that->handle == 0)
		goto err;

	// Send a Feature Report to the device
	// (see http://www-user.tu-chemnitz.de/~heha/bastelecke/Rund%20um%20den%20PC/hid-ser.en.htm)
	buf[0] = 0x0;
	buf[1] = bps;
	buf[2] = bps>>8;
	buf[3] = bps>>16;
	buf[4] = bps>>24;
	buf[5] = 0x03;
	if (hid_send_feature_report(that->handle, buf, 6) < 0)
		goto err_close;

	that->buf_pos = 0;
	that->buf_len = 0;
	return that;

err_close:
	hid_close(that->handle);
err:
	perror("VC830_HE2325U::openDev()");
	delete that;
	return NULL;
}

int VC830_HE2325U::readByte()
{
	// Read data from chip
	// (see http://www-user.tu-chemnitz.de/~heha/bastelecke/Rund%20um%20den%20PC/hid-ser.en.htm)
	while (buf_pos >= buf_len)
	{
		int rc = hid_read(handle, buf, sizeof(buf));
		if (rc < 0) {
			perror("VC830_HE2325U::readByte()");
			return -1;
		}
		if (rc == 0)
			continue;

		buf_pos = 1;
		buf_len = (buf[0] & 7) + 1;

#ifdef VC830_PRINT_RAW_DATA
		if (buf_len == 1)
		{
			printf("[]");
		}
		else
		{
			printf("\n[%d:", buf_len-1);
			for (int i = buf_pos; i < buf_len; i++)
				printf(" %02x", buf[i]);
			printf("]");
		}
#endif
	}

#ifdef VC830_PRINT_RAW_DATA
	printf("%c", buf[buf_pos]);
	fflush(stdout);
#endif
	return buf[buf_pos++];
}

VC830_HE2325U::~VC830_HE2325U()
{
	hid_close(handle);
}

VC830* VC830_RS232::openDev(const char *dev)
{
	struct termios tio;
	VC830_RS232 *that = new VC830_RS232;

	that->fd = open(dev, O_RDONLY);
	if (that->fd < 0)
		goto err;

	memset(&tio, 0, sizeof(tio));
	tcgetattr(that->fd, &tio);
	cfsetospeed(&tio, B2400);
	if (tcsetattr(that->fd, TCSANOW, &tio) < 0)
		goto err_close;

	return that;

err_close:
	close(that->fd);
err:
	perror("VC830_RS232::openDev()");
	delete that;
	return NULL;
}

int VC830_RS232::readByte()
{
	unsigned char buf;
	int rc = read(fd, &buf, 1);
	if (rc <= 0) {
		perror("VC830_RS232::readByte()");
		return -1;
	}
#ifdef VC830_PRINT_RAW_DATA
	printf("%c", buf);
	fflush(stdout);
#endif
	return buf;
}

VC830_RS232::~VC830_RS232()
{
	close(fd);
}

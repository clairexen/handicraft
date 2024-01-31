#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#ifndef WIN32
#include <endian.h>
#endif
#include "bbclientlib.h"

static char *usbErrorMessage(int errCode)
{
	static char buffer[80];

	switch (errCode) {
	case USBOPEN_ERR_ACCESS:
		return "Access to device denied";
	case USBOPEN_ERR_NOTFOUND:
		return "The specified device was not found";
	case USBOPEN_ERR_IO:
		return "Communication error with device";
	default:
		sprintf(buffer, "Unknown USB error %d", errCode);
		return buffer;
	}
	return NULL;
}

usbDevice_t *bblabOpen(void)
{
	usbDevice_t *dev = NULL;
	unsigned char rawVid[2] = { USB_CFG_VENDOR_ID }, rawPid[2] = { USB_CFG_DEVICE_ID};
	char vendorName[] = { USB_CFG_VENDOR_NAME, 0 }, productName[] = { USB_CFG_DEVICE_NAME, 0};
	int vid = rawVid[0] + 256 * rawVid[1];
	int pid = rawPid[0] + 256 * rawPid[1];
	int err;

	if ((err = usbhidOpenDevice(&dev, vid, vendorName, pid, productName, 1)) != 0) {
		fprintf(stderr, "error finding %s: %s\n", productName, usbErrorMessage(err));
		return NULL;
	}
	return dev;
}

static int bblabSend(usbDevice_t *dev, int id, const void *data, int len)
{
	int i, err;
	char buffer[len+1];

	buffer[0] = id;
	memcpy(buffer+1, data, len);

	for (i=0; (err = usbhidSetReport(dev, buffer, len+1)) != 0 && i < 100; i++) {
		fprintf(stderr, "error writing data [%d]: %s\n", i, usbErrorMessage(err));
	}

	if (i >= 100)
		abort();

	return err == 0 ? 0 : -1;
}

static int bblabRecv(usbDevice_t *dev, int id, void *data, int len)
{
	int i, err;
	char buffer[len+1];

	len++;
	for (i=0; (err = usbhidGetReport(dev, id, buffer, &len)) != 0 && i < 100; i++) {
		fprintf(stderr, "error reading data [%d]: %s\n", i, usbErrorMessage(err));
	}

	if (i >= 100)
		abort();

	memcpy(data, buffer+1, len-1);
	return err == 0 ? 0 : -1;
}

void bblabMemSet(usbDevice_t *dev, int addr, const void *data, int len)
{
	uint8_t ptrdata[6] = { 1, 0, addr & 0xff, addr >> 8, len & 0xff, len >> 8 };
	bblabSend(dev, 2, ptrdata, 6);
	for (int i=0; i<len; i+=64) {
		uint8_t buffer[64];
		for (int j = 0; j < 64; j++)
			buffer[j] = i+j < len ? ((uint8_t*)data)[i+j] : 0;
		bblabSend(dev, 3, buffer, 64);
	}
}

void bblabMemGet(usbDevice_t *dev, int addr, void *data, int len)
{
	uint8_t ptrdata[6] = { 1, 0, addr & 0xff, addr >> 8, len & 0xff, len >> 8 };
	bblabSend(dev, 2, ptrdata, 6);
	for (int i=0; i<len; i+=64) {
		uint8_t buffer[64];
		bblabRecv(dev, 3, buffer, 64);
		for (int j = 0; j < 64 && i + j < len; j++)
			((uint8_t*)data)[i+j] = buffer[j];
	}
}

int bblabParameter(usbDevice_t *dev, int parameter)
{
	uint16_t buf = parameter;
	bblabSend(dev, 4, &buf, 2);
	bblabRecv(dev, 4, &buf, 2);
	return buf;
}

int bblabStateGet(usbDevice_t *dev)
{
	uint8_t buf;
	bblabRecv(dev, 1, &buf, 1);
	return buf;
}

void bblabStateSet(usbDevice_t *dev, int state)
{
	uint8_t buf = state;
	bblabSend(dev, 1, &buf, 1);
}

void bblabConfigConvert(struct bblab_config_t *cfg)
{
#ifndef WIN32
#define REG_8(_n) do { } while (0)
#define REG_16(_n) cfg->_n = htole16(cfg->_n);

	REG_8(clkdiv);
	REG_8(trigger_state);
	REG_16(trigger_counter);
	REG_16(trigger_init_ticks);
	REG_8(pretrig_mode_channel);
	REG_8(pretrig_value);
	REG_8(trigger_mode_channel);
	REG_8(trigger_value);
	REG_16(posttrig_nticks);
	REG_8(outclkdiv);
	REG_8(outputmask);
	REG_16(outlen);
	REG_16(outidx);
	REG_16(digioutptr);
	REG_16(pwmoutptr);
	REG_16(samplelen);
	REG_16(sampleidx);
	REG_16(sampleoutidx);
	REG_16(digisampleptr);
	REG_16(adcsampleptr[0]);
	REG_16(adcsampleptr[1]);
	REG_16(adcsampleptr[2]);
	REG_16(adcsampleptr[3]);
	REG_16(collision_tick[0]);
	REG_16(collision_tick[1]);
	REG_16(collision_tick[2]);
	REG_16(collision_tick[3]);
	REG_16(collision_duration[0]);
	REG_16(collision_duration[1]);
	REG_16(collision_duration[2]);
	REG_16(collision_duration[3]);

#undef REG_8
#undef REG_16
#endif
}

void bblabConfigDump(const struct bblab_config_t *cfg, const char *prefix)
{
#define REG_8(_n) fprintf(stderr, "%s" #_n " = 0x%02x (%d)\n", prefix, cfg->_n, cfg->_n)
#define REG_16(_n) fprintf(stderr, "%s" #_n " = 0x%04x (%d)\n", prefix, cfg->_n, cfg->_n)

	REG_8(clkdiv);
	REG_8(trigger_state);
	REG_16(trigger_counter);
	REG_16(trigger_init_ticks);
	REG_8(pretrig_mode_channel);
	REG_8(pretrig_value);
	REG_8(trigger_mode_channel);
	REG_8(trigger_value);
	REG_16(posttrig_nticks);
	REG_8(outclkdiv);
	REG_8(outputmask);
	REG_16(outlen);
	REG_16(outidx);
	REG_16(digioutptr);
	REG_16(pwmoutptr);
	REG_16(samplelen);
	REG_16(sampleidx);
	REG_16(sampleoutidx);
	REG_16(digisampleptr);
	REG_16(adcsampleptr[0]);
	REG_16(adcsampleptr[1]);
	REG_16(adcsampleptr[2]);
	REG_16(adcsampleptr[3]);
	REG_16(collision_tick[0]);
	REG_16(collision_tick[1]);
	REG_16(collision_tick[2]);
	REG_16(collision_tick[3]);
	REG_16(collision_duration[0]);
	REG_16(collision_duration[1]);
	REG_16(collision_duration[2]);
	REG_16(collision_duration[3]);

#undef REG_8
#undef REG_16
}


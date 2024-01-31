
#ifndef BBLAB_COMMON_H
#define BBLAB_COMMON_H

#define BBLAB_VERSION 0x005a /* (major << 8) | minor */

#define USB_CFG_VENDOR_ID       0xc0, 0x16 /* = 0x16c0 = 5824 = voti.nl */
#define USB_CFG_DEVICE_ID       0xdf, 0x05 /* obdev's shared PID for HIDs */
#define USB_CFG_DEVICE_VERSION  0x00, 0x01

#define USB_CFG_VENDOR_NAME     'C', 'l', 'i', 'f', 'f', 'o', 'r', 'd', ' ', 'W', 'o', 'l', 'f', ' ', 'h', 't', 't', 'p', ':', '/', '/', 'c', 'l', 'i', 'f', 'f', 'o', 'r', 'd', '.', 'a', 't', '/'
#define USB_CFG_VENDOR_NAME_LEN 33

#define USB_CFG_DEVICE_NAME     'B', 'B', 'L', 'a', 'b'
#define USB_CFG_DEVICE_NAME_LEN 5

#define SYSPAR_VERSION   0
#define SYSPAR_MEMSIZE   1
#define SYSPAR_FREQUENCY 2

#define STATE_INIT     0  /* Device Transition: HALT */
#define STATE_HALT     1  /* Client Transistion: CONFIG */
#define STATE_CONFIG   2  /* Device Transistion: RECORD */
#define STATE_RECORD   3  /* Device Transistion: STANDBY, Client Transistion: SHUTDOWN */
#define STATE_STANDBY  4  /* Client Transistions: RECORD, SHUTDOWN */
#define STATE_SHUTDOWN 5  /* Device Transistion: HALT */

#define TRIGGER_STATE_INIT            0
#define TRIGGER_STATE_FILL            1
#define TRIGGER_STATE_PRETRIG         2
#define TRIGGER_STATE_PRETRIG_PHASE2  3
#define TRIGGER_STATE_TRIGGER         4
#define TRIGGER_STATE_TRIGGER_PHASE2  5
#define TRIGGER_STATE_POSTTRIGGER     6

#define CHANNEL_NONE    0x00
#define CHANNEL_DIGI1   0x01
#define CHANNEL_DIGI2   0x02
#define CHANNEL_DIGI3   0x03
#define CHANNEL_DIGI4   0x04
#define CHANNEL_ANALOG1 0x05
#define CHANNEL_ANALOG2 0x06
#define CHANNEL_ANALOG3 0x07
#define CHANNEL_ANALOG4 0x08
#define CHANNEL_OUTIDX  0x09

#define TRIGMODE_NONE   0x00
#define TRIGMODE_LT     0x10 // <
#define TRIGMODE_LE     0x20 // <=
#define TRIGMODE_EQ     0x30 // ==
#define TRIGMODE_GE     0x40 // >=
#define TRIGMODE_GT     0x50 // >
#define TRIGMODE_LT_GE  0x60 // raising edge
#define TRIGMODE_GT_LE  0x70 // falling edge

#ifndef __ASSEMBLER__

struct bblab_config_t
{
	/* clock configuration */
	uint8_t clkdiv;

	/* trigger configuration */
	uint8_t trigger_state;
	uint16_t trigger_counter;
	uint16_t trigger_init_ticks;
	uint8_t pretrig_mode_channel;
	uint8_t pretrig_value;
	uint8_t trigger_mode_channel;
	uint8_t trigger_value;
	uint16_t posttrig_nticks;

	/* output pattern */
	uint8_t outclkdiv;
	uint8_t outputmask;
	uint16_t outlen, outidx;
	uint16_t digioutptr, pwmoutptr;

	/* sample buffer config */
	uint16_t samplelen, sampleidx, sampleoutidx;
	uint16_t digisampleptr, adcsampleptr[4];

	/* collision handling */
	uint16_t collision_tick[4];
	uint16_t collision_duration[4];

} __attribute__((packed));

#endif

#endif /* BBLAB_COMMON_H */


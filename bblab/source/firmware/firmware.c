#include <avr/io.h>
#include <avr/wdt.h>
#include <avr/interrupt.h>	/* for sei() */
#include <util/delay.h>		/* for _delay_ms() */
#include <avr/eeprom.h>

#include <avr/pgmspace.h>	/* required by usbdrv.h */
#include "usbdrv.h"

#undef UART_DEBUG_OUTPUT
#undef UART_DEBUG_PRINT_USB


/* ------------------------------------------------------------------------- */
/* -------------------------------- Pinouts -------------------------------- */
/* ------------------------------------------------------------------------- */

#if PLATFORM_bblab
// Pinout on ATtiny85 (BBLab):
//
//	USB D- ....... PB1
//	USB D+ ....... PB0
//
//	Channel 1 .... PB2
//	Channel 2 .... PB3
//	Channel 3 .... PB4
//	Channel 4 .... PB5
//
//	PWM Out ...... PB4
#endif

#if PLATFORM_arduino
// Pinout on ATmega168 (Arduino):
//
//	USB D- ....... PB1  (Pin 9)
//	USB D+ ....... PB0  (Pin 8)
//
//	Channel 1 .... PC0  (Analog In 0)
//	Channel 2 .... PC1  (Analog In 1)
//	Channel 3 .... PC2  (Analog In 2)
//	Channel 4 .... PC3  (Analog In 3)
//
//	PWM Out ...... PD6  (Pin 6)
#endif


/* ------------------------------------------------------------------------- */
/* ------------------------------- Serial IO ------------------------------- */
/* ------------------------------------------------------------------------- */

#if PLATFORM_arduino
#  ifdef UART_DEBUG_OUTPUT
static void uart_setup()
{
	UBRR0L = (uint8_t) (F_CPU / (19200 * 16L) - 1);
	UBRR0H = (F_CPU / (19200 * 16L) - 1) >> 8;
	UCSR0B = (1 << RXEN0) | (1 << TXEN0);
	UCSR0C = (1 << UCSZ00) | (1 << UCSZ01);

	DDRD &= ~_BV(PIND0);
	PORTD |= _BV(PIND1);
}

static void putch(char ch)
{
	while (!(UCSR0A & _BV(UDRE0))) ;
	UDR0 = ch;
}

static void putstr(char *str)
{
	while (*str)
		putch(*(str++));
}

static void puthex(uint16_t v, uchar len)
{
	while (len-- > 0) {
		uchar digit = (v >> (4*len)) & 15;
		putch("0123456789abcdef"[digit]);
	}
}
#  else
#    define uart_setup(...)  do { } while (0)
#    define putch(...) do { } while (0)
#    define putstr(...) do { } while (0)
#    define puthex(...) do { } while (0)
#  endif
#  ifdef UART_DEBUG_PRINT_USB
#    define PUTCH_USB  putch
#    define PUTSTR_USB putstr
#    define PUTHEX_USB puthex
#  else
#    define PUTCH_USB(...)  do { } while (0)
#    define PUTSTR_USB(...) do { } while (0)
#    define PUTHEX_USB(...) do { } while (0)
#  endif
#else
#    define uart_setup(...)  do { } while (0)
#    define putch(...) do { } while (0)
#    define putstr(...) do { } while (0)
#    define puthex(...) do { } while (0)
#    define PUTCH_USB(...)  do { } while (0)
#    define PUTSTR_USB(...) do { } while (0)
#    define PUTHEX_USB(...) do { } while (0)
#endif


/* ------------------------------------------------------------------------- */
/* ----------------------------- USB interface ----------------------------- */
/* ------------------------------------------------------------------------- */

// Protocol Description:
//   Report ID 1:
//     Set/Pull engine state
//     Payload: 1x 8-Bit
//   Report ID 2:
//     Configure read/write pointer
//     Payload: 3x 16-Bit (memory id, addr, len)
//   Report ID 3:
//     Write to memory / read from memory
//     Payload: 64x 8-Bit
//   Report ID 4:
//     Read system parameter
//     Payload: 1x 16-Bit
PROGMEM char usbHidReportDescriptor[USB_CFG_HID_REPORT_DESCRIPTOR_LENGTH] = {
	0x06, 0x00, 0xff,	// USAGE_PAGE (Vendor Defined Page 1)
	0x09, 0x01,		// USAGE (Vendor Usage 1)
	0xa1, 0x01,		// COLLECTION (Application)
	0x15, 0x00,		//   LOGICAL_MINIMUM (0)
	0x26, 0xff, 0x00,	//   LOGICAL_MAXIMUM (255)
	0x75, 0x08,		//   REPORT_SIZE (8)
	0x85, 0x01,		//   REPORT_ID (1)
	0x95, 0x01,		//   REPORT_COUNT (1)
	0x09, 0x01,		//   USAGE (Vendor Usage 1)
	0xb2, 0x02, 0x01,	//   FEATURE (Data,Var,Abs,Buf)
	0x85, 0x02,		//   REPORT_ID (2)
	0x95, 0x06,		//   REPORT_COUNT (6)
	0x09, 0x01,		//   USAGE (Vendor Usage 1)
	0xb2, 0x02, 0x01,	//   FEATURE (Data,Var,Abs,Buf)
	0x85, 0x03,		//   REPORT_ID (3)
	0x95, 0x40,		//   REPORT_COUNT (64)
	0x09, 0x01,		//   USAGE (Vendor Usage 1)
	0xb2, 0x02, 0x01,	//   FEATURE (Data,Var,Abs,Buf)
	0x85, 0x04,		//   REPORT_ID (4)
	0x95, 0x02,		//   REPORT_COUNT (2)
	0x09, 0x01,		//   USAGE (Vendor Usage 1)
	0xb2, 0x02, 0x01,	//   FEATURE (Data,Var,Abs,Buf)
	0xc0			// END_COLLECTION
};


/* The following variables store the status of the current data transfer */
static uint8_t usbif_report_id;
static int8_t usbif_bytecount;
static uint16_t usbif_syspar;
static struct {
	uint16_t memid, addr, len;
} __attribute__((packed)) usbif_memptr;

/* This array is the 'main memory' for the engine */
uchar engine_mem[MEMSIZE];
uchar engine_state;

#define bbcfg ((struct bblab_config_t*)&engine_mem)

/* usbFunctionRead() is called when the host requests a chunk of data from
 * the device. For more information see the documentation in usbdrv/usbdrv.h.
 */
uchar usbFunctionRead(uchar *data, uchar len)
{
	uint8_t report_id_len = 0;

	PUTSTR_USB("** usbFunctionRead ** ");
	PUTHEX_USB(len, 2);
	PUTSTR_USB("\r\n");

	if (usbif_bytecount == -1 && len != 0) {
		data[0] = usbif_report_id;
		usbif_bytecount++;
		report_id_len++;
		data++; len--;
	}

	/* handle engine state read */
	if (usbif_report_id == 1)
	{
		if (len == 0 || usbif_bytecount > 1)
			return report_id_len;
		data[0] = engine_state;
		usbif_bytecount++;
		return 1 + report_id_len;
	}

	/* handle rw pointer config */
	if (usbif_report_id == 2)
	{
		if (len > 6 - usbif_bytecount)
			len = 6 - usbif_bytecount;

		for (uint8_t i = 0; i < len; i++, usbif_bytecount++)
			data[i] = ((uint8_t*)&usbif_memptr)[usbif_bytecount];

		return len + report_id_len;
	}

	/* handle memory read */
	if (usbif_report_id == 3)
	{
		if (len > 64 - usbif_bytecount)
			len = 64 - usbif_bytecount;

		for (uint8_t i = 0; i < len; i++, usbif_bytecount++)
		{
			if (usbif_memptr.len == 0)
				continue;

			if (usbif_memptr.memid == 1 && usbif_memptr.addr < sizeof(engine_mem))
				data[i] = engine_mem[usbif_memptr.addr++];

			usbif_memptr.len--;
		}

		return len + report_id_len;
	}

	/* handle syspar read */
	if (usbif_report_id == 4)
	{
		uint16_t syspar = 0;

		if (usbif_syspar == SYSPAR_VERSION)
			syspar = BBLAB_VERSION;

		if (usbif_syspar == SYSPAR_MEMSIZE)
			syspar = sizeof(engine_mem);

		if (usbif_syspar == SYSPAR_FREQUENCY)
			syspar = F_CPU / 256;

		if (len > 2 - usbif_bytecount)
			len = 2 - usbif_bytecount;

		for (uint8_t i = 0; i < len; i++, usbif_bytecount++)
			data[i] = ((uint8_t*)&syspar)[usbif_bytecount];

		return len + report_id_len;
	}

	/* don't know what to read... */
	return report_id_len;
}

/* usbFunctionWrite() is called when the host sends a chunk of data to the
 * device. For more information see the documentation in usbdrv/usbdrv.h.
 */
uchar usbFunctionWrite(uchar *data, uchar len)
{
	PUTSTR_USB("** usbFunctionWrite ** ");
	PUTHEX_USB(len, 2);
	PUTSTR_USB("\r\n");

	if (usbif_bytecount == -1 && len != 0) {
		PUTSTR_USB("  Report ID in payload: ");
		PUTHEX_USB(data[0], 2);
		PUTSTR_USB("\r\n");
		usbif_bytecount++;
		data++; len--;
	}

	/* handle engine state write */
	if (usbif_report_id == 1)
	{
		engine_state = data[0];
		return 1;
	}

	/* handle rw pointer config */
	if (usbif_report_id == 2)
	{
		if (len > 6 - usbif_bytecount)
			len = 6 - usbif_bytecount;

		for (uint8_t i = 0; i < len; i++, usbif_bytecount++)
			((uint8_t*)&usbif_memptr)[usbif_bytecount] = data[i];

		if (usbif_bytecount == 6) {
			PUTSTR_USB("  memptr: memid=");
			PUTHEX_USB(usbif_memptr.memid, 4);
			PUTSTR_USB(", addr=");
			PUTHEX_USB(usbif_memptr.addr, 4);
			PUTSTR_USB(", len=");
			PUTHEX_USB(usbif_memptr.len, 4);
			PUTSTR_USB("\r\n");
		}

		return usbif_bytecount == 6;
	}

	/* handle memory write */
	if (usbif_report_id == 3)
	{
		if (len > 64 - usbif_bytecount)
			len = 64 - usbif_bytecount;

		for (uint8_t i = 0; i < len; i++, usbif_bytecount++)
		{
			if (usbif_memptr.len == 0)
				continue;

			if (usbif_memptr.memid == 1 && usbif_memptr.addr < sizeof(engine_mem))
				engine_mem[usbif_memptr.addr++] = data[i];

			usbif_memptr.len--;
		}

		return usbif_bytecount == 64;
	}

	/* handle syspar pointer config */
	if (usbif_report_id == 4)
	{
		if (len > 2 - usbif_bytecount)
			len = 2 - usbif_bytecount;

		for (uint8_t i = 0; i < len; i++, usbif_bytecount++)
			((uint8_t*)&usbif_syspar)[usbif_bytecount] = data[i];

		if (usbif_bytecount == 2) {
			PUTSTR_USB("  syspar ptr=");
			PUTHEX_USB(usbif_syspar, 4);
			PUTSTR_USB("\r\n");
		}

		return usbif_bytecount == 2;
	}

	/* just say ok to everything else.. */
	return 1;
}

usbMsgLen_t usbFunctionSetup(uchar data[8])
{
	usbRequest_t *rq = (void *)data;
	PUTSTR_USB("** usbFunctionSetup **\r\n");

	/* HID class request */
	if ((rq->bmRequestType & USBRQ_TYPE_MASK) == USBRQ_TYPE_CLASS) {
		if (rq->bRequest == USBRQ_HID_GET_REPORT) {
			/* wValue: ReportType (highbyte), ReportID (lowbyte) */
			usbif_report_id = rq->wValue.bytes[0];
			usbif_bytecount = -1;
			/* use usbFunctionRead() to obtain data */
			PUTSTR_USB("  USBRQ_HID_GET_REPORT with report id ");
			PUTHEX_USB(usbif_report_id, 2);
			PUTSTR_USB("\r\n");
			return USB_NO_MSG;
		} else if (rq->bRequest == USBRQ_HID_SET_REPORT) {
			/* wValue: ReportType (highbyte), ReportID (lowbyte) */
			usbif_report_id = rq->wValue.bytes[0];
			usbif_bytecount = -1;
			/* use usbFunctionWrite() to receive data from host */
			PUTSTR_USB("  USBRQ_HID_SET_REPORT with report id ");
			PUTHEX_USB(usbif_report_id, 2);
			PUTSTR_USB("\r\n");
			return USB_NO_MSG;
		}
	} else {
		/* ignore vendor type requests, we don't use any */
	}
	return 0;
}


/* ------------------------------------------------------------------------- */
/* ---------------------------- Clock Interface ---------------------------- */
/* ------------------------------------------------------------------------- */

uint8_t clk_in, clk_out;

static void clock_poll()
{
	static uint8_t last_value, reminder_in, reminder_out;
	uint8_t this_value;

#if PLATFORM_bblab
	this_value = TCNT0;
#elif PLATFORM_arduino
	this_value = TCNT1L;
#else
#  error Platform support is missing!
#endif

	reminder_in += this_value - last_value;
	reminder_out += this_value - last_value;
	last_value = this_value;

	if (bbcfg->clkdiv == 0)
		bbcfg->clkdiv = 1;

	if (bbcfg->outclkdiv == 0)
		bbcfg->outclkdiv = bbcfg->clkdiv;

	clk_in = 0;
	while (reminder_in >=  bbcfg->clkdiv) {
		reminder_in -= bbcfg->clkdiv;
		clk_in++;
	}

	clk_out = 0;
	while (reminder_out >=  bbcfg->outclkdiv) {
		reminder_out -= bbcfg->outclkdiv;
		clk_out++;
	}
}

static void clock_setup()
{
#if PLATFORM_bblab
	// reset timer
	TCCR0A = 0;
	TCCR0B = 0;
	TIMSK = 0;

	// start timer (div = 256)
	TCCR0B = 4;
#elif PLATFORM_arduino
	// reset timer
	TCCR1A = 0;
	TCCR1B = 0;
	TCCR1C = 0;
	TIMSK1 = 0;

	// start timer (div = 256)
	TCCR1B = 4;
#else
#  error Platform support is missing!
#endif

	// reset clock monitor
	clock_poll();
}

static void clock_shutdown()
{
#if PLATFORM_bblab
	TCCR0A = 0;
	TCCR0B = 0;
	TIMSK = 0;
#elif PLATFORM_arduino
	TCCR1A = 0;
	TCCR1B = 0;
	TCCR1C = 0;
	TIMSK1 = 0;
#else
#  error Platform support is missing!
#endif
}


/* ------------------------------------------------------------------------- */
/* ---------------------------- PWM Out Interface -------------------------- */
/* ------------------------------------------------------------------------- */

static void pwm_setup()
{
#if PLATFORM_bblab
	// use OC1B (PB4) pin
	TCCR1 = _BV(CS10);
	GTCCR = _BV(PWM1B) | _BV(COM1B1);

	// start with a middle voltage level
	OCR1B = 0x40;
	OCR1C = 0x7f;

	// configure pin as output
	DDRB |= (1 << 4);
#elif PLATFORM_arduino
	// use OC0A (PD6) pin in Fast PWM mode
	TCCR0A = _BV(COM0A1) | _BV(WGM01) | _BV(WGM00);

	// use normal clock without prescaling (--> PWM at approx. 64 kHz)
	TCCR0B = _BV(CS00);

	// start with a middle voltage level
	OCR0A = 0x80;

	// no interrupts
	TIMSK0 = 0;

	// configure pin as output
	DDRD |= (1 << 6);
#else
#  error Platform support is missing!
#endif
}

static void pwm_shutdown()
{
#if PLATFORM_bblab
	TCCR1 = 0;
	GTCCR = 0;
	DDRB &= ~(1 << 4);
#elif PLATFORM_arduino
	TCCR0A = 0;
	TCCR0B = 0;
	TIMSK0 = 0;
	DDRD &= ~(1 << 6);
#else
#  error Platform support is missing!
#endif
}

static inline void pwm_set(uint8_t value)
{
#if PLATFORM_bblab
	if ((value & 0x80) == 0)
		OCR1C = 0x7f;
	else if ((value & 0xc0) == 0x80)
		OCR1C = 0x3f;
	else if ((value & 0xe0) == 0xc0)
		OCR1C = 0x1f;
	OCR1B = value;
#elif PLATFORM_arduino
	OCR0A = value;
#else
#  error Platform support is missing!
#endif
}


/* ------------------------------------------------------------------------- */
/* ----------------------------- ADC In Interface -------------------------- */
/* ------------------------------------------------------------------------- */

uint8_t adc_idx;
uint8_t adc_value[4];

#if PLATFORM_bblab
uint8_t adc_mux_cfg[4] PROGMEM = {
	_BV(ADLAR) |         0 | _BV(MUX0), // PB2
	_BV(ADLAR) | _BV(MUX1) | _BV(MUX0), // PB3
	_BV(ADLAR) | _BV(MUX1) |         0, // PB4
	_BV(ADLAR) |         0 |         0  // PB5
};
#endif

static void adc_poll()
{
#if PLATFORM_bblab
	if ((ADCSRA & _BV(ADIF)) != 0)
	{
		// PORTB |= _BV(5);

		adc_value[adc_idx] = ADCH;
		if (adc_value[adc_idx] == 0xff)
			adc_value[adc_idx]--;

		adc_idx = (adc_idx + 1) % 4;

		// start next conversion
		ADCSRA |= _BV(ADSC) | _BV(ADIF);

		// configure mux for next channel
		ADMUX = pgm_read_byte(&adc_mux_cfg[(adc_idx + 1) % 4]);

		// PORTB &= ~_BV(5);
	}
#elif PLATFORM_arduino
	if ((ADCSRA & _BV(ADIF)) != 0)
	{
		PORTC |= 0x20;

		adc_value[adc_idx] = ADCH;
		if (adc_value[adc_idx] == 0xff)
			adc_value[adc_idx]--;

		adc_idx = (adc_idx + 1) % 4;

		// start next conversion
		ADCSRA |= _BV(ADSC) | _BV(ADIF);

		// configure mux for next channel
		ADMUX = _BV(REFS0) | _BV(ADLAR) | ((adc_idx + 1) % 4);

		PORTC &= ~0x20;
	}
#else
#  error Platform support is missing!
#endif
}

static void adc_setup()
{
	adc_idx = 0;
	adc_value[0] = 0xff;
	adc_value[1] = 0xff;
	adc_value[2] = 0xff;
	adc_value[3] = 0xff;

#if PLATFORM_bblab
	// DDRB |= _BV(5);
	// PORTB &= ~_BV(5);

	// configure mux for channel 0
	ADMUX = pgm_read_byte(&adc_mux_cfg[0]);

	// configure ADC clock (div = 8)
	ADCSRB = 0;
	ADCSRA = _BV(ADPS1) |  _BV(ADPS0);

	// start ADC
	ADCSRA |= _BV(ADEN) | _BV(ADSC) | _BV(ADIF);

	// configure mux for next channel
	ADMUX = pgm_read_byte(&adc_mux_cfg[1]);
#elif PLATFORM_arduino
	DDRC |= 0x20;
	PORTC &= ~0x20;

	// configure mux for channel 0
	ADMUX = _BV(REFS0) | _BV(ADLAR);

	// configure ADC clock (div = 8)
	ADCSRB = 0;
	ADCSRA = _BV(ADPS1) | _BV(ADPS0);

	// start ADC
	ADCSRA |= _BV(ADEN) | _BV(ADSC) | _BV(ADIF);

	// configure mux for channel 1
	ADMUX = _BV(REFS0) | _BV(ADLAR) | 1;
#else
#  error Platform support is missing!
#endif
}

static void adc_shutdown()
{
#if PLATFORM_bblab
	while ((ADCSRA & _BV(ADSC)) != 0) { }
	ADCSRA |= _BV(ADIF);

	ADCSRA = 0;
	ADCSRB = 0;
#elif PLATFORM_arduino
	while ((ADCSRA & _BV(ADSC)) != 0) { }
	ADCSRA |= _BV(ADIF);

	DDRC |= 0x20;
	PORTC &= ~0x20;

	ADCSRA = 0;
	ADCSRB = 0;
#else
#  error Platform support is missing!
#endif

	adc_idx = 0;
	adc_value[0] = 0xff;
	adc_value[1] = 0xff;
	adc_value[2] = 0xff;
	adc_value[3] = 0xff;
}

static inline uint8_t adc_peek(uint8_t channel)
{
	return adc_value[channel];
}

static inline uint8_t adc_get(uint8_t channel)
{
	uint8_t value = adc_value[channel];
	adc_value[channel] = 0xff;
	return value;
}


/* ------------------------------------------------------------------------- */
/* --------------------------- Digital I/O Interface ----------------------- */
/* ------------------------------------------------------------------------- */

static void digi_setup(uint8_t out_channels)
{
#if PLATFORM_bblab
	for (uint8_t i = 0; i < 4; i++) {
		if ((out_channels & (1 << i)) == 0)
			continue;
		DDRB |= (4 << i);
		PORTB &= ~(4 << i);
	}
#elif PLATFORM_arduino
	for (uint8_t i = 0; i < 4; i++) {
		if ((out_channels & (1 << i)) == 0)
			continue;
		DDRC |= (1 << i);
		PORTC &= ~(1 << i);
	}
#else
#  error Platform support is missing!
#endif
}

static void digi_shutdown()
{
#if PLATFORM_bblab
	for (uint8_t i = 0; i < 4; i++) {
		DDRB &= ~(4 << i);
		PORTB &= ~(4 << i);
	}
#elif PLATFORM_arduino
	for (uint8_t i = 0; i < 4; i++) {
		DDRC &= ~(1 << i);
		PORTC &= ~(1 << i);
	}
#else
#  error Platform support is missing!
#endif
}

static inline void digi_set(uint8_t bits)
{
#if PLATFORM_bblab
	PORTB = (PORTB & 0x03) | ((bits & 0x0f) << 2);
#elif PLATFORM_arduino
	PORTC = (PORTC & 0xf0) | (bits & 0x0f);
#else
#  error Platform support is missing!
#endif
}

static inline uint8_t digi_get()
{
#if PLATFORM_bblab
	return  (PINB >> 2) & 0x0f;
#elif PLATFORM_arduino
	return PINC & 0x0f;
#else
#  error Platform support is missing!
#endif
}


/* ------------------------------------------------------------------------- */
/* -------------------------------- Engine --------------------------------- */
/* ------------------------------------------------------------------------- */

static void engine_shutdown()
{
	clock_shutdown();
	pwm_shutdown();
	adc_shutdown();
	digi_shutdown();
}

static void engine_setup()
{
	uint8_t bits;

	engine_shutdown();

	adc_setup();

	bits = 0;
	if ((bbcfg->outputmask & 0x01) != 0)
		bits |= 1 << 0;
	if ((bbcfg->outputmask & 0x02) != 0)
		bits |= 1 << 1;
	if ((bbcfg->outputmask & 0x04) != 0)
		bits |= 1 << 2;
	if ((bbcfg->outputmask & 0x08) != 0)
		bits |= 1 << 3;
	digi_setup(bits);

	if (bbcfg->pwmoutptr != 0)
		pwm_setup();

	clock_setup();
}

static void engine_output()
{
	if (bbcfg->outlen == 0)
		return;

	bbcfg->outidx += clk_out;
	while (bbcfg->outidx >= bbcfg->outlen)
		bbcfg->outidx -= bbcfg->outlen;

	if (bbcfg->digioutptr != 0) {
		uint8_t bits = engine_mem[bbcfg->digioutptr + (bbcfg->outidx >> 1)];
		if ((bbcfg->outidx & 1) != 0)
			bits = bits >> 4;
		else
			bits = bits & 0x0f;
		digi_set(bits);
	}

	if (bbcfg->pwmoutptr != 0) {
		pwm_set(engine_mem[bbcfg->pwmoutptr + bbcfg->outidx]);
	}
}

static void engine_input()
{
	if (bbcfg->samplelen == 0)
		return;

	bbcfg->sampleidx += clk_in;
	while (bbcfg->sampleidx >= bbcfg->samplelen)
		bbcfg->sampleidx -= bbcfg->samplelen;

	if (clk_in > 1) {
		for (int8_t i = 3; i > 0; i--) {
			bbcfg->collision_tick[i] = bbcfg->collision_tick[i-1] + clk_in < 65000 ? bbcfg->collision_tick[i-1] + clk_in : 65000;
			bbcfg->collision_duration[i] = bbcfg->collision_duration[i-1];
		}
		bbcfg->collision_tick[0] = 0;
		bbcfg->collision_duration[0] = clk_in;
	} else {
		for (int8_t i = 0; i < 4; i++)
			bbcfg->collision_tick[i] = bbcfg->collision_tick[i] < 65000 ? bbcfg->collision_tick[i] + 1 : 65000;
	}

	uint8_t bits = digi_get();

	if (bbcfg->trigger_state == TRIGGER_STATE_INIT)
	{
		bbcfg->trigger_counter = bbcfg->trigger_init_ticks;
		if (bbcfg->trigger_counter > 0)
			bbcfg->trigger_state = TRIGGER_STATE_FILL;
	}
	if (bbcfg->trigger_state == TRIGGER_STATE_FILL)
	{
		if (bbcfg->trigger_counter > clk_in) {
			bbcfg->trigger_counter -= clk_in;
		} else {
			if (bbcfg->pretrig_mode_channel != 0)
				bbcfg->trigger_state = TRIGGER_STATE_PRETRIG;
			else if (bbcfg->trigger_mode_channel != 0)
				bbcfg->trigger_state = TRIGGER_STATE_TRIGGER;
			else
				bbcfg->trigger_state = TRIGGER_STATE_POSTTRIGGER;
			bbcfg->trigger_counter = 0;
		}
	}
	if (bbcfg->trigger_state >= TRIGGER_STATE_PRETRIG && bbcfg->trigger_state <= TRIGGER_STATE_TRIGGER_PHASE2)
	{
		uint8_t mode_channel, ref_value, current_val = 0xff;

		if (bbcfg->trigger_state == TRIGGER_STATE_PRETRIG || bbcfg->trigger_state == TRIGGER_STATE_PRETRIG_PHASE2) {
			mode_channel = bbcfg->pretrig_mode_channel;
			ref_value = bbcfg->pretrig_value;
		} else {
			mode_channel = bbcfg->trigger_mode_channel;
			ref_value = bbcfg->trigger_value;
		}

		switch (mode_channel & 0x0f)
		{
		case CHANNEL_DIGI1:
			current_val = (bits & (1 << 0)) ? 1 : 0;
			break;
		case CHANNEL_DIGI2:
			current_val = (bits & (1 << 1)) ? 1 : 0;
			break;
		case CHANNEL_DIGI3:
			current_val = (bits & (1 << 2)) ? 1 : 0;
			break;
		case CHANNEL_DIGI4:
			current_val = (bits & (1 << 3)) ? 1 : 0;
			break;
		case CHANNEL_ANALOG1:
			current_val = adc_peek(0);
			break;
		case CHANNEL_ANALOG2:
			current_val = adc_peek(1);
			break;
		case CHANNEL_ANALOG3:
			current_val = adc_peek(2);
			break;
		case CHANNEL_ANALOG4:
			current_val = adc_peek(3);
			break;
		case CHANNEL_OUTIDX:
			current_val = bbcfg->outidx > 255 ? 255 : bbcfg->outidx;
			break;
		}

		if (current_val != 0xff)
		{
			switch (mode_channel & 0xf0)
			{
			case TRIGMODE_LT:
				if (current_val < ref_value)
					goto trigger_met;
				break;
			case TRIGMODE_LE:
				if (current_val <= ref_value)
					goto trigger_met;
				break;
			case TRIGMODE_EQ:
				if (current_val == ref_value)
					goto trigger_met;
				break;
			case TRIGMODE_GE:
				if (current_val >= ref_value)
					goto trigger_met;
				break;
			case TRIGMODE_GT:
				if (current_val > ref_value)
					goto trigger_met;
				break;
			case TRIGMODE_LT_GE:
				if (bbcfg->trigger_state == TRIGGER_STATE_PRETRIG && current_val < ref_value)
					bbcfg->trigger_state = TRIGGER_STATE_PRETRIG_PHASE2;
				else if (bbcfg->trigger_state == TRIGGER_STATE_TRIGGER && current_val < ref_value)
					bbcfg->trigger_state = TRIGGER_STATE_TRIGGER_PHASE2;
				else if (bbcfg->trigger_state == TRIGGER_STATE_PRETRIG_PHASE2 && current_val >= ref_value)
					goto trigger_met;
				else if (bbcfg->trigger_state == TRIGGER_STATE_TRIGGER_PHASE2 && current_val >= ref_value)
					goto trigger_met;
				break;
			case TRIGMODE_GT_LE:
				if (bbcfg->trigger_state == TRIGGER_STATE_PRETRIG && current_val > ref_value)
					bbcfg->trigger_state = TRIGGER_STATE_PRETRIG_PHASE2;
				else if (bbcfg->trigger_state == TRIGGER_STATE_TRIGGER && current_val > ref_value)
					bbcfg->trigger_state = TRIGGER_STATE_TRIGGER_PHASE2;
				else if (bbcfg->trigger_state == TRIGGER_STATE_PRETRIG_PHASE2 && current_val <= ref_value)
					goto trigger_met;
				else if (bbcfg->trigger_state == TRIGGER_STATE_TRIGGER_PHASE2 && current_val <= ref_value)
					goto trigger_met;
				break;
			}
		}

		if (0) {
trigger_met:
			bbcfg->trigger_counter = 0;
			if (bbcfg->trigger_state == TRIGGER_STATE_PRETRIG || bbcfg->trigger_state == TRIGGER_STATE_PRETRIG_PHASE2) {
				if (bbcfg->trigger_mode_channel != 0)
					bbcfg->trigger_state = TRIGGER_STATE_TRIGGER;
				else
					bbcfg->trigger_state = TRIGGER_STATE_POSTTRIGGER;
			} else {
				bbcfg->trigger_state = TRIGGER_STATE_POSTTRIGGER;
			}
		} else {
			bbcfg->trigger_counter = current_val;
		}

	}
	if (bbcfg->trigger_state == TRIGGER_STATE_POSTTRIGGER)
	{
		bbcfg->trigger_counter += clk_in;
		if (bbcfg->trigger_counter > bbcfg->posttrig_nticks) {
			bbcfg->trigger_state = TRIGGER_STATE_INIT;
			engine_state = STATE_STANDBY;
		}
	}

	if (bbcfg->digisampleptr != 0) {
		if ((bbcfg->sampleidx & 1) != 0)
			engine_mem[bbcfg->digisampleptr + (bbcfg->sampleidx >> 1)] =
					(engine_mem[bbcfg->digisampleptr + (bbcfg->sampleidx >> 1)] & 0x0f) | (bits << 4);
		else
			engine_mem[bbcfg->digisampleptr + (bbcfg->sampleidx >> 1)] =
					(engine_mem[bbcfg->digisampleptr + (bbcfg->sampleidx >> 1)] & 0xf0) | (bits & 0x0f);
	}

	for (uint8_t i = 0; i < 4; i++) {
		if (bbcfg->adcsampleptr[i] != 0)
			engine_mem[bbcfg->adcsampleptr[i] + bbcfg->sampleidx] = adc_get(i);
	}

	bbcfg->sampleoutidx = bbcfg->outidx;
}


/* ------------------------------------------------------------------------- */
/* ------------------------ Oscillator Calibration ------------------------- */
/* ------------------------------------------------------------------------- */

#if PLATFORM_bblab
/* Calibrate the RC oscillator to 8.25 MHz. The core clock of 16.5 MHz is
 * derived from the 66 MHz peripheral clock by dividing. Our timing reference
 * is the Start Of Frame signal (a single SE0 bit) available immediately after
 * a USB RESET. We first do a binary search for the OSCCAL value and then
 * optimize this value with a neighboorhod search.
 * This algorithm may also be used to calibrate the RC oscillator directly to
 * 12 MHz (no PLL involved, can therefore be used on almost ALL AVRs), but this
 * is wide outside the spec for the OSCCAL value and the required precision for
 * the 12 MHz clock! Use the RC oscillator calibrated to 12 MHz for
 * experimental purposes only!
 */
static void calibrateOscillator(void)
{
	uchar step = 128;
	uchar trialValue = 0, optimumValue;
	int x, optimumDev, targetValue = (unsigned)(1499 * (double)F_CPU / 10.5e6 + 0.5);

	/* do a binary search: */
	do {
		OSCCAL = trialValue + step;
		x = usbMeasureFrameLength(); /* proportional to current real frequency */
		if (x < targetValue) /* frequency still too low */
			trialValue += step;
		step >>= 1;
	} while (step > 0);
	/* We have a precision of +/- 1 for optimum OSCCAL here */
	/* now do a neighborhood search for optimum value */
	optimumValue = trialValue;
	optimumDev = x; /* this is certainly far away from optimum */
	for (OSCCAL = trialValue - 1; OSCCAL <= trialValue + 1; OSCCAL++) {
		x = usbMeasureFrameLength() - targetValue;
		if (x < 0)
			x = -x;
		if (x < optimumDev) {
			optimumDev = x;
			optimumValue = OSCCAL;
		}
	}
	OSCCAL = optimumValue;
}

/*
Note: This calibration algorithm may try OSCCAL values of up to 192 even if
the optimum value is far below 192. It may therefore exceed the allowed clock
frequency of the CPU in low voltage designs!
You may replace this search algorithm with any other algorithm you like if
you have additional constraints such as a maximum CPU clock.
For version 5.x RC oscillators (those with a split range of 2x128 steps, e.g.
ATTiny25, ATTiny45, ATTiny85), it may be useful to search for the optimum in
both regions.
*/

void usbEventResetReady(void)
{
	calibrateOscillator();
	/* store the calibrated value in EEPROM */
	eeprom_write_byte((void*)0, OSCCAL);
}
#endif


/* ------------------------------------------------------------------------- */
/* --------------------------------- Main ---------------------------------- */
/* ------------------------------------------------------------------------- */

int main(void)
{
#if PLATFORM_bblab
	/* calibration value from last time */
	uint8_t calibrationValue = eeprom_read_byte((void*)0);
	if(calibrationValue != 0xff)
		OSCCAL = calibrationValue;
#endif

	uart_setup();
	putstr("\r\n\r\n*** Just a test - booting up ***\r\n");

	wdt_enable(WDTO_1S);
	/* Even if you don't use the watchdog, turn it off here. On newer devices,
	 * the status of the watchdog (on/off, period) is PRESERVED OVER RESET!
	 */

	putstr("usbInit...\r\n");
	/* RESET status: all port bits are inputs without pull-up.
	 * That's the way we need D+ and D-. Therefore we don't need any
	 * additional hardware initialization.
	 */
	usbInit();

	/* enforce re-enumeration, do this while interrupts are disabled! */
	putstr("usbDeviceDisconnect...\r\n");
	usbDeviceDisconnect();
	/* fake USB disconnect for > 250 ms */
	for (uchar i = 0; --i; ) {
		wdt_reset();
		_delay_ms(1);
	}
	putstr("usbDeviceConnect...\r\n");
	usbDeviceConnect();

	putstr("enable clock...\r\n");
	clock_setup();

	putstr("enable interrupts...\r\n");
	sei();

	putstr("main loop...\r\n");
	while (1)
	{
		wdt_reset();
		usbPoll();
		adc_poll();

		switch (engine_state)
		{
		case STATE_INIT:
			engine_state = STATE_HALT;
			break;
		case STATE_HALT:
			break;
		case STATE_CONFIG:
			putstr("setup...\r\n");
			engine_setup();
			engine_state = STATE_RECORD;
			break;
		case STATE_RECORD:
			clock_poll();
			if (clk_out > 0) {
				putch('r');
				engine_output();
			}
			if (clk_in > 0) {
				putch('R');
				engine_input();
			}
			break;
		case STATE_STANDBY:
			clock_poll();
			if (clk_out > 0) {
				putch('S');
				engine_output();
			}
			break;
		case STATE_SHUTDOWN:
			putstr("\r\nshutdown...\r\n");
			engine_shutdown();
			engine_state = STATE_HALT;
			break;
		default:
			putstr("invalid engine state!\r\n");
			break;
		}
	}

	return 0;
}


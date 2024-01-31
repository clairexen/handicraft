/*
 *  icestickprog -- simple programming tool for modified Lattice iCEstick dev boards
 *
 *  Copyright (C) 2014  Clifford Wolf <clifford@clifford.at>
 *  
 *  Permission to use, copy, modify, and/or distribute this software for any
 *  purpose with or without fee is hereby granted, provided that the above
 *  copyright notice and this permission notice appear in all copies.
 *  
 *  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 *  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 *  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 *  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 *  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 *  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 *
 *  Usage example: ./icestickprog < LED_Rotation_bitmap.bin
 *
 *  This tool is programming the iCE FPGA directly, not the serial prom. In
 *  order for this to work you have to desolder the flash chip and one zero
 *  ohm resistor and connect the FT2232H SI pin directly to the iCE SPI_SI
 *  pin, as shown on this picture:
 *
 *     http://www.clifford.at/gallery/2014-elektronik/IMG_20141115_183838
 *
 *  See "icestickflash" in this directory for a programming tool that programs
 *  the serial flash chip on an unhacked icestick board.
 */

#define _GNU_SOURCE

#include <ftdi.h>
#include <stdio.h>
#include <unistd.h>

struct ftdi_context ftdic;

void check_rx()
{
	while (1) {
		unsigned char data;
		int rc = ftdi_read_data(&ftdic, &data, 1);
		if (rc <= 0) break;
		printf("unexpected rx byte: %02x\n", data);
	}
}

void error()
{
	check_rx();
	printf("ABORT.\n");
	ftdi_usb_close(&ftdic);
	ftdi_deinit(&ftdic);
	exit(1);
}

unsigned char recv_byte()
{
	unsigned char data;
	while (1) {
		int rc = ftdi_read_data(&ftdic, &data, 1);
		if (rc < 0) {
			printf("Read error.\n");
			error();
		}
		if (rc == 1)
			break;
		usleep(100);
	}
	return data;
}

void send_byte(unsigned char data)
{
	int rc = ftdi_write_data(&ftdic, &data, 1);
	if (rc != 1) {
		printf("Write error (single byte, rc=%d, expected %d).\n", rc, 1);
		error();
	}
}

void send_spi(unsigned char *data, int n)
{
	send_byte(0x11);
	send_byte(n-1);
	send_byte((n-1) >> 8);

	int rc = ftdi_write_data(&ftdic, data, n);
	if (rc != n) {
		printf("Write error (chunk, rc=%d, expected %d).\n", rc, n);
		error();
	}
}

void set_gpio(int slavesel_b, int creset_b)
{
	unsigned char gpio = 1;

	if (slavesel_b) {
		// ADBUS4 (GPIOL0)
		gpio |= 0x10;
	}

	if (creset_b) {
		// ADBUS7 (GPIOL3)
		gpio |= 0x80;
	}

	send_byte(0x80);
	send_byte(gpio);
	send_byte(0x93);
}

int get_cdone()
{
	unsigned char data;
	send_byte(0x81);
	data = recv_byte();
	// ADBUS6 (GPIOL2)
	return (data & 0x40) != 0;
}

int main()
{
	// ---------------------------------------------------------
	// Initialize USB connection to FT2232H
	// ---------------------------------------------------------

	printf("init..\n");

	ftdi_init(&ftdic);
	ftdi_set_interface(&ftdic, INTERFACE_A);

	if (ftdi_usb_open(&ftdic, 0x0403, 0x6010)) {
		printf("Can't find iCEstick USB device (vedor_id 0x0403, device_id 0x6010).\n");
		error();
	}

	if (ftdi_usb_reset(&ftdic)) {
		printf("Failed to reset iCEstick USB device.\n");
		error();
	}

	if (ftdi_usb_purge_buffers(&ftdic)) {
		printf("Failed to purge buffers on iCEstick USB device.\n");
		error();
	}

	if (ftdi_set_bitmode(&ftdic, 0xff, BITMODE_MPSSE) < 0) {
		printf("Failed set BITMODE_MPSSE on iCEstick USB device.\n");
		error();
	}

	// enable clock divide by 5
	send_byte(0x8b);

	// set 6 MHz clock
	send_byte(0x86);
	send_byte(0x00);
	send_byte(0x00);

	printf("cdone: %s\n", get_cdone() ? "high" : "low");

	set_gpio(1, 1);
	usleep(100);


	// ---------------------------------------------------------
	// Reset
	// ---------------------------------------------------------

	printf("reset..\n");

	set_gpio(0, 0);
	usleep(100);

	set_gpio(0, 1);
	usleep(2000);

	printf("cdone: %s\n", get_cdone() ? "high" : "low");


	// ---------------------------------------------------------
	// Program
	// ---------------------------------------------------------

	printf("prog..\n");

	while (1)
	{
		static unsigned char buffer[4096];
		int rc = read(0, buffer, 4096);
		if (rc <= 0) break;
		printf("sending %d bytes.\n", rc);
		send_spi(buffer, rc);
	}

	// add 48 dummy bits
	send_byte(0x8f);
	send_byte(0x05);
	send_byte(0x00);

	// add 1 more dummy bit
	send_byte(0x8e);
	send_byte(0x00);

	printf("cdone: %s\n", get_cdone() ? "high" : "low");


	// ---------------------------------------------------------
	// Exit
	// ---------------------------------------------------------

	printf("Bye.\n");
	ftdi_disable_bitbang(&ftdic);
	ftdi_usb_close(&ftdic);
	ftdi_deinit(&ftdic);
	return 0;

}


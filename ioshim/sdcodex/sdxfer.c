#include <stdint.h>
#include <string.h>
#include <assert.h>

#define IOSHIM_CTRLPC (*(volatile uint32_t *)0x10000000)
#define IOSHIM_MEM ((volatile uint8_t *)0x10008000)

#define SDXFER_CNT0_OFFSET 100
#define SDXFER_CNT1_OFFSET 101

#define SDXFER_BUF0_OFFSET 128
#define SDXFER_BUF1_OFFSET 192

#define SDXFER_CNT0 (IOSHIM_MEM[SDXFER_CNT0_OFFSET])
#define SDXFER_CNT1 (IOSHIM_MEM[SDXFER_CNT1_OFFSET])

#define SDXFER_BUF0 (IOSHIM_MEM + SDXFER_BUF0_OFFSET)
#define SDXFER_BUF1 (IOSHIM_MEM + SDXFER_BUF1_OFFSET)

// generated using ioshim_asm from sdxfer.ios
static const uint8_t sdxfer_kernel[] = {
	0xf4, 0xc6, 0xe5, 0xc6, 0xd0, 0xc0, 0xb2, 0xc0, 0xaa, 0xc2, 0x1f, 0xf8,
	0x05, 0xe8, 0x00, 0x00, 0x20, 0xc8, 0xd2, 0xfa, 0x2b, 0x21, 0x0a, 0xb7,
	0x9b, 0x02, 0x0a, 0xf0, 0xd2, 0xfa, 0xdf, 0xf9, 0x1e, 0xf8, 0x10, 0xe8,
	0x00, 0x00, 0x20, 0xcc, 0xd2, 0xfa, 0x2b, 0x21, 0x0a, 0xb7, 0x9b, 0x02,
	0x15, 0xf0, 0xd2, 0xfa, 0xde, 0xf9, 0x05, 0xe0, 0x00, 0x00
};

static int sdxfer_nextbuf;

static void sdxfer_begin()
{
	// reset ioshim
	IOSHIM_CTRLPC = -1;

	// load kernel
	assert(sizeof(sdxfer_kernel) < 100);
	memcpy((void*)IOSHIM_MEM, sdxfer_kernel, sizeof(sdxfer_kernel));
	__sync_synchronize();

	// initialize state variable
	sdxfer_nextbuf = 0;

	// reset counters
	SDXFER_CNT0 = 0;
	SDXFER_CNT1 = 0;

	// start ioshim
	IOSHIM_CTRLPC = 0;
}

static void sdxfer_block(void *data, int size)
{
	if (size == 0)
		return;

	assert(size > 0 && size <= 64 && size % 2 == 0);

	// wait for buffer to become ready
	while ((sdxfer_nextbuf ? SDXFER_CNT1 : SDXFER_CNT0) == 0) { /* wait */ }

	// copy data
	memcpy((void*)(sdxfer_nextbuf ? SDXFER_BUF1 : SDXFER_BUF0), data, size);
	__sync_synchronize();

	// notify ioshim
	if (sdxfer_nextbuf)
		SDXFER_CNT1 = size;
	else
		SDXFER_CNT0 = size;
	
	// flip buffers
	sdxfer_nextbuf = !sdxfer_nextbuf;
}

static void sdxfer_end()
{
	// wait for xfers to finish
	while (SDXFER_CNT0 != 0) { /* wait */ }
	while (SDXFER_CNT1 != 0) { /* wait */ }

	// reset ioshim
	IOSHIM_CTRLPC = -1;
}

void sdxfer(void *data, int size)
{
	sdxfer_begin();
	while (size > 0) {
		int blocksize = size > 64 ? 64 : size;
		sdxfer_block(data, blocksize);
		data += blocksize;
		size -= blocksize;
	}
	sdxfer_end();
}


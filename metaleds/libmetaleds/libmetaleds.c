
#include <SDL/SDL.h>
#include <SDL/SDL_net.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "libmetaleds.h"

int metaleds_width;
int metaleds_height;
int metaleds_size;

int metaleds_led_intensity[16] = { 0, 51, 115, 135, 150, 163, 176, 189, 199, 209, 217, 224, 232, 240, 247, 255 };
int metaleds_intensity_led[256];

enum {
	MODE_NONE,
	MODE_EXIT,
	MODE_FIO,
	MODE_FIO_SIM,
	MODE_NET,
	MODE_NET_SIM,
	MODE_SIM
} mode;

static unsigned char start_stop_seq[1] = { 0x80 };

static FILE *fp = NULL;
static SDL_Surface *s = 0;
static TCPsocket tcpsock;

static unsigned char ledlayout[8][8] = {
	{ 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 1, 1, 1, 1, 0, 0 },
	{ 0, 1, 2, 3, 3, 2, 1, 0 },
	{ 0, 1, 3, 4, 4, 3, 1, 0 },
	{ 0, 1, 3, 4, 4, 3, 1, 0 },
	{ 0, 1, 2, 3, 3, 2, 1, 0 },
	{ 0, 0, 1, 1, 1, 1, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0 },
};

static unsigned char ledcolor[256][2];

static void init_ledcolor()
{
	int i;
	for (i=0; i<256; i++) {
		int v = i < 32 ? 32 : i;
		ledcolor[i][0] = v < 128 ? 0 : (v-128)*2;
		ledcolor[i][1] = v < 128 ? v*2 : 255;
	}
}

static void init_intensity_led()
{
	int i, j, best_led, best_dif;
	for (i=0; i<256; i++)
	{
		best_led = 0;
		best_dif = abs(i-metaleds_led_intensity[0]);
		for (j=1; j<16; j++) {
			int this_dif =  abs(i-metaleds_led_intensity[j]);
			if (this_dif < best_dif) {
				best_led = j;
				best_dif = this_dif;
			}
		}
		metaleds_intensity_led[i] = best_led;
	}
}

int metaleds_init(const char *device_desc)
{
	char devstr[strlen(device_desc)+1];
	strcpy(devstr, device_desc);

	init_intensity_led();

	if (mode != MODE_NONE)
		return -1;

	char *type = strtok(devstr, ":");

	if (!type)
		return -1;

	if (!strcmp(type, "fio") || !strcmp(type, "fio+sim"))
	{
		char *filename = strtok(NULL, ":");
		char *width_str = strtok(NULL, ":");
		char *height_str = strtok(NULL, ":");

		if (!height_str)
			return -1;

		if (!(fp = fopen(filename, "wb")))
			return -1;

		if (fwrite(start_stop_seq, sizeof(start_stop_seq), 1, fp) != 1)
			return -1;

		metaleds_width = strtol(width_str, &width_str, 10);
		if (metaleds_width <= 0 || (width_str && width_str[0]))
			return -1;

		metaleds_height = strtol(height_str, &height_str, 10);
		if (metaleds_height <= 0 || (height_str && height_str[0]))
			return -1;

		if (!strcmp(type, "fio+sim"))
			goto plus_sim;

		metaleds_size = metaleds_width*metaleds_height;
		mode = MODE_FIO;
		return 0;
	}

	if (!strcmp(type, "net") || !strcmp(type, "net+sim"))
	{
		char *host_str = strtok(NULL, ":");
		char *port_str = strtok(NULL, ":");

		if (!port_str)
			return -1;

		int port = strtol(port_str, &port_str, 10);
		if (port <= 0 || port > 65535|| (port_str && port_str[0]))
			return -1;

		if (SDLNet_Init() < 0)
			return -1;

		IPaddress ip;
		if (SDLNet_ResolveHost(&ip, host_str, port) == -1) {
			fprintf(stderr, "SDLNet_ResolveHost: %s\n", SDLNet_GetError());
			return -1;
		}
		tcpsock = SDLNet_TCP_Open(&ip);
		if (!tcpsock) {
			fprintf(stderr, "SDLNet_TCP_Open: %s\n", SDLNet_GetError());
			return -1;
		}

		metaleds_width = 72;
		metaleds_height = 48;

		if (!strcmp(type, "net+sim"))
			goto plus_sim;

		metaleds_size = metaleds_width*metaleds_height;
		mode = MODE_NET;
		return 0;
	}

	if (!strcmp(type, "sim"))
	{
		char *width_str = strtok(NULL, ":");
		char *height_str = strtok(NULL, ":");

		if (!height_str)
			return -1;

		metaleds_width = strtol(width_str, &width_str, 10);
		if (metaleds_width <= 0 || (width_str && width_str[0]))
			return -1;

		metaleds_height = strtol(height_str, &height_str, 10);
		if (metaleds_height <= 0 || (height_str && height_str[0]))
			return -1;

plus_sim:
		if (SDL_Init(SDL_INIT_VIDEO) < 0)
			return -1;

		s = SDL_SetVideoMode(metaleds_width*8, metaleds_height*8, 24, 0);
		if (!s)
			return -1;

		init_ledcolor();

		metaleds_size = metaleds_width*metaleds_height;
		mode = MODE_SIM;

		if (!strcmp(type, "fio+sim"))
			mode = MODE_FIO_SIM;

		if (!strcmp(type, "net+sim"))
			mode = MODE_NET_SIM;

		return 0;
	}

	return -1;
}

metaleds_frame_p metaleds_malloc()
{
	metaleds_frame_p frame = malloc(metaleds_size);
	memset(frame, 0, metaleds_size);
	return frame;
}

extern void metaleds_clrsrc(metaleds_frame_p frame)
{
	memset(frame, 0, metaleds_size);
}

static int metaleds_frame_fio(metaleds_frame_p frame)
{
	if (fwrite(frame, metaleds_size, 1, fp) != 1)
		return -1;
	if (fwrite(start_stop_seq, sizeof(start_stop_seq), 1, fp) != 1)
		return -1;
	fflush(fp);
	return 0;
}

static int metaleds_frame_net(metaleds_frame_p frame)
{
	if (SDLNet_TCP_Send(tcpsock, frame, metaleds_size) != metaleds_size)
		return -1;
	if (SDLNet_TCP_Send(tcpsock, start_stop_seq, sizeof(start_stop_seq)) != sizeof(start_stop_seq))
		return -1;
	return 0;
}

static int metaleds_frame_sim(metaleds_frame_p frame)
{
	SDL_Event event;
	while (SDL_PollEvent(&event)) {
		if (event.type == SDL_KEYDOWN) {
			switch (event.key.keysym.sym)
			{
				case SDLK_q:
				case SDLK_ESCAPE:
					mode = MODE_EXIT;
					return -1;
			}
		}
		if (event.type == SDL_QUIT) {
			mode = MODE_EXIT;
			return -1;
		}
	}

	int i, py, px;
	for (i=0; i<metaleds_size; i++)
	{
		int x = METALEDS_IDX_TO_X(i);
		int y = METALEDS_IDX_TO_Y(i);

		int v = metaleds_led_intensity[frame[i]];

		int vrg = ledcolor[v][0];
		int vb  = ledcolor[v][1];

		for (py=0; py<8; py++)
		for (px=0; px<8; px++)
		{
			int i = (x*8+px) + (y*8+py)*(metaleds_width*8);
			int pvrg = vrg * ledlayout[py][px] / 4;
			int pvb  = vb  * ledlayout[py][px] / 4;
			((unsigned char*)s->pixels)[i*3 + 0] = pvb;
			((unsigned char*)s->pixels)[i*3 + 1] = pvrg;
			((unsigned char*)s->pixels)[i*3 + 2] = pvrg;
		}
	}

	SDL_UpdateRect(s, 0, 0, metaleds_width*8, metaleds_height*8);
	return 0;
}

int metaleds_frame(metaleds_frame_p frame)
{
	switch (mode)
	{
	case MODE_FIO:
		return metaleds_frame_fio(frame);
	case MODE_FIO_SIM:
		if (metaleds_frame_fio(frame) < 0)
			return -1;
		return metaleds_frame_sim(frame);
	case MODE_NET:
		return metaleds_frame_net(frame);
	case MODE_NET_SIM:
		if (metaleds_frame_net(frame) < 0)
			return -1;
		return metaleds_frame_sim(frame);
	case MODE_SIM:
		return metaleds_frame_sim(frame);
	default:
		return -1;
	}
}

extern void metaleds_free(metaleds_frame_p frame)
{
	free(frame);
}


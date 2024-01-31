#include <SDL/SDL.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <termios.h>
#include <stdint.h>
#include "beatlock.h"

SDL_Surface *screen;
struct beatlock_state bs;

int noise_countdown = 0;
bool beats[BEATLOCK_ENERGY_SAMPLES];

uint32_t xorshift32() {
	static uint32_t x = 314159265;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return x;
}

void sdl_update()
{
	struct SDL_Rect rect = { 0, 0, BEATLOCK_ENERGY_SAMPLES, 200 };
	SDL_FillRect(screen, &rect, 0x00000000);

	// signal energy over time

	float energy_min = bs.energy_samples[0];
	float energy_max = bs.energy_samples[0];

	for (int i = 0; i < BEATLOCK_ENERGY_SAMPLES; i++) {
		if (energy_min > bs.energy_samples[i])
			energy_min = bs.energy_samples[i];
		if (energy_max < bs.energy_samples[i])
			energy_max = bs.energy_samples[i];
	}

	rect.w = 1;
	for (rect.x = 0; rect.x < BEATLOCK_ENERGY_SAMPLES; rect.x++) {
		if (beats[rect.x]) {
			rect.w = 2, rect.h = 100, rect.y = 0;
			SDL_FillRect(screen, &rect, 0x00660066);
			rect.w = 1;
		}
		int lv = 5 + 80 * (bs.energy_samples[rect.x] - energy_min) / (energy_max - energy_min);
		rect.h = lv, rect.y = 100 - lv;
		SDL_FillRect(screen, &rect, 0x00ff8800);
	}

	rect.x = bs.energy_idx, rect.y = 0, rect.w = 10, rect.h = 100;
	SDL_FillRect(screen, &rect, 0x00000000);

	rect.x = bs.energy_idx, rect.y = 0, rect.w = 1, rect.h = 100;
	SDL_FillRect(screen, &rect, 0x00ffffff);

	// autocorrelation over period

	float conv_min = bs.conv_data[0];
	float conv_max = bs.conv_data[0];

	for (int i = 0; i < BEATLOCK_CONV_MAX - BEATLOCK_CONV_MIN + 1; i++) {
		if (conv_min > bs.conv_data[i])
			conv_min = bs.conv_data[i];
		if (conv_max < bs.conv_data[i])
			conv_max = bs.conv_data[i];
	}

	rect.w = 1;
	for (int i = 0; i < BEATLOCK_CONV_MAX - BEATLOCK_CONV_MIN + 1; i++) {
		int lv = 5 + 80 * (bs.conv_data[i] - conv_min) / (conv_max - conv_min);
		rect.x = BEATLOCK_CONV_MIN + i, rect.h = lv, rect.y = 200 - lv;
		SDL_FillRect(screen, &rect, 0x00880088);
	}

	rect.x = BEATLOCK_CONV_MIN + bs.conv_best, rect.y = 150, rect.h = 50;
	SDL_FillRect(screen, &rect, 0x00ffffff);

	rect.x = 0, rect.w = BEATLOCK_CONV_MIN, rect.y = 200-5, rect.h = 5;
	SDL_FillRect(screen, &rect, 0x00880088);

	// autocorrelation over period with selection filter

	float conv_filt_min = bs.conv_filtered[0];
	float conv_filt_max = bs.conv_filtered[0];

	for (int i = 0; i < BEATLOCK_CONV_MAX - BEATLOCK_CONV_MIN + 1; i++) {
		if (conv_filt_min > bs.conv_filtered[i])
			conv_filt_min = bs.conv_filtered[i];
		if (conv_filt_max < bs.conv_filtered[i])
			conv_filt_max = bs.conv_filtered[i];
	}

	rect.w = 1;
	for (int i = 0; i < BEATLOCK_CONV_MAX - BEATLOCK_CONV_MIN + 1; i++) {
		int lv = 5 + 90 * (bs.conv_filtered[i] - conv_filt_min) / (conv_filt_max - conv_filt_min);
		rect.x = 50 + BEATLOCK_CONV_MAX + BEATLOCK_CONV_MIN + i, rect.h = lv, rect.y = 200 - lv;
		if (bs.conv_best - BEATLOCK_CONV_WIN <= i && i <= bs.conv_best + BEATLOCK_CONV_WIN)
			SDL_FillRect(screen, &rect, 0x00008888);
		else
			SDL_FillRect(screen, &rect, 0x00006666);
	}

	rect.x = 50 + BEATLOCK_CONV_MAX + BEATLOCK_CONV_MIN + bs.conv_best, rect.y = 150, rect.h = 50;
	SDL_FillRect(screen, &rect, 0x00ffffff);

	rect.x = 50 + BEATLOCK_CONV_MAX, rect.w = BEATLOCK_CONV_MIN, rect.y = 200-5, rect.h = 5;
	SDL_FillRect(screen, &rect, 0x00006666);

	// left/right energy balance

	int lv_left = 90 * bs.left_energy / (bs.left_energy + bs.right_energy + 1);
	rect.x = 100 + 2*BEATLOCK_CONV_MAX, rect.w = 50, rect.y = 200-lv_left, rect.h = lv_left;
	SDL_FillRect(screen, &rect, 0x00008800);

	int lv_right = 90 * bs.right_energy / (bs.left_energy + bs.right_energy + 1);
	rect.x = 170 + 2*BEATLOCK_CONV_MAX, rect.w = 50, rect.y = 200-lv_right, rect.h = lv_right;
	SDL_FillRect(screen, &rect, 0x00008800);

	// double freq counter

	for (int i = 0; i < bs.double_freq_cnt; i++) {
		rect.x = 250 + 2*BEATLOCK_CONV_MAX, rect.w = 20, rect.y = 190 - 7*i, rect.h = 5;
		SDL_FillRect(screen, &rect, 0x00ffff00);
	}

	SDL_UpdateRect(screen, 0, 0, BEATLOCK_ENERGY_SAMPLES, 200);
}

int main(int argc, char **argv)
{
	SDL_Init(SDL_INIT_VIDEO);
	SDL_WM_SetCaption("BeatPlot", "BeatPlot");
	screen = SDL_SetVideoMode(BEATLOCK_ENERGY_SAMPLES, 200, 32, SDL_SWSURFACE);

	if (screen == NULL)
		return -1;

	if (argc == 1)
	{
		FILE *p = popen("aplay -fU8 -c1 -r22050 -", "w");
		int p_fd = fileno(p);

		uint8_t sample = 0;
		Uint32 last_sdl_update = 0;
		while (read(0, &sample, 1) == 1)
		{
			SDL_Event event;
			while (SDL_PollEvent(&event)) {
				if (event.type == SDL_QUIT)
					return 0;
				if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_q)
					return 0;
			}

			bool beat = beatlock_sample(&bs, sample);

			if (bs.energy_count == 0)
			{
				beats[bs.energy_idx] = beat;
				sdl_update();

				Uint32 now =  SDL_GetTicks();
				while (last_sdl_update + 8 > now) {
					SDL_Delay(1);
					now =  SDL_GetTicks();
				}
				last_sdl_update = now;
			}

			if (beat)
				noise_countdown = 1000;
			if (noise_countdown > 0) {
				noise_countdown--;
				sample += (xorshift32() % 32) - 16;
			}

			// sample = bs.last_sample;
			if (write(p_fd, &sample, 1) != 1)
				return 1;
		}
	}
	else
	{
		int fd = open(argv[1], O_RDONLY);

		if (fd < 0)
			return -1;

		struct termios newtio;
		tcgetattr(fd, &newtio);
		newtio.c_cflag = B115200 | CS8 | CREAD;
		newtio.c_iflag = IGNPAR;
		newtio.c_oflag = 0;
		newtio.c_lflag = 0;
		newtio.c_cc[VMIN] = 1;
		newtio.c_cc[VTIME] = 0;

		tcflush(fd, TCIFLUSH);
		tcsetattr(fd, TCSANOW, &newtio);

	next_frame:;
		uint8_t data = 0;
		int magic_idx = 0;
		const uint8_t magic[8] = { 0x79, 0xba, 0x24, 0x8f, 0x50, 0xb2, 0x1e, 0x78 };
		int timeout = 2 * sizeof(struct beatlock_state);
		while (read(fd, &data, 1) == 1) {
			fprintf(stderr, "<%02x>", data);
			if (data == magic[magic_idx])
				magic_idx++;
			else if (data == magic[0]) {
				fprintf(stderr, "?\n?");
				magic_idx = 1;
			} else {
				fprintf(stderr, magic_idx ? "?\n" : "?");
				magic_idx = 0;
			}
			if (magic_idx == 8)
				break;
			if (--timeout == 0) {
				fprintf(stderr, "TIMEOUT [%zd]\n", sizeof(struct beatlock_state));
				return -1;
			}
		}
		fprintf(stderr, "\n");

		size_t pos = 0;
		ssize_t rc;
		while ((rc = read(fd, ((uint8_t*)&bs) + pos, sizeof(struct beatlock_state)-pos)) > 0)
		{
			SDL_Event event;
			while (SDL_PollEvent(&event)) {
				if (event.type == SDL_QUIT)
					return 0;
				if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_q)
					return 0;
			}

			pos += rc;
			fprintf(stderr, ".");
			if (pos >= sizeof(struct beatlock_state))
			{
				fprintf(stderr, "\n");
				sdl_update();
				goto next_frame;
			}
		}
	}

	return 0;
}


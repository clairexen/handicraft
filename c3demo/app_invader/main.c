#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

static inline void setled(int v)
{
	*(volatile uint32_t*)0x20000000 = v;
}

uint32_t level_screens[] = {
	0b00010001110100010111010000100000,
	0b00010001000100010100010000100000,
	0b00010001100010100110010000100000,
	0b00010001000010100100010000100000,
	0b00011101110001000111011100100000,

	0b00010001110100010111010000101000,
	0b00010001000100010100010000101000,
	0b00010001100010100110010000101000,
	0b00010001000010100100010000101000,
	0b00011101110001000111011100101000,

	0b00100011101000101110100001010100,
	0b00100010001000101000100001010100,
	0b00100011000101001100100001010100,
	0b00100010000101001000100001010100,
	0b00111011100010001110111001010100,

	0b01000111010001011101000010100010,
	0b01000100010001010001000010100010,
	0b01000110001010011001000010010100,
	0b01000100001010010001000010010100,
	0b01110111000100011101110010001000,

	0b00100011101000101110100001000100,
	0b00100010001000101000100001000100,
	0b00100011000101001100100000101000,
	0b00100010000101001000100000101000,
	0b00111011100010001110111000010000,

	0b01000111010001011101000010001010,
	0b01000100010001010001000010001010,
	0b01000110001010011001000001010010,
	0b01000100001010010001000001010010,
	0b01110111000100011101110000100010,

	0b00100010101001010010111011001100,
	0b00100010101101011010100010101100,
	0b00101010101111011110110011101100,
	0b00101010101011010110100011000000,
	0b00010100101001010010111010101100
};

int level;
int player_x;
int player_bullet_x;
int player_bullet_y;

bool autopilot;
int autopilot_state;

bool invaders_chdir;
int invaders_dir;
int invaders_down;
int active_invader;
int invader_bullet_x;
int invader_bullet_y;
int invader_max_y;
int next_invader_max_y;

uint32_t blocks[32];
bool reset_blocks;

struct invader_pos_t {
	int x;
	int y;
} invaders[16];

uint32_t xorshift32() {
	static uint32_t x32 = 314159265;
	x32 ^= x32 << 13;
	x32 ^= x32 >> 17;
	x32 ^= x32 << 5;
	return x32;
}

void setpixel(int x, int y, uint8_t r, uint8_t g, uint8_t b)
{
	if (0 <= x && x < 32 && 0 <= y && y < 32) {
		uint32_t rgb = (r << 16) | (g << 8) | b;
		uint32_t addr = 4*x + 32*4*y + 0x10000000;
		*(volatile uint32_t*)addr = rgb;
	}
}

void place_block(int x, int y)
{
	blocks[y] |= 1 << x;
}

void place_barrier(int x, int y)
{
	place_block(x-3, y+1);
	place_block(x-2, y+1);
	place_block(x-1, y+1);
	place_block(x+1, y+1);
	place_block(x+2, y+1);
	place_block(x+3, y+1);

	place_block(x-3, y);
	place_block(x-2, y);
	place_block(x-1, y);
	place_block(x+0, y);
	place_block(x+1, y);
	place_block(x+2, y);
	place_block(x+3, y);

	place_block(x-2, y-1);
	place_block(x-1, y-1);
	place_block(x+0, y-1);
	place_block(x+1, y-1);
	place_block(x+2, y-1);

	place_block(x-1, y-2);
	place_block(x+0, y-2);
	place_block(x+1, y-2);

}

bool move_invader(int n)
{
	if (!n)
	{
		int n2 = xorshift32() % 16;

		int t = invaders[n].x;
		invaders[n].x = invaders[n2].x;
		invaders[n2].x = t;

		t = invaders[n].y;
		invaders[n].y = invaders[n2].y;
		invaders[n2].y = t;

		invader_max_y = next_invader_max_y;
		next_invader_max_y = 0;
	}

	int x = invaders[n].x;
	int y = invaders[n].y;

	setpixel(x, y, 0, 0, 0);
	setpixel(x-1, y+1, 0, 0, 0);
	setpixel(x+1, y+1, 0, 0, 0);

	if (invaders_dir > 0 && y > 0 && x == 31)
		invaders_chdir = true;

	if (invaders_dir < 0 && y > 0 && x == 0)
		invaders_chdir = true;

	if (invaders_chdir && !n) {
		invaders_chdir = false;
		invaders_dir = invaders_dir > 0 ? -1 : +1;
		invaders_down = 16;
	}

	x = x + invaders_dir;
	if (invaders_down) {
		invaders_down--;
		y++;
	}

	setpixel(x, y, 50, 30, 30);
	setpixel(x-1, y+1, 50, 30, 30);
	setpixel(x+1, y+1, 50, 30, 30);

	invaders[n].x = x;
	invaders[n].y = y;

	if (0 <= y && y < 31) {
		if (0 <= x && x <  31) blocks[y+1] &= ~(1 << (x+1));
		if (0 <  x && x <= 31) blocks[y+1] &= ~(1 << (x-1));
	}

	if (next_invader_max_y < y)
		next_invader_max_y = y;

	if (invader_max_y == y && player_x == x && invader_bullet_y < 0) {
		invader_bullet_x = x;
		invader_bullet_y = y+1;
		setpixel(invader_bullet_x, invader_bullet_y, 255, 0, 0);
	}

	return y >= 29 && x >= player_x-2 && x <= player_x+2;
}

void place_invader(int n, int x, int y)
{
	invaders[n].x = x;
	invaders[n].y = y;

	setpixel(x, y, 50, 30, 30);
	setpixel(x-1, y+1, 50, 30, 30);
	setpixel(x+1, y+1, 50, 30, 30);
}

void move_player_left()
{
	if (player_x > 0)
		player_x--;

	setpixel(player_x-1, 31, 50, 100, 100);
	setpixel(player_x+0, 31, 50, 100, 100);
	setpixel(player_x+1, 31, 50, 100, 100);
	setpixel(player_x+2, 31, 0, 0, 0);

	setpixel(player_x+0, 30, 50, 100, 100);
	setpixel(player_x+1, 30, 0, 0, 0);
}

void move_player_right()
{
	if (player_x < 31)
		player_x++;

	setpixel(player_x-2, 31, 0, 0, 0);
	setpixel(player_x-1, 31, 50, 100, 100);
	setpixel(player_x+0, 31, 50, 100, 100);
	setpixel(player_x+1, 31, 50, 100, 100);

	setpixel(player_x-1, 30, 0, 0, 0);
	setpixel(player_x+0, 30, 50, 100, 100);
}

void mysleep(int n)
{
	while (n--)
		for (int i = 0; i < 500000; i++)
			asm volatile ("");
}

void game()
{
	player_bullet_x = 0;
	player_bullet_y = -1;

	invaders_chdir = false;
	invaders_dir = +1;
	invaders_down = 0;
	active_invader = 0;
	invader_bullet_x = 0;
	invader_bullet_y = -1;
	invader_max_y = 0;
	next_invader_max_y = 0;

	if (reset_blocks)
		level = 0;
	else
		level++;

	for (int x = 0; x < 32; x++)
	for (int y = 0; y < 32; y++)
		if (y > 12 && y < 18 && level_screens[level*5+y-13] & (1 << (31-x)))
			setpixel(x, y, 200, 200, 0);
		else if (level == 0)
			setpixel(x, y, 0, 0, 0);

	if (level == 6) {
		mysleep(3);
		while (!autopilot)
			mysleep(1);
		reset_blocks = true;
		return;
	}

	for (int i = 0; i < level; i++)
	for (int x = 0; x < 4; x++)
		setpixel(2 + i*6 + x, 0, 50, 50, 0);

	mysleep(1);

	if (reset_blocks) {
		place_barrier(3, 27);
		place_barrier(11, 27);
		place_barrier(20, 27);
		place_barrier(28, 27);
		reset_blocks = false;
	}

	for (int x = 0; x < 32; x++)
	for (int y = 1; y < 32; y++)
		if (blocks[y] & (1 << x))
			setpixel(x, y, 0, 50, 50);
		else
			setpixel(x, y, 0, 0, 0);

	for (int x = 0; x < 4; x++)
	for (int y = 0; y < 4; y++)
		place_invader(4*x+y, 6 + 5*x, 2 + 4*y);

	if (level == 0)
		player_x = 10;
	else
		move_player_left();
	move_player_right();

	*(volatile uint32_t*)0x20000010 = 0;
	*(volatile uint32_t*)0x20000020 = 0;
	*(volatile uint32_t*)0x20000030 = 0;
	*(volatile uint32_t*)0x20000040 = 0;

	uint8_t ctrlbits[8];
	for (int k = 0; k < 8; k++)
		ctrlbits[k] = ((*(volatile uint32_t*)(0x20000014 + 0x10*(k / 2))) >> ((k % 2) ? 5 : 1)) & 7;
	
	uint32_t num_cycles_now, num_cycles_last;
	asm volatile ("rdcycle %0" : "=r"(num_cycles_last));

	for (int iter = 0;; iter++)
	{
		bool move_left = false;
		bool move_right = false;
		bool fire = false;

		for (int k = 0; k < 8; k++)
		{
			uint8_t bits = ((*(volatile uint32_t*)(0x20000014 + 0x10*(k / 2))) >> ((k % 2) ? 5 : 1)) & 7;

			if (bits == ctrlbits[k])
				continue;

			int old_pos = ctrlbits[k] >> 1;
			int new_pos = bits >> 1;

			old_pos = old_pos ^ (old_pos >> 1);
			new_pos = new_pos ^ (new_pos >> 1);

			bool old_fire = (ctrlbits[k] & 1) == 0;
			bool new_fire = (bits & 1) == 0;

			if (new_pos == ((old_pos+1) & 3))
				move_left = true;

			if (new_pos == ((old_pos-1) & 3))
				move_right = true;

			if (new_fire && !old_fire)
				fire = true;

			ctrlbits[k] = bits;
		}

		if (autopilot)
		{
			if (move_left || move_right || fire) {
				autopilot = false;
				if (level > 0) {
					reset_blocks = true;
					return;
				}
			}

			int x = 0;
			switch (autopilot_state) {
				case 0: x = 24; break;
				case 1: x = 16; break;
				case 2: x =  7; break;
				case 3: x = 15; break;
			}

			if (player_x-1 <= invader_bullet_x && player_x+1 >= invader_bullet_x && invader_bullet_y > 20 && x == player_x)
				autopilot_state = (autopilot_state+1) % 4;
			else if (((iter + 0x800) & 0x3fff) == 0 && player_bullet_y > 0)
				autopilot_state = (autopilot_state+1) % 4;

			if (x < player_x)
				move_left = (iter & 0xff) == 0;
			else if (x > player_x)
				move_right = (iter & 0xff) == 0;
			else if (player_bullet_y < 0)
				fire = (iter & 0xfff) == 0;
		}

		if (move_left)
			move_player_left();

		if (move_right)
			move_player_right();

		if (fire) {
			setpixel(player_bullet_x, player_bullet_y, 0, 0, 0);
			player_bullet_x = player_x;
			player_bullet_y = 29;
			setpixel(player_bullet_x, player_bullet_y, 255, 255, 255);
		}

		asm volatile ("rdcycle %0" : "=r"(num_cycles_now));
		if (num_cycles_now - num_cycles_last > 100000)
		{
			num_cycles_last = num_cycles_last + 100000;

			if (invader_bullet_y >= 0)
			{
				setpixel(invader_bullet_x, invader_bullet_y, 0, 0, 0);
				invader_bullet_y++;

				if (invader_bullet_y >= 32) {
					invader_bullet_y = -1;
				} else if (blocks[invader_bullet_y] & (1 << invader_bullet_x)) {
					setpixel(invader_bullet_x, invader_bullet_y, 0, 0, 0);
					blocks[invader_bullet_y] &= ~(1 << invader_bullet_x);
					invader_bullet_y = -1;
				} else {
					setpixel(invader_bullet_x, invader_bullet_y, 255, 0, 0);
					if (invader_bullet_y >= 30 && player_x-1 <= invader_bullet_x && player_x+1 >= invader_bullet_x) {
						autopilot = true;
						reset_blocks = true;
						return;
					}
				}
			}

			if (player_bullet_y >= 0)
			{
				setpixel(player_bullet_x, player_bullet_y, 0, 0, 0);
				player_bullet_y--;

				if (player_bullet_y == 0)
					player_bullet_y = -1;

				bool player_won = true;

				for (int i = 0; i < 16; i++)
				{
					if (invaders[i].y+1 == player_bullet_y &&
							invaders[i].x-1 <= player_bullet_x &&
							invaders[i].x+1 >= player_bullet_x)
					{
						int x = invaders[i].x;
						int y = invaders[i].y;

						setpixel(x, y, 0, 0, 0);
						setpixel(x-1, y+1, 0, 0, 0);
						setpixel(x+1, y+1, 0, 0, 0);

						invaders[i].y = -1000;
						player_bullet_y = -10;
					}

					if (invaders[i].y > 0)
						player_won = false;
				}

				if (player_won)
					return;

				if (blocks[player_bullet_y] & (1 << player_bullet_x)) {
					setpixel(player_bullet_x, player_bullet_y, 0, 0, 0);
					blocks[player_bullet_y] &= ~(1 << player_bullet_x);
					player_bullet_y = -1;
				} else
					setpixel(player_bullet_x, player_bullet_y, 255, 255, 255);
			}

			if (move_invader(active_invader)) {
				autopilot = true;
				reset_blocks = true;
				return;
			}

			active_invader = (active_invader + 1) % 16;
		}
	}
}

void main()
{
	printf("Space Invaders!\n");
	reset_blocks = true;
	autopilot = true;
	while (1) game();
}

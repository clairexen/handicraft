/*
 *  UltimateTicTacToe
 *
 *  Copyright (C) 2015  Clifford Wolf <clifford@clifford.at>
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
 *  See http://pr0gramm.com/top/404217 for rules.
 *
 *  Start with -o or -x to play against the computer. Without option
 *  the computer will play against itself. Use numeric keypad.
 */

#include <time.h>
#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>

int ticTacToeSolver(char board[3][3], char player, int px, int py)
{
	assert(board[px][py] == ' ');
	board[px][py] = player;

	int ret = +1;
	if (board[px][0] == player && board[px][1] == player && board[px][2] == player) goto ret;
	if (board[0][py] == player && board[1][py] == player && board[2][py] == player) goto ret;
	if (board[0][0] == player && board[1][1] == player && board[2][2] == player) goto ret;
	if (board[0][2] == player && board[1][1] == player && board[2][0] == player) goto ret;

	ret = +2;
	for (int x = 0; x < 3; x++)
	for (int y = 0; y < 3; y++)
		if (board[x][y] == ' ') {
			int r = -ticTacToeSolver(board, player == 'X' ? 'O' : 'X', x, y);
			if (r < ret) ret = r;
		}
	if (ret == +2)
		ret = 0;

ret:
	board[px][py] = ' ';
	return ret;
}

int ticTacToeProber(char board[3][3], char player)
{
	int ret = -2;
	for (int x = 0; x < 3; x++)
	for (int y = 0; y < 3; y++)
		if (board[x][y] == ' ') {
			int r = ticTacToeSolver(board, player, x, y);
			if (r > ret) ret = r;
		}
	if (ret == -2)
		ret = 0;

	return ret;
}

struct Board
{
	// dims: big-x, big-y, small-x, small-y
	char s_board[3][3][3][3];

	// dims: big-x, big-y
	char b_board[3][3];
	uint8_t b_count[3][3];

	// next move constraints
	int8_t current_bx, current_by;
	char current_player;

	// game status
	char winner;

	struct BoardRollbackBuffer
	{
		int8_t bx, by, sx, sy;
		int8_t current_bx, current_by;
	};

	struct PrintInfo
	{
		int line_nr;
		int score_x, score_o;
		int prognosis_x;
		int itercount;
		int max_queue_n;
		int max_depth;
		int min_stop_depth;
		int histcounters[96];

		void reset() {
			line_nr = 0;
			score_x = 0;
			score_o = 0;
			prognosis_x = 0;
			itercount = 0;
			max_queue_n = 0;
			max_depth = 0;
			min_stop_depth = 1000;
			memset(histcounters, 0, sizeof(histcounters));
		}
	} pi;

	Board()
	{
		memset(s_board, ' ', 3*3*3*3);
		memset(b_board, ' ', 3*3);
		memset(b_count, 0, 3*3);

		current_bx = -1;
		current_by = -1;
		current_player = 'X';
		winner = ' ';

		pi.reset();
	}

	int set(int bx, int by, int sx, int sy, BoardRollbackBuffer *rbuf = NULL)
	{
		// ret=0 nothing gained
		// ret=1 win in this b_board
		// ret=100 win on entire board
		int ret = 0;

		if (rbuf) {
			rbuf->bx = bx;
			rbuf->by = by;
			rbuf->sx = sx;
			rbuf->sy = sy;
			rbuf->current_bx = current_bx;
			rbuf->current_by = current_by;
		}

		assert(s_board[bx][by][sx][sy] == ' ');
		assert(b_board[bx][by] == ' ');

		s_board[bx][by][sx][sy] = current_player;
		b_count[bx][by]++;

		if (s_board[bx][by][sx][0] == current_player && s_board[bx][by][sx][1] == current_player && s_board[bx][by][sx][2] == current_player)
			goto win_b_board;

		if (s_board[bx][by][0][sy] == current_player && s_board[bx][by][1][sy] == current_player && s_board[bx][by][2][sy] == current_player)
			goto win_b_board;

		if (sx == sy && s_board[bx][by][0][0] == current_player && s_board[bx][by][1][1] == current_player && s_board[bx][by][2][2] == current_player)
			goto win_b_board;

		if (sx == (2-sy) && s_board[bx][by][0][2] == current_player && s_board[bx][by][1][1] == current_player && s_board[bx][by][2][0] == current_player)
			goto win_b_board;

		if (0) {
	win_b_board:
			b_board[bx][by] = current_player;
			ret = 1;

			if (b_board[bx][0] == current_player && b_board[bx][1] == current_player && b_board[bx][2] == current_player)
				goto win_game;

			if (b_board[0][by] == current_player && b_board[1][by] == current_player && b_board[2][by] == current_player)
				goto win_game;

			if (bx == by && b_board[0][0] == current_player && b_board[1][1] == current_player && b_board[2][2] == current_player)
				goto win_game;

			if (bx == (2-by) && b_board[0][2] == current_player && b_board[1][1] == current_player && b_board[2][0] == current_player)
				goto win_game;
		} else
		if (b_count[bx][by] == 9) {
			b_board[bx][by] = '#';
		}

		if (0) {
	win_game:
			winner = current_player;
			ret = 100;
		}

		current_bx = b_board[sx][sy] == ' ' ? sx : -1;
		current_by = b_board[sx][sy] == ' ' ? sy : -1;

		if (current_player == 'X')
			current_player = 'O';
		else
			current_player = 'X';

		return ret;
	}

	void rollback(const BoardRollbackBuffer &rbuf)
	{
		s_board[rbuf.bx][rbuf.by][rbuf.sx][rbuf.sy] = ' ';
		b_board[rbuf.bx][rbuf.by] = ' ';
		b_count[rbuf.bx][rbuf.by]--;
		current_bx = rbuf.current_bx;
		current_by = rbuf.current_by;
		winner = ' ';

		if (current_player == 'X')
			current_player = 'O';
		else
			current_player = 'X';
	}

	int find_move(int cost, int *bx_ = NULL, int *by_ = NULL, int *sx_ = NULL, int *sy_ = NULL)
	{
		static int depth = 1;
		int8_t queue_bx[96], queue_by[96], queue_sx[96], queue_sy[96];
		int queue_n = 0;

		if (current_bx >= 0)
		{
			for (int sx = 0; sx < 3; sx++)
			for (int sy = 0; sy < 3; sy++)
				if (s_board[current_bx][current_by][sx][sy] == ' ') {
					assert(queue_n < 96);
					queue_bx[queue_n] = current_bx;
					queue_by[queue_n] = current_by;
					queue_sx[queue_n] = sx;
					queue_sy[queue_n] = sy;
					queue_n++;
				}
		}
		else
		{
			for (int bx = 0; bx < 3; bx++)
			for (int by = 0; by < 3; by++)
			{
				if (b_board[bx][by] != ' ')
					continue;

				for (int sx = 0; sx < 3; sx++)
				for (int sy = 0; sy < 3; sy++)
					if (s_board[bx][by][sx][sy] == ' ') {
						assert(queue_n < 96);
						queue_bx[queue_n] = bx;
						queue_by[queue_n] = by;
						queue_sx[queue_n] = sx;
						queue_sy[queue_n] = sy;
						queue_n++;
					}
			}
		}

		if (pi.max_queue_n < queue_n)
			pi.max_queue_n = queue_n;

		pi.histcounters[queue_n]++;

		if (queue_n != 0 && cost == 0) {
			int n = rand() % queue_n;
			if (bx_) *bx_ = queue_bx[n];
			if (by_) *by_ = queue_by[n];
			if (sx_) *sx_ = queue_sx[n];
			if (sy_) *sy_ = queue_sy[n];
			return 0;
		}

		if (queue_n == 0 || queue_n > cost) {
			if (bx_) *bx_ = -1;
			return 0;
		}

		int ret = -1000;
		int sub_cost = cost / queue_n;

		uint8_t good_solutions[96];
		int good_solutions_n = 0;

		for (int i = 0; i < queue_n; i++)
		{
			BoardRollbackBuffer rbuf;
			int this_ret = set(queue_bx[i], queue_by[i], queue_sx[i], queue_sy[i], &rbuf);
			pi.itercount++;

			if (sub_cost > 10 && this_ret < 10) {
				if (++depth > pi.max_depth)
					pi.max_depth = depth;
				this_ret -= find_move(sub_cost);
				depth--;
			} else
			if (sub_cost <= 10) {
				if (pi.min_stop_depth > depth)
					pi.min_stop_depth = depth;
			}

			if (this_ret > ret) {
				good_solutions_n = 1;
				good_solutions[0] = i;
				ret = this_ret;
			}
			else if (this_ret == ret) {
				good_solutions[good_solutions_n++] = i;
			}

			rollback(rbuf);
		}

		assert(ret > -1000);
		assert(good_solutions_n > 0);

		if (bx_ || by_ || sx_ || sy_)
		{
			if (good_solutions_n == 1)
			{
				if (bx_) *bx_ = queue_bx[good_solutions[0]];
				if (by_) *by_ = queue_by[good_solutions[0]];
				if (sx_) *sx_ = queue_sx[good_solutions[0]];
				if (sy_) *sy_ = queue_sy[good_solutions[0]];
			}
			else
			{
				uint8_t best_solutions[96];
				int best_solutions_n = 0;
				int best_score = -1000;

				for (int i = 0; i < good_solutions_n; i++)
				{
					int bx = queue_bx[good_solutions[i]];
					int by = queue_by[good_solutions[i]];
					int sx = queue_sx[good_solutions[i]];
					int sy = queue_sy[good_solutions[i]];
					assert(s_board[bx][by][sx][sy] == ' ');

					int this_score = 2 * ticTacToeSolver(s_board[bx][by], current_player, sx, sy);

					int tmp;
					if (this_score >= 0 && (tmp = ticTacToeSolver(b_board, current_player, bx, by)) >= 0)
						this_score += 1 + tmp;
					else
					if (this_score <= 0 && ticTacToeSolver(b_board, current_player == 'X' ? 'O' : 'X', bx, by) < 0)
						this_score--;

					if (best_score < this_score) {
						best_solutions_n = 1;
						best_solutions[0] = good_solutions[i];
						best_score = this_score;
					} else
					if (best_score == this_score) {
						best_solutions[best_solutions_n++] = good_solutions[i];
					}
				}

				int n = rand() % best_solutions_n;
				if (bx_) *bx_ = queue_bx[best_solutions[n]];
				if (by_) *by_ = queue_by[best_solutions[n]];
				if (sx_) *sx_ = queue_sx[best_solutions[n]];
				if (sy_) *sy_ = queue_sy[best_solutions[n]];
			}
		}

		return ret;
	}

	int get_hist(int idx)
	{
		if (idx < 10)
			return pi.histcounters[idx];

		if (idx < 20)
			return (pi.histcounters[10 + 2*(idx-10)] + pi.histcounters[10 + 2*(idx-10) + 1]) / 2;

		int sum = 0;
		int i_start = 30 + 5*(idx-20), i_stop = i_start + 5;
		for (int i = i_start; i < i_stop; i++)
			sum += pi.histcounters[i];
		return sum / 5;
	}

	char get_pred_bb(int bx, int by, char player)
	{
		if (b_board[bx][by] != ' ')
			return b_board[bx][by];

		int r = ticTacToeProber(s_board[bx][by], player);

		if (player == 'X') {
			if (r > 0) return 'x';
			if (r < 0) return 'o';
		}

		if (player == 'O') {
			if (r > 0) return 'o';
			if (r < 0) return 'x';
		}

		return '-';
	}

	void print_info_line()
	{
		int n = pi.line_nr++;

		if (!n--) printf("  %c|%c|%c", b_board[0][0],  b_board[1][0], b_board[2][0]);
		if (!n--) printf("  -+-+-");
		if (!n--) printf("  %c|%c|%c", b_board[0][1],  b_board[1][1], b_board[2][1]);
		if (!n--) printf("  -+-+-");
		if (!n--) printf("  %c|%c|%c", b_board[0][2],  b_board[1][2], b_board[2][2]);

		n += 5;
		if (!n--) printf("  Player <X> points: %d", pi.score_x);
		if (!n--) printf("  Player <O> points: %d", pi.score_o);
		if (!n--) printf("  Score:     %+2d", pi.score_x - pi.score_o);
		if (!n--) printf("  Prognosis: %+2d", pi.score_x - pi.score_o + pi.prognosis_x);
		if (!n--) printf("  Checked %d moves.", pi.itercount);

		if (!n--) printf("");
		if (!n--) printf("  Maximum depth in search tree: %d", pi.max_depth);
		if (!n--) printf("  Maximum breadth in search tree: %d", pi.max_queue_n);
		if (!n--) printf("  Shortest exhaustion path: %d", pi.min_stop_depth);
		if (!n--) printf("");
		if (!n--) printf("  Predictions for");
		if (!n--) printf("  individual Tic-");
		if (!n--) printf("  Tac-Toe boards:");

		n += 3;
		if (!n--) printf("  %c%c%c", get_pred_bb(0, 0, 'X'), get_pred_bb(1, 0, 'X'), get_pred_bb(2, 0, 'X'));
		if (!n--) printf("  %c%c%c", get_pred_bb(0, 1, 'X'), get_pred_bb(1, 1, 'X'), get_pred_bb(2, 1, 'X'));
		if (!n--) printf("  %c%c%c", get_pred_bb(0, 2, 'X'), get_pred_bb(1, 2, 'X'), get_pred_bb(2, 2, 'X'));

		n += 3;
		if (!n--) printf("  %c%c%c", get_pred_bb(0, 0, 'O'), get_pred_bb(1, 0, 'O'), get_pred_bb(2, 0, 'O'));
		if (!n--) printf("  %c%c%c", get_pred_bb(0, 1, 'O'), get_pred_bb(1, 1, 'O'), get_pred_bb(2, 1, 'O'));
		if (!n--) printf("  %c%c%c", get_pred_bb(0, 2, 'O'), get_pred_bb(1, 2, 'O'), get_pred_bb(2, 2, 'O'));

		if (!n--) printf("");
		if (!n--) printf("  Histogramm search tree breadth:");
		if (!n--) printf("  0        10        30        80");
		//                  |....x....|....x....|....x....|

		n -= 5;
		if (-6 < n && n < 0)
		{
			int peak = 0;
			int l = (-3-n)-3;
			printf("  ");

			for (int i = 0; i <= 30; i++) {
				int k = get_hist(i);
				if (peak < k) peak = k;
			}

			for (int i = 0; i <= 30; i++)
			{
				float q = (1.5 * get_hist(i)) / peak;
				int k = q > 0 ? log2f(q) : -100;

				if (l < k) printf(":");
				else if (l == k) printf(".");
				else printf(l == -5 ? "." : " ");
			}
		}

		printf("\n");
	}

	void print()
	{
		pi.line_nr = 0;
		pi.score_x = 0;
		pi.score_o = 0;

		for (int bx = 0; bx < 3; bx++)
		for (int by = 0; by < 3; by++) {
			if (b_board[bx][by] == 'X') pi.score_x++;
			if (b_board[bx][by] == 'O') pi.score_o++;
		}

		printf("+-------------------------+");
		print_info_line();
		printf("|                         |");
		print_info_line();

		for (int by = 0; by < 3; by++) {
			for (int sy = 0; sy < 3; sy++) {
				if (sy > 0) {
					printf("| ");
					for (int bx = 0; bx < 3; bx++) {
						char frame_ch = (current_bx == bx && current_by == by) ? '*' : ' ';
						for (int sx = 0; sx < 3; sx++)
							printf("%c-", sx ? '+' : frame_ch);
						printf("%c ", frame_ch);
					}
					printf("|");
					print_info_line();
				}
				printf("| ");
				for (int bx = 0; bx < 3; bx++) {
					char frame_ch = (current_bx == bx && current_by == by) ? '*' : ' ';
					for (int sx = 0; sx < 3; sx++) {
						char ch = s_board[bx][by][sx][sy];
						if (b_board[bx][by] != ' ' && b_board[bx][by] != ch)
							ch = tolower(ch);
						printf("%c%c", sx ? '|' : frame_ch, ch);
					}
					printf("%c ", frame_ch);
				}
				printf("|");
				print_info_line();
			}
			printf("|                         |");
			print_info_line();
		}

		printf("+-------------------------+");
		print_info_line();
	}
};

int xy_to_numpad(int x, int y)
{
	switch (y*10 + x)
	{
		case  0: return 7;
		case  1: return 8;
		case  2: return 9;
		case 10: return 4;
		case 11: return 5;
		case 12: return 6;
		case 20: return 1;
		case 21: return 2;
		case 22: return 3;
		default: return -1;
	}
}

int numpad_to_x(int n)
{
	switch (n)
	{
		case  1: return 0;
		case  2: return 1;
		case  3: return 2;
		case  4: return 0;
		case  5: return 1;
		case  6: return 2;
		case  7: return 0;
		case  8: return 1;
		case  9: return 2;
		default: return -1;
	}
}

int numpad_to_y(int n)
{
	switch (n)
	{
		case  1: return 2;
		case  2: return 2;
		case  3: return 2;
		case  4: return 1;
		case  5: return 1;
		case  6: return 1;
		case  7: return 0;
		case  8: return 0;
		case  9: return 0;
		default: return -1;
	}
}

int main(int argc, char **argv)
{
	srand(time(NULL) ^ (getpid() << 10));

	bool human_x = false;
	bool human_o = false;
	bool fool_x = false;
	bool fool_o = false;
	bool smart_x = false;
	bool smart_o = false;
	bool silent = false;

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-x")) human_x = true;
		if (!strcmp(argv[i], "-o")) human_o = true;
		if (!strcmp(argv[i], "-x0")) fool_x = true;
		if (!strcmp(argv[i], "-o0")) fool_o = true;
		if (!strcmp(argv[i], "-x1")) smart_x = true;
		if (!strcmp(argv[i], "-o1")) smart_o = true;
		if (!strcmp(argv[i], "-s")) silent = true;
		if (!strcmp(argv[i], "-z")) srand(42);
	}

	Board brd;
	int move_count = 0;

	if (!silent)
		brd.print();

	while (brd.winner == ' ')
	{
		int bx, by, sx, sy, r;
		move_count++;

		if ((brd.current_player == 'X' && human_x) || (brd.current_player == 'O' && human_o))
		{
			r = brd.find_move(100, &bx, &by, &sx, &sy);
			if (bx < 0) break;

			while (1)
			{
				printf("[%d] Player <%c>: ", move_count, brd.current_player);
				fflush(stdout);

				int n1 = -1, n2 = -1;
				scanf("%1d%1d", &n1, &n2);

				bx = numpad_to_x(n1);
				by = numpad_to_y(n1);

				sx = numpad_to_x(n2);
				sy = numpad_to_y(n2);

				if (bx < 0 && by < 0 && sx < 0 && sy < 0)
					continue;

				if (brd.current_bx >= 0 && brd.current_bx != bx)
					continue;

				if (brd.current_by >= 0 && brd.current_by != by)
					continue;

				if (brd.s_board[bx][by][sx][sy] != ' ' || brd.b_board[bx][by] != ' ')
					continue;

				break;
			}
		}
		else
		{
			int cost = 10000000;

			if (brd.current_player == 'X' && smart_x) cost *= 5;
			if (brd.current_player == 'O' && smart_o) cost *= 5;

			if (brd.current_player == 'X' && fool_x) cost = 0;
			if (brd.current_player == 'O' && fool_o) cost = 0;

			brd.pi.reset();
			r = brd.find_move(cost, &bx, &by, &sx, &sy);
			if (bx < 0) break;

			brd.pi.prognosis_x = brd.current_player == 'X' ? r : -r;

			printf("[%d] Player <%c>: %d%d\n", move_count, brd.current_player, xy_to_numpad(bx, by), xy_to_numpad(sx, sy));
		}

		brd.set(bx, by, sx, sy);

		if (!silent)
			brd.print();
	}

	if (brd.winner != ' ') {
		printf("Player <%c> won!\n", brd.winner);
	} else {
		int points_x = 0, points_o = 0;
		for (int bx = 0; bx < 3; bx++)
		for (int by = 0; by < 3; by++)
			switch (brd.b_board[bx][by]) {
				case 'X': points_x++; break;
				case 'O': points_o++; break;
			}
		if (points_x > points_o)
			printf("Player <X> won by %d points.\n", points_x - points_o);
		else if (points_o > points_x)
			printf("Player <O> won by %d points.\n", points_o - points_x);
		else
			printf("Draw!\n");
	}

	return 0;
}


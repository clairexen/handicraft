/*
 *  The Teeny Tiny Mansion (TTTM) -- mockup text adventure game do show
 *  how formal verification of adventure game logic can be implemented
 *
 *  Copyright (C) 2017  Clifford Wolf <clifford@clifford.at>
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
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <unistd.h>
#include <time.h>

// Set this define for verbose output about states and actions
#undef VERBOSE

// Maximum number of possible actions in any state
#define MAX_ACTIONS 6

// Maximum number of actions required to complete the game from any state (for bounded liveness)
#define MAX_FINISH_DEPTH 18

// Maximum number of actions required to make progress from any state (for unbounded liveness)
#define MAX_PROGRESS_DEPTH 4

// Counterexample state as written by cbmc
//#define CEX_STATE { .character=2, .alice_location=6, .bob_location=3, .red_key_location=1, .blue_key_location=1, .green_key_location=2 }

// Counterexample action as written by cbmc
//#define CEX_ACTION { .op=10, .arg1=6 }

// For @brouhaha, who really seems to like games where the player is eaten by a grue
#define ENABLE_GRUE

//  _____ _   _ _____    ____    _    __  __ _____
// |_   _| | | | ____|  / ___|  / \  |  \/  | ____|
//   | | | |_| |  _|   | |  _  / _ \ | |\/| |  _|
//   | | |  _  | |___  | |_| |/ ___ \| |  | | |___
//   |_| |_| |_|_____|  \____/_/   \_\_|  |_|_____|
// 

#define CHAR_ALICE    1
#define CHAR_BOB      2

#define LOC_WEST_ROOM 3
#define LOC_EAST_ROOM 4
#define LOC_RED_ROOM  5
#define LOC_BLUE_ROOM 6

#define OBJ_RED_KEY   7
#define OBJ_BLUE_KEY  8
#define OBJ_GREEN_KEY 9

#define OP_GOTO      10
#define OP_PICKUP    11
#define OP_GIVE      12
#define OP_SWITCH    13
#define OP_FINISH    14

const char *id2name(int id)
{
#define X(_n) if (id == _n) return #_n;
	X(CHAR_ALICE)
	X(CHAR_BOB)
	X(LOC_WEST_ROOM)
	X(LOC_EAST_ROOM)
	X(LOC_RED_ROOM)
	X(LOC_BLUE_ROOM)
	X(OBJ_RED_KEY)
	X(OBJ_BLUE_KEY)
	X(OBJ_GREEN_KEY)
	X(OP_GOTO)
	X(OP_PICKUP)
	X(OP_GIVE)
	X(OP_SWITCH)
	X(OP_FINISH)
#undef X
	return "****";
}

typedef struct {
	int character;
	int alice_location;
	int bob_location;
	int red_key_location;
	int blue_key_location;
	int green_key_location;
} game_state_t;

typedef struct {
	int op;
	int arg1;
} game_action_t;

void banner()
{
	printf("\n");
	printf("  =====================  The Teeny Tiny Mansion  ========================\n");
	printf("\n");
	printf("  +----------------------------+           +----------------------------+\n");
	printf("  |                            |           |                            |\n");
	printf("  |                            |           |                            |\n");
	printf("  |                            |           |                            |\n");
	printf("  |                            |           |                            |\n");
	printf("  |                            |           |                            |\n");
	printf("  |           Red              |           |             Blue           |\n");
	printf("  |           Room             |           |             Room           |\n");
	printf("  |                            |           |                            |\n");
	printf("  |                            |           |                            |\n");
	printf("  |                            |           |                            |\n");
	printf("  |                            |           |                            |\n");
	printf("  +--------           ---------+           +---------           --------+\n");
	printf("          |           |                             |           |\n");
	printf("          |  Red Door |                             | Blue Door |\n");
	printf("          |           |                             |           |\n");
	printf("  +--------           ---------+           +---------           --------+\n");
	printf("  |                            |           |                            |\n");
	printf("  |                            |           |                            |\n");
	printf("  |                            |           |                            |\n");
	printf("  |                            |           |                            |\n");
	printf("  |                            |-----------|                            |\n");
	printf("  |           West                Green                  East           |\n");
	printf("  |           Room                 Door                  Room           |\n");
	printf("  |                            |-----------|                            |\n");
	printf("  |                            |           |                            |\n");
	printf("  |                            |           |                            |\n");
	printf("  |                            |           |                            |\n");
	printf("  |                            |           |                            |\n");
	printf("  +----------------------------+           +----------------------------+\n");
	printf("\n");
	printf("\n");
	printf("Help Alice get into the Red Room and help Bob get into the Blue Room.\n");
	printf("\n");
	printf("\n");
}

uint32_t xorshift32(uint32_t x32)
{
	x32 ^= x32 << 13;
	x32 ^= x32 >> 17;
	x32 ^= x32 << 5;
	return x32;
}

void make_initstate(game_state_t *state, uint32_t seed)
{
	seed = xorshift32(seed);
	seed = xorshift32(seed);
	seed = xorshift32(seed);
	seed = xorshift32(seed);
	seed = xorshift32(seed);

	state->character = (seed & 1) ? CHAR_ALICE : CHAR_BOB;
	state->alice_location = (seed & 2) ? LOC_WEST_ROOM : LOC_EAST_ROOM;
	state->bob_location = (seed & 4) ? LOC_WEST_ROOM : LOC_EAST_ROOM;
	state->red_key_location = (seed & 8) ? LOC_WEST_ROOM : LOC_EAST_ROOM;
	state->blue_key_location = (seed & 16) ? LOC_WEST_ROOM : LOC_EAST_ROOM;
	state->green_key_location = (seed & 32) ? LOC_WEST_ROOM : LOC_EAST_ROOM;

#ifndef BAD_GAME_DESIGN
	if (state->alice_location == state->bob_location)
		state->green_key_location = state->alice_location;
#endif
}

int query_actions(const game_state_t *state, game_action_t *action_list)
{
	int num_actions = 0;
	int current_location = 0;
	int other_location = 0;
	const char *other_character_name = 0;

#ifdef VERBOSE
	printf("-----------------------------\n");
	printf("character          %s\n", id2name(state->character));
	printf("alice_location     %s\n", id2name(state->alice_location));
	printf("bob_location       %s\n", id2name(state->bob_location));
	printf("red_key_location   %s\n", id2name(state->red_key_location));
	printf("blue_key_location  %s\n", id2name(state->blue_key_location));
	printf("green_key_location %s\n", id2name(state->green_key_location));
	printf("-----------------------------\n\n");
#endif

	if (state->character == CHAR_ALICE)
	{
		printf("You are Alice.\n");
		current_location = state->alice_location;
		other_location = state->bob_location;
		other_character_name = "Bob";

		if (current_location != LOC_RED_ROOM)
			printf("You want to be in the Red Room.\n");
		else
			printf("You are happy because you are in the Red Room.\n");
	}

	if (state->character == CHAR_BOB)
	{
		printf("You are Bob.\n");
		current_location = state->bob_location;
		other_location = state->alice_location;
		other_character_name = "Alice";

		if (current_location != LOC_BLUE_ROOM)
			printf("You want to be in the Blue Room.\n");
		else
			printf("You are happy because you are in the Blue Room.\n");
	}

	if (state->red_key_location == state->character)
		printf("You have the Red Key.\n");

	if (state->blue_key_location == state->character)
		printf("You have the Blue Key.\n");

	if (state->green_key_location == state->character)
		printf("You have the Green Key.\n");

	printf("\n");

	if (current_location == LOC_WEST_ROOM) {
		printf("You are in the West Room.\n");
		printf("You see a Red Door (to Red Room).\n");
		printf("You see a Green Door (to East Room).\n");
	}

	if (current_location == LOC_EAST_ROOM) {
		printf("You are in the East Room.\n");
		printf("You see a Blue Door (to Blue Room).\n");
		printf("You see a Green Door (to West Room).\n");
	}

	if (current_location == LOC_RED_ROOM) {
		printf("You are in the Red Room.\n");
		printf("You see a Red Door (to West Room).\n");
	}

	if (current_location == LOC_BLUE_ROOM) {
		printf("You are in the Blue Room.\n");
		printf("You see a Blue Door (to East Room).\n");
	}

	if (state->red_key_location == current_location)
		printf("You see the Red Key on the floor.\n");

	if (state->blue_key_location == current_location)
		printf("You see the Blue Key on the floor.\n");

	if (state->green_key_location == current_location)
		printf("You see the Green Key on the floor.\n");

	if (current_location == other_location)
		printf("%s is with you in this room.\n", other_character_name);

	printf("\n");

	if (state->alice_location == LOC_RED_ROOM && state->bob_location == LOC_BLUE_ROOM) {
		printf("  %d) Finish Game\n", num_actions);
		action_list[num_actions].op = OP_FINISH;
		action_list[num_actions].arg1 = 0;
		num_actions++;
	}

	if (state->character == CHAR_ALICE /* && state->bob_location != LOC_BLUE_ROOM */) {
		printf("  %d) Switch to Bob\n", num_actions);
		action_list[num_actions].op = OP_SWITCH;
		action_list[num_actions].arg1 = 0;
		num_actions++;
	}

	if (state->character == CHAR_BOB /* && state->alice_location != LOC_RED_ROOM */) {
		printf("  %d) Switch to Alice\n", num_actions);
		action_list[num_actions].op = OP_SWITCH;
		action_list[num_actions].arg1 = 0;
		num_actions++;
	}

	if (state->red_key_location == current_location) {
		printf("  %d) Pick up Red Key.\n", num_actions);
		action_list[num_actions].op = OP_PICKUP;
		action_list[num_actions].arg1 = OBJ_RED_KEY;
		num_actions++;
	}

	if (state->blue_key_location == current_location) {
		printf("  %d) Pick up Blue Key.\n", num_actions);
		action_list[num_actions].op = OP_PICKUP;
		action_list[num_actions].arg1 = OBJ_BLUE_KEY;
		num_actions++;
	}

	if (state->green_key_location == current_location) {
		printf("  %d) Pick up Green Key.\n", num_actions);
		action_list[num_actions].op = OP_PICKUP;
		action_list[num_actions].arg1 = OBJ_GREEN_KEY;
		num_actions++;
	}

	if (state->alice_location == state->bob_location) {
		if (state->red_key_location == state->character) {
			printf("  %d) Give Red Key to %s.\n", num_actions, other_character_name);
			action_list[num_actions].op = OP_GIVE;
			action_list[num_actions].arg1 = OBJ_RED_KEY;
			num_actions++;
		}
		if (state->blue_key_location == state->character) {
			printf("  %d) Give Blue Key to %s.\n", num_actions, other_character_name);
			action_list[num_actions].op = OP_GIVE;
			action_list[num_actions].arg1 = OBJ_BLUE_KEY;
			num_actions++;
		}
		if (state->green_key_location == state->character) {
			printf("  %d) Give Green Key to %s.\n", num_actions, other_character_name);
			action_list[num_actions].op = OP_GIVE;
			action_list[num_actions].arg1 = OBJ_GREEN_KEY;
			num_actions++;
		}
	}

	if (current_location == LOC_WEST_ROOM) {
		printf("  %d) Go to Red Room.\n", num_actions);
		action_list[num_actions].op = OP_GOTO;
		action_list[num_actions].arg1 = LOC_RED_ROOM;
		num_actions++;
	}

	if (current_location == LOC_EAST_ROOM) {
		printf("  %d) Go to Blue Room.\n", num_actions);
		action_list[num_actions].op = OP_GOTO;
		action_list[num_actions].arg1 = LOC_BLUE_ROOM;
		num_actions++;
	}

	if (current_location == LOC_RED_ROOM || current_location == LOC_EAST_ROOM) {
		printf("  %d) Go to West Room.\n", num_actions);
		action_list[num_actions].op = OP_GOTO;
		action_list[num_actions].arg1 = LOC_WEST_ROOM;
		num_actions++;
	}

	if (current_location == LOC_BLUE_ROOM || current_location == LOC_WEST_ROOM) {
		printf("  %d) Go to East Room.\n", num_actions);
		action_list[num_actions].op = OP_GOTO;
		action_list[num_actions].arg1 = LOC_EAST_ROOM;
		num_actions++;
	}

	printf("\n");
	assert(num_actions <= MAX_ACTIONS);
	return num_actions;
}

int read_user_action(game_state_t *state, int num_actions)
{
	while (1) {
		int user_input;
		if (state->character == CHAR_ALICE)
			printf("Alice> ");
		if (state->character == CHAR_BOB)
			printf("Bob> ");
		fflush(stdout);
		char buffer[32];
		char *p = fgets(buffer, 32, stdin);
		if (p == NULL) {
			printf("EOF\n");
			exit(1);
		}
		if (sscanf(p, "%d", &user_input) == 1 && 0 <= user_input && user_input < num_actions) {
			printf("\n");
			return user_input;
		}
	}
}

bool apply_action(game_state_t *state, const game_action_t *action)
{
	int current_location = 0;
	int other_character_id = 0;
	const char *other_character_name = 0;

#ifdef VERBOSE
	printf("---------------\n");
	printf("op    %s\n", id2name(action->op));
	printf("arg1  %s\n", id2name(action->arg1));
	printf("---------------\n\n");
#endif

	if (state->character == CHAR_ALICE) {
		current_location = state->alice_location;
		other_character_id = CHAR_BOB;
		other_character_name = "Bob";
	}

	if (state->character == CHAR_BOB) {
		current_location = state->bob_location;
		other_character_id = CHAR_ALICE;
		other_character_name = "Alice";
	}

	if (action->op == OP_GOTO)
	{
		if (state->red_key_location != state->character && action->arg1 == LOC_RED_ROOM) {
			printf("==> You need the Red Key to do that. <==\n\n");
			return false;
		}

		if (state->blue_key_location != state->character && action->arg1 == LOC_BLUE_ROOM) {
			printf("==> You need the Blue Key to do that. <==\n\n");
			return false;
		}

		if (state->green_key_location != state->character &&
				((current_location == LOC_WEST_ROOM && action->arg1 == LOC_EAST_ROOM) ||
				(current_location == LOC_EAST_ROOM && action->arg1 == LOC_WEST_ROOM))) {
			printf("==> You need the Green Key to do that. <==\n\n");
			return false;
		}

		if (state->character == CHAR_ALICE && current_location == LOC_RED_ROOM) {
			printf("==> It's nice here. I'm not leaving. <==\n\n");
			return false;
		}

		if (state->character == CHAR_BOB && current_location == LOC_BLUE_ROOM) {
			printf("==> It's nice here. I'm not leaving. <==\n\n");
			return false;
		}

#ifndef BAD_GAME_DESIGN
		if (state->character == CHAR_ALICE && action->arg1 == LOC_RED_ROOM)
		{
			bool help_bob_first = false;

			if (state->blue_key_location == CHAR_ALICE)
				help_bob_first = true;

			if (state->green_key_location == CHAR_ALICE)
			{
				if (state->blue_key_location == LOC_WEST_ROOM)
					help_bob_first = true;

				if (state->bob_location == LOC_WEST_ROOM)
					help_bob_first = true;
			}

			if (state->green_key_location == LOC_WEST_ROOM && state->bob_location != LOC_WEST_ROOM)
				help_bob_first = true;

			if (help_bob_first) {
				printf("==> I have to help Bob first. <==\n\n");
				return false;
			}
		}

		if (state->character == CHAR_BOB && action->arg1 == LOC_BLUE_ROOM)
		{
			bool help_alice_first = false;

			if (state->red_key_location == CHAR_BOB)
				help_alice_first = true;

			if (state->green_key_location == CHAR_BOB)
			{
				if (state->red_key_location == LOC_EAST_ROOM)
					help_alice_first = true;

				if (state->alice_location == LOC_EAST_ROOM)
					help_alice_first = true;
			}

			if (state->green_key_location == LOC_EAST_ROOM && state->alice_location != LOC_EAST_ROOM)
				help_alice_first = true;

			if (help_alice_first) {
				printf("==> I have to help Alice first. <==\n\n");
				return false;
			}
		}
#endif

		if (state->character == CHAR_ALICE)
			state->alice_location = action->arg1;
		else
			state->bob_location = action->arg1;
	}

	if (action->op == OP_PICKUP)
	{
		if (action->arg1 == OBJ_RED_KEY)
			state->red_key_location = state->character;

		if (action->arg1 == OBJ_BLUE_KEY)
			state->blue_key_location = state->character;

		if (action->arg1 == OBJ_GREEN_KEY)
			state->green_key_location = state->character;
	}

	if (action->op == OP_GIVE)
	{
		if (action->arg1 == OBJ_RED_KEY)
			state->red_key_location = other_character_id;

		if (action->arg1 == OBJ_BLUE_KEY)
			state->blue_key_location = other_character_id;

		if (action->arg1 == OBJ_GREEN_KEY)
			state->green_key_location = other_character_id;
	}

	if (action->op == OP_SWITCH) {
		if (state->character == CHAR_ALICE)
			state->character = CHAR_BOB;
		else
			state->character = CHAR_ALICE;
	}

	return true;
}

#ifndef FORMAL
int main()
{
	banner();

#ifdef CEX_STATE
	game_state_t state = CEX_STATE;
#else
	game_state_t state;
	uint32_t initseed = time(NULL) * 100 + getpid();
	make_initstate(&state, initseed);
#endif

	game_action_t action_list[MAX_ACTIONS];
	int num_actions, user_input;

	do {
		num_actions = query_actions(&state, action_list);
		do {
			user_input = read_user_action(&state, num_actions);
			if (action_list[user_input].op == OP_FINISH) break;
		} while (!apply_action(&state, &action_list[user_input]));
	} while (action_list[user_input].op != OP_FINISH);

#ifdef ENABLE_GRUE
	time_t uxtime;
	struct tm *lctime;
	time(&uxtime);
	lctime = localtime(&uxtime);
	if (lctime->tm_hour == 0 && lctime->tm_min == 0) {
		printf("Out of the dark comes a grue and eats Alice and Bob. Good night.\n\n");
	}
#endif

	printf("Congratulations! You have finished The Teeny Tiny Mansion.\n\n");

	return 0;
}
#endif


//  _____ ___  ____  __  __    _    _       ____  ____   ___   ___  _____ ____
// |  ___/ _ \|  _ \|  \/  |  / \  | |     |  _ \|  _ \ / _ \ / _ \|  ___/ ___|
// | |_ | | | | |_) | |\/| | / _ \ | |     | |_) | |_) | | | | | | | |_  \___ \
// |  _|| |_| |  _ <| |  | |/ ___ \| |___  |  __/|  _ <| |_| | |_| |  _|  ___) |
// |_|   \___/|_| \_\_|  |_/_/   \_\_____| |_|   |_| \_\\___/ \___/|_|   |____/
// 

#ifdef FORMAL

typedef int critic_score_t;

bool eq_actions(const game_action_t *action1, const game_action_t *action2)
{
	return action1->op == action2->op && action1->arg1 == action2->arg1;
}

bool lt_scores(critic_score_t score1, critic_score_t score2)
{
	return score1 < score2;
}

bool formal_state_valid(const game_state_t *state)
{
	// The simple stuff: Check ranges for all state variables

	if (state->character != CHAR_ALICE && state->character != CHAR_BOB)
		return false;
	
	if (state->alice_location < LOC_WEST_ROOM || state->alice_location > LOC_BLUE_ROOM)
		return false;

	if (state->bob_location < LOC_WEST_ROOM || state->bob_location > LOC_BLUE_ROOM)
		return false;

	if (state->red_key_location < CHAR_ALICE || state->red_key_location > LOC_EAST_ROOM)
		return false;

	if (state->blue_key_location < CHAR_ALICE || state->blue_key_location > LOC_EAST_ROOM)
		return false;

	if (state->green_key_location < CHAR_ALICE || state->green_key_location > LOC_EAST_ROOM)
		return false;
	
	// A character in red/blue room must have the red/blue key

	if (state->alice_location == LOC_RED_ROOM && state->red_key_location != CHAR_ALICE)
		return false;

	if (state->alice_location == LOC_BLUE_ROOM && state->blue_key_location != CHAR_ALICE)
		return false;

	if (state->bob_location == LOC_RED_ROOM && state->red_key_location != CHAR_BOB)
		return false;

	if (state->bob_location == LOC_BLUE_ROOM && state->blue_key_location != CHAR_BOB)
		return false;
	
	// The green key can not be isolated from the characters, unless it isn't needed anymore

	bool need_green_key = false;

	if (state->alice_location == LOC_EAST_ROOM || state->alice_location == LOC_BLUE_ROOM || state->red_key_location == LOC_EAST_ROOM ||
			(state->red_key_location == CHAR_BOB && state->bob_location == LOC_EAST_ROOM))
		need_green_key = true;

	if (state->bob_location == LOC_WEST_ROOM || state->bob_location == LOC_RED_ROOM || state->blue_key_location == LOC_WEST_ROOM ||
			(state->blue_key_location == CHAR_ALICE && state->alice_location == LOC_WEST_ROOM))
		need_green_key = true;

	if (need_green_key)
	{
		if (state->green_key_location == LOC_WEST_ROOM) {
			bool alice_has_access = state->alice_location == LOC_WEST_ROOM;
			bool bob_has_access = state->bob_location == LOC_RED_ROOM || state->bob_location == LOC_WEST_ROOM;
			if (!alice_has_access && !bob_has_access) return false;
		}

		if (state->green_key_location == LOC_EAST_ROOM) {
			bool alice_has_access = state->alice_location == LOC_EAST_ROOM || state->alice_location == LOC_BLUE_ROOM;
			bool bob_has_access = state->bob_location == LOC_EAST_ROOM;
			if (!alice_has_access && !bob_has_access) return false;
		}
	}

	// A character will not leave the other character stranded

	if (state->alice_location == LOC_RED_ROOM)
	{
		if (state->blue_key_location == CHAR_ALICE)
			return false;

		if (state->green_key_location == CHAR_ALICE)
		{
			if (state->blue_key_location == LOC_WEST_ROOM)
				return false;

			if (state->bob_location == LOC_WEST_ROOM)
				return false;
		}
	}

	if (state->bob_location == LOC_BLUE_ROOM)
	{
		if (state->red_key_location == CHAR_BOB)
			return false;

		if (state->green_key_location == CHAR_BOB)
		{
			if (state->red_key_location == LOC_EAST_ROOM)
				return false;

			if (state->alice_location == LOC_EAST_ROOM)
				return false;
		}
	}

	return true;
}

bool formal_action_valid(const game_state_t *state, const game_action_t *action)
{
	game_action_t action_list[MAX_ACTIONS];
	int num_actions = query_actions(state, action_list);

	for (int i = 0; i < MAX_ACTIONS; i++) {
		if (i < num_actions && eq_actions(&action_list[i], action))
			return true;
	}

	return false;
}

void formal_actor(const game_state_t *state, game_action_t *action)
{
	int current_location = 0;
	int other_location = 0;
	int other_character_id = 0;

	if (state->character == CHAR_ALICE) {
		current_location = state->alice_location;
		other_character_id = CHAR_BOB;
		other_location =  state->bob_location;
	}

	if (state->character == CHAR_BOB) {
		current_location = state->bob_location;
		other_character_id = CHAR_ALICE;
		other_location = state->alice_location;
	}

	// Pick up everything

	if (state->red_key_location == current_location) {
		action->op = OP_PICKUP;
		action->arg1 = OBJ_RED_KEY;
		return;
	}

	if (state->blue_key_location == current_location) {
		action->op = OP_PICKUP;
		action->arg1 = OBJ_BLUE_KEY;
		return;
	}

	if (state->green_key_location == current_location) {
		action->op = OP_PICKUP;
		action->arg1 = OBJ_GREEN_KEY;
		return;
	}

	if (state->red_key_location == other_location || state->blue_key_location == other_location || state->green_key_location == other_location) {
		action->op = OP_SWITCH;
		action->arg1 = 0;
		return;
	}

	// Pick up everything in the other room(s)

	if (state->red_key_location == LOC_WEST_ROOM || state->red_key_location == LOC_EAST_ROOM ||
			state->blue_key_location == LOC_WEST_ROOM || state->blue_key_location == LOC_EAST_ROOM)
	{
		if (state->green_key_location == other_character_id) {
			action->op = OP_SWITCH;
			action->arg1 = 0;
			return;
		}

		if ((current_location == LOC_WEST_ROOM || current_location == LOC_EAST_ROOM) &&
				state->green_key_location != state->character) {
			action->op = OP_SWITCH;
			action->arg1 = 0;
			return;
		}

		if (current_location == LOC_RED_ROOM) {
			action->op = OP_GOTO;
			action->arg1 = LOC_WEST_ROOM;
			return;
		}

		if (current_location == LOC_WEST_ROOM) {
			action->op = OP_GOTO;
			action->arg1 = LOC_EAST_ROOM;
			return;
		}

		if (current_location == LOC_EAST_ROOM) {
			action->op = OP_GOTO;
			action->arg1 = LOC_WEST_ROOM;
			return;
		}

		if (current_location == LOC_BLUE_ROOM) {
			action->op = OP_GOTO;
			action->arg1 = LOC_EAST_ROOM;
			return;
		}
	}

	// Can we solve directly from current state?

	bool direct_solve = true;

	if (state->red_key_location != CHAR_ALICE)
		direct_solve = false;

	if (state->blue_key_location != CHAR_BOB)
		direct_solve = false;

	if (state->alice_location == LOC_EAST_ROOM && state->green_key_location != CHAR_ALICE)
		direct_solve = false;

	if (state->bob_location == LOC_WEST_ROOM && state->green_key_location != CHAR_BOB)
		direct_solve = false;
	
	if (direct_solve)
	{
		if (state->alice_location == LOC_RED_ROOM && state->bob_location == LOC_BLUE_ROOM) {
			action->op = OP_FINISH;
			action->arg1 = 0;
			return;
		}

		if (state->alice_location == LOC_RED_ROOM && state->character == CHAR_ALICE) {
			action->op = OP_SWITCH;
			action->arg1 = 0;
			return;
		}

		if (state->bob_location == LOC_BLUE_ROOM && state->character == CHAR_BOB) {
			action->op = OP_SWITCH;
			action->arg1 = 0;
			return;
		}

		if (state->character == CHAR_ALICE)
		{
			if (current_location == LOC_WEST_ROOM) {
				action->op = OP_GOTO;
				action->arg1 = LOC_RED_ROOM;
				return;
			}

			if (current_location == LOC_EAST_ROOM) {
				action->op = OP_GOTO;
				action->arg1 = LOC_WEST_ROOM;
				return;
			}
		}

		if (state->character == CHAR_BOB)
		{
			if (current_location == LOC_EAST_ROOM) {
				action->op = OP_GOTO;
				action->arg1 = LOC_BLUE_ROOM;
				return;
			}

			if (current_location == LOC_WEST_ROOM) {
				action->op = OP_GOTO;
				action->arg1 = LOC_EAST_ROOM;
				return;
			}
		}
	}

	// Bring Alice and Bob into the same room

	if (current_location != other_location)
	{
		if (current_location == LOC_RED_ROOM) {
			action->op = OP_GOTO;
			action->arg1 = LOC_WEST_ROOM;
			return;
		}

		if (current_location == LOC_BLUE_ROOM) {
			action->op = OP_GOTO;
			action->arg1 = LOC_EAST_ROOM;
			return;
		}

		if (other_location == LOC_RED_ROOM || other_location == LOC_BLUE_ROOM) {
			action->op = OP_SWITCH;
			action->arg1 = 0;
			return;
		}

		if (state->green_key_location == other_character_id) {
			action->op = OP_SWITCH;
			action->arg1 = 0;
			return;
		}

		if (current_location != other_location) {
			action->op = OP_GOTO;
			action->arg1 = other_location;
			return;
		}
	}

	// Exchange keys

	if (state->red_key_location != CHAR_ALICE) {
		if (state->character == CHAR_ALICE) {
			action->op = OP_SWITCH;
			action->arg1 = 0;
			return;
		}
		action->op = OP_GIVE;
		action->arg1 = OBJ_RED_KEY;
		return;
	}

	if (state->blue_key_location != CHAR_BOB) {
		if (state->character == CHAR_BOB) {
			action->op = OP_SWITCH;
			action->arg1 = 0;
			return;
		}
		action->op = OP_GIVE;
		action->arg1 = OBJ_BLUE_KEY;
		return;
	}

	if (current_location == LOC_WEST_ROOM && state->green_key_location != CHAR_BOB) {
		if (state->character == CHAR_BOB) {
			action->op = OP_SWITCH;
			action->arg1 = 0;
			return;
		}
		action->op = OP_GIVE;
		action->arg1 = OBJ_GREEN_KEY;
		return;
	}

	if (current_location == LOC_EAST_ROOM && state->green_key_location != CHAR_ALICE) {
		if (state->character == CHAR_ALICE) {
			action->op = OP_SWITCH;
			action->arg1 = 0;
			return;
		}
		action->op = OP_GIVE;
		action->arg1 = OBJ_GREEN_KEY;
		return;
	}

	assert(0);
}

critic_score_t formal_critic(const game_state_t *state)
{
	int score = 0;

	if (state->red_key_location == CHAR_ALICE)
		score += 2;

	if (state->red_key_location == CHAR_BOB)
		score += 1;

	if (state->blue_key_location == CHAR_BOB)
		score += 2;

	if (state->blue_key_location == CHAR_ALICE)
		score += 1;

	if (state->green_key_location == CHAR_ALICE || state->green_key_location == CHAR_BOB)
		score += 1;
	
	if (state->alice_location == LOC_RED_ROOM)
		score += 1;

	if (state->alice_location == LOC_BLUE_ROOM)
		score -= 1;

	if (state->bob_location == LOC_BLUE_ROOM)
		score += 1;

	if (state->bob_location == LOC_RED_ROOM)
		score -= 1;

	return score;
}

void prove_completeness(game_state_t state, game_action_t action)
{
	if (!formal_state_valid(&state))
		return;

	if (!formal_action_valid(&state, &action))
		return;
	
	apply_action(&state, &action);
	assert(formal_state_valid(&state));
}

void prove_init_validness(uint32_t seed)
{
	game_state_t state;
	make_initstate(&state, seed);
	assert(formal_state_valid(&state));
}

void prove_actor_validness(game_state_t state)
{
	if (!formal_state_valid(&state))
		return;

	game_action_t action;
	formal_actor(&state, &action);
	assert(formal_action_valid(&state, &action));
}

void prove_bounded_liveness(game_state_t state)
{
	if (!formal_state_valid(&state))
		return;
	
	for (int i = 0; i < MAX_FINISH_DEPTH; i++)
	{
		game_action_t action;
		formal_actor(&state, &action);
		// assert(formal_action_valid(&state, &action));

		if (action.op == OP_FINISH)
			return;

		apply_action(&state, &action);
	}

	assert(0);
}

void prove_unbounded_liveness(game_state_t state)
{
	if (!formal_state_valid(&state))
		return;

	critic_score_t old_score = formal_critic(&state);

	for (int i = 0; i < MAX_PROGRESS_DEPTH; i++)
	{
		game_action_t action;
		formal_actor(&state, &action);
		// assert(formal_action_valid(&state, &action));

		if (action.op == OP_FINISH)
			return;

		apply_action(&state, &action);
	}

	critic_score_t new_score = formal_critic(&state);

	assert(lt_scores(old_score, new_score));
}

void prove_transitiveness(game_state_t state1, game_state_t state2, game_state_t state3)
{
	if (!formal_state_valid(&state1))
		return;

	if (!formal_state_valid(&state2))
		return;

	if (!formal_state_valid(&state3))
		return;

	critic_score_t score1 = formal_critic(&state1);
	critic_score_t score2 = formal_critic(&state2);
	critic_score_t score3 = formal_critic(&state3);

	if (lt_scores(score1, score2) && lt_scores(score2, score3))
		assert(lt_scores(score1, score3));
}

int main()
{
	banner();

#ifdef CEX_STATE
	game_state_t state = CEX_STATE;
#else
	game_state_t state;
	make_initstate(&state, 123456789);
#endif

	while (1)
	{
		assert(formal_state_valid(&state));

#ifdef VERBOSE
		critic_score_t score = formal_critic(&state);
		printf("Current score: %d\n\n", (int)score);
#endif

		game_action_t action_list[MAX_ACTIONS];
		int num_actions = query_actions(&state, action_list);
		int user_input = -1;

#ifdef CEX_ACTION
		game_action_t action = CEX_ACTION;
#else
		game_action_t action;
		formal_actor(&state, &action);
#endif

		for (int i = 0; i < num_actions; i++) {
			if (!eq_actions(&action_list[i], &action)) continue;
			printf("%s> %d\n\n", state.character == CHAR_ALICE ? "Alice" : "Bob", i);
			assert(user_input == -1);
			user_input = i;
		}

		assert(user_input != -1);
		if (action_list[user_input].op == OP_FINISH) break;

		bool action_did_something = apply_action(&state, &action_list[user_input]);
		assert(action_did_something);

#ifdef CEX_ACTION
		assert(formal_state_valid(&state));
		return 0;
#endif
	}

	printf("Congratulations! The bot has finished The Teeny Tiny Mansion.\n\n");

#if 0
	int num_states = 0, num_valid_states = 0;
	for (state.character = CHAR_ALICE; state.character <= CHAR_BOB; state.character++)
	for (state.alice_location = LOC_WEST_ROOM; state.alice_location <= LOC_BLUE_ROOM; state.alice_location++)
	for (state.bob_location = LOC_WEST_ROOM; state.bob_location <= LOC_BLUE_ROOM; state.bob_location++)
	for (state.red_key_location = CHAR_ALICE; state.red_key_location <= LOC_BLUE_ROOM; state.red_key_location++)
	for (state.blue_key_location = CHAR_ALICE; state.blue_key_location <= LOC_BLUE_ROOM; state.blue_key_location++)
	for (state.green_key_location = CHAR_ALICE; state.green_key_location <= LOC_BLUE_ROOM; state.green_key_location++) {
		if (formal_state_valid(&state))
			num_valid_states++;
		num_states++;
	}
	printf("Number of states: %d\n", num_states);
	printf("Number of valid states: %d\n", num_valid_states);
#endif

	return 0;
}

#endif // FORMAL

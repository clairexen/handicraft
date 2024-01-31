/*
 *  ftseval - Fast Transistor Switches Evaluator  (a proof-of-concept)
 *
 *  Copyright (C) 2011  Clifford Wolf <clifford@clifford.at>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

#include "ftseval.h"
#include <assert.h>
#include <stdio.h>

// #define debugf(args...) fprintf(stderr, args)
#define debugf(args...) do { } while(0)

#define FTSEVAL_MAX_QUEUE_SIZE 100000

int ftseval_transitionCounter;
int ftseval_netevalCounter;

int ftseval_queue1_top, ftseval_queue2_top;
ftseval_netid_t ftseval_queue1[FTSEVAL_MAX_QUEUE_SIZE];
ftseval_netid_t ftseval_queue2[FTSEVAL_MAX_QUEUE_SIZE];

#define S(_s) FTSEVAL_STATE_ ## _s,
static ftseval_state_t combine_table[FTSEVAL_NUM_STATES][FTSEVAL_NUM_STATES] = {
	//PULLDOWN    FLOAT       CHARGE      PULLUP     
	{ S(PULLDOWN) S(PULLDOWN) S(PULLDOWN) S(PULLDOWN) }, // PULLDOWN
	{ S(PULLDOWN) S(FLOAT)    S(CHARGE)   S(PULLUP)   }, // FLOAT
	{ S(PULLDOWN) S(CHARGE)   S(CHARGE)   S(PULLUP)   }, // CHARGE
	{ S(PULLDOWN) S(PULLUP)   S(PULLUP)   S(PULLUP)   }, // PULLUP
};
static ftseval_state_t successor_table[FTSEVAL_NUM_STATES + 1] = {
	// PULLDOWN FLOAT    CHARGE    PULLUP    *INIT*
	S(FLOAT)    S(FLOAT) S(CHARGE) S(CHARGE) S(FLOAT)
};
static ftseval_bool_t switch_table[FTSEVAL_NUM_STATES] = {
	// PULLDOWN FLOAT CHARGE PULLUP
	0, 0, 1, 1 /* NMOS Logic */
};

void ftseval_set(ftseval_netid_t id, ftseval_state_t state)
{
	struct ftseval_net_s *net = &ftseval_nets[id];
	assert(id >= 0 && id < ftseval_numNets);
	net->init = state;
	if (net->stage == 0) {
		assert(ftseval_queue1_top < FTSEVAL_MAX_QUEUE_SIZE);
		ftseval_queue1[ftseval_queue1_top++] = id;
		net->queueBuf = net->init;
		net->queuePtr = id;
		net->stage = 1;
	}
}

ftseval_state_t ftseval_get(ftseval_netid_t id)
{
	struct ftseval_net_s *net = &ftseval_nets[id];
	assert(id >= 0 && id < ftseval_numNets);
	return net->state;
}

void ftseval_init()
{
	ftseval_transitionCounter = 0;
	ftseval_netevalCounter = 0;

	ftseval_queue1_top = 0;
	ftseval_queue2_top = 0;

	ftseval_netid_t i;
	for (i = 0; i < ftseval_numNets; i++)
	{
		assert(ftseval_queue1_top < FTSEVAL_MAX_QUEUE_SIZE);
		ftseval_queue1[ftseval_queue1_top++] = i;

		struct ftseval_net_s *net = &ftseval_nets[i];
		net->state = FTSEVAL_NUM_STATES;
		net->queueBuf = net->init;
		net->queuePtr = i;
		net->stage = 1;
	}
}

void ftseval_run()
{
	debugf("\nftseval_run()\n");
restart:
	if (ftseval_queue1_top == 0 && ftseval_queue2_top == 0)
		return;

	ftseval_transitionCounter++;

	debugf("queue1 iteration:\n");
	while (ftseval_queue1_top > 0)
	{
		ftseval_netid_t id = ftseval_queue1[--ftseval_queue1_top];
		struct ftseval_net_s *net = &ftseval_nets[id];
		assert(id >= 0 && id < ftseval_numNets);

		debugf("  net %d: (stage=%d, supply=%d, init=%d, state=%d)\n",
			id, net->stage, net->supply, net->init, net->state);
		ftseval_netevalCounter++;

		if (net->stage != 1)
			continue;

		debugf("    buffer @%d: %d", net->queuePtr,
				ftseval_nets[net->queuePtr].queueBuf);

		ftseval_state_t st = ftseval_nets[net->queuePtr].queueBuf;
		st = combine_table[st][net->init];
		st = combine_table[st][successor_table[net->state]];
		ftseval_nets[net->queuePtr].queueBuf = st;

		debugf(" -> %d\n", st);

		if (net->supply) {
			net->stage = 0;
			continue;
		}

		assert(ftseval_queue2_top < FTSEVAL_MAX_QUEUE_SIZE);
		ftseval_queue2[ftseval_queue2_top++] = id;
		net->stage = 2;

		ftseval_neigh_num_t i;
		for (i = 0; i < net->neighNum; i++)
		{
			ftseval_netid_t nid = net->neighPtrLst[i];
			struct ftseval_net_s *neigh = &ftseval_nets[nid];
			assert(nid >= 0 && nid < ftseval_numNets);

			debugf("    neigh %d: (net=%d, stage=%d, on=%d)\n", i,
					nid, neigh->stage, net->neighOnLst[i]);

			if (!net->neighOnLst[i] || neigh->stage == 2)
				continue;

			assert(ftseval_queue1_top < FTSEVAL_MAX_QUEUE_SIZE);
			ftseval_queue1[ftseval_queue1_top++] = nid;
			neigh->queuePtr = net->queuePtr;
			neigh->stage = 1;
		}
	}

	debugf("queue2 iteration:\n");
	while (ftseval_queue2_top > 0)
	{
		ftseval_netid_t id = ftseval_queue2[--ftseval_queue2_top];
		struct ftseval_net_s *net = &ftseval_nets[id];
		assert(id >= 0 && id < ftseval_numNets);

		debugf("  net %d: (stage=%d, supply=%d, state=%d, newState=%d @%d)\n", id, net->stage,
				net->supply, net->state, ftseval_nets[net->queuePtr].queueBuf, net->queuePtr);

		if (net->stage == 2)
			net->stage = 0;

		ftseval_state_t newState = ftseval_nets[net->queuePtr].queueBuf;
		if (newState == net->state)
			continue;
		net->state = newState;

		ftseval_cc_num_t i;
		ftseval_bool_t newOn = switch_table[newState];
		for (i = 0; i < net->ccNum; i++)
		{
			ftseval_netid_t cid = net->ccPtrLst[i];
			ftseval_neigh_num_t cidx = net->ccIdxLst[i];
			struct ftseval_net_s *cnet = &ftseval_nets[cid];
			assert(cid >= 0 && cid < ftseval_numNets);

			debugf("    channel %d: (net=%d, oldOn=%d, newOn=%d)\n",
					i, cid, cnet->neighOnLst[cidx], newOn);

			if (newOn != cnet->neighOnLst[cidx]) {
				cnet->neighOnLst[cidx] = newOn;
				if (cnet->stage != 1) {
					assert(ftseval_queue1_top < FTSEVAL_MAX_QUEUE_SIZE);
					ftseval_queue1[ftseval_queue1_top++] = cid;
					cnet->queueBuf = cnet->init;
					cnet->queuePtr = cid;
					cnet->stage = 1;
				}
			}
		}
	}

	// tail recursion
	goto restart;
}


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

#ifndef FTSVAL_H
#define FTSVAL_H

#define FTSEVAL_NUM_STATES 4
#define FTSEVAL_STATE_PULLDOWN  0
#define FTSEVAL_STATE_FLOAT     1
#define FTSEVAL_STATE_CHARGE    2
#define FTSEVAL_STATE_PULLUP    3

typedef int ftseval_netid_t;
typedef unsigned char ftseval_bool_t;
typedef unsigned char ftseval_state_t;
typedef unsigned char ftseval_stage_t;
typedef unsigned char ftseval_flags_t;
typedef unsigned char ftseval_cctype_t;
typedef int ftseval_neigh_num_t;
typedef int ftseval_cc_num_t;

struct ftseval_net_s
{
	ftseval_state_t queueBuf;
	ftseval_netid_t queuePtr;

	ftseval_state_t init;
	ftseval_state_t state;
	ftseval_stage_t stage;
	ftseval_bool_t supply;

	ftseval_neigh_num_t neighNum;
	ftseval_bool_t *neighOnLst;
	ftseval_netid_t *neighPtrLst;

	ftseval_cc_num_t ccNum;
	ftseval_netid_t *ccPtrLst;
	ftseval_neigh_num_t *ccIdxLst;
};

extern int ftseval_transitionCounter;
extern int ftseval_netevalCounter;

extern ftseval_netid_t ftseval_numNets;
extern struct ftseval_net_s ftseval_nets[];

extern int ftseval_queue1_top, ftseval_queue2_top;
extern ftseval_netid_t ftseval_queue1[];
extern ftseval_netid_t ftseval_queue2[];

extern void ftseval_set(ftseval_netid_t id, ftseval_state_t state);
extern ftseval_state_t ftseval_get(ftseval_netid_t id);

extern void ftseval_init();
extern void ftseval_run();

#endif

#include <assert.h>
#include <stdio.h>

//	=IN=	=A=	=B=	=C=	=D=	=E=
//	0	B1L	A0R	C0R	E1L	B0L
//	1	A1R	C0L	D1L	A0R	H1L
//
// source: http://www.drb.insel.de/~heiner/BB/mabu90.html (#6)

#define A 0
#define B 1
#define C 2
#define D 3
#define E 4
#define H 5

#define R i++
#define L i--

#define STATE(_S, _N0, _V0, _D0, _N1, _V1, _D1) case _S: if (!tape[i]) state = _N0, tape[_D0] = _V0; else state = _N1, tape[_D1] = _V1; break;

#define TAPE_LEN 1000
char tape[TAPE_LEN], ping[TAPE_LEN];
int min_idx, max_idx;

int main()
{
	long long cycles = 0;
	int score = 0, state = A, i = TAPE_LEN / 2, k;

	min_idx = i;
	max_idx = i;

	while (state != H)
	{
		assert(0 < i && i+1 < TAPE_LEN);
		cycles++;

		switch (state) {
			STATE(A, B,1,L, A,1,R)
			STATE(B, A,0,R, C,0,L)
			STATE(C, C,0,R, D,1,L)
			STATE(D, E,1,L, A,0,R)
			STATE(E, B,0,L, H,1,L)
		}

		ping[i] = 1;
		if (i < min_idx) min_idx = i;
		if (i > max_idx) max_idx = i;

		if (cycles % 1000000 == 0) {
			for (score = k = 0; k < TAPE_LEN; k++)
				score += tape[k];
			printf("cycles=%lld, score=%d, position=%d\n", cycles, score, i - TAPE_LEN / 2);
			for (k = min_idx; k <= max_idx; k++)
				printf("%d", tape[k]);
			printf("\n");
			for (k = min_idx; k <= max_idx; k++)
				printf("%c", k == i ? '^' : ping[k] ? '-' : ' '), ping[k] = 0;
			printf("\n");
		}
	}

	for (score = k = 0; k < TAPE_LEN; k++)
		score += tape[k];
	printf("cycles=%lld, score=%d, position=%d\n", cycles, score, i - TAPE_LEN / 2);

	return 0;
}


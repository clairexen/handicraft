
TARGETS = tttm tttm_badgame tttm_autorun
TARGETS += prove_completeness.log
TARGETS += prove_init_validness.log
TARGETS += prove_actor_validness.log
TARGETS += prove_bounded_liveness.log
TARGETS += prove_unbounded_liveness.log
TARGETS += prove_transitiveness.log

all: $(TARGETS)

tttm: tttm.c
	clang -Wall -std=c99 -Os -o tttm tttm.c

tttm_badgame: tttm.c
	clang -D BAD_GAME_DESIGN -Wall -std=c99 -Os -o tttm_badgame tttm.c

tttm_autorun: tttm.c
	clang -D FORMAL -Wall -std=c99 -Os -o tttm_autorun tttm.c

prove_%.log: tttm.c
	rm -f $@
	cbmc -D FORMAL --function $(basename $@) tttm.c | tee $(basename $@).out
	grep "VERIFICATION SUCCESSFUL" $(basename $@).out
	mv $(basename $@).out $@

clean:
	rm -f $(TARGETS)


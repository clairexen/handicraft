
all: bitread bitmatch bitcorr bitstat bitgrep bitdiff

bitread: bitread.cc
	clang -Wall -Wextra -O0 -ggdb -static -std=c++11 -o $@ $< -lstdc++

bitmatch: bitmatch.cc
	clang -Wall -Wextra -O0 -ggdb -static -std=c++11 -o $@ $< -lstdc++

bitcorr: bitcorr.cc
	clang -Wall -Wextra -O0 -ggdb -static -std=c++11 -o $@ $< -lstdc++

bitstat: bitstat.cc
	clang -Wall -Wextra -O0 -ggdb -static -std=c++11 -o $@ $< -lstdc++

bitgrep: bitgrep.cc
	clang -Wall -Wextra -O0 -ggdb -static -std=c++11 -o $@ $< -lstdc++

bitdiff: bitdiff.cc
	clang -Wall -Wextra -O0 -ggdb -static -std=c++11 -o $@ $< -lstdc++

clean:
	rm -f bitread
	rm -f bitmatch
	rm -f bitcorr
	rm -f bitstat
	rm -f bitgrep
	rm -f bitdiff

.PHONY: all clean


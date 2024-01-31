
all: bcode_check bcode_xor bitpack3 bitpack7 bitpack15 bitpack31 bitpack63 bitpack127 mseqgen

%: %.cc
	clang -o $@ -Wall -Wextra $< -O3 -lstdc++

bitpack3: bitpack.cc
	clang -o $@ -Wall -Wextra $< -DLEN=3 -O3 -lstdc++

bitpack7: bitpack.cc
	clang -o $@ -Wall -Wextra $< -DLEN=7 -O3 -lstdc++

bitpack15: bitpack.cc
	clang -o $@ -Wall -Wextra $< -DLEN=15 -O3 -lstdc++

bitpack31: bitpack.cc
	clang -o $@ -Wall -Wextra $< -DLEN=31 -O3 -lstdc++

bitpack63: bitpack.cc
	clang -o $@ -Wall -Wextra $< -DLEN=63 -O3 -lstdc++

bitpack127: bitpack.cc
	clang -o $@ -Wall -Wextra $< -DLEN=127 -O3 -lstdc++

run: all
	for n in 3 7 15 31 63; do ./bitpack$$n; echo; done | tee bitpack.txt
	./bcode_check `grep ^Code: bitpack.txt  | cut -f2 -d' '` | tee -a bitpack.txt

clean:
	rm -rf bitpack[0-9] bitpack[0-9][0-9] bitpack[0-9][0-9][0-9]
	rm -rf bcode_check bcode_xor mseqgen bitpack.txt

.PHONY: all run clean


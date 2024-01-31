
all: matmul_plain matmul_fast matmul_native pdae_plain pdae_fast pdae_native

%_plain: %.c
	gcc -std=c99 -o $@ $< -lm

%_fast: %.c
	gcc -std=c99 -Ofast -o $@ $< -lm

%_native: %.c
	gcc -std=c99 -march=native -Ofast -o $@ $< -lm


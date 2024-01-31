
all: cudamemtest

cudamemtest: cudamemtest.cu
	nvcc -o cudamemtest -O3 -gencode arch=compute_20,code=sm_21 cudamemtest.cu

clean:
	rm -f cudamemtest

.PHONY: all clean


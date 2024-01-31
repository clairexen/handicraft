CFLAGS = -O3 -ggdb -Wall -D_GNU_SOURCE
LDFLAGS = -L /usr/local/cuda/lib64
LDLIBS = -lm -lcudart -lstdc++

NVCC = nvcc
NVCCFLAGS = -lineinfo -gencode arch=compute_30,code=compute_30

perfmeas: perfmeas.o primes_cpu.o primes_gpu.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

primes_gpu.o: primes_gpu.cu
	$(NVCC) --compiler-bindir $(CC) $(NVCCFLAGS) -c $<

clean:
	rm -f perfmeas perfmeas.o primes_cpu.o primes_gpu.o


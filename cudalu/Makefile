
all: demo culademo

demo: cudalu.cu cudalu.h demo.cc profiler.h
	nvcc -arch=sm_20 --ptxas-options=-v -o demo -g -O3 demo.cc cudalu.cu -lcublas

culademo: culademo.cc profiler.h
	gcc -m64 -o culademo culademo.cc -O3 -I/usr/local/cula/include -I/usr/local/cuda/include -L/usr/local/cula/lib64 -lcula_core -lcula_lapack -lcublas -lcudart

clean:
	rm -f demo culademo


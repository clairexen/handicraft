
CC = clang
CXX = clang
CXXFLAGS += -MD -Wall -O2 -ggdb -std=c++11
LDFLAGS += -ggdb
LDLIBS += -lstdc++ -lm

CUDA_DIR = /usr/local/cuda-5.0
CUFLAGS += -O2 -gencode arch=compute_20,code=compute_20
CXXFLAGS += -I$(CUDA_DIR)/include/
LDFLAGS += -L$(CUDA_DIR)/lib64/
LDLIBS += -lcusparse

demo: demo.o culsqr.o kernels.o

kernels.o: kernels.cu
	$(CUDA_DIR)/bin/nvcc -o kernels.d --generate-dependencies kernels.cu
	$(CUDA_DIR)/bin/nvcc -o kernels.o -c $(CUFLAGS) kernels.cu

clean:
	rm -f demo *.o *.d

-include *.d


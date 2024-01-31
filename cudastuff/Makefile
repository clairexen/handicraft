NVCC := nvcc
LDLIBS := -lstdc++ -lm -lcudart
NVCC_OPTS := -use_fast_math -Xcompiler -fPIC -Xcompiler -fopenmp -lineinfo -src-in-ptx
NVCC_OPTS += -arch sm_32 -maxrregcount 16 -restrict -res-usage -Xptxas -dlcm=ca

all: cudamemcpy

cudamemcpy: cudamemcpy.cu
	$(NVCC) $(NVCC_OPTS) -o $@ $< $(LDLIBS)

clean:
	rm -f cudamemcpy


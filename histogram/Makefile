MPICC=mpicc
NVCC=nvcc
CCFLAGS=-g -O0

mpiHistogrammingA: mpiHistogramming.c
	$(MPICC) $(CCFLAGS) $< -o $@ -DSUBQ_A

mpiHistogrammingB: mpiHistogramming.c
	$(MPICC) $(CCFLAGS) $< -o $@ -DSUBQ_B


cudaHistogramming: cudaHistogramming.cu
	nvcc $< -arch=sm_35 -O3 -o $@

cudaEdgeDetector: cudaEdgeDetector.cu
	$(NVCC) $(CCFLAGS) $< -o $@ `pkg-config --cflags --libs opencv`

vi: cudaEdgeDetector.cu
	vim ./$<

test: cuda
	./$<

gdb: cuda
	gdb --args ./$<

.PHONY: clean
clean:
	rm -f cuda test mpiHistogrammingA mpiHistogrammingB \
	      cudaHistogramming cudaEdgeDetector

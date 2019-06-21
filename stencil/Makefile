NVCC=nvcc
FLAGS=-g -O0 -arch=sm_35
# FLAGS=-g -O0

stencil-tiled: stencil.cu
	$(NVCC) $(FLAGS) -o $@ $< -DTILED

stencil: stencil.cu
	$(NVCC) $(FLAGS) -o $@ $<

vi: stencil.cu
	vim $<

.PHONY: clean
clean: 
	rm -f stencil-tiled stencil

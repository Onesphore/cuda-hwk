#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda.h>

#define MAX 1000
#define DISPLAY 1

__global__ void histoKernel(int*, int, int*);
__global__ void reduceKernel(int*, int*, int);

static double CLOCK();

int main(int argc, char* argv[]) {
  double start, end; // Timing
  // Host
  int *data, data_sz;
  int class_cnt;
  int class_sz;
  int node_cnt;      // To compare with Q1.
  int *result;

  data_sz   = atoi(argv[1]);
  class_cnt = atoi(argv[2]);
  node_cnt  = atoi(argv[3]);
  class_sz    = MAX/class_cnt;

  data         = (int*) malloc(sizeof(int) * data_sz);
  result       = (int*) malloc(sizeof(int) * class_cnt);
  for (int i=0; i<data_sz; ++i)
    data[i] = rand() % MAX;

  // Device
  int* data_device;
  int* result_extd_device; // extended
  int* result_device;

  cudaMalloc(&data_device,        sizeof(int) * data_sz);
  cudaMalloc(&result_extd_device, sizeof(int) * class_cnt * node_cnt);
  cudaMalloc(&result_device,      sizeof(int) * class_cnt);
  cudaMemcpy(data_device, data,   sizeof(int) * data_sz,
		                cudaMemcpyHostToDevice);
  // Actual computation
  start = CLOCK();
  cudaMemset(result_extd_device, 0, sizeof(int) * class_cnt * node_cnt);
  histoKernel<<<node_cnt, class_cnt>>>(data_device, data_sz, 
		                                    result_extd_device);
  // Reduction
  cudaMemset(result_device, 0, sizeof(int) * class_cnt);
  reduceKernel<<<1, class_cnt>>>(result_extd_device, result_device, node_cnt);
  cudaDeviceSynchronize();
  end = CLOCK();

  cudaMemcpy(result, result_device, sizeof(int) * class_cnt,
		                      cudaMemcpyDeviceToHost);
  printf("[CUDA] Histogramming took %.3f milliseconds.\n", end-start);

#ifdef DISPLAY
  for (int i=0; i<class_cnt; ++i) {
    printf("[%3d - %4d[: %d\n", i*class_sz, (i+1)*class_sz,
	                                      result[i]);
  }
#endif // DISPLAY
  return 0;
}

__global__ void
histoKernel(int* data, int data_sz, int* result) {
  int i, j;
  int left_bound, right_bound;
  int class_sz;
  int chunk_sz;

  chunk_sz = data_sz/blockDim.x;
  class_sz = MAX/blockDim.x;

  for (i = blockIdx.x; i < blockDim.x; i += gridDim.x) {
    left_bound  = i * chunk_sz;
    right_bound = (i + 1) * chunk_sz;
    for (j = left_bound; j < right_bound; ++j) {
      if ((data[j] >= threadIdx.x*class_sz) && 
	  (data[j] < (threadIdx.x+1)*class_sz)) {
        result[blockIdx.x * blockDim.x + threadIdx.x] += 1;
      }
    }
  }
}

__global__ void
reduceKernel(int* input, int* output, int node_cnt) {
  for (int i=0; i<node_cnt; ++i) {
    output[threadIdx.x] += input[blockDim.x * i + threadIdx.x];
  }
}

double CLOCK() {
  struct timespec t = {0, 0};
  clock_gettime(CLOCK_MONOTONIC, &t);

  return (double) (t.tv_sec*1.0e3 + t.tv_nsec*1.0e-6);  
}

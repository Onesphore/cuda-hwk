#include <stdio.h>
#include <time.h>

#define LEN     256
#define TILESZ  16

// Uncomment this line if you want to display //
// the result of the computation.             //
// #define DISPLAY 1                          

static double CLOCK();

__global__ void matInit(float*);
__global__ void stencil(float*, float*);
__global__ void stencil_tiled(float*, float*);


int main(int argc, char** argv) {
  float *a, *a_host, *b;

  a_host = (float*) malloc(sizeof(float) * LEN*LEN*LEN);
  cudaMalloc(&a, sizeof(float) * LEN*LEN*LEN);
  cudaMalloc(&b, sizeof(float) * LEN*LEN*LEN);
  cudaMemset(a, 0, sizeof(float) * LEN*LEN*LEN);
  cudaMemset(b, 0, sizeof(float) * LEN*LEN*LEN);
  

  dim3 Grid, Block;

#ifdef TILED
  Grid  = dim3(LEN, LEN/TILESZ, LEN/TILESZ);
  // Block = dim3(TILESZ, TILESZ);
  Block = dim3(TILESZ);

#else
  Grid  = dim3(LEN);
  Block = dim3(LEN);
#endif // ifdef TILED
 

  ////////////////////////////
  // Initialize matrix b    //
  //////////////////////////// 
  matInit<<<LEN, LEN>>>(b);

  ////////////////////////////
  // stencil computation    //
  ////////////////////////////

  double start, end;
  start = CLOCK();

#ifdef TILED
  stencil_tiled<<<Grid, Block>>>(a, b);
#else
  stencil<<<Grid, Block>>>(a, b);
#endif // #ifdef TILED

  cudaDeviceSynchronize();
  end = CLOCK();


  /////////////////////////
  // Display the result  //
  ///////////////////////// 
#ifdef DISPLAY
  cudaMemcpy(a_host, a, sizeof(float) * LEN*LEN*LEN, cudaMemcpyDeviceToHost);
  for (int i=0; i<LEN; ++i)
    for (int j=0; j<LEN; ++j)
      for (int k=0; k<LEN; ++k) {
        printf("(i=%d, j=%d, k=%d) = %.2f\n", 
		i, j, k, a_host[i*LEN*LEN+j*LEN+k]);
      }
#endif // DISPLAY

#ifdef TILED
  printf("stencil-tiled took %.2f ms\n", end-start);
#else
  printf("stencil took %.2f ms\n", end-start);
#endif // #ifdef TILED
  return 0;
}

__global__ void 
matInit(float* mat) {
  int i = blockIdx.x;   // int M = gridDim.x;
  int j = threadIdx.x;  int N = blockDim.x;
                        int L = LEN;

  for (int k=0; k<L; ++k) {
    mat[i*N*L + j*L + k] =  i*N*L + j*L +k;
  }

}

__global__ void
stencil(float *a, float *b) {
  int x = blockIdx.x,  X = gridDim.x,
      y = threadIdx.x, Y = gridDim.x,
                       Z = Y; 

  int tId = x*Y + y;
  if ((x > 0 && x < X-1) && 
      (y > 0 && y < Y-1)) {
     for (int z = 1; z<Z-1; ++z) {
       float b1 = b[(x-1)*Y*Z + y*Z + z],
             b2 = b[(x+1)*Y*Z + y*Z + z],
             b3 = b[x*Y*Z + (y-1)*Z + z],
             b4 = b[x*Y*Z + (y+1)*Z + z],
             b5 = b[x*Y*Z + y*Z + (z-1)],
             b6 = b[x*Y*Z + y*Z + (z+1)];

       a[tId*Z + z] = 0.8*(b1+b2+b3+b4+b5+b6);
     }
  }
}

__global__ void
stencil_tiled(float *a, float *b) {
  int x = blockIdx.x,  X = gridDim.x,
      y = blockIdx.y,  Y = gridDim.y,
      z = blockIdx.z,  Z = gridDim.z,
      s = threadIdx.x, S = blockDim.x,
                       T = S;

  int tId = x*Y*Z*S + y*Z*S +s*Z + z;

  if ((x > 0 && x < X-1) &&
      (y != 0 || s != 0) && (y != Y-1 || s != S-1))
    for (int t=0; t<T; ++t)
      if ((z != 0 || t != 0) && (z != Z-1 || t != T-1)) {

        float b1 = b[(x-1)*Y*Z*S*T + y*Z*S*T +s*Z*T + z*T + t],
              b2 = b[(x+1)*Y*Z*S*T + y*Z*S*T +s*Z*T + z*T + t],
              b3 = b[x*Y*Z*S*T + y*Z*S*T +s*Z*T + z*T + t - (T*Z)],
              b4 = b[x*Y*Z*S*T + y*Z*S*T +s*Z*T + z*T + t + (T*Z)],
              b5 = b[x*Y*Z*S*T + y*Z*S*T +s*Z*T + z*T + (t-1)],
              b6 = b[x*Y*Z*S*T + y*Z*S*T +s*Z*T + z*T + (t+1)];
   
        a[tId*T+t] = 0.8*(b1+b2+b3+b4+b5+b6);
      }
}

double CLOCK() {
  struct timespec t = {0, 0};
  clock_gettime(CLOCK_MONOTONIC, &t);

  return (double) (t.tv_sec*1.0e3 + t.tv_nsec*1.0e-6);
}

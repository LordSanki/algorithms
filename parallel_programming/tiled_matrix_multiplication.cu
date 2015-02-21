#include <wb.h>
#include <cuda.h>
#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows,
    int numAColumns, int numBRows,
    int numBColumns, int numCRows,
    int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int m = numARows; int n = numBRows; int k = numBColumns;
  float cSum = 0.0f;
  int iterations = ((n-1)/TILE_WIDTH)+1;

  for(int t=0; t<iterations; t++)
  {
    if( (t*TILE_WIDTH + threadIdx.x) < n && row < m)
      tileA[threadIdx.y][threadIdx.x] = A[row*n + t*TILE_WIDTH + threadIdx.x];
    else
      tileA[threadIdx.y][threadIdx.x] = 0.0f;

    if( (t*TILE_WIDTH + threadIdx.y) < n && col < k)
      tileB[threadIdx.y][threadIdx.x] = B[(t*TILE_WIDTH + threadIdx.y)*k + col];
    else
      tileB[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();
    for(int j=0;j<TILE_WIDTH; j++)
    {
      cSum = cSum + (tileA[threadIdx.y][j]*tileB[j][threadIdx.x]);

    }
    __syncthreads();

  }

  if(row < m && col < k)
  {
    C[row*k + col] = cSum;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA =
    ( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
  hostB =
    ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float*)malloc(sizeof(float)*numCRows*numCColumns);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void**)&deviceA,sizeof(float)*numARows*numAColumns);
  cudaMalloc((void**)&deviceB,sizeof(float)*numBRows*numBColumns);
  cudaMalloc((void**)&deviceC,sizeof(float)*numCRows*numCColumns);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, sizeof(float)*numARows*numAColumns, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeof(float)*numBRows*numBColumns, cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 numBlocks(((numCColumns-1)/TILE_WIDTH)+1,((numCRows-1)/TILE_WIDTH)+1,1);
  dim3 threadsPerBlock(TILE_WIDTH,TILE_WIDTH,1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here

  matrixMultiplyShared <<< numBlocks, threadsPerBlock >>>(deviceA,deviceB,deviceC,
      numARows, numAColumns,
      numBRows, numBColumns,
      numCRows, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, sizeof(float)*numCRows*numCColumns, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");
  //hostC[0] = numCColumns;
  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}


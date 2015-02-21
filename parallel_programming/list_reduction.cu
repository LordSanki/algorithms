// MP Reduction
// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#include    <wb.h>

//@@ You can change this

#define BLOCK_SIZE 512


#define wbCheck(stmt) do {                                                    \
  cudaError_t err = stmt;                                               \
  if (err != cudaSuccess) {                                             \
    wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
    wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
    return -1;                                                        \
  }                                                                     \
} while(0)

__global__ void reductionKernel(float * input, float * output, int len) {
  //@@ Load a segment of the input vector into shared memory
  __shared__ float mem[BLOCK_SIZE<<1];
  int tidx = threadIdx.x;
  int bidx = blockIdx.x;
  int stride;
  if(bidx*BLOCK_SIZE*2 + tidx < len){
    mem[tidx] = input[bidx*BLOCK_SIZE*2 + tidx];
  }
  else{
    mem[tidx] = 0.0f;
  }
  if((bidx*BLOCK_SIZE*2 + tidx + BLOCK_SIZE) < len){
    mem[tidx + BLOCK_SIZE] = input[bidx*BLOCK_SIZE*2 + tidx + BLOCK_SIZE];
  }
  else{
    mem[tidx + BLOCK_SIZE] = 0.0f;
  }

  //@@ Traverse the reduction tree
  for(stride = BLOCK_SIZE; tidx < stride; stride = stride/2){
    __syncthreads();
    mem[tidx] = mem[tidx] + mem[tidx+stride];
  }
  //@@ Write the computed sum of the block to the output vector at the 
  //@@ correct index
  if(tidx == 0){
    output[bidx] = mem[tidx];
  }
}

int main(int argc, char ** argv) {
  int ii;
  wbArg_t args;
  float * hostInput; // The input 1D list
  //float * hostOutput; // The output list
  float * deviceInput;
  float * deviceOutput;
  int numInputElements; // number of elements in the input list
  int numOutputElements; // number of elements in the output list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numInputElements);

  numOutputElements = ((numInputElements-1) / (BLOCK_SIZE<<1)) +1;

  //hostOutput = (float*) malloc(numOutputElements * sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ", numInputElements);
  //wbLog(TRACE, "The number of output elements in the input is ", numOutputElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void**)&deviceInput, sizeof(float)*numInputElements);
  cudaMalloc((void**)&deviceOutput, sizeof(float)*numOutputElements);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, sizeof(float)*numInputElements, cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");
  //@@ Initialize the grid and block dimensions here

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  while(numInputElements >= BLOCK_SIZE)
  {
    float *temp;
    dim3 numThreads(BLOCK_SIZE,1,1);
    dim3 numBlocks(numOutputElements, 1, 1);
    reductionKernel<<< numBlocks, numThreads >>>(deviceInput, deviceOutput, numInputElements);
    cudaDeviceSynchronize();
    numInputElements = numOutputElements;
    numOutputElements = ((numInputElements-1)/(BLOCK_SIZE<<1)) +1;
    temp = deviceInput; deviceInput = deviceOutput; deviceOutput = temp;
  }
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostInput, deviceInput, sizeof(float)*numInputElements, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  /********************************************************************
   * Reduce output vector on the host
   * NOTE: One could also perform the reduction of the output vector
   * recursively and support any size input. For simplicity, we do not
   * require that for this lab.
   ********************************************************************/
  for (ii = 1; ii < numInputElements; ii++) {
    hostInput[0] += hostInput[ii];
  }

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostInput, 1);

  free(hostInput);
  //free(hostOutput);

  return 0;
}



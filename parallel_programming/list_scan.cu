// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
  cudaError_t err = stmt;                                               \
  if (err != cudaSuccess) {                                             \
    wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
    wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
    return -1;                                                        \
  }                                                                     \
} while(0)

#define BLOCK_WIDTH 16
__global__ void block_sums(float * input, float * output, int len, int step) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from here

  __shared__ float smem[BLOCK_WIDTH*2];
  int id1,id2;
  id1 = step*(blockIdx.x*blockDim.x*2 + threadIdx.x) + (step-1);
  id2 = step*(blockIdx.x*blockDim.x*2 + (blockDim.x + threadIdx.x)) + (step-1);
  // loading data in shared memory
  if(id1 < len){
    smem[threadIdx.x] = input[id1];
  }
  else{
    smem[threadIdx.x] = 0;
  }
  if(id2 < len){
    smem[threadIdx.x+blockDim.x] = input[id2];
  }
  else{
    smem[threadIdx.x+blockDim.x] = 0;
  }

  __syncthreads();
  //stage1 reduction
  int stride;
  for(stride = 1; stride <= blockDim.x; stride *= 2){

    int index = ((threadIdx.x+1)*stride*2) -1;
    if(index < 2*BLOCK_WIDTH){
      smem[index] += smem[index-stride];
    }
    __syncthreads();
  }
  // stage2 expansion
  for(stride = BLOCK_WIDTH/2; stride > 0; stride /= 2){
    int index = ((threadIdx.x+1)*stride*2) -1;
    __syncthreads();
    if(index < 2*BLOCK_WIDTH){
      smem[index + stride] += smem[index];
    }

  }
  __syncthreads();
  if(id1 < len){
    output[id1] = smem[threadIdx.x];
  }
  if(id2 < len){
    output[id2] = smem[threadIdx.x+blockDim.x];
  }
}

__global__ void spread(float * input, int len) {
  int id = blockIdx.x*blockDim.x*2 + threadIdx.x-1;
  int id2 = id + blockDim.x;
  __shared__ float val;
  if(threadIdx.x == 0){
    if(id < len && id > 0){
      val = input[id];
    }
    else{
      val = 0;
    }
  }
  __syncthreads();

  if(id2 < len && id2 > 0){
    input[id2] += val;
  }
  if(threadIdx.x != 0){
    if(id < len && id > 0){
      input[id] += val;
    }
  }
}

__global__ void addKernel (float *in, float *val, int len,  int step){
  int id = threadIdx.x;
  float v = val[step-1];
  if(id < len)
    in[id+step] += v;
}

int main(int argc, char ** argv) {
  wbArg_t args;
  float * hostInput; // The input 1D list
  float * hostOutput; // The output list
  float * deviceInput;
  float * deviceOutput;
  int numElements; // number of elements in the list
  int i;
  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float*) malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ", numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 numThreads(BLOCK_WIDTH,1,1);
  dim3 numBlocks(((numElements-1)/(2*BLOCK_WIDTH))+1,1,1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  block_sums <<<numBlocks, numThreads>>> (deviceInput, deviceOutput, numElements, 1);
  for( i =1; i<((numElements-1)/(2*BLOCK_WIDTH))+1; i++){
    int step = i*BLOCK_WIDTH*2;
    addKernel <<<1 ,2*BLOCK_WIDTH>>> (deviceOutput, deviceOutput, numElements, step);
  }

  //block_sums <<<numBlocks, numThreads>>> (deviceOutput, deviceOutput, numElements, BLOCK_WIDTH*2);
  //spread <<<numBlocks, numThreads>>> (deviceOutput, numElements);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");
  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}



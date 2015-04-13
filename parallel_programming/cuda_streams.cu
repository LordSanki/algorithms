#include<wb.h>

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
  //@@ Insert code to implement vector addition here
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if(idx < len)
    out[idx] = in1[idx] + in2[idx];
}

#define NUM_STREAMS 4
int main(int argc, char ** argv) {
  wbArg_t args;
  int inputLength;
  float * hostInput1;
  float * hostInput2;
  float * hostOutput;
  float * deviceInput1[NUM_STREAMS];
  float * deviceInput2[NUM_STREAMS];
  float * deviceOutput[NUM_STREAMS];
  cudaStream_t st[NUM_STREAMS];
  int sS [NUM_STREAMS];
  int sL [NUM_STREAMS];

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *) malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");


  int eps = ((inputLength-1)/NUM_STREAMS) +1; // elements per stream

  dim3 numThreads(256, 1, 1);
  dim3 numBlocks( ((eps-1)/256) +1, 1, 1);

  for(int i=0; i<NUM_STREAMS; i++){
    sS[i] = eps*i;
    sL[i] = (eps > (inputLength-(eps*i))) ? (inputLength-(eps*i)) : eps;
    sL[i] *= sizeof(float);

    cudaStreamCreate(&st[i]);
    cudaMalloc((void**)&deviceInput1[i], sL[i] * sizeof(float));
    cudaMalloc((void**)&deviceInput2[i], sL[i] * sizeof(float));
    cudaMalloc((void**)&deviceOutput[i], sL[i] * sizeof(float));
  }

  wbTime_start(Generic, "Streamed Add");
  for(int i=0; i<NUM_STREAMS; i++){
    cudaMemcpyAsync(deviceInput1[i], &hostInput1[sS[i]], sL[i], cudaMemcpyHostToDevice, st[i]);
    cudaMemcpyAsync(deviceInput2[i], &hostInput2[sS[i]], sL[i], cudaMemcpyHostToDevice, st[i]);
    //cudaMemcpy(deviceInput1[i], &hostInput1[sS[i]], sL[i], cudaMemcpyHostToDevice);
    //cudaMemcpy(deviceInput2[i], &hostInput2[sS[i]], sL[i], cudaMemcpyHostToDevice);
  }
  for(int i=0; i<NUM_STREAMS; i++){
    vecAdd <<<numBlocks, numThreads, 0, st[i]>>> (deviceInput1[i], deviceInput2[i], deviceOutput[i], sL[i]);
    //vecAdd <<<numBlocks, numThreads, 0>>> (deviceInput1[i], deviceInput2[i], deviceOutput[i], sL[i]);
  }
  for(int i=0; i<NUM_STREAMS; i++){
    cudaMemcpyAsync(&hostOutput[sS[i]], deviceOutput[i], sL[i], cudaMemcpyDeviceToHost, st[i]);
    //cudaMemcpy(&hostOutput[sS[i]], deviceOutput[i], sL[i], cudaMemcpyDeviceToHost);
  }

  cudaDeviceSynchronize();

  wbTime_stop(Generic, "Streamed Add");

  for(int i=0; i<NUM_STREAMS; i++){
    cudaFree(deviceInput1[i]);
    cudaFree(deviceInput2[i]);
    cudaFree(deviceOutput[i]);
  }

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}



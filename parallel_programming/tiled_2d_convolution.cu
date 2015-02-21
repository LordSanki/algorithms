#include    <wb.h>


#define wbCheck(stmt) do {                                                    \
  cudaError_t err = stmt;                                               \
  if (err != cudaSuccess) {                                             \
    wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
    wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
    return -1;                                                        \
  }                                                                     \
} while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2

//@@ INSERT CODE HERE
#define OTILE_SIZE 12
#define ITILE_SIZE (OTILE_SIZE+(Mask_width-1))
#define NCHANNELS 3

__global__ void conv2D(float *img, float *out, const float *__restrict__ mask, int W, int H)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int ro = blockIdx.y*OTILE_SIZE + ty;
  int co = blockIdx.x*OTILE_SIZE + tx;
  int ri = ro - Mask_radius;
  int ci = co - Mask_radius;
  int i,j;
  float acc = 0.0f;
  __shared__ float iTile[ITILE_SIZE*ITILE_SIZE*NCHANNELS];

  if(ri >= 0 && ci >= 0 && ri < H && ci < W){
    iTile[(ty*ITILE_SIZE + tx)*NCHANNELS + tz] = img[(ri*W + ci)*NCHANNELS + tz];
  }
  else{
    iTile[(ty*ITILE_SIZE + tx)*NCHANNELS + tz] = 0.0f;
  }

  __syncthreads();

  if( tx < OTILE_SIZE && ty < OTILE_SIZE){
    for(i=0; i<Mask_width; i++){
      for(j=0; j<Mask_width; j++){
        acc += ( mask[i*Mask_width + j]*iTile[((ty+i)*ITILE_SIZE + tx + j)*NCHANNELS + tz] );
      }
    }
    if(ro < H && co < W){
      out[(ro*W + co)*NCHANNELS + tz] = acc;
    }
  }
}

int main(int argc, char* argv[]) {
  wbArg_t args;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char * inputImageFile;
  char * inputMaskFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float * hostInputImageData;
  float * hostOutputImageData;
  float * hostMaskData;
  float * deviceInputImageData;
  float * deviceOutputImageData;
  float * deviceMaskData;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);
  inputMaskFile = wbArg_getInputFile(args, 1);

  inputImage = wbImport(inputImageFile);
  hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

  assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
  assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);

  assert(imageChannels == NCHANNELS);

  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");


  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData,
      hostInputImageData,
      imageWidth * imageHeight * imageChannels * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMaskData,
      hostMaskData,
      maskRows * maskColumns * sizeof(float),
      cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");


  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE
  dim3 numThreads(ITILE_SIZE, ITILE_SIZE, NCHANNELS);
  dim3 numBlocks( ((imageWidth-1)/OTILE_SIZE)+1, ((imageHeight-1)/OTILE_SIZE)+1, 1);
  conv2D<<< numBlocks, numThreads >>>(deviceInputImageData,
      deviceOutputImageData, deviceMaskData, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");


  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputImageData,
      deviceOutputImageData,
      imageWidth * imageHeight * imageChannels * sizeof(float),
      cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(args, outputImage);

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceMaskData);

  free(hostMaskData);
  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}


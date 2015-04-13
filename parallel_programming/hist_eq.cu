// Histogram Equalization

#include    <wb.h>

#define HISTOGRAM_LENGTH 256
typedef unsigned char uchar;
typedef unsigned int uint;

//@@ insert code here

__global__ void f2c(float *input, uchar *op, int length){
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  if(idx < length)
    op[idx] = (uchar)(input[idx]*255.0);
}

__global__ void c2f(uchar *input, float *op, int length){
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  if(idx < length)
    op[idx] = (float)(input[idx]/255.0);
}

#define RGB2G_BLOCK_WIDTH 512


__global__ void rgb2g(uchar* input, uchar *output, int length, int length2){
  __shared__ uchar smem[RGB2G_BLOCK_WIDTH*3];
  int idx = blockIdx.x*blockDim.x + threadIdx.x;

  if(idx < length)
    smem[threadIdx.x] = input[idx];
  else
    smem[threadIdx.x] = 0;

  if(idx+RGB2G_BLOCK_WIDTH < length)
    smem[threadIdx.x + RGB2G_BLOCK_WIDTH] = input[idx+RGB2G_BLOCK_WIDTH];
  else
    smem[threadIdx.x + RGB2G_BLOCK_WIDTH] = 0;

  if(idx+(RGB2G_BLOCK_WIDTH*2) < length)
    smem[threadIdx.x + (RGB2G_BLOCK_WIDTH*2)] = input[idx + (RGB2G_BLOCK_WIDTH*2)];
  else
    smem[threadIdx.x+ 2*RGB2G_BLOCK_WIDTH] = 0;

  __syncthreads();

  if(idx < length2)
    output[idx] = (uchar)(0.21*smem[3*threadIdx.x] + 0.71*smem[3*threadIdx.x +1] + 0.07*smem[3*threadIdx.x +2]);
}


__global__ void histo(uchar *input, uint *output, int length)
{
  __shared__ uint phist[HISTOGRAM_LENGTH];
  int tid = threadIdx.x;
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = blockDim.x*gridDim.x;

  if(tid < HISTOGRAM_LENGTH)
    phist[tid] = 0;
  __syncthreads();

  while( i< length){
    atomicAdd(&(phist[input[i]]),1);
    i += stride;
  }
  __syncthreads();
  if(tid < HISTOGRAM_LENGTH){
    atomicAdd(&(output[tid]),phist[tid]);
  }
}

__global__ void pdf(uint *input, float *output, int length){
  int tid = threadIdx.x;
  output[tid] = ((float)input[tid])/length;
}
__global__ void cdf(float *input){
  // prefix sum kernel
  if (threadIdx.x == 0)
    for(int i=1; i<HISTOGRAM_LENGTH; i++)
      input[i] += input[i-1];
}

__global__ void mincdf(float *input, float *output)
{
  // min reduction kernel
  int tid = threadIdx.x;
  int stride = HISTOGRAM_LENGTH/2;

  __shared__ float smem[HISTOGRAM_LENGTH];
  smem[tid] = input[tid];

  for(int i=0; i<stride; stride = stride /2){
    __syncthreads();
    smem[tid]  = (smem[tid]<smem[tid+stride]) ? smem[tid] : smem[tid+stride];
  }
  if(threadIdx.x == 0)
    output[0] = smem[0];
}

__global__ void genHistEqMap(float *cdf, const float *mincdf, uchar *map){
  //clamp(255*(cdf[val] - cdfmin)/(1 - cdfmin), 0, 255)
  int tid = threadIdx.x;
  float v = cdf[tid];
  v = ((v-mincdf[0])/(1-mincdf[0]))*255.0;
  v = (v<0.0)?0.0:v;
  v = (v>255.0)?255.0:v;
  map[tid] = (uchar)v;
}

__global__ void applyMap(uchar *input, uchar *output, const uchar * __restrict__ map, int length)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = blockDim.x*gridDim.x;

  while(idx < length){
    output[idx] = map[input[idx]];
    idx += stride;
  }
}

void cpu_impl(float *img, float *output, int length){
  unsigned int hist[256] = {0};
  unsigned char map[256];
  float cdf[256] = {0};
  float mincdf = 0;
  int i;
  uchar *cimg = (uchar*)malloc(length*3*sizeof(uchar));
  uchar *gcimg = (uchar*)malloc(length*sizeof(uchar));
  for(i=0; i<length*3; i++){
    cimg[i] = (uchar)(img[i]*255.0);
  }
  for(i=0; i<length; i++){
    gcimg[i] = (uchar)(0.21*cimg[3*i] + 0.71*cimg[3*i +1] + 0.07*cimg[3*i +2]);
    hist[gcimg[i]]++;
  }
  cdf[0] = (float)hist[0]/(float)length;
  for(i=1; i<256; i++){
    cdf[i] = cdf[i-1] + (float)hist[i]/(float)length;
  }
  for(i=0; i<256; i++)
    if(mincdf < cdf[i]){ mincdf = cdf[i]; break;}

  for(i=0; i<256; i++){
    float f = ((cdf[i] - mincdf)/(1-mincdf))*255;
    f = (f<0)?0:f; f = (f>255)?255:f;
    map[i] = (uchar)f;
  }
  for(i=0; i<length*3; i++){
    cimg[i] = map[cimg[i]];
    output[i] = ((float)cimg[i])/255.0f;
  }
  free(cimg); free(gcimg);
}

int main(int argc, char ** argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float * hostInputImageData;
  float * hostOutputImageData;
  const char * inputImageFile;
  int size; int length;
  //@@ Insert more code here
  float *dfImage;
  uchar *dcImage;
  uchar *dcgImage;
  uint *dhist;
  float *dcdf;
  float *dmincdf;
  uchar *dhistEqMap;


  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  wbTime_stop(Generic, "Importing data and creating memory on host");
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  cpu_impl(hostInputImageData, hostOutputImageData, imageWidth*imageHeight);
#if 0f    //@@ insert code here
  length = imageWidth*imageHeight;
  size = length*imageChannels;
  dim3 castThreads(512,1,1);
  dim3 castBlocks(((size-1)/512)+1,1,1);
  dim3 rgbThreads(RGB2G_BLOCK_WIDTH,1,1);
  dim3 rgbBlocks(((length-1)/RGB2G_BLOCK_WIDTH)+1,1,1);

  cudaMalloc((void**)&dfImage, size*sizeof(float));
  cudaMalloc((void**)&dcImage, size*sizeof(uchar));

  cudaMemcpy(dfImage, hostInputImageData, size*sizeof(float), cudaMemcpyHostToDevice);

  f2c <<<castBlocks, castThreads>>> (dfImage, dcImage, size);

  cudaFree(dfImage);dfImage
    cudaMalloc((void**)&dcgImage, length*sizeof(uchar));

  rgb2g <<<rgbBlocks, rgbThreads>>> (dcImage, dcgImage, size, length);

  cudaFree(dcImage);

  cudaMalloc((void**)&dhist, HISTOGRAM_LENGTH*sizeof(uint));
  cudaMalloc((void**)&dcdf, HISTOGRAM_LENGTH*sizeof(float));
  cudaMalloc((void**)&dmincdf, sizeof(float));
  cudaMalloc((void**)&dhistEqMap, HISTOGRAM_LENGTH*sizeof(uchar));

  dim3 histThreads(512,1,1);
  dim3 histBlocks(((length-1)/256)+1,1,1);
  histo <<<histBlocks ,histThreads>>> (dcgImage, dhist, length);

  dim3 cdfThreads(HISTOGRAM_LENGTH,1,1);
  dim3 cdfBlocks(1,1,1);
  pdf <<<cdfBlocks, cdfThreads>>> (dhist, dcdf, length);

  cdf <<<cdfBlocks, cdfThreads>>> (dcdf);

  mincdf <<<cdfBlocks, cdfThreads>>> (dcdf, dmincdf);

  genHistEqMap <<<cdfBlocks, cdfThreads>>> (dcdf, dmincdf, dhistEqMap);

  cudaFree(dhist);
  cudaFree(dcdf);
  cudaFree(dmincdf);

  cudaMalloc((void**)&dcImage, length*sizeof(uchar));

  applyMap <<<rgbBlocks, rgbThreads>>> (dcgImage, dcImage, dhistEqMap, length);

  cudaFree(dcgImage);
  cudaMalloc((void**)&dfImage, length*sizeof(float));

  c2f <<<rgbBlocks, rgbThreads>>> (dcImage, dfImage, length);

  cudaMemcpy(hostOutputImageData, dfImage, length*sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(dhistEqMap);dhistEqMapcudaFree(dfImage);
  cudaFree(dcImage);
#endif

  wbSolution(args, outputImage);

  //@@ insert code here

  return 0;
}


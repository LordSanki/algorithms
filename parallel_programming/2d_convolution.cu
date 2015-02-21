/*
   * Prashant Solanki (Unity: psolank)
   * Simple Image convolutions implementation without tiling
   * Convolutions mask is stored in constant memory
   * Tested with CUDA Toolkit 3.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <cuda.h>

#define BUF_SIZE 200000

#define ThreadsPerBlockX 8
#define ThreadsPerBlockY 8

#define CUDA_CALL(X) if(cudaSuccess != X) printf("Call Failed at %s\n",__LINE__);

int count_cols(char *buff);
void count_rc(char *fname, int *r1, int *c1, int *r2, int *c2);
void print2D(float *arr, int r, int c);
float* alloc2D(int r, int c);
void free2D(float *arr);
void parse2D(FILE *f, float *arr, int r, int c);
void parse2DPadded(FILE *f, float *arr, int r, int c, int px, int py);
void flip_kernel(float * arr, int r, int c);

// Constant cache memory to store convolution mask and its size
__constant__ float dMask[100];
__constant__ int dHalfW;
__constant__ int dHalfH;

// kernel to convolve image with mask
// one thread processes one pixel in input image
__global__ void conv2DKernel(float *in, float *out, int r1, int c1) {

  int i,j; int x,y; 
  int maskIndex = 0;

  // computing row and column of pixel for which convolution os to be done
  int r = blockIdx.y*blockDim.y + threadIdx.y;
  int c = blockIdx.x*blockDim.x + threadIdx.x;
  float acc = 0.0f;

  // accessing neighbouring pixels and multiplying with mask
  for(i = -dHalfH; i <= dHalfH; i++){
    for(j = -dHalfW; j <= dHalfW; j++){
      x = c + j;
      y = r + i;
      // condition to check if element is outside the image
      if(x >= 0 && x < c1 && y >= 0 && y < r1){
        acc = acc + (dMask[maskIndex] * in[ y*c1 + x ]);
      }
      maskIndex++;
    }
  }
  // condition to check if element is outside image
  if(r < r1 && c < c1){
    out[ r*c1 + c ] = acc;
  }
}

int main(int argc, char **argv) {
  float *hInput;
  float *hMask;
  float *hOutput;
  float *dInput;
  float *dOutput;
  int r1,c1,r2,c2, R, C;
  FILE *fptr;
  if(argc < 2) { printf(" Please specify input filename\n"); return -1;}
  // Finding dimensions of input matricex
  count_rc(argv[1],&r1, &c1, &r2, &c2);
  if(r1 == 0) return -1;

  // conputing dimensions of output matrix 
  R = (r1 + r2) -1;
  C = (c1 + c2) -1;
 
  // allocating input matrices
  hInput = alloc2D(R, C);

  // zeroing the input matrix
  memset(hInput, 0, sizeof(float)*R*C);

  // allocation mask
  hMask  = alloc2D(10, 10);

  // allocating output matix
  hOutput = alloc2D(R, C);

  // opening input file
  fptr = fopen(argv[1], "rb");

  // parsing first matrix withing the padded region defined as c2/2 and r2/2
  parse2DPadded(fptr, hInput, r1, c1, c2/2, r2/2);

  // parsing mask
  parse2D(fptr, hMask, r2, c2);

  // closing the file
  fclose(fptr);

  // flipping kernel  vertically and horizontally
  flip_kernel(hMask, r2, c2);

//  print2D(hMask, r2, c2);

  r2 = r2/2;
  c2 = c2/2;

  // allocating gpu memory
  CUDA_CALL(cudaMalloc((void**)&dInput, R*C*sizeof(float)));
  //err = cudaMalloc((void**)&dMask, r2*c2*sizeof(float));
  CUDA_CALL(cudaMalloc((void**)&dOutput, R*C*sizeof(float)));

  // Copy memory to the GPU
  CUDA_CALL(cudaMemcpy(dInput, hInput, sizeof(float)*R*C, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpyToSymbol(dMask, hMask, sizeof(float)*10*10, 0, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpyToSymbol(dHalfW, (const int*)&r2, sizeof(int), 0, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpyToSymbol(dHalfH, (const int*)&c2, sizeof(int), 0, cudaMemcpyHostToDevice));

  // Initialize the grid and block dimensions
  dim3 numThreads(ThreadsPerBlockX,ThreadsPerBlockY,1);
  dim3 numBlocks( ((C-1)/ThreadsPerBlockX)+1, ((R-1)/ThreadsPerBlockY)+1, 1 );
  
  // Launch the GPU Kernel
  conv2DKernel<<<numBlocks, numThreads>>>(dInput, dOutput, R, C);

  //cudaDeviceSynchronize();
  CUDA_CALL(cudaThreadSynchronize());

  // Copy the GPU memory back to the CPU
  CUDA_CALL(cudaMemcpy(hOutput, dOutput, R*C*sizeof(float), cudaMemcpyDeviceToHost));

  // free the GPU memory
  CUDA_CALL(cudaFree(dInput));
  CUDA_CALL(cudaFree(dOutput));

  // printing result
  print2D(hOutput, R, C);

  // free the host memory
  free2D(hInput);
  free2D(hMask);
  free2D(hOutput);

  return 0;
}

// count number of rows and columns for the given input file
void count_rc(char *fname, int *r1, int *c1, int *r2, int *c2)
{
  *r1 = 0; *c1 = 0; *r2 = 0; *c2 =0;
  char *buff = (char*)malloc(BUF_SIZE);
  FILE *f = fopen(fname, "rb");
  if(f == NULL){ printf("Unable to open file %s\n",fname); free(buff); return; }

  fgets(buff, BUF_SIZE, f);
  *c1 = count_cols(buff);
  while(strlen(buff) > 1){
    (*r1)++;
    fgets(buff, BUF_SIZE, f);
  }

  fgets(buff, BUF_SIZE, f);
  *c2 = count_cols(buff);
  while(strlen(buff) > 1){
    (*r2)++;
    if(NULL == fgets(buff, BUF_SIZE, f)) break;
    if((feof(f)) && (strlen(buff) > 1) ){(*r2)++; break;}
  }
  free(buff);
  fclose(f);
}

// count number of columns in given buffer
int count_cols(char *buff)
{
  int i;int n=1;
  for(i=0; i<strlen(buff)-1; i++)
  {
    if(buff[i] == ' '){
      if(buff[i+1] != '\n' && buff[i+1] != '\r' && buff[i+1] != ' '){
        n++;
      }
    }
  }
  return n;
}

// print a 2D matrix
void print2D(float *arr, int r, int c)
{
  int i,j;
  for(i=0; i<r; i++){
    for(j=0; j<c; j++){
      if(j>0) printf(" ");
      printf("%f",arr[ i*r + j]);
    }
    printf("\n");
  }
}

// allocate memory for matrix of size rxc
float* alloc2D(int r, int c)
{
  return (float*)malloc( r*c*sizeof(float) );
}

// free memory
void free2D(float *arr)
{
  free(arr);
}

// parsing a matrix of size rxc
void parse2D(FILE *f, float *arr, int r, int c)
{
  int i,j;
  for(i=0; i<r; i++){
    for(j=0; j<c; j++){
      fscanf( f, "%f", &arr[ (i*c) + j] );
    }
  }
}

void parse2DPadded(FILE *f, float *arr, int r, int c, int px, int py)
{
  int i,j;
  int wStep = c + 2*px;
  int offset = py*wStep + px;
  for(i=0; i<r; i++){
    for(j=0; j<c; j++){
      fscanf( f, "%f", &arr[ offset + (i*wStep) + j] );
    }
  }
}

void flip_kernel(float * arr, int r, int c)
{
  float f;
  int i,j;
  int R = r-1;
  int C = c-1;
  for(i=0; i<=r/2; i++){
    for(j=0; j<c; j++){
      f = arr[i*c +j];
      arr[i*c +j] = arr[(R-i)*c + (C-j)];
      arr[(R-i)*c + (C-j)] = f;
    }
  }
}







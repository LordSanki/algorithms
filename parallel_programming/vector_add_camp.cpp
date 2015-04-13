#include <wb.h>
#include <amp.h>

using namespace concurrency;

void vecAdd(float *a, float *b, float *c, int n)
{
  array_view<float,1> A(n,a), B(n,b), C(n,c);
  C.discard_data();
  parallel_for_each(C.get_extent(), [=](index<1> i)
      restrict(amp){
      C[i] = A[i] + B[i]; } );
  C.synchronize();
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;

  args = wbArg_read(argc, argv);

  hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  //@@ Insert C++AMP code here
  vecAdd(hostInput1, hostInput2, hostOutput, inputLength);

  wbSolution(args, hostOutput, inputLength);

  return 0;
}


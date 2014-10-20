#include <iostream>
#include <mergesort.h>

using namespace std;

int main()
{
  int a[16] = {0,2,3,6,7,3,7,3,7,74,5,2,67,8,5,16};

  for(int i=0; i<16; i++)
  {
    cout<<a[i]<<" ";
  }
  cout<<endl;

  MergeSort::sortArray(a, 16);

  for(int i=0; i<16; i++)
  {
    cout<<a[i]<<" ";
  }
  cout<<endl;
  return 0;
}


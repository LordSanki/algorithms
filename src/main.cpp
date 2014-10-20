#include <iostream>
#include <mergesort.h>

using namespace std;

bool test_merge_sort();

int main()
{
  cout<<"Testing Merge Sort: "<<test_merge_sort()<<endl;
  return 0;
}

bool test_merge_sort()
{
  int size = 100000;
  float f = 3.333;
  int *a = (int*)&f;
  unsigned short int *arr = new unsigned short int [size];
  unsigned long int addr = ((unsigned long int)arr);
  for (int i=1; i<size; i++)
  {
    arr[i] = ((addr|i)*arr[i-1])^*a;
  }
  
  MergeSort::sortArray(arr, size);

  bool result = true;
  for(int i=1; i<size; i++)
    if(arr[i] < arr[i-1]) { result = false; break; }
  
  delete [] arr;
  return result;
}


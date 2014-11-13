#ifndef __QUICK_SORT_H__
#define __QUICK_SORT_H__
#include <debug_print.h>
#include <cstdlib>
#include <cstdio>
namespace QuickSort
{
  void print_array(int * arr, int size)
{
  for(int i=0; i<size; i++)
  {
    printf("%d,",arr[i]);
  }
  printf("\n");
}
  int choose_pivot(int *arr, int size)
  {
    return size/2;
  }
  int partition(int *arr, int size, int pivot)
  {
    int split = 0;
    int t = arr[0];
    arr[0] = arr[pivot];
    arr[pivot] = t;
    for(int i=1; i<size; i++)
    {
      if(arr[i] < arr[0])
      {
        split++;
        t = arr[split];
        arr[split] = arr[i];
        arr[i] = t;
      }
    }
    t = arr[split];
    arr[split] = arr[0];
    arr[0] = t;
    return split;
  }

  void sortArray(int *arr, int size)
  {
    if(size > 1)
    { // Divide
      int pivot = choose_pivot(arr, size);
      pivot = partition(arr, size, pivot);
      if(pivot > 0)
        sortArray( arr, pivot );
      pivot++;
      size = size - pivot;
      if(size > 0)
        sortArray( &arr[pivot], size );
    }
  }
};

#endif //__QUICK_SORT_H__

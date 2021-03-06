#ifndef __QUICK_SORT_H__
#define __QUICK_SORT_H__
#include <debug_print.h>
#include <cstdlib>
#include <cstdio>
//#define MED3
//#define LAST
#define FIRST
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
#ifdef FIRST
      return 0;
#endif
#ifdef LAST
      return size-1;
#endif
#ifdef MID
      return size/2;
#endif
#ifdef MED3
      int first = arr[0];
      int last = arr[size-1];
      int mid; int mid_index;
      if(size%2 == 0)
        mid_index = (size/2)-1;
      else
        mid_index = size/2;
      mid = arr[mid_index];
      if((first < last && first > mid)||(first >last && first <mid))
        return 0;
      if((last <first && last >mid)||(last >first && last <mid))
        return size -1;
      return mid_index;
#endif
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
  long long count_comparisons(int *arr, int size, long long comp = 0)
  {
    if(size > 1)
    { // Divide
      int pivot = choose_pivot(arr, size);
      comp = comp + (size-1);
      pivot = partition(arr, size, pivot);
      if(pivot > 0)
      {
        //comp = comp + (pivot-1);
        comp = count_comparisons( arr, pivot, comp);
      }
      pivot++;
      size = size - pivot;
      if(size > 0)
      {
        //comp = comp + (size-1);
        comp = count_comparisons( &arr[pivot], size,comp );
      }
    }
    return comp;
  }
  int rselect_order(int * arr, int size, int pos)
  {
    int pivot = choose_pivot(arr, size);
    pivot = partition(arr, size, pivot);
    if(pivot > pos-1)
      return rselect_order(arr, pivot, pos);
    else if(pivot < pos-1)
      return rselect_order(&arr[pivot+1],size-(pivot+1),pos-(pivot+1));
    else
      return arr[pivot];
  }
};

#endif //__QUICK_SORT_H__

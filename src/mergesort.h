#ifndef __MERGE_SORT_H__
#define __MERGE_SORT_H__
#include <debug_print.h>
namespace MergeSort
{
  template <typename Type>
    void sortArray(Type *arr, int size)
    {
#pragma omp parallel
      {
#pragma omp single nowait
        {
          if(size > 1)
          { // Divide
#pragma omp task
            sortArray( arr, size/2 );
#pragma omp task
            sortArray( &arr[size/2], size-(size/2) );
#pragma omp taskwait
            // Merge
            int k = size/2;
            for(int i=0; i<size; i++)
            {
              if((arr[i] > arr[k]) && (k<size) )
              {
                Type t = arr[k];
                int l=k;
                while(l>i)
                {
                  arr[l] = arr[l-1];
                  l--;
                }
                arr[i] = t;
                k++;
              }
            }
          }
        }
      }
    }
};

#endif //__MERGE_SORT_H__

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
              if((arr[i] > arr[k]) && (k<size) && (i<k))
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

  template <typename Type>
    unsigned long int countInversions(Type *arr, int size)
    {
      unsigned long int count=0;
      if(size > 1)
      { // Divide
        count += countInversions( arr, size/2 );
        count += countInversions( &arr[size/2], size-(size/2) );
        // Merge
        int k = size/2;
        for(int i=0; i<size; i++)
        {
          if((arr[i] > arr[k]) && (k<size) && (i<k))
          {
            Type t = arr[k];
            int l=k;
            while(l>i)
            {
              arr[l] = arr[l-1];
              l--;
              count++;
            }
            arr[i] = t;
            k++;
          }
        }
      }
      return count;
    }

};

#endif //__MERGE_SORT_H__

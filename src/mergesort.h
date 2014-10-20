#ifndef __MERGE_SORT_H__
#define __MERGE_SORT_H__

namespace MergeSort
{
  template <typename Type>
  void sortArray(Type *arr, int size)
  {
    if(size > 1)
    { // Divide
      sortArray( arr, size/2 );
      sortArray( &arr[size/2], size/2 );
    }
    else
    { // Conquer
      return;
    }
    // Merge
    int k = size/2;
    for(int i=0; i<size; i++)
    {
      if(arr[i] < arr[k] && i<k )
      {
      }
      else if(k<size)
      {
        int l=k;
        while(l>i)
        {
          Type t = arr[l];
          arr[l] = arr[l-1];
          arr[l-1] = t;
          l--;
        }
        k++;
      }
    }
  }

};

#endif //__MERGE_SORT_H__

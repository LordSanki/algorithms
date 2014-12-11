#include <iostream>
#include <mergesort.h>
#include <cstdio>
#include <quicksort.h>

using namespace std;

#define NUM_VALS 10000
//#define NUM_VALS 1000
//#define NUM_VALS 100
typedef void(*Sorter)(int*,int);
bool test_sort(Sorter sort);
int * read_input();
void print_array(int * arr, int size);
bool test_rselect();
int main()
{
//  cout<<"Testing Merge Sort: "<<test_merge_sort()<<endl;
//  int arr[] = { 1, 3, 4, 2 };
//  int arr[] = { 4, 80, 70, 23, 9, 60, 68, 27, 66, 78, 12, 40, 52, 53, 44, 8, 49, 28, 18, 46, 21, 39, 51, 7, 87, 99, 69, 62, 84, 6, 79, 67, 14, 98, 83, 0, 96, 5, 82, 10, 26, 48, 3, 2, 15, 92, 11, 55, 63, 97, 43, 45, 81, 42, 95, 20, 25, 74, 24, 72, 91, 35, 86, 19, 75, 58, 71, 47, 76, 59, 64, 93, 17, 50, 56, 94, 90, 89, 32, 37, 34, 65, 1, 73, 41, 36, 57, 77, 30, 22, 13, 29, 38, 16, 88, 61, 31, 85, 33, 54 };
//  int arr[] = { 9, 12, 3, 1, 6, 8, 2, 5, 14, 13, 11, 7, 10, 4, 0 };
//  int arr[] = { 37, 7, 2, 14, 35, 47, 10, 24, 44, 17, 34, 11, 16, 48, 1, 39, 6, 33, 43, 26, 40, 4, 28, 5, 38, 41, 42, 12, 13, 21, 29, 18, 3, 19, 0, 32, 46, 27, 31, 25, 15, 36, 20, 8, 9, 49, 22, 23, 30, 45 };
//    int arr[] = { 1, 6, 3, 2, 4, 5 };
//  cout<<"Inversions: "<<MergeSort::countInversions(arr, sizeof(arr)/sizeof(int))<<endl;
//  cout<<"Inversions: "<<MergeSort::countInversions(read_input(),100000)<<endl;
//  std::cout<<"Testing: "<< test_sort(QuickSort::sortArray)<<std::endl;

//    std::cout<<"Testing Rselect: "<<test_rselect()<<std::endl;
//    std::cout<<"Testing Rselect: "<<QuickSort::rselect_order(arr,6,3)<<std::endl;

  int * arr = read_input();
//  int arr[] = {3,9,8,4,6,10,2,5,7,1};
  cout<<"Comparisons: "<<QuickSort::count_comparisons(arr, NUM_VALS)<<endl;
  return 0;
}

void print_array(int * arr, int size)
{
  printf("Array :\n");
  for(int i=0; i<size; i++)
  {
    printf("%d,",arr[i]);
  }
  printf("\n");
}

bool test_sort(Sorter sort)
{
  int size = 100000;
  float f = 3.333;
  int *a = (int*)&f;
  int *arr = new int [size];
  unsigned long int addr = ((unsigned long int)arr);
  for (int i=1; i<size; i++)
  {
    arr[i] = ((addr|i)*arr[i-1])^*a;
  }
  
  sort(arr, size);

  bool result = true;
  for(int i=1; i<size; i++)
    if(arr[i] < arr[i-1]) { result = false; break; }
  
  delete [] arr;
  return result;
}

bool test_rselect()
{
  int size = 10000;
  float f = 3.333;
  int *a = (int*)&f;
  int *arr = new int [size];
  int *arr2 = new int [size];
  unsigned long int addr = ((unsigned long int)arr);
  for (int i=1; i<size; i++)
  {
    arr[i] = ((addr|i)*arr[i-1])^*a;
    arr2[i] = arr[i];
  }
  QuickSort::sortArray(arr, size);

  bool result = (arr[size/10] == QuickSort::rselect_order(arr2,size,(size/10)+1));
  result = result && (arr[size/5] == QuickSort::rselect_order(arr2,size,(size/5)+1));
  result = result && (arr[size/4] == QuickSort::rselect_order(arr2,size,(size/4)+1));
  result = result && (arr[size/2] == QuickSort::rselect_order(arr2,size,(size/2)+1));
  result = result && (arr[3*size/4] == QuickSort::rselect_order(arr2,size,(3*size/4)+1));
  return result;
}

int * read_input()
{
  int size = NUM_VALS;
  int *arr = new int [size];
  for(int i=0; i<size; i++)
  {
    scanf("%d",&arr[i]);
  }
  return arr;
}


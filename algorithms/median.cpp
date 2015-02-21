#include <iostream>
#include <cstdio>
#include <cstdlib>

typedef bool(*Func)(int,int);

static bool min(int child, int parent) { return child < parent; }
static bool max(int child, int parent) { return child > parent; }

void bubble_down(int *heap, int size, Func f);
void bubble_up(int *heap, int size, Func f);

using namespace std;
int main(int argc, char **argv)
{
  if (argc < 2) return -1;
  int size = atoi(argv[1]);

  // min heap will hold the largest size/2 values
  int *high = new int[size/2 +1];
  // max heap will hold the smallest size/2 values
  int *low = new int[size/2 +1];

  // initializing
  int high_size =0,low_size =0;
  cin>>high[1];high_size++;

  int median_sum = high[1];
  for (int i=2; i<=size; i++)
  {
    // reading input
    int num;
    cin>>num;
    
    // checking if element is larger than min of high heap
    if(num > high[1])
    {
      // adding element to high heap
      high_size++; high[high_size] = num; 
      bubble_up(high, high_size, min);
    }
    else
    {
      // adding element to low heap
      low_size++; low[low_size] = num; 
      bubble_up(low, low_size, max);
    }

    // adjusting imbalance when high heap is larger
    if( high_size > low_size + 1)
    {
      // taking min element of high heap and adding to low heap
      low_size++; low[low_size] = high[1];
      bubble_up(low, low_size, max);

      // deleting min element from high heap by swapping with last element
      high[1] = high[high_size]; high_size--;
      // correcting via bubbling down
      bubble_down(high, high_size, min);
    }
    // adjusting imbalance when low heap is larger
    else if (low_size > high_size + 1)
    {
      // taking max element of low heap and adding to high heap
      high_size++; high[high_size] = low[1];
      // bubbling up the element in high heap
      bubble_up(high, high_size, min);

      // removing max element of low heap by swapping with last element
      low[1] = low[low_size]; low_size--;
      // correcting via bubbling down
      bubble_down(low, low_size, max);
    }
    if(low_size > high_size)
    {
      median_sum += low[1];
    }
    else if (low_size < high_size)
    {
      median_sum += high[1];
    }
    else
    {
      median_sum += low[1];
    }
  }
  cout<<"Median Sum:"<<median_sum%10000<<endl;
  return 0;
}


void bubble_up(int *heap, int size, Func f)
{
  int n = size;
  while (n>1)
  {
    if( f(heap[n],heap[n/2]) )
    {
      int k = heap[n];
      heap[n] = heap[n/2];
      heap[n/2] = k;
    }
    else
      break;
    n=n/2;
  }
}

void bubble_down(int *heap, int size, Func f)
{
  unsigned int n = 1;
  unsigned int l = n*2;

  while(size > l)
  {
    int child;
    // finding the candidate child to compare and swap
    if( f(heap[l], heap[l+1]) )
      child = l;
    else
      child = l+1;

    // checking if candidate child is to be swapped
    if( f(heap[child],heap[n]) )
    {
      int t = heap[child];
      heap[child] = heap[n];
      heap[n] = t;
    }
    else
      break;
    n = child;
    l = 2*n;
  }

  // special case when last node has only 1 child
  // then the while loop terminates at size == l
  if(size == l)
  {
    if( f(heap[l],heap[n]))
    {
      int t = heap[l];
      heap[l] = heap[n];
      heap[n] = t;
    }
  }
}


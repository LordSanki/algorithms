#include <cstring>
#include <cstdlib>
#include <cstdio>
int main(int argc, char **argv)
{
  int size = atoi(argv[1]);
  bool *arr = new bool[size];

  memset(arr, 1,sizeof(bool)*size);
//  for(int i=2; i<size; i++)
//    arr[i] = true;
  arr[0] = false; arr[1] = false;

  for(int i=2; i<size; i++)
  {
    if(arr[i] == true)
    {
      int mult = 2;
      while((mult * i) < size)
      {
        arr[mult*i] = false;
        mult++;
      }
    }
  }
  for(int i=0; i<size; i++)
    if(arr[i])
      printf("%d ",i);
  printf("\n");
  
  return 0;
}



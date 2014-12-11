#include <stdio.h>
#include <algorithm>
int main()
{
  printf("Enter n diff followed by array\n");
  int n,k; char c; int count = 0;
  scanf("%d",&n); scanf("%c",&c);scanf("%d",&k);
  int *arr = new int[n];
  for(int i=0; i< n; i++)
  {
    scanf("%d", &arr[i]);
  }
  std::sort(arr, &arr[n]);
  for(int i=0; i< n; i++)
  {
    if(std::binary_search(&arr[i],&arr[n],arr[i]+k))
    {
        count++;
    }
  }
  printf("%d\n",count);
  return 0;
}

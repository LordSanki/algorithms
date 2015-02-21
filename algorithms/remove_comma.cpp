#include <cstdio>
#include <cstring>

int main()
{
  char c[] = {'1',',','2',',','3',',','4',',','5',',','6'};

  int j=0,i=0,size = sizeof(c);
  for(int i=0; i<size; i++)
  {
    if(c[i] != ',')
    {
      c[j] = c[i];
      j++;
    }
  }
  if(j<i)
    c[j] = '\0';

  printf("%s\n",c);
  return 0;
}

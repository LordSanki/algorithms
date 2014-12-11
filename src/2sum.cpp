#include <unordered_map>
#include <iostream>
#include <cstdio>
using namespace std;

struct node
{
  node *next;
  int val;
  node(int i) {next = NULL;val = i;}
};


int main(int argc, char** argv)
{
  typedef unordered_map<long,bool> DPairs;
  typedef unordered_map<long, DPairs > Hash;
  if(argc < 2) return -1;
  
  Hash table;
  node * nums = NULL;
  
  int size = atoi(argv[1]);
  table.reserve(size);

  for(int i=0; i<size; i++)
  {
    long l;
    cin>>l;
    if(table.count(l) == 0)
    {
      table.insert(pair<long,DPairs>(l,DPairs()));
      node * n = new node(l);
      n->next = nums;
      nums = n;
    }
  }

  long pairs_found = 0;
  cout<<"Counting pairs:"<<endl;

  node *sums = NULL;

  for (int i=-10000; i<=10000; i++)
  {
    node *n = new node(i);
    n->next = sums;
    sums = n;
  }

  node *k = nums;
  // iterating over the k numbers
  while(k)
  {
    node *n = sums;
    node *p = NULL;
    long first = k->val;
    // iterating over n possible sums
    while(n)
    {
      long sum = n->val;
      // computing the second number required to form the current sum with first
      long second = sum - first;
      // condition to check if it exists in table
      if((table.count(second) == 1) && (first != second))
      {
        cout<<"first:"<<first<<" second:"<<second<<" sum:"<<sum<<endl;
        // making sure the pair was not prevoisly formed
        if(table[second].count(first) == 0)
        {
          table[first].insert(pair<long,bool>(second,true));

          pairs_found++;
          // deleting the sums for which a pair of numbers has been found
          if(n == sums)
          {
            sums = sums->next;
            delete n;
            n = sums;
          }
          else
          {
            p->next = n->next;
            delete n;
            n = p->next;
          }
        }
      }
      else
      {
        p = n;
        n = n->next;
      }
    }
    k = k->next;
  }

  cout<<endl<<"Number of pairs: "<<pairs_found<<endl;
  return 0;
}

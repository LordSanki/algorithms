#include <unordered_map>
#include <unordered_set>
#include <map>
#include <iostream>
#include <cstdio>
using namespace std;

struct node
{
  node *next;
  long val;
  node(long i) {next = NULL;val = i;}
};


int main(int argc, char** argv)
{
  typedef unordered_set<long> DPairs;
  typedef unordered_map<long, DPairs > Hash;
  if(argc < 2) return -1;
  
  long size = atoi(argv[1]);

  Hash table;
  node * nums = NULL;
  
  table.max_load_factor(0.01);
  table.rehash(size*100);

  node *tail = NULL;
  for(long i=0; i<size; i++)
  {
    long l;
    long n = scanf("%ld",&l);
    if(table.find(l) == table.end())
    {
      table.insert(pair<long,DPairs>(l,DPairs()));
      node * n = new node(l);
      n->next = nums;
      nums = n;
    }
  }
  long pairs_found = 0;
  cout<<"Counting pairs("<<table.size()<<"):"<<endl;

  node *sums = NULL;

  //for (long i=-10000; i<=10000; i++)
  for (long i=-10000; i<=10000; i++)
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
      if((table.find(second) != table.end()) && (first != second))
      {
        // making sure the pair was not prevoisly formed
        if(table[second].find(first) == table[second].end())
        {
          table[first].insert(second);
      //    cout<<"first:"<<first<<" second:"<<second<<" sum:"<<sum<<endl;
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

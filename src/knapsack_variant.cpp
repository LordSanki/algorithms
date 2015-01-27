/* This a variant of the knapsack problem
 * We are given N cans of food every 2 days. Each can has some fixed calories.
 * We have to finish all the cans in 2 days, Also if a can is opened it has to b finished.
 * Challange is to eat equal(or best split) number of calories over the 2 days.
 * Input: First integer is number of cans followed by calories of each can.
 * Output: THe amount of calories we should eat each day.
 * Sample 12 528 129 376 504 543 363 213 138 206 440 504 418 Solution: 2181 2181
 * Sample 8 529 382 130 462 223 167 235 529 Solution: 1344 1313
 * */

#include <iostream>
#include <vector>
#include <list>

using namespace std;

typedef vector<int> vi;
typedef list<int> li;
int main(int argc, char**argv)
{
  int n; cin >> n;
  vi arr(n);
  int total =0;
  for(int i=0; i<n; i++){
    cin >> arr[i];
    total = total + arr[i];
  }
  li table; table.push_back(0);
  for(int i=0; i<n; i++){
    li::reverse_iterator it = table.rbegin();
    while(it != table.rend()){
      table.push_back( *it + arr[i]);
      it++;
    }
  }
  int best = total; int s1 = total;
  for(li::iterator it = table.begin(); it != table.end(); it++)
  {
    int res = total - (2*(*it));
    if( res < best && res >= 0){
      best = res;
      s1 = *it;
    }
  }
  cout << total-s1 <<" "<< s1 << endl;
  return 0;
}


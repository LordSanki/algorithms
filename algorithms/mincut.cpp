#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <list>
#include <algorithm>

using namespace std;
struct Edge
{
  int str;
  int vtx;
};
typedef list<Edge> EdgeList;
struct Vertex
{
  int vtx;
  EdgeList edges;
  int num_edges;
  bool operator == (const Vertex &other)
  {
    if(other.vtx == vtx)
      return true;
    else
      return false;
  }
};
typedef list< Vertex > AdjList;
struct self_edge
{
  self_edge(int v_)
  {
    v= v_;
  }
  int v;
  bool operator() (const Edge &val) { return (val.vtx == v);}
};

bool edge_compare(const Edge & e1, const Edge &e2) { return (e1.vtx < e2.vtx);}
bool edge_remove(const Edge & e1, const Edge &e2) {return (e1.vtx == e2.vtx);}

void mincut(AdjList & vertex, int seed);
void update_edges(Vertex &vtx, vector<int> &map);
//void update_edges(EdgeList &edges, vector<int> &map);
void find_vtx(AdjList &list, int v, AdjList::iterator &it);
void print_list(EdgeList &list);
int find_min_cut(int seed);

int main(int argc, char ** argv)
{
  int min = 999999;
  int k = 0;
  if(argc > 1)
    k = atoi(argv[1]);
  while(1)
  {
    int t = find_min_cut(k);
    k++;
    if(t <min)
    {
      printf("MIN: %d\n",t);
      min = t;
    }
  }
  return 0;
}

int find_min_cut(int seed)
{
  FILE *f = fopen("kargerMinCut.txt","r");
  AdjList vertices;

  while( 0 == feof(f) )
  {
    EdgeList edges;
    int i = 0;
    char buf[1000] = {0};
    fgets(buf, 1000, f);
    char * c = strtok(buf,"\t");
    if(c)
    {
      vertices.push_back(Vertex()); 
      Vertex &v = vertices.back();
      v.vtx = atoi(c);
      c = strtok(NULL,"\t\n\r");
      while(c)
      {
        Edge e;
        e.str = 1; e.vtx = atoi(c);
        v.edges.push_back(e);
        c = strtok(NULL,"\t\n\r");
      }
      v.edges.sort(edge_compare);
      v.num_edges = v.edges.size();
    }
  }
  mincut(vertices, seed);
  fclose(f);
  int min = vertices.front().edges.front().str;
  if(min ==1)
  {
  }
  return min; 
}

void mincut(AdjList & vertices, int seed)
{
  srand(seed);
  vector<int> map;
  int size = vertices.size();
  for(int i=0; i<=size; i++)
    map.push_back(i);

  while(size > 2)
  {
    AdjList::iterator v1 = vertices.begin();
    int random = rand() % size;
    if(random != 0)
      std::advance(v1, random);
    EdgeList & edges = v1->edges;
    EdgeList::iterator e1 = edges.begin();
    random = rand() % edges.size();
    if(random != 0)
      std::advance(e1, random );

    int vV2 = e1->vtx;

    if(vV2 == 0) exit(0);
    while(vV2 != map[vV2])
      vV2 = map[vV2];
    map[vV2] = v1->vtx;
    
    AdjList::iterator v2;
    find_vtx(vertices, vV2,  v2);
    // merging v1 & v2

//    printf("RAND V2 %d->%d\n",random, v2->vtx);print_list(v2->edges);
//    printf("Merging %d & %d\n",v1->vtx,v2->vtx);// print_list(v1->edges); print_list(v2->edges);
    v1->edges.merge(v2->edges, edge_compare);


    vertices.erase(v2);
    update_edges(*v1, map);
    v1->edges.unique(edge_remove);
    v1->edges.remove_if(self_edge(v1->vtx));
    v1->edges.sort(edge_compare);
    //print_list(v1->edges);
    size--;
  }
  AdjList::iterator v1 = vertices.begin();
  update_edges(*v1, map);
  v1->edges.unique(edge_remove);
  v1->edges.remove_if(self_edge(v1->vtx));
  v1++;
  update_edges(*v1, map);
  v1->edges.unique(edge_remove);
  v1->edges.remove_if(self_edge(v1->vtx));
}

void find_vtx(AdjList &list, int v, AdjList::iterator &it)
{
  for(it = list.begin();
      it != list.end(); it++)
  {
    if (it->vtx == v)
      return;
  }
}

void update_edges(Vertex &vtx, vector<int> &map)
{
  EdgeList & edges = vtx.edges;
  for(EdgeList::iterator it = edges.begin(); it != edges.end();
      it++)
  {
    while(it->vtx != map[it->vtx])
      it->vtx = map[it->vtx];
  }
  edges.sort(edge_compare);
  EdgeList::iterator first = edges.begin();
  int str = 0;
  for(EdgeList::iterator it = edges.begin(); it != edges.end();)
  {
    if(it->vtx != first->vtx)
    {
      for(EdgeList::iterator it2 = first; it2 != it; it2++)
      {
        it2->str = str;
      }
      str = it->str;
      first = it;
    }
    else
    {
      str += it->str;
    } 
    it++;
    if(it == edges.end())
    {
      for(EdgeList::iterator it2 = first; it2 != it; it2++)
      {
        it2->str = str;
      }
    }
  }
}

void print_list(EdgeList &edges)
{
  for(EdgeList::iterator it = edges.begin(); it != edges.end();
      it++)
  {
    printf ("%d[%d] ",it->vtx, it->str);
  }
  printf("\n");
}

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <list>
#include <iostream>

using namespace std;
struct Edge
{
  int length;
  int vtx;
};
typedef vector < Edge > EdgeList;

struct Vertex
{
  int vtx;
  EdgeList edges;
};

struct VertexDetails
{
  Vertex *vtx_ptr;
  int dist;
  int heap_idx;
  bool explored;
};

typedef vector < Vertex > VertexList;

void swap(int *heap, VertexDetails *vertices, int idx1, int idx2);
void bubble_up(int idx, int *heap, int heap_size, VertexDetails *vertices);
void bubble_down(unsigned int n, int *heap, int size, VertexDetails *vertices);
void add_heap(int *heap, int &heap_size, VertexDetails *vertices, int vtx);
int ext_min(int *heap, int &heap_size, VertexDetails *vertices);
VertexDetails* compute_all_paths(VertexList &vlist);
void read_graph(VertexList &vertices, const char* filename);

int main(int argc, char ** argv)
{
  if(argc < 2) { printf("./djk <adjlist.txt> <node>\n"); return -1;}
  VertexList vertices;
  read_graph(vertices, argv[1]);
  VertexDetails *result = compute_all_paths(vertices);
  printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",result[7].dist,result[37].dist,result[59].dist,result[82].dist,result[99].dist,result[115].dist,result[133].dist,result[165].dist,result[188].dist,result[197].dist);
  if(argc > 2)
    cout<<result[atoi(argv[2])].dist<<endl;
  return 0;
}

void read_graph(VertexList &vertices, const char* filename)
{
  FILE *f = fopen(filename,"r");

  while( 0 == feof(f) )
  {
    int i = 0;
    char buf[1000] = {0};
    fgets(buf, 1000, f);
    char * c = strtok(buf,"\t");
    if(c)
    {
      vertices.push_back(Vertex()); 
      Vertex &v = vertices.back();
      v.vtx = atoi(c);
      c = strtok(NULL,",\n\r");
      while(c)
      {
        Edge e;
        e.vtx = atoi(c);
        c = strtok(NULL,"\t\n\r");
        e.length = atoi(c);
        v.edges.push_back(e);
        c = strtok(NULL,",\n\r");
      }
    }
  }
  if(f)
    fclose(f);
}

VertexDetails* compute_all_paths(VertexList &vlist)
{
  int num_vtx = vlist.size();
  VertexDetails *vertices = new VertexDetails[num_vtx+1];
  int *heap = new int[num_vtx+1];

  for(int i=1; i<=num_vtx; i++)
  {
    vertices[i].vtx_ptr = &vlist[i-1];
    vertices[i].explored = false;
    vertices[i].dist = 1000000;
    vertices[i].heap_idx = -1;
  }
  int heap_size = 1;
  vertices[1].dist = 0;
  vertices[1].heap_idx = 1;
  heap[1] = 1;
 
  while(heap_size > 0)
  {
    VertexDetails &v = vertices[ ext_min(heap,heap_size,vertices) ];
    v.explored = true;
    EdgeList &edges = v.vtx_ptr->edges;
    for(unsigned int i=0; i<edges.size(); i++)
    {
      VertexDetails &w = vertices[edges[i].vtx];
      if( w.explored == false )
      {
        int new_dist = v.dist + edges[i].length;
        if(w.heap_idx == -1)
        {
          w.dist = new_dist;
          add_heap(heap, heap_size, vertices, edges[i].vtx);
        }
        else if(w.dist > new_dist)
        {
          w.dist = new_dist;
          bubble_up(w.heap_idx, heap, heap_size, vertices);
        }
      }
    }
  }
  return vertices;
}

int ext_min(int *heap, int &heap_size, VertexDetails *vertices)
{
  // extracting first element
  int ret = heap[1];
  heap[1] = heap[heap_size];
  vertices[heap[1]].heap_idx = 1;
  heap_size--;

  //bubble_down heap
  bubble_down(1, heap, heap_size, vertices);

  return ret;
}

void add_heap(int *heap, int &heap_size, VertexDetails *vertices, int vtx)
{
  heap_size++;
  heap[heap_size] = vtx;
  vertices[vtx].heap_idx = heap_size;
  bubble_up(heap_size, heap, heap_size, vertices);
}

void bubble_up(int idx, int *heap, int heap_size, VertexDetails *vertices)
{
  int n = idx;
  while(n > 1)
  {
    if( vertices[heap[n]].dist < vertices[heap[n/2]].dist )
    {
      swap(heap, vertices, n, n/2);
    }
    else
      break;
    n=n/2;
  }
}

void bubble_down(unsigned int n, int *heap, int size, VertexDetails *vertices)
{
  unsigned int l = n*2;

  while(size > l)
  {
    int child;
    // finding the candidate child to compare and swap
    if( vertices[heap[l]].dist < vertices[heap[l+1]].dist )
      child = l;
    else
      child = l+1;

    // checking if candidate child is to be swapped
    if( vertices[heap[child]].dist < vertices[heap[n]].dist )
    {
      swap(heap, vertices, child, n);
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
    if( vertices[heap[l]].dist < vertices[heap[n]].dist )
    {
      swap(heap, vertices, l, n);
    }
  }
}

void swap(int *heap, VertexDetails *vertices, int idx1, int idx2)
{
  int t = vertices[heap[idx1]].heap_idx;
  vertices[heap[idx1]].heap_idx = vertices[heap[idx2]].heap_idx;
  vertices[heap[idx2]].heap_idx = t;

  t = heap[idx1];
  heap[idx1] = heap[idx2];
  heap[idx2] = t;
}















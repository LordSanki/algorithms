#include <cstdio>
#define SIZE 10
struct Node
{
  int data;
  Node * next;
  Node(){next = NULL;};
  Node(int i){next = NULL; data = i;};
};

void print_node(Node *n);
void print_list(Node *n);
void recursive_reverse(Node *& head);
void recursive_reverse(Node ** head);
void iterative_reverse(Node *& head);

void recursive_reverse(Node *& head)
{
  if(head)
  {
    Node *n1 = head;
    Node *n2 = n1->next;
    n1->next = NULL;
    if(n2)
    {
      head = n2;
      recursive_reverse(head);
      n2->next = n1;
    }
  }
}

void recursive_reverse(Node ** head)
{
  if(head)
  {
    Node *n1 = *head;
    Node *n2 = n1->next;
    n1->next = NULL;
    if(n2)
    {
      *head = n2;
      recursive_reverse(head);
      n2->next = n1;
    }
  }
}
void iterative_reverse(Node *& head)
{
  Node *n1 = head;
  if(n1 == NULL) return;
  Node *n2 = n1->next;
  while(n2)
  {
    n1->next = n2->next;
    n2->next = head;
    head = n2;
    n2 = n1->next;
  }
}

void print_list(Node *head)
{
  Node * next = head;
  while(next)
  {
    printf("%d\n", next->data);
    next = next->next;
  }
}

void print_node(Node *n)
{
  printf("Node: %d\n",n->data);
}

int main()
{
  Node *head = new Node(0);
  Node *next = head;
  for(int i=1; i<SIZE; i++)
  {
     next->next = new Node(i);
     next = next->next;
  }
  print_list(head);
  recursive_reverse(head);
  iterative_reverse(head);
  print_list(head);
  return 0;
}

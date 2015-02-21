
void insert_bot(stack **s, data *d)
{
  if(empty(s) == true)
  {
    push(s,d);
  }
  else
  {
    data *k = pop(s);
    insert_bot(s,d);
    push(s,k);
  }
}

void reverse(stack ** s)
{
  if(s)
  {
    data * d = NULL;
    if(empty(s) == false)
    {
      d = pop(s);
      reverse(s);
    }
    if(d)
      insert_bot(d, s);
  }
}

int main()
{
}


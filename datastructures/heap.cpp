
template <class T>
struct min{
  bool operator() (const T &a, const T &b){return a<b;}
};
template <class T>
struct max{
  bool operator() (const T &a, const T &b){return a>b;}
};

template <class T, class Q>
class Heap{
  public:
    typedef T value_type;
    typedef int key_type;
    typedef Q comp_type;

    Heap(int _max_size){
      size = 0;
      max_size = _max_size;
      arr = new value_type[max_size+1];
    }
    int size(){ return size; }
    void push(T &t){
      if(size < max_size){
        size++;
        arr[size] = t;
        bubble_up(size);
      }
    }
    value_type top(){
      return arr[1];
    }
    void pop(){
      if(size >= 1){
        arr[1] = arr[size];
        size--;
      }
      if(size > 1){
        bubble_down(1);
      }
    }
    ~Heap(){ delete [] arr; }
  private:
    int size;
    int max_size;
    value_type * arr;
    comp_type op;
    void bubble_up(int n){
      key_type k=n;
      key_type k2 = k/2;
      while( k>1 && op(arr[k],arr[k2]) ){
        value_type t = arr[k2];
        arr[k2] = arr[k];
        arr[k] = t;
        k = k2; k2 = k/2;
      }
    }
    void bubble_down(int n){
      int k=n;
      while(k <= size){
        key_type k1 = 2*k;
        key_type k2 = 2*k+1;
        if( op(arr[k1],arr[k2]) ){
          if( op(arr[k1], arr[k]) ){
            value_type t = arr[k1];
            arr[k1] = arr[k];
            arr[k] = t;
            k = k1;
          }
          else break;
        }
        else{
          if( op(arr[k2], arr[k]) ){
            value_type t = arr[k2];
            arr[k2] = arr[k];
            arr[k] = t;
            k = k2;
          }
          else break;
        }
      }
    }
};













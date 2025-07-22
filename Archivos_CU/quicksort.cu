#include <iostream>
#include <stack>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

using namespace std;

struct Rango{
  int inicio,fin;
};

//Kernel para marcar los elementos segun el pivote
struct menorQue{
  int pivote;
  __host__ __device__ menorQue(int p):pivote(p){}
  __host__ __device__ bool operator()(int x) const{
    return (x<pivote);
  }
};
struct igualA{
  int pivote;
  __host__ __device__ igualA(int p):pivote(p){}
  __host__ __device__ bool operator()(int x) const{
    return (x==pivote);
  }
};
struct mayorQue{
  __host__ __device__
  int operator()(int esMenor, int esIgual) const{
    return !(esMenor||esIgual);
  }
};

void printArray(vector<int> vec){
  for(int i=0;i<vec.size();i++) cout<<vec[i]<<" ";
  cout<<"\n";
}

void quicksort(thrust::device_vector<int> &d_arr){
  int arr_size=d_arr.size();
  thrust::device_vector<int> d_buffer(arr_size);
  stack<Rango> rg_stack;
  rg_stack.push({0,arr_size-1});
  while(!rg_stack.empty()){
    Rango rg=rg_stack.top();
    rg_stack.pop();
    int rg_size=rg.fin-rg.inicio+1;
    if(rg_size<=1) continue;
    int pivote=d_arr[rg.fin];
    thrust::device_vector<int> lessFlags(rg_size),equalFlags(rg_size),greaterFlags(rg_size);
    thrust::device_vector<int> lessPos(rg_size),equalPos(rg_size),greaterPos(rg_size);
    auto begin=d_arr.begin()+rg.inicio;
    auto finish=d_arr.begin()+rg.fin+1;
    thrust::transform(begin,finish,lessFlags.begin(),menorQue(pivote));
    thrust::transform(begin,finish,equalFlags.begin(),igualA(pivote));
    thrust::transform(
      lessFlags.begin(),
      lessFlags.end(),
      equalFlags.begin(),
      greaterFlags.begin(),
      mayorQue()
    );
    thrust::exclusive_scan(lessFlags.begin(),lessFlags.end(),lessPos.begin());
    thrust::exclusive_scan(equalFlags.begin(),equalFlags.end(),equalPos.begin());
    thrust::exclusive_scan(greaterFlags.begin(),greaterFlags.end(),greaterPos.begin());
    int numMenor=thrust::reduce(lessFlags.begin(),lessFlags.end());
    int numIgual=thrust::reduce(equalFlags.begin(),equalFlags.end());
    for(int i=0;i<rg_size;i++){
      int val=d_arr[rg.inicio+i];
      if(lessFlags[i]){
        int indice=rg.inicio+lessPos[i];
        d_buffer[indice]=val;
      }else if(equalFlags[i]){
        int indice=rg.inicio+numMenor+equalPos[i];
        d_buffer[indice]=val;
      }else{
        int indice=rg.inicio+numMenor+numIgual+greaterPos[i];
        d_buffer[indice]=val;
      }
    }
    thrust::copy(
      d_buffer.begin()+rg.inicio,
      d_buffer.begin()+rg.fin+1,
      d_arr.begin()+rg.inicio
    );
    if(numMenor>1) rg_stack.push({rg.inicio,rg.inicio+numMenor-1});
    if(rg_size-numMenor-numIgual>1) rg_stack.push({rg.inicio+numMenor+numIgual,rg.fin});
  }
}

int main(int argc, char** argv){
  srand(time(NULL));
  int size=atoi(argv[1]);
  vector<int> arr;
  for(int i=0;i<size;i++) arr.push_back(1+rand()%200);
  auto start=chrono::high_resolution_clock::now();
  thrust::device_vector<int> d_arr(arr.begin(),arr.end());
  quicksort(d_arr);
  thrust::copy(d_arr.begin(),d_arr.end(),arr.begin());
  auto finish=chrono::high_resolution_clock::now();
  auto duration=chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
  cout<<"Tiempo demorado en ordenar arreglo: "<<duration/1000000000.0<<" s\n";
  return 0;
}
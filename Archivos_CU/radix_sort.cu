#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

using namespace std;
typedef unsigned int usi;
typedef unsigned char uch;

const int BITS=8;
const int RADIO=1<<BITS;

void printArray(usi* arr, int size){
  for(int i=0;i<size;i++) cout<<arr[i]<<" ";
  cout<<"\n";
}

__global__ void computarDigitos(usi* input, uch* digitos, int shift, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) digitos[i]=(input[i]>>shift)&0xFF;
}

__global__ void histogramaKernel(uch* digitos, int* histograma, int n){
  __shared__ int histLocal[RADIO];
  int tid=threadIdx.x;
  if(tid<RADIO) histLocal[tid]=0;
  __syncthreads();
  int i=blockIdx.x*blockDim.x+tid;
  if(i<n) atomicAdd(&histLocal[digitos[i]],1);
  __syncthreads();
  if(tid<RADIO) atomicAdd(&histograma[tid],histLocal[tid]);
}

__global__ void reordenarKernel(usi* input, uch* digitos, int* digit_offsets,
 int* counters, usi* output, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n){
    uch digito=digitos[i];
    int pos=atomicAdd(&counters[digito],1);
    output[digit_offsets[digito]+pos]=input[i];
  }
}

void radixSort(usi* d_input, usi* d_output, int n){
  uch* d_digitos;
  int* d_histograma;
  int* d_pos;
  int* d_counters;
  cudaMalloc(&d_digitos, n*sizeof(uch));
  cudaMalloc(&d_histograma, RADIO*sizeof(int));
  cudaMalloc(&d_pos, RADIO*sizeof(int));
  cudaMalloc(&d_counters, RADIO*sizeof(int));
  dim3 bloque(256);
  dim3 grid((n+bloque.x-1)/bloque.x);
  for(int shift=0;shift<32;shift+=BITS){
    computarDigitos<<<grid,bloque>>>(d_input,d_digitos,shift,n);
    cudaMemset(d_histograma,0,RADIO*sizeof(int));
    histogramaKernel<<<grid,bloque>>>(d_digitos,d_histograma,n);
    thrust::device_ptr<int> hist_ptr(d_histograma);
    thrust::device_ptr<int> pos_ptr(d_pos);
    thrust::exclusive_scan(hist_ptr,hist_ptr+RADIO,pos_ptr);
    cudaMemset(d_counters,0,RADIO*sizeof(int));
    reordenarKernel<<<grid,bloque>>>(d_input,d_digitos,d_pos,d_counters,d_output,n);
    swap(d_input,d_output);
  }
  cudaFree(d_digitos);
  cudaFree(d_histograma);
  cudaFree(d_pos);
  cudaFree(d_counters);
}

int main(int argc, char** argv){
  srand(time(NULL));
  int size=atoi(argv[1]);
  usi* arr=new usi[size];
  for(int i=0;i<size;i++) arr[i]=1+rand()%200;
  auto start=chrono::high_resolution_clock::now();
  usi *d_in,*d_out;
  cudaMalloc(&d_in,size*sizeof(usi));
  cudaMalloc(&d_out,size*sizeof(usi));
  cudaMemcpy(d_in,arr,size*sizeof(usi),cudaMemcpyHostToDevice);
  radixSort(d_in,d_out,size);
  cudaMemcpy(arr,d_in,size*sizeof(usi),cudaMemcpyDeviceToHost);
  auto finish=chrono::high_resolution_clock::now();
  auto duration=chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
  cout<<"Tiempo demorado en ordenar arreglo: "<<duration/1000000000.0<<" s\n";
  cudaFree(d_in);
  cudaFree(d_out);
  delete[] arr;
  return 0;
}
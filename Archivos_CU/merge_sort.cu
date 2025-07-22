#include <iostream>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

void printArray(int* arr, int size){
  for(int i=0;i<size;i++) cout<<arr[i]<<" ";
  cout<<"\n";
}

__global__ void merge(int* in, int* out, int width, int size){
  int indice=blockIdx.x*blockDim.x+threadIdx.x;
  int inicio=2*indice*width;
  if(inicio>=size) return;
  int mid=min(inicio+width,size),fin=min(inicio+2*width,size);
  int i=inicio,j=mid,k=inicio;
  while(i<mid && j<fin){
    if(in[i]<=in[j]){
      out[k]=in[i];
      i++;
    }else{
      out[k]=in[j];
      j++;
    }
    k++;
  }
  while(i<mid){
    out[k]=in[i];
    i++;
    k++;
  }
  while(j<fin){
    out[k]=in[j];
    j++;
    k++;
  }
}

void mergeSort(int* arr, int size){
  int *d_in,*d_out;
  cudaMalloc(&d_in,size*sizeof(int));
  cudaMalloc(&d_out,size*sizeof(int));
  cudaMemcpy(d_in,arr,size*sizeof(int),cudaMemcpyHostToDevice);
  int hebras=256;
  for(int width=1;width<size;width*=2){
    int bloques=(size+2*width*hebras-1)/(2*width*hebras);
    merge<<<bloques,hebras>>>(d_in,d_out,width,size);
    cudaDeviceSynchronize();
    swap(d_in,d_out);
  }
  cudaMemcpy(arr,d_in,size*sizeof(int),cudaMemcpyDeviceToHost);
  cudaFree(d_in);
  cudaFree(d_out);
}

int main(int argc, char** argv){
  srand(time(NULL));
  int size=atoi(argv[1]);
  int* arr=new int[size];
  for(int i=0;i<size;i++) arr[i]=1+rand()%200;
  auto start=chrono::high_resolution_clock::now();
  mergeSort(arr,size);
  auto finish=chrono::high_resolution_clock::now();
  auto duration=chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
  cout<<"Tiempo demorado en ordenar arreglo: "<<duration/1000000000.0<<" s\n";
  delete[] arr;
  return 0;
}
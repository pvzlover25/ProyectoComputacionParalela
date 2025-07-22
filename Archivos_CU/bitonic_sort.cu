#include <iostream>
#include <chrono>
#include <climits>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

void printArray(int* arr, int size){
  for(int i=0;i<size;i++) cout<<arr[i]<<" ";
  cout<<"\n";
}

// Kernel de Bitonic Sort
__global__ void bitonic_sort_kernel(int* data, int j, int k, int n) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;

    unsigned int ixj = i ^ j;
    if (ixj > i && ixj < n) {
        bool ascendente = ((i & k) == 0);
        if ((data[i] > data[ixj]) == ascendente) {
            int temp = data[i];
            data[i] = data[ixj];
            data[ixj] = temp;
        }
    }
}

// Funcion para redondear al siguiente numero potencia de 2
int sigtePotenciaDos(int n) {
    int ret = 1;
    while (ret < n) ret <<= 1;
    return ret;
}

void bitonic_sort(int* h_data, int n) {
    int padded_n = sigtePotenciaDos(n);
    int* h_padded = new int[padded_n];
    for (int i = 0; i < n; i++) h_padded[i] = h_data[i];
    for (int i = n; i < padded_n; i++) h_padded[i] = INT_MAX;

    int* d_data;
    cudaMalloc(&d_data, padded_n * sizeof(int));
    cudaMemcpy(d_data, h_padded, padded_n * sizeof(int), cudaMemcpyHostToDevice);

    int hebras = 512;
    int numBloques = (padded_n + hebras - 1) / hebras;

    for (int k = 2; k <= padded_n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_sort_kernel<<<numBloques,hebras>>>(d_data, j, k, padded_n);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(h_padded, d_data, padded_n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Copiar solo los n elementos ordenados reales
    for (int i = 0; i < n; i++) {
        h_data[i] = h_padded[i];
    }
    delete[] h_padded;
}

int main(int argc, char** argv){
  srand(time(NULL));
  int size=atoi(argv[1]);
  int* arr=new int[size];
  for(int i=0;i<size;i++) arr[i]=1+rand()%200;
  auto start=chrono::high_resolution_clock::now();
  bitonic_sort(arr,size);
  auto finish=chrono::high_resolution_clock::now();
  auto duration=chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
  cout<<"Tiempo demorado en ordenar arreglo: "<<duration/1000000000.0<<" s\n";
  return 0;
}
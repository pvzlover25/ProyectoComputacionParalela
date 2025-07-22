#include <iostream>
#include <chrono>
#include <unordered_set>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/iterator/constant_iterator.h>

using namespace std;
const int THRESHOLD=16;

void printArray(vector<int> &arr){
  for(int i=0;i<arr.size();i++) cout<<arr[i]<<" ";
  cout<<"\n";
}

int getMin(int a, int b){
  if(a<b) return a;
  return b;
}

vector<int> randSample(vector<int> &arr, int s){
  vector<int> ret;
  unordered_set<int> miSet;
  int arr_size=arr.size();
  while(miSet.size()<s){
    int r=rand()%arr_size;
    miSet.insert(r);
  }
  for(int indice:miSet) ret.push_back(arr[indice]);
  return ret;
}

vector<int> getPivotes(vector<int> &sample, int numPivotes){
  int sample_size=sample.size();
  vector<int> ret;
  if(numPivotes==0||sample_size==0) return ret;
  if(sample_size<=numPivotes) return ret;
  for(int i=1;i<=numPivotes;i++){
    int indice=i*sample_size/(numPivotes+1);
    if(indice>=sample_size) indice=sample_size-1;
    ret.push_back(sample[indice]);
  }
  return ret;
}

__global__ void assignToBuckets(int* arr, int* pivotes, int* bucket_ids, int size, int numPivotes){
  int indice=blockIdx.x*blockDim.x+threadIdx.x;
  if(indice>=size) return;
  int val=arr[indice];
  int inf=0,sup=numPivotes;
  while(inf<sup){
    int mid=(inf+sup)/2;
    if(val<=pivotes[mid]) sup=mid;
    else inf=mid+1;
  }
  bucket_ids[indice]=inf;
}

void sampleSort(vector<int> &arr, int p){
  int arr_size=arr.size();
  if(arr_size<=THRESHOLD){
    sort(arr.begin(),arr.end());
    return;
  }
  int s=getMin(arr_size,4*p*log2(arr_size));
  vector<int> sample=randSample(arr,s);
  if(sample.size()<=p-1){
    sort(arr.begin(),arr.end());
    return;
  }
  sort(sample.begin(),sample.end());
  vector<int> pivotes=getPivotes(sample,p-1);
  if(pivotes.empty()){
    sort(arr.begin(),arr.end());
    return;
  }
  thrust::device_vector<int> d_arr(arr.begin(),arr.end());
  thrust::device_vector<int> d_pivotes(pivotes.begin(),pivotes.end());
  thrust::device_vector<int> d_bucket_ids(arr_size);
  int hebras=256;
  int bloques=(arr_size+hebras-1)/hebras;
  assignToBuckets<<<bloques,hebras>>>(
    thrust::raw_pointer_cast(d_arr.data()),
    thrust::raw_pointer_cast(d_pivotes.data()),
    thrust::raw_pointer_cast(d_bucket_ids.data()),
    arr_size,
    p-1
  );
  cudaDeviceSynchronize();
  thrust::stable_sort_by_key(d_bucket_ids.begin(),d_bucket_ids.end(),d_arr.begin());
  thrust::device_vector<int> keys_out(p);
  thrust::device_vector<int> counts(p);
  auto rbk_end=thrust::reduce_by_key(
    d_bucket_ids.begin(),
    d_bucket_ids.end(),
    thrust::constant_iterator<int>(1),
    keys_out.begin(),
    counts.begin()
  );
  int num_buckets=rbk_end.first-keys_out.begin();
  thrust::device_vector<int> d_offsets(p+1);
  d_offsets[0]=0;
  thrust::inclusive_scan(
    counts.begin(),
    counts.begin()+num_buckets,
    d_offsets.begin()+1
  );
  for(int i=0;i<num_buckets;i++){
    int inicio=d_offsets[i];
    int fin=d_offsets[i+1];
    if(fin>inicio) thrust::sort(d_arr.begin()+inicio,d_arr.begin()+fin);
  }
  thrust::copy(d_arr.begin(),d_arr.end(),arr.begin());
}

int main(int argc, char** argv){
  srand(time(NULL));
  int size=atoi(argv[1]),p=atoi(argv[2]);
  vector<int> arr;
  for(int i=0;i<size;i++) arr.push_back(1+rand()%200);
  auto start=chrono::high_resolution_clock::now();
  sampleSort(arr,p);
  auto finish=chrono::high_resolution_clock::now();
  auto duration=chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
  cout<<"Tiempo demorado en ordenar arreglo: "<<duration/1000000000.0<<" s\n";
  return 0;
}
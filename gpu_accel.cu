#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gpu_accel.cuh"
#include <NTL/ZZVec.h>
#include <thread>
#include <assert.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;

#define USE_BARRET 1
#define USE_LOCAL_SYNC 0
#define USE_GLOBAL_MUTEX 0 //global mutex does not work when the sync beyond a grid, must use multi-grid sync
#define debug_blue 0

unsigned long long computeMu(unsigned int bit_length, unsigned long long q)
{

  #if 0//efficient NTT
  unsigned __int128 tmp = 1;
  tmp = tmp << (2 * bit_length);
  __float128 tmp2= tmp;
  unsigned long long res = tmp2/q;
  return res;
  #endif

  #if 1//SEED 2022: Accelerating Polynomial Multiplication for Homomorphic Encryption on GPUs
  unsigned __int128 tmp = 1;
  tmp = tmp << (2 * bit_length + 1);
  __float128 tmp2= tmp;
  unsigned long long mu = tmp2/q;
  return mu;
  #endif

}

// __device__ __forceinline__ void singleBarrett(unsigned long long& a, unsigned& q, unsigned& mu, int& qbit)  // ??
__device__ __forceinline__ void singleBarrett(unsigned __int128& a, unsigned long long& q, unsigned long long& mu, int& qbit)  // ??
{  
    #if 0//efficient NTT paper
    unsigned __int128 rx;
    rx = a >> (qbit - 2);
    rx *= mu;
    rx >>= qbit + 2;
    rx *= q;
    a -= rx;

    a -= q * (a >= q);
    #endif

    #if 1//SEED 2022: Accelerating Polynomial Multiplication for Homomorphic Encryption on GPUs
    unsigned __int128 rx;
    rx = a >> (qbit - 2);
    rx *= mu;
    rx >>= qbit + 3;
    rx *= q;
    a -= rx;

    a -= q * (a >= q);
    #endif
}

__global__ void KernelMulMod(unsigned long long a[], const unsigned long long b[], unsigned long long q){
  int global_tid = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;

  // __extension__ unsigned __int128 temp_storage = a[global_tid]; //just for testing don't use this
  __extension__ unsigned __int128 temp_storage = b[global_tid];
  temp_storage *= a[global_tid];
  a[global_tid] = temp_storage % q;
}

__global__ void KernelMulMod(unsigned long long a[], const unsigned long long b[], unsigned long long q, int n){
  int global_tid = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;

  if(global_tid<n){
    __extension__ unsigned __int128 temp_storage = b[global_tid];
    temp_storage *= a[global_tid];
    a[global_tid] = temp_storage % q;
  }
}


#if 1 //CT & GS for 2048 ntt points
__global__ void CTBasedNTTInnerSingle(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[], int n_of_groups)
{
    int local_tid = threadIdx.x;

    extern __shared__ unsigned long long shared_array[];

    //Ardhi: To my understanding, this is just load coeffs into shared memory, one thread pickup two coeffs to shmem
    //for ex: shared_array[tid]=a[tid]
    // #pragma unroll
    // for (int iteration_num = 0; iteration_num < 2; iteration_num++)
    // {  // copying to shared memory from global
    //     int global_tid = local_tid + iteration_num * 1024;
    //     shared_array[global_tid] = a[global_tid + blockIdx.x * 2048]; //Ardhi: if blockIdx.x is always zero, why this is needed??
    // }

    int blockT2048 = blockIdx.x * 2048;
    shared_array[local_tid] = a[local_tid + blockT2048]; //Ardhi: if blockIdx.x is always zero, why this is needed??
    int global_tid = local_tid + 1024;
    shared_array[global_tid] = a[global_tid + blockT2048]; //Ardhi: if blockIdx.x is always zero, why this is needed??


    int blockT1024 = blockIdx.x * THREADS_PER_BLOCK;
    #pragma unroll
    // for (int length = 1; length < 2048; length *= 2)
    for (int length = 0; length < 11; length++)
    {  // iterations for ntt
        // int step = (1024 / length);
        int step = (1024 >> length);         // int step = (2048 / length) / 2;
        int log_step = 10 - length;

        // int psi_step = local_tid / step;
        int psi_step = local_tid >> log_step;   
        
        int target_index = psi_step * step * 2 + (local_tid & (step - 1));

        // psi_step = (local_tid + blockIdx.x * 1024) / step;
        psi_step = (local_tid + blockT1024) >> log_step;
        
        // unsigned long long psi = psi_powers[length + psi_step];

        // unsigned long long psi = psi_powers[n_of_groups + (threadIdx.x + blockIdx.x * THREADS_PER_BLOCK)/step];
        unsigned long long psi = psi_powers[n_of_groups + ((threadIdx.x + blockT1024) >> log_step)];

        unsigned long long first_target_value = shared_array[target_index];
        // unsigned long long temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow
        __extension__ unsigned __int128 temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

        temp_storage *= psi;

        #if USE_BARRET
        singleBarrett(temp_storage, q, mu, qbit);
        #else
        temp_storage %= q;
        #endif

        unsigned long long second_target_value = temp_storage;

        unsigned long long target_result = first_target_value + second_target_value;

        target_result -= q * (target_result >= q);

        shared_array[target_index] = target_result;

        first_target_value += q * (first_target_value < second_target_value);

        shared_array[target_index + step] = first_target_value - second_target_value;

        n_of_groups *= 2;

        __syncthreads();
    }

    // #pragma unroll
    // for (int iteration_num = 0; iteration_num < 2; iteration_num++)
    // {  // copying back to global from shared
    //     int global_tid = local_tid + iteration_num * 1024;
    //     a[global_tid + blockIdx.x * 2048] = shared_array[global_tid];
    // }

      a[local_tid + blockT2048] = shared_array[local_tid];
      a[global_tid + blockT2048] = shared_array[global_tid];
}

__global__ void GSBasedINTTInnerSingle(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[], int n, int n_of_groups)
{
    int local_tid = threadIdx.x;

    extern __shared__ unsigned long long shared_array[];

    #pragma unroll
    for (int iteration_num = 0; iteration_num < 2; iteration_num++)
    {  // copying to shared memory from global
        int global_tid = local_tid + iteration_num * 1024;
        shared_array[global_tid] = a[global_tid + blockIdx.x * 2048];
    }

    __syncthreads();


    int group_size = 2;// =n/n_of_groups
    n_of_groups = n/2;
    #pragma unroll
    for (int length = 1024; length >= 1; length /= 2)
    {  // iterations for intt
        int step = (2048 / length) / 2;

        int psi_step = local_tid / step;
        int target_index = psi_step * step * 2 + local_tid % step;

        psi_step = (local_tid + blockIdx.x * 1024) / step;

        // unsigned long long psiinv = psiinv_powers[length + psi_step];
        int n_of_thread_per_group = (group_size/2);
        unsigned long long psiinv = psiinv_powers[n_of_groups + (threadIdx.x + blockIdx.x * THREADS_PER_BLOCK)/n_of_thread_per_group];
        //unsigned long long psiinv = psiinv_powers[n_of_groups + global_tid/n_of_thread_per_group];
        // printf("n_of_groups: %d, global_tid %d, target_index: %d, psi_index: %d\n", length, (threadIdx.x + blockIdx.x * THREADS_PER_BLOCK), target_index, length + (threadIdx.x + blockIdx.x * THREADS_PER_BLOCK)/n_of_thread_per_group);

        unsigned long long first_target_value = shared_array[target_index];
        unsigned long long second_target_value = shared_array[target_index + step];

        unsigned long long target_result = first_target_value + second_target_value;

        target_result -= q * (target_result >= q);

        // shared_array[target_index] = (target_result >> 1) + q2 * (target_result & 1);
        shared_array[target_index] =target_result;

        first_target_value += q * (first_target_value < second_target_value);

        // unsigned long long temp_storage = first_target_value - second_target_value;
        __extension__ unsigned __int128 temp_storage = first_target_value - second_target_value;

        temp_storage *= psiinv;

        #if USE_BARRET
        singleBarrett(temp_storage, q, mu, qbit);
        #else
        temp_storage %= q;
        #endif

        unsigned long long temp_storage_low = temp_storage;

        // shared_array[target_index + step] = (temp_storage_low >> 1) + q2 * (temp_storage_low & 1);
        shared_array[target_index + step] = temp_storage_low;
        
        n_of_groups /= 2;
        group_size *= 2;
        
        __syncthreads();
    }

    #pragma unroll
    for (int iteration_num = 0; iteration_num < 2; iteration_num++)
    {  // copying back to global from shared
        int global_tid = local_tid + iteration_num * 1024;
        a[global_tid + blockIdx.x * 2048] = shared_array[global_tid];
    }
    
}
#endif

//Multi-kernel launch
__global__ void CTBasedNTTInnerSingle(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[], int n, int n_of_groups)
{
    int global_tid = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;

    int group_size = n/n_of_groups;
    int n_of_thread_per_group = group_size/2; //we assume that 1 thread manages 2 coefficients
    int step = n_of_thread_per_group;

    // int target_index = (global_tid/n_of_thread_per_group)*group_size+(global_tid % n_of_thread_per_group);
    int target_index = (global_tid/n_of_thread_per_group)*group_size+(global_tid & (n_of_thread_per_group - 1));

    unsigned long long psi = psi_powers[n_of_groups + global_tid/n_of_thread_per_group];

    unsigned long long first_target_value = a[target_index];

    unsigned __int128 temp_storage = a[target_index + step];  // this is for eliminating the possibility of overflow

    temp_storage *= psi;

    #if USE_BARRET
    singleBarrett(temp_storage, q, mu, qbit);
    #else
    temp_storage %= q;
    #endif

    unsigned long long second_target_value = temp_storage;

    unsigned long long target_result = first_target_value + second_target_value;

    target_result -= q * (target_result >= q);

    a[target_index] = target_result;

    first_target_value += q * (first_target_value < second_target_value);

    a[target_index + step] = first_target_value - second_target_value;

}

__global__ void CTBasedNTTInnerSingleGlobalMutex(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[], int n)
{
    int global_tid = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
    grid_group g = this_grid();

    for (int n_of_groups = 1; (n/n_of_groups>2048); n_of_groups *= 2) //Ardhi: do the ntt until the working size is 2048 elements
    { 
    int group_size = n/n_of_groups;
    int n_of_thread_per_group = group_size/2; //we assume that 1 thread manages 2 coefficients
    int step = n_of_thread_per_group;

    // int target_index = (global_tid/n_of_thread_per_group)*group_size+(global_tid % n_of_thread_per_group);
    int target_index = (global_tid/n_of_thread_per_group)*group_size+(global_tid & (n_of_thread_per_group - 1));

    unsigned long long psi = psi_powers[n_of_groups + global_tid/n_of_thread_per_group];

    unsigned long long first_target_value = a[target_index];

    unsigned __int128 temp_storage = a[target_index + step];  // this is for eliminating the possibility of overflow

    temp_storage *= psi;

    #if USE_BARRET
    singleBarrett(temp_storage, q, mu, qbit);
    #else
    temp_storage %= q;
    #endif

    unsigned long long second_target_value = temp_storage;

    unsigned long long target_result = first_target_value + second_target_value;

    target_result -= q * (target_result >= q);

    a[target_index] = target_result;

    first_target_value += q * (first_target_value < second_target_value);

    a[target_index + step] = first_target_value - second_target_value;
    

    g.sync();
    }

}

__device__  void sync_with_trigger_bit(int first, int second, int iteration, int trigger_vectors[])
{     
  long int counter;
  while (!(trigger_vectors[first] == iteration && trigger_vectors[second] == iteration))
  {
    //wait until the required data is ready
    counter++;
  }
  
  if(threadIdx.x + blockIdx.x * THREADS_PER_BLOCK == 1)
    printf("iteration %d first %d second %d counter %ld\n", iteration, first, second, counter);
}


#if USE_LOCAL_SYNC
//Single Kernel Call
__global__ void CTBasedNTTInnerSingleKernel(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[], int n, int trigger_vectors[])
{
  int global_tid = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;

  // if(global_tid == 0)
  //   printf("counter: %d\n", trigger_vectors[0]);

  int loop_counter = 1;
  for (int n_of_groups = 1; (n/n_of_groups>2048); n_of_groups *= 2) //Ardhi: do the ntt until the working size is 2048 elements
  {

    int group_size = n/n_of_groups;
    int n_of_thread_per_group = group_size/2; //we assume that 1 thread manages 2 coefficients
    int step = n_of_thread_per_group;

    // int target_index = (global_tid/n_of_thread_per_group)*group_size+(global_tid % n_of_thread_per_group);
    int target_index = (global_tid/n_of_thread_per_group)*group_size+(global_tid & (n_of_thread_per_group - 1));

    unsigned long long psi = psi_powers[n_of_groups + global_tid/n_of_thread_per_group];

    //local sync here
    while (true)
    {
      //wait until the required data is ready
      
      //release the lock for the first iteration
      if(trigger_vectors[target_index] == 0 && trigger_vectors[target_index + step] == 0)
      {
        // if(global_tid == 0)
          // printf("first iter\n");
        atomicExch(&trigger_vectors[target_index], 1);
        atomicExch(&trigger_vectors[target_index + step], 1);
        break;
      }

      if(trigger_vectors[target_index] == n_of_groups && trigger_vectors[target_index + step] == n_of_groups)
      {
        //release the lock when the previous threads finish
        break;
      }

      // if(global_tid == 0)
        // printf("trigger_vector %d n_of_groups %d \n", trigger_vectors[target_index], n_of_groups);
    }
    
    unsigned long long first_target_value = a[target_index];

    unsigned __int128 temp_storage = a[target_index + step];  // this is for eliminating the possibility of overflow

    temp_storage *= psi;

    #if USE_BARRET
    singleBarrett(temp_storage, q, mu, qbit);
    #else
    temp_storage %= q;
    #endif

    unsigned long long second_target_value = temp_storage;

    unsigned long long target_result = first_target_value + second_target_value;

    target_result -= q * (target_result >= q);

    a[target_index] = target_result;

    //update trigger bit for the next iteration
    // trigger_vectors[target_index] *= 2;
    atomicExch(&trigger_vectors[target_index], trigger_vectors[target_index] * 2);
    // printf("trigger_target %d\n", trigger_vectors[target_index]);
    first_target_value += q * (first_target_value < second_target_value);

    a[target_index + step] = first_target_value - second_target_value;

    //update trigger bit for the next iteration
    // trigger_vectors[target_index + step] *= 2;
    atomicExch(&trigger_vectors[target_index + step], trigger_vectors[target_index + step] * 2);
    // printf("trigger_target+step %d\n", trigger_vectors[target_index + step]);

  }

}
#endif

__global__ void GSBasedINTTInnerSingle(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[], int n, unsigned long long n_inv, int n_of_groups)
{
    int global_tid = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;

    // unsigned long q2 = (q + 1) >> 1;

    int group_size = n/n_of_groups;
    int n_of_thread_per_group = group_size/2; //we assume that 1 thread manages 2 coefficients
    int step = n_of_thread_per_group;

    int target_index = (global_tid/n_of_thread_per_group)*group_size+(global_tid % n_of_thread_per_group);

    unsigned long long psiinv = psiinv_powers[n_of_groups + global_tid/n_of_thread_per_group];

    // printf("n_of_groups: %d, global_tid %d, target_index: %d, psi_index: %d\n", n_of_groups, global_tid, target_index, n_of_groups + global_tid/n_of_thread_per_group);

    unsigned long long first_target_value = a[target_index];

    unsigned long long second_target_value = a[target_index + step];

    unsigned long long target_result = first_target_value + second_target_value;

    target_result -= q * (target_result >= q);

    // a[target_index] = (target_result >> 1) + q2 * (target_result & 1);
    a[target_index] = target_result;

    first_target_value += q * (first_target_value < second_target_value);

    unsigned __int128 temp_storage = first_target_value - second_target_value;

    temp_storage *= psiinv;

    #if USE_BARRET
    singleBarrett(temp_storage, q, mu, qbit);
    #else
    temp_storage %= q;
    #endif

    unsigned long long temp_storage_low = temp_storage;

    // a[target_index + step] = (temp_storage_low >> 1) + q2 * (temp_storage_low & 1);
    a[target_index + step] = temp_storage_low;

    if(n_of_groups == 1){
    //Ardhi: below code for normalization of the result i.e a[i]*(1/n)
        group_size = n/1;
        n_of_thread_per_group = group_size/2; //we assume that 1 thread manages 2 coefficients
        step = n_of_thread_per_group;

        target_index = (global_tid/n_of_thread_per_group)*group_size+(global_tid % n_of_thread_per_group);

        // unsigned __int128 temp_storage;
        
        temp_storage = a[target_index];
        temp_storage *= n_inv;
        // singleBarrett(temp_storage, q, mu, qbit);
        temp_storage %= q;
        a[target_index] = temp_storage;

        temp_storage = a[target_index+step];
        temp_storage *= n_inv;
        // singleBarrett(temp_storage, q, mu, qbit);
        temp_storage %= q;
        a[target_index+step] = temp_storage;
    }
        
}

__global__ void CTBasedNTTInnerSingleAndMulMod(unsigned long long a[], unsigned long long b[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[], int n, int n_of_groups)
{
    int global_tid = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;

    int group_size = n/n_of_groups;
    int n_of_thread_per_group = group_size/2; //we assume that 1 thread manages 2 coefficients
    int step = n_of_thread_per_group;

    int target_index = (global_tid/n_of_thread_per_group)*group_size+(global_tid % n_of_thread_per_group);

    unsigned long long psi = psi_powers[n_of_groups + global_tid/n_of_thread_per_group];

    unsigned long long first_target_value = a[target_index];

    unsigned __int128 temp_storage = a[target_index + step];  // this is for eliminating the possibility of overflow

    temp_storage *= psi;

    // singleBarrett(temp_storage, q, mu, qbit);
    temp_storage %= q;
    unsigned long long second_target_value = temp_storage;

    unsigned long long target_result = first_target_value + second_target_value;

    target_result -= q * (target_result >= q);

    a[target_index] = target_result;

    first_target_value += q * (first_target_value < second_target_value);

    a[target_index + step] = first_target_value - second_target_value;

    if(n_of_groups == n/2){

    }
}

#include <stdlib.h>
#include <random>

unsigned long long modpow64(unsigned long long a, unsigned long long b, unsigned long long mod)  // calculates (<a> ** <b>) mod <mod>
{
    unsigned long long res = 1;

    if (1 & b)
        res = a;

    while (b != 0)
    {
        b = b >> 1;
        // unsigned long long t64 = (unsigned long long)a * a;
        // a = t64 % mod;
        __extension__ unsigned __int128 t128 = (unsigned __int128)a * a;
        a = t128 % mod;

        if (b & 1)
        {
            // unsigned long long r64 = (unsigned long long)a * res;
            // res = r64 % mod;
            __extension__ unsigned __int128 r128 = (unsigned __int128)a * res;
            res = r128 % mod;
        }

    }
    return res;
}

unsigned long long bitReverse(unsigned long long a, int bit_length)  // reverses the bits for twiddle factor calculation
{
    // cout<<"a: "<<a;
    unsigned long long res = 0;

    for (int i = 0; i < bit_length; i++)
    {
        res <<= 1;
        res = (a & 1) | res;
        a >>= 1;
    }

    // cout<<"\n reverse A: "<<res<<endl;
    return res;
}

std::random_device dev;  // uniformly distributed integer random number generator that produces non-deterministic random numbers
std::mt19937_64 rng(dev());  // pseudo-random generator of 64 bits with a state size of 19937 bits

void randomArray64(unsigned long long a[], int n, unsigned long long q)
{
    std::uniform_int_distribution<unsigned long long> randnum(0, q - 1);  // uniformly distributed random integers on the closed interval [a, b] according to discrete probability

    for (int i = 0; i < n; i++)
    {
        a[i] = randnum(rng);
    }
}

void fillTablePsi64(unsigned long long psi, unsigned long long q, unsigned long long psiinv, unsigned long long psiTable[], unsigned long long psiinvTable[], unsigned int n)  // twiddle factors computation
{
    for (unsigned int i = 0; i < n; i++)
    {
        psiTable[i] = modpow64(psi, bitReverse(i, log2(n)), q);
        // cout<<"\npsi: "<<psi<<" bitRev: "<<bitReverse(i, log2(n))<<" mod: "<<q<<" psi^bitRev mod q: "<<psiTable[i];
        psiinvTable[i] = modpow64(psiinv, bitReverse(i, log2(n)), q);
    }
}

//device variable;
#define magicNumber 3
// long *d_A, *d_B, *d_C, *d_modulus, *d_scalar;
// size_t bytes;
// long d_phim, d_n_rows;

// long *contiguousHostMapA, *contiguousHostMapB, *contiguousModulus, *scalarPerRow;

CPU_GPU_Buffer buf[n_threads];



void InitContiguousHostMapModulus(long phim, int n_rows){
  for(int i = 0; i < n_threads; i++)
  {
    // buf[i].contiguousHostMapA = (long *)malloc(magicNumber*phim*n_rows*sizeof(long));
    // buf[i].contiguousHostMapB = (long *)malloc(magicNumber*phim*n_rows*sizeof(long));
    // buf[i].contiguousModulus = (long *)malloc(magicNumber*n_rows*sizeof(long));
    // buf[i].scalarPerRow = (long *)malloc(magicNumber*n_rows*sizeof(long));

    cudaMallocHost(&buf[i].contiguousHostMapA, magicNumber*phim*n_rows*sizeof(long));
    cudaMallocHost(&buf[i].contiguousHostMapB, magicNumber*phim*n_rows*sizeof(long));
    cudaMallocHost(&buf[i].contiguousModulus, magicNumber*n_rows*sizeof(long));
    cudaMallocHost(&buf[i].scalarPerRow, magicNumber*n_rows*sizeof(long));
  }
}

// GPU_Buffer myBuf[n_threads];
// std::map<long, GPU_Buffer[n_threads]> CModBuffer;
std::vector<std::pair<long, std::vector<GPU_Buffer>>> CModBuffer_forward;
std::vector<std::pair<long, std::vector<GPU_Buffer>>> CModBuffer_inverse;

//Ardhi: for each thread's buffer allocate and copy the TF to GPU
void SetCModBuffer(long p, long k2, long mm, NTL::zz_pX& powers, NTL::zz_pX& b, long psi, 
bool inverse)
{
  GPU_Buffer newBuf[n_threads];
  long k2_inv = NTL::InvMod(k2, p);
  unsigned int bit_length = ceil(std::log2(p)); 
  unsigned long long mu = computeMu(bit_length, p);

  std::vector<unsigned long long> gpu_powers;
  //Ardhi: generate twiddle factors for gpu ntt with n=2^k
  for (unsigned int i = 0; i < k2; i++)
  {
      gpu_powers.push_back(NTL::PowerMod(psi, bitReverse(i, std::log2(k2)), p));
  }

  //Buffer allocation for Bluestein
  for(int j = 0; j < n_threads; j++)
  {
    // CHECK(cudaMalloc(&newBuf[j].x_dev, k2 * sizeof(unsigned long long)));

    //TF
    CHECK(cudaMalloc(&newBuf[j].gpu_powers_m_dev, mm * sizeof(unsigned long long)));
    CHECK(cudaMalloc(&newBuf[j].gpu_powers_dev, k2 * sizeof(unsigned long long)));
    CHECK(cudaMalloc(&newBuf[j].ntt_b, k2 * sizeof(unsigned long long)));

    // CHECK(cudaMalloc(&newBuf[j].x_pinned, mm * sizeof(unsigned long long)));
    // cudaMalloc(&iRbInVec, k2 * sizeof(unsigned long long));  
    // cudaMalloc(&RaInVec, k2 * sizeof(unsigned long long));
    // cudaMalloc(&iRaInVec, k2 * sizeof(unsigned long long));
    // cudaMalloc(&gpu_ipowers_dev, k2 * sizeof(unsigned long long));
    // cudaMalloc(&zMStar_dev, mm * sizeof(long));
    // cudaMalloc(&target_dev, mm * sizeof(long));

    newBuf[j].ownerThreadId = -1;
    newBuf[j].k2 = k2;
    newBuf[j].k2_inv = k2_inv;
    newBuf[j].debug_buffer = (unsigned long long*)malloc(k2 * sizeof(unsigned long long));
    newBuf[j].bit_length = bit_length;
    newBuf[j].mu = mu;
  }

  //copy TF to GPU
  for(int i = 0; i < n_threads; i++)
  {
    CHECK(cudaMemcpy(newBuf[i].gpu_powers_m_dev, powers.rep.data(), powers.rep.length() * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(newBuf[i].gpu_powers_dev, gpu_powers.data(), k2 * sizeof(unsigned long long), cudaMemcpyHostToDevice));

    // CHECK(cudaMemcpy(zMStar_dev, zMStar_h, mm*sizeof(long), cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(target_dev, target_h, mm*sizeof(long), cudaMemcpyHostToDevice));
  }

  unsigned long long *tmp = (unsigned long long*) malloc(k2 * sizeof(unsigned long long));

  //convert b to ntt_b
  gpu_ntt_forward(tmp, k2, b, p, newBuf[0].gpu_powers_dev); //Ardhi: convert b->ntt_b aka unsigned long[]

  //copy ntt_b to GPU
  for(int i = 0; i < n_threads; i++)
  {
    CHECK(cudaMemcpy(newBuf[i].ntt_b, tmp, k2 * sizeof(unsigned long long), cudaMemcpyHostToDevice));
  }

  std::vector<GPU_Buffer> bufferVec(newBuf, newBuf + n_threads);

  //Insert the buffer
   if(inverse)
    CModBuffer_inverse.push_back(std::make_pair(p, bufferVec));
  else
    CModBuffer_forward.push_back(std::make_pair(p, bufferVec));

  free(tmp);
}

//multistream
//Ardhi: for each thread's buffer allocate and copy the TF to GPU
void SetCModBuffer(long p, long k2, long mm, NTL::zz_pX& powers, NTL::zz_pX& b, long psi, 
long *zMStar_h, long *target_h, bool inverse)
{
  GPU_Buffer newBuf[n_threads];
  long k2_inv = NTL::InvMod(k2, p);
  unsigned int bit_length = ceil(std::log2(p)); 
  unsigned long long mu = computeMu(bit_length, p);

  std::vector<unsigned long long> gpu_powers;
  //Ardhi: generate twiddle factors for gpu ntt with n=2^k
  for (unsigned int i = 0; i < k2; i++)
  {
      gpu_powers.push_back(NTL::PowerMod(psi, bitReverse(i, std::log2(k2)), p));
  }

  //Buffer allocation for Bluestein
  for(int j = 0; j < n_threads; j++)
  {
    // CHECK(cudaMalloc(&newBuf[j].x_dev, k2 * sizeof(unsigned long long)));
    // CHECK(cudaMalloc(&newBuf[j].x_dev_filtered, k2 * sizeof(long)));

    //TF
    CHECK(cudaMalloc(&newBuf[j].gpu_powers_m_dev, mm * sizeof(unsigned long long)));
    CHECK(cudaMalloc(&newBuf[j].gpu_powers_dev, k2 * sizeof(unsigned long long)));
    CHECK(cudaMalloc(&newBuf[j].ntt_b, k2 * sizeof(unsigned long long)));
    CHECK(cudaMalloc(&newBuf[j].zMStar_dev, mm * sizeof(long)));
    CHECK(cudaMalloc(&newBuf[j].target_dev, mm * sizeof(long)));

    newBuf[j].ownerThreadId = -1;
    newBuf[j].k2 = k2;
    newBuf[j].k2_inv = k2_inv;
    newBuf[j].debug_buffer = (unsigned long long*)malloc(k2 * sizeof(unsigned long long));
    newBuf[j].bit_length = bit_length;
    newBuf[j].mu = mu;
  }

  //copy TF & zMstar+target to GPU
  for(int i = 0; i < n_threads; i++)
  {
    CHECK(cudaMemcpy(newBuf[i].gpu_powers_m_dev, powers.rep.data(), powers.rep.length() * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(newBuf[i].gpu_powers_dev, gpu_powers.data(), k2 * sizeof(unsigned long long), cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(newBuf[i].zMStar_dev, zMStar_h, mm*sizeof(long), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(newBuf[i].target_dev, target_h, mm*sizeof(long), cudaMemcpyHostToDevice));
  }

  unsigned long long *tmp = (unsigned long long*) malloc(k2 * sizeof(unsigned long long));

  //convert b to ntt_b
  gpu_ntt_forward(tmp, k2, b, p, newBuf[0].gpu_powers_dev); //Ardhi: convert b->ntt_b aka unsigned long[]

  //copy ntt_b to GPU
  for(int i = 0; i < n_threads; i++)
  {
    CHECK(cudaMemcpy(newBuf[i].ntt_b, tmp, k2 * sizeof(unsigned long long), cudaMemcpyHostToDevice));
  }

  std::vector<GPU_Buffer> bufferVec(newBuf, newBuf + n_threads);

  //Insert the buffer
   if(inverse)
    CModBuffer_inverse.push_back(std::make_pair(p, bufferVec));
  else
    CModBuffer_forward.push_back(std::make_pair(p, bufferVec));

  free(tmp);
}

//Ardhi: return the GPU buffer based on p and tid
GPU_Buffer& GetCModBuffer(long p, bool inverse)
{
  int tid;
  if(inverse)
  {
    //Ardhi: get the vec<GPU_buffer> based on p
    for(int i = 0; i < CModBuffer_inverse.size(); i++)
    {
      if(CModBuffer_inverse[i].first == p)
      {
        tid = getBufferIndex(CModBuffer_inverse[i].second);
        assert(tid != -1);
        return CModBuffer_inverse[i].second[tid];
      }
    }
  }
  else
  { //Forward
    //Ardhi: get the vec<GPU_buffer> based on p
    for(int i = 0; i < CModBuffer_forward.size(); i++)
    {
      if(CModBuffer_forward[i].first == p)
      {
        tid = getBufferIndex(CModBuffer_forward[i].second);
        assert(tid != -1);
        return CModBuffer_forward[i].second[tid];
      }
    }
  }

  return; //not found
}

std::hash<std::thread::id> hasher;

int getBufferIndex()
{
  auto tid = std::this_thread::get_id();

  for(int i = 0; i < n_threads; i++)
  {
    if(buf[i].ownerThreadId == (long) hasher(tid))
      return i;

    if(buf[i].ownerThreadId == -1)
    {
      buf[i].ownerThreadId = (long) hasher(tid);
      return i;
    }
  }

  return -1; //Ardhi: no buffer allocated for the requestor thread
}

int getBufferIndex(std::vector<GPU_Buffer> &myBuf)
{
  auto tid = std::this_thread::get_id();

  for(int i = 0; i < n_threads; i++)
  {
    if(myBuf[i].ownerThreadId == (long) hasher(tid))
      return i;

    if(myBuf[i].ownerThreadId == -1)
    {
      myBuf[i].ownerThreadId = (long) hasher(tid);
      return i;
    }
  }

  return -1; //Ardhi: no buffer allocated for the requestor thread
}

CPU_GPU_Buffer getCPU_GPU_Buffer()
{
  return buf[getBufferIndex()];
}

std::vector<cudaStream_t> getThreadStreams()
{
  return buf[getBufferIndex()].streams;
}

void setScalar(long index, long data){
  buf[getBufferIndex()].scalarPerRow[index] = data;
}

void setMapA(long index, long data){
  buf[getBufferIndex()].contiguousHostMapA[index] = data;
}

void setMapB(long index, long data){
  buf[getBufferIndex()].contiguousHostMapB[index] = data;
}

void setRowMapA(long offset, long *source)
{
  memcpy(buf[getBufferIndex()].contiguousHostMapA+offset, source, buf[getBufferIndex()].d_phim*sizeof(long));
}

void setRowMapB(long offset, const long *source)
{
  memcpy(buf[getBufferIndex()].contiguousHostMapB+offset, source, buf[getBufferIndex()].d_phim*sizeof(long));
}

long *getRowMapB(long index){
  return &buf[getBufferIndex()].contiguousHostMapB[index];
}

long *getRowMapA(long index){
  return &buf[getBufferIndex()].contiguousHostMapA[index];
}

long getMapA(long index){
  return buf[getBufferIndex()].contiguousHostMapA[index];
}

long getMapB(long index){
  return buf[getBufferIndex()].contiguousHostMapB[index];
}

void setModulus(long index, long data){
  buf[getBufferIndex()].contiguousModulus[index] = data;
}

void InitGPUBuffer(long phim, int n_rows, long m){
  long k = NTL::NextPowerOfTwo(2 * m - 1);
  long k2 = 1L << k; // k2 = 2^k

  for(int i = 0; i < n_threads; i++)
  {
    buf[i].d_phim = phim;
    buf[i].d_n_rows = n_rows;
    long bytes = magicNumber*phim*n_rows*sizeof(long);
    buf[i].bytes = bytes;
    buf[i].ownerThreadId = -1;
    buf[i].k2 = k2;

    // Allocate memory for arrays d_A, d_B, and d_C on device
    cudaMalloc(&buf[i].d_A, bytes);
    cudaMalloc(&buf[i].d_B, bytes);
    cudaMalloc(&buf[i].d_C, bytes);
    cudaMalloc(&buf[i].d_modulus, magicNumber*n_rows*sizeof(long));
    cudaMalloc(&buf[i].d_scalar, magicNumber*n_rows*sizeof(long));

    cudaMalloc(&buf[i].x_dev, magicNumber*k2*n_rows*sizeof(unsigned long long));
    cudaMalloc(&buf[i].x_dev_filtered, magicNumber*k2*n_rows*sizeof(long));

    CHECK_CUFFT_ERRORS(cufftPlan1d(&buf[i].plan, m, CUFFT_Z2Z, 1));
    CHECK(cudaMalloc(&buf[i].buf_dev, m*sizeof(cufftDoubleComplex)));
    initializeStreams(n_rows * magicNumber, buf[i].streams);

    // CHECK(cudaMalloc(&buf[i].x_dev, k2 * sizeof(unsigned long long)));
    // CHECK(cudaMalloc(&buf[i].x_pinned, m * sizeof(unsigned long long)));


  }

}



void DestroyGPUBuffer(){
    int idx = getBufferIndex();
    // Free GPU memory
    cudaFree(buf[idx].d_C);
    cudaFree(buf[idx].d_A);
    cudaFree(buf[idx].d_B);

    cudaFree(buf[idx].d_modulus);
    cudaFree(buf[idx].d_scalar);
}


__global__ void kernel_addMod(long *a, long *b, long *result, long size, long *d_modulus, long phim){
    // printf("==kernel code== Hi there, this is Ardhi\n");
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
      result[tid] = a[tid] + b[tid];
      // result[tid] %= modulus;
      if(result[tid] >= d_modulus[tid/phim])
        result[tid] -= d_modulus[tid/phim];
    }
}

#define debug_impl 0
void CudaEltwiseAddMod(long actual_nrows){
    int idx = getBufferIndex();
    // Copy data from host arrays A and B to device arrays d_A and d_B
    cudaMemcpy(buf[idx].d_A, buf[idx].contiguousHostMapA, buf[idx].bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(buf[idx].d_B, buf[idx].contiguousHostMapB, buf[idx].bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(buf[idx].d_modulus, buf[idx].contiguousModulus, actual_nrows*sizeof(long), cudaMemcpyHostToDevice);

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(buf[idx].d_phim*actual_nrows) / thr_per_blk );

    // Launch kernel
    kernel_addMod<<< blk_in_grid, thr_per_blk >>>(buf[idx].d_A, buf[idx].d_B, buf[idx].d_C, buf[idx].d_phim*actual_nrows, buf[idx].d_modulus, buf[idx].d_phim);
    // cudaDeviceSynchronize();

    // Copy data from device array d_C to host array result
    cudaMemcpy(buf[idx].contiguousHostMapA, buf[idx].d_C, buf[idx].bytes, cudaMemcpyDeviceToHost);

}

__global__ void kernel_addModScalar(long *a, long *scalar, long *result, long size, long *d_modulus, long phim){
    // printf("==kernel code== Hi there, this is Ardhi\n");
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
      result[tid] = a[tid] + scalar[tid/phim];
      // result[tid] %= modulus;
      if(result[tid] >= d_modulus[tid/phim])
        result[tid] -= d_modulus[tid/phim];
    }
}

void CudaEltwiseAddMod(long actual_nrows, long scalar){
    int idx = getBufferIndex();
    // Copy data from host arrays A and B to device arrays d_A and d_B
    cudaMemcpy(buf[idx].d_A, buf[idx].contiguousHostMapA, buf[idx].bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(buf[idx].d_scalar, buf[idx].scalarPerRow, actual_nrows*sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(buf[idx].d_modulus, buf[idx].contiguousModulus, actual_nrows*sizeof(long), cudaMemcpyHostToDevice);

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(buf[idx].d_phim*actual_nrows) / thr_per_blk );

    // Launch kernel
    kernel_addModScalar<<< blk_in_grid, thr_per_blk >>>(buf[idx].d_A, buf[idx].d_scalar, buf[idx].d_C, buf[idx].d_phim*actual_nrows, buf[idx].d_modulus, buf[idx].d_phim);
    // cudaDeviceSynchronize();

    // Copy data from device array d_C to host array result
    cudaMemcpy(buf[idx].contiguousHostMapA, buf[idx].d_C, buf[idx].bytes, cudaMemcpyDeviceToHost);

#if 0
  //allocate data in device memory
  thrust::device_vector<long> d_result(result,result + size);
  thrust::device_vector<long> d_a(a,a + size);
  thrust::device_vector<long> d_b(size);
  thrust::device_vector<long> d_modululus(size);

  //fill b with scalar
  thrust::fill(d_b.begin(), d_b.end(), scalar);

  //fill d_modulus with modulus
  thrust::fill(d_modululus.begin(), d_modululus.end(), modulus);

  //result = a + b
  thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_result.begin(), thrust::plus<long>());

  //result = result mod modulus
  thrust::transform(d_result.begin(), d_result.end(), d_modululus.begin(), d_result.begin(), thrust::modulus<long>());

  //copy the result back to CPU
  thrust::copy(d_result.begin(), d_result.end(), result);
#endif
}

__global__ void kernel_subMod(long *a, long *b, long *result, long size, long *d_modulus, long phim){
    // printf("==kernel code== Hi there, this is Ardhi\n");
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
      if (a[tid] >= b[tid]) {
        result[tid] = a[tid] - b[tid];
      } else {
        result[tid] = a[tid] + d_modulus[tid/phim] - b[tid];
      }
    }
}

void CudaEltwiseSubMod(long actual_nrows){
    int idx = getBufferIndex();

    // Copy data from host arrays A and B to device arrays d_A and d_B
    cudaMemcpy(buf[idx].d_A, buf[idx].contiguousHostMapA, buf[idx].bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(buf[idx].d_B, buf[idx].contiguousHostMapB, buf[idx].bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(buf[idx].d_modulus, buf[idx].contiguousModulus, actual_nrows*sizeof(long), cudaMemcpyHostToDevice);

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(buf[idx].d_phim*actual_nrows) / thr_per_blk );

    // Launch kernel
    kernel_subMod<<< blk_in_grid, thr_per_blk >>>(buf[idx].d_A, buf[idx].d_B, buf[idx].d_C, buf[idx].d_phim*actual_nrows, buf[idx].d_modulus, buf[idx].d_phim);
    // cudaDeviceSynchronize();

    // Copy data from device array d_C to host array result
    cudaMemcpy(buf[idx].contiguousHostMapA, buf[idx].d_C, buf[idx].bytes, cudaMemcpyDeviceToHost);

#if 0
  //allocate data in device memory
  thrust::device_vector<long> d_result(result,result + size);
  thrust::device_vector<long> d_a(a,a + size);
  thrust::device_vector<long> d_b(b,b + size);
  thrust::device_vector<long> d_modululus(size);

  //fill d_modulus with modulus
  thrust::fill(d_modululus.begin(), d_modululus.end(), modulus);

  //b = -b
  thrust::transform(d_b.begin(), d_b.end(), d_b.begin(), thrust::negate<long>());

  //result = a + b
  thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_result.begin(), thrust::plus<long>());

  //result = result mod modulus
  thrust::transform(d_result.begin(), d_result.end(), d_modululus.begin(), d_result.begin(), thrust::modulus<long>());

  //copy the result back to CPU
  thrust::copy(d_result.begin(), d_result.end(), result);
#endif
}

__global__ void kernel_subModScalar(long *a, long *scalar, long *result, long size, long *d_modulus, long phim){
    // printf("==kernel code== Hi there, this is Ardhi\n");
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
      if (a[tid] >= scalar[tid/phim]) {
        result[tid] = a[tid] - scalar[tid/phim];
      } else {
        result[tid] = a[tid] + d_modulus[tid/phim] - scalar[tid/phim];
      }
    }
}

void CudaEltwiseSubMod(long actual_nrows, long scalar){
    int idx = getBufferIndex();
    // Copy data from host arrays A and B to device arrays d_A and d_B
    cudaMemcpy(buf[idx].d_A, buf[idx].contiguousHostMapA, buf[idx].bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(buf[idx].d_scalar, buf[idx].scalarPerRow, actual_nrows*sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(buf[idx].d_modulus, buf[idx].contiguousModulus, actual_nrows*sizeof(long), cudaMemcpyHostToDevice);

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(buf[idx].d_phim*actual_nrows) / thr_per_blk );

    // Launch kernel
    kernel_subModScalar<<< blk_in_grid, thr_per_blk >>>(buf[idx].d_A, buf[idx].d_scalar, buf[idx].d_C, buf[idx].d_phim*actual_nrows, buf[idx].d_modulus, buf[idx].d_phim);
    // cudaDeviceSynchronize();

    // Copy data from device array d_C to host array result
    cudaMemcpy(buf[idx].contiguousHostMapA, buf[idx].d_C, buf[idx].bytes, cudaMemcpyDeviceToHost);

#if 0
  //allocate data in device memory
  thrust::device_vector<long> d_result(result,result + size);
  thrust::device_vector<long> d_a(a,a + size);
  thrust::device_vector<long> d_b(size);
  thrust::device_vector<long> d_modululus(size);

  //negate scalar
  scalar = -scalar;

  //fill b with -scalar
  thrust::fill(d_b.begin(), d_b.end(), scalar);

  //fill d_modulus with modulus
  thrust::fill(d_modululus.begin(), d_modululus.end(), modulus);

  //result = a + b
  thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_result.begin(), thrust::plus<long>());

  //result = result mod modulus
  thrust::transform(d_result.begin(), d_result.end(), d_modululus.begin(), d_result.begin(), thrust::modulus<long>());

  //copy the result back to CPU
  thrust::copy(d_result.begin(), d_result.end(), result);
#endif
}

// inline uint64_t DivideUInt128UInt64Lo(uint64_t x1, uint64_t x0, uint64_t y) {
//   uint128_t n =
//       (static_cast<uint128_t>(x1) << 64) | (static_cast<uint128_t>(x0));
//   uint128_t q = n / y;

//   return static_cast<uint64_t>(q);
// }

__device__ __int128 myMod(__int128 x, long m) {
    return (x%m + m)%m;
}



__device__ long mul_mod(long a, long b, long m) {
    if (!((a | b) & (0xFFFFFFFFULL << 32))) return a * b % m;

    long d = 0, mp2 = m >> 1;
    int i;
    if (a >= m) a %= m;
    if (b >= m) b %= m;
    for (i = 0; i < 64; ++i) {
        d = (d > mp2) ? (d << 1) - m : d << 1;
        if (a & 0x8000000000000000ULL) d += b;
        if (d >= m) d -= m;
        a <<= 1;
    }
    return d;
}

__device__ __int128 myMod3(__int128 a,long b)
{
    if(a < 0)
      a *= -1;
    
    return b - (a%b);
}

long sp_CorrectDeficit(long a, long n)
{
   return a >= 0 ? a : a+n;
}

long sp_CorrectExcess(long a, long n)
{
   return a-n >= 0 ? a-n : a;
}

// long MulMod(long a, long b, long n, double ninv) {
//   long q = long( double(a) * (double(b) * ninv) );
//   unsigned long rr = cast_unsigned(a)*cast_unsigned(b) - cast_unsigned(q)*cast_unsigned(n);
//   long r = sp_CorrectDeficit(rr, n);
//   return sp_CorrectExcess(r, n);
// }



std::ostream&
operator<<( std::ostream& dest, __int128_t value )
{
    std::ostream::sentry s( dest );
    if ( s ) {
        __uint128_t tmp = value < 0 ? -value : value;
        char buffer[ 128 ];
        char* d = std::end( buffer );
        do
        {
            -- d;
            *d = "0123456789"[ tmp % 10 ];
            tmp /= 10;
        } while ( tmp != 0 );
        if ( value < 0 ) {
            -- d;
            *d = '-';
        }
        int len = std::end( buffer ) - d;
        if ( dest.rdbuf()->sputn( d, len ) != len ) {
            dest.setstate( std::ios_base::badbit );
        }
    }
    return dest;
}

// Returns most-significant bit of the input
inline uint64_t MSB(uint64_t input) {
  return static_cast<uint64_t>(std::log2l(input));
}

inline uint64_t Log2(uint64_t x) { return MSB(x); }

template <int InputModFactor>
uint64_t ReduceMod(uint64_t x, uint64_t modulus,
                   const uint64_t* twice_modulus = nullptr,
                   const uint64_t* four_times_modulus = nullptr) {

  if (InputModFactor == 1) {
    return x;
  }
  if (InputModFactor == 2) {
    if (x >= modulus) {
      x -= modulus;
    }
    return x;
  }
  if (InputModFactor == 4) {
    if (x >= *twice_modulus) {
      x -= *twice_modulus;
    }
    if (x >= modulus) {
      x -= modulus;
    }
    return x;
  }
  if (InputModFactor == 8) {

    if (x >= *four_times_modulus) {
      x -= *four_times_modulus;
    }
    if (x >= *twice_modulus) {
      x -= *twice_modulus;
    }
    if (x >= modulus) {
      x -= modulus;
    }
    return x;
  }

  return x;
}

__extension__ typedef __int128 int128_t;
__extension__ typedef unsigned __int128 uint128_t;

// Returns low 64bit of 128b/64b where x1=high 64b, x0=low 64b
inline uint64_t DivideUInt128UInt64Lo(uint64_t x1, uint64_t x0, uint64_t y) {
  uint128_t n =
      (static_cast<uint128_t>(x1) << 64) | (static_cast<uint128_t>(x0));
  uint128_t q = n / y;

  return static_cast<uint64_t>(q);
}

/// @brief Pre-computes a Barrett factor with which modular multiplication can
/// be performed more efficiently
class MultiplyFactor {
 public:
  MultiplyFactor() = default;

  /// @brief Computes and stores the Barrett factor floor((operand << bit_shift)
  /// / modulus). This is useful when modular multiplication of the form
  /// (x * operand) mod modulus is performed with same modulus and operand
  /// several times. Note, passing operand=1 can be used to pre-compute a
  /// Barrett factor for multiplications of the form (x * y) mod modulus, where
  /// only the modulus is re-used across calls to modular multiplication.
  MultiplyFactor(uint64_t operand, uint64_t bit_shift, uint64_t modulus)
      : m_operand(operand) {
    uint64_t op_hi = operand >> (64 - bit_shift);
    uint64_t op_lo = (bit_shift == 64) ? 0 : (operand << bit_shift);

    m_barrett_factor = DivideUInt128UInt64Lo(op_hi, op_lo, modulus);
  }

  /// @brief Returns the pre-computed Barrett factor
  inline uint64_t BarrettFactor() const { return m_barrett_factor; }

  /// @brief Returns the operand corresponding to the Barrett factor
  inline uint64_t Operand() const { return m_operand; }

 private:
  uint64_t m_operand;
  uint64_t m_barrett_factor;
};

inline uint128_t MultiplyUInt64(uint64_t x, uint64_t y) {
  return uint128_t(x) * uint128_t(y);
}

inline void MultiplyUInt64(uint64_t x, uint64_t y, uint64_t* prod_hi,
                           uint64_t* prod_lo) {
  uint128_t prod = MultiplyUInt64(x, y);
  *prod_hi = static_cast<uint64_t>(prod >> 64);
  *prod_lo = static_cast<uint64_t>(prod);
}

__device__ __int128 flooredDivision(__int128 a, long b)
{
    if(a/b > 0)
      return a/b;

    if(a%b == 0)
      return (a/b);
    else
      return (a/b)-1;

}
// __device__ __int128 myMod2(__int128 a,long b)
// {
//     return a - b * flooredDivision(a, b);
// }

__global__ void kernel_mulMod(long *a, long *b, long *result, long size, long *d_modulus, long phim){
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if(tid < size){
    // __int128 temp_res=0;
    // __int128 temp_a=0;
    // __int128 temp_b=0;
    __int128 temp_storage= a[tid];
    // temp_a = a[tid];
    // temp_b = b[tid];

    // temp_res = temp_a * temp_b;
    temp_storage *= b[tid];
    // temp_res = temp_res%modulus;
    // temp_res = myMod2(temp_res, d_modulus[tid/phim]);

    // d_result[tid] = temp_res;
    // result[tid] %= modulus;
    result[tid]=temp_storage % d_modulus[tid/phim];

    // result[tid]=mul_mod(a[tid], b[tid], modulus);
  }
}

void CudaEltwiseMultMod(long actual_nrows){
	// HELIB_NTIMER_START(CudaEltwiseMultMod_CudaMemCpyHD);
    int idx = getBufferIndex();
    // Copy data from host arrays A and B to device arrays d_A and d_B
    cudaMemcpy(buf[idx].d_A, buf[idx].contiguousHostMapA, buf[idx].bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(buf[idx].d_B, buf[idx].contiguousHostMapB, buf[idx].bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(buf[idx].d_modulus, buf[idx].contiguousModulus, actual_nrows*sizeof(long), cudaMemcpyHostToDevice);
	// HELIB_NTIMER_STOP(CudaEltwiseMultMod_CudaMemCpyHD);

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(buf[idx].d_phim*actual_nrows) / thr_per_blk );
	// HELIB_NTIMER_START(CudaEltwiseMultMod_kernel);
    // Launch kernel
    kernel_mulMod<<< blk_in_grid, thr_per_blk >>>(buf[idx].d_A, buf[idx].d_B, buf[idx].d_C, buf[idx].d_phim*actual_nrows, buf[idx].d_modulus, buf[idx].d_phim);
    // cudaDeviceSynchronize();
	// HELIB_NTIMER_STOP(CudaEltwiseMultMod_kernel);

	// HELIB_NTIMER_START(CudaEltwiseMultMod_CudaMemCpyDH);
    // Copy data from device array d_C to host array result
    cudaMemcpy(buf[idx].contiguousHostMapA, buf[idx].d_C, buf[idx].bytes, cudaMemcpyDeviceToHost);
	// HELIB_NTIMER_STOP(CudaEltwiseMultMod_CudaMemCpyDH);

#if 0
//HEXL Naive MulMod
  constexpr int64_t beta = -2;
  constexpr int64_t alpha = 62;  // ensures alpha - beta = 64
  int const InputModFactor=1;

  uint64_t gamma = Log2(InputModFactor);
  // HEXL_UNUSED(gamma);

  const uint64_t ceil_log_mod = Log2(modulus) + 1;  // "n" from Algorithm 2
  uint64_t prod_right_shift = ceil_log_mod + beta;

  // Barrett factor "mu"
  // TODO(fboemer): Allow MultiplyFactor to take bit shifts != 64
  uint64_t barr_lo =
      MultiplyFactor(uint64_t(1) << (ceil_log_mod + alpha - 64), 64, modulus)
          .BarrettFactor();

  const uint64_t twice_modulus = 2 * modulus;

  for (size_t i = 0; i < size; ++i) {
    uint64_t prod_hi, prod_lo, c2_hi, c2_lo, Z;

    uint64_t x = ReduceMod<InputModFactor>(*a, modulus, &twice_modulus);
    uint64_t y = ReduceMod<InputModFactor>(*b, modulus, &twice_modulus);

    // Multiply inputs
    MultiplyUInt64(x, y, &prod_hi, &prod_lo);

    // floor(U / 2^{n + beta})
    uint64_t c1 = (prod_lo >> (prod_right_shift)) +
                  (prod_hi << (64 - (prod_right_shift)));

    // c2 = floor(U / 2^{n + beta}) * mu
    MultiplyUInt64(c1, barr_lo, &c2_hi, &c2_lo);

    // alpha - beta == 64, so we only need high 64 bits
    uint64_t q_hat = c2_hi;

    // only compute low bits, since we know high bits will be 0
    Z = prod_lo - q_hat * modulus;

    // Conditional subtraction
    *result = (Z >= modulus) ? (Z - modulus) : Z;

    if(NTL::MulMod(a[i], b[i], modulus) != *result)
        std::cout<<"MulMod Missmatch Detected. j="<<i<<"/"<<size<<" CPU: "<<NTL::MulMod(a[i], b[i], modulus)<<" GPU: "<<*result<<" A: "<<a[i]<<" B: "<<b[i]<<" Mod: "<<modulus<<std::endl;
    else 
        std::cout<<"MulMod Matched. j="<<i<<"/"<<size<<" CPU: "<<NTL::MulMod(a[i], b[i], modulus)<<" GPU: "<<*result<<" A: "<<a[i]<<" B: "<<b[i]<<" Mod: "<<modulus<<std::endl;

    ++a;
    ++b;
    ++result;
  }
  #endif

}

__global__ void kernel_mulModScalar(long *a, long *scalar, long *result, long size, long *d_modulus, long phim){
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if(tid < size){
    // __int128 temp_res=0;
    // __int128 temp_a=0;
    // __int128 temp_b=0;

    __int128 temp_storage = a[tid];
    // temp_b = scalar[tid/phim];

    // temp_res = temp_a * temp_b;
    // temp_res = temp_res%modulus;
    // temp_res = myMod2(temp_res, d_modulus[tid/phim]);

    temp_storage *= scalar[tid/phim];
    // d_result[tid] = temp_res;
    // result[tid] %= modulus;
    result[tid]= temp_storage % d_modulus[tid/phim];

    // result[tid]=mul_mod(a[tid], b[tid], modulus);
  }
}

void CudaEltwiseMultMod(long actual_nrows, long scalar){
   // Copy data from host arrays A and B to device arrays d_A and d_B
    int idx = getBufferIndex();

    cudaMemcpy(buf[idx].d_A, buf[idx].contiguousHostMapA, buf[idx].bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(buf[idx].d_scalar, buf[idx].scalarPerRow, actual_nrows*sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(buf[idx].d_modulus, buf[idx].contiguousModulus, actual_nrows*sizeof(long), cudaMemcpyHostToDevice);

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(buf[idx].d_phim*actual_nrows) / thr_per_blk );

    // Launch kernel
    kernel_mulModScalar<<< blk_in_grid, thr_per_blk >>>(buf[idx].d_A, buf[idx].d_scalar, buf[idx].d_C, buf[idx].d_phim*actual_nrows, buf[idx].d_modulus, buf[idx].d_phim);
    // cudaDeviceSynchronize();

    // Copy data from device array d_C to host array result
    cudaMemcpy(buf[idx].contiguousHostMapA, buf[idx].d_C, buf[idx].bytes, cudaMemcpyDeviceToHost);

#if 0
  //allocate data in device memory
  thrust::device_vector<long> d_result(result,result + size);
  thrust::device_vector<long> d_a(a,a + size);
  thrust::device_vector<long> d_b(size);
  thrust::device_vector<long> d_modululus(size);

  //fill b with scalar
  thrust::fill(d_b.begin(), d_b.end(), scalar);

  //fill d_modulus with modulus
  thrust::fill(d_modululus.begin(), d_modululus.end(), modulus);

  //result = a * b
  thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_result.begin(), thrust::multiplies<long>());

  //result = result mod modulus
  thrust::transform(d_result.begin(), d_result.end(), d_modululus.begin(), d_result.begin(), thrust::modulus<long>());

  //copy the result back to CPU
  thrust::copy(d_result.begin(), d_result.end(), result);
#endif
}


//device buffers
unsigned long long* psi_powers, * psiinv_powers;
unsigned long long* a;
unsigned long long* d_a;
unsigned long long* d_b;
unsigned long long* psiTable;
unsigned long long* psiinvTable;
cudaStream_t stream[32];

void init_gpu_ntt(unsigned int n){
    int size_array = sizeof(unsigned long long) * n;
    cudaMalloc(&psi_powers, size_array);
    cudaMalloc(&psiinv_powers, size_array);
    cudaMallocHost(&a, sizeof(unsigned long long) * n);
    cudaMalloc(&d_a, size_array);
    cudaMalloc(&d_b, size_array);
    psiTable = (unsigned long long*)malloc(size_array);
    psiinvTable = (unsigned long long*)malloc(size_array);

    // for (int i = 0; i < 32; ++i)
    //   cudaStreamCreate(&stream[i]); 
}

void moveTwFtoGPU(unsigned long long gpu_powers_dev[], std::vector<unsigned long long>& gpu_powers, int k2, NTL::zz_pX& powers, unsigned long long gpu_powers_m_dev[], long zMStar_dev[], long zMStar_h[], long mm, long target_dev[], long target_h[]){
    CHECK(cudaMemcpy(gpu_powers_dev, gpu_powers.data(), k2 * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_powers_m_dev, powers.rep.data(), powers.rep.length() * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(zMStar_dev, zMStar_h, mm*sizeof(long), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(target_dev, target_h, mm*sizeof(long), cudaMemcpyHostToDevice));
}

void gpu_ntt_backward(NTL::vec_zz_p& res, unsigned int n, const NTL::vec_zz_p& x, unsigned long long q, const std::vector<unsigned long long>& gpu_ipowers, unsigned long long psi, unsigned long long psiinv){
    int size_array = sizeof(unsigned long long) * n;
    unsigned int q_bit = ceil(std::log2(q));

    /****************************************************************
    BEGIN
    cudamalloc, memcpy, etc... for gpu
    */

#if 0 //Ardhi: check psi computation
    unsigned long long* psiTable = (unsigned long long*)malloc(size_array);
    unsigned long long* psiinvTable = (unsigned long long*)malloc(size_array);
    fillTablePsi64(psi, q, psiinv, psiTable, psiinvTable, n); //gel psi psi

    for(long i=0; i<n; i++){
      if(psiinvTable[i] != gpu_ipowers[i]){
          throw std::runtime_error("psiTable and gpu_powers missmatch");
      }
    }

#endif

    // const unsigned long long *twiddle_factors = gpu_powers.data();
    cudaMemcpy(psiinv_powers, gpu_ipowers.data(), size_array, cudaMemcpyHostToDevice);

    // we print these because we forgot them every time :)
    // std::cout << "n = " << n << std::endl;
    // std::cout << "q = " << q << std::endl;
    // std::cout << "Psi = " << psi << std::endl;
    // std::cout << "Psi Inverse = " << psiinv << std::endl;

    //generate parameters for barrett
    unsigned int bit_length = q_bit;
    // double mu1 = powl(2, 2 * bit_length);
    // unsigned mu = mu1 / q;
    unsigned __int128 mu1 = 1;
    mu1 = mu1 << (2*bit_length);
    unsigned long long mu = mu1/q;

    for(int i=0; i < n; i++)
      a[i] = rep(x[i]);

    cudaMemcpy(d_a, a, size_array, cudaMemcpyHostToDevice);

    /*
    END
    cudamalloc, memcpy, etc... for gpu
    ****************************************************************/

    
    /****************************************************************
    BEGIN
    Kernel Calls
    */

    long n_inv = NTL::InvMod(n, q);
    int num_blocks = n/(THREADS_PER_BLOCK*2);

#if 1
    GSBasedINTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK, 2048 * sizeof(unsigned long long)>>>(d_a, q, mu, bit_length, psiinv_powers, n, n/2);

    if(n>2048){
      #pragma unroll
      // for (int n_of_groups = n/2; n_of_groups >= 1; n_of_groups /= 2) 
      for (int n_of_groups = (n/2048)/2; n_of_groups >= 1; n_of_groups /= 2) 
      {
          GSBasedINTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK>>>(d_a, q, mu, bit_length, psiinv_powers, n, n_inv, n_of_groups);
      }
    }
#else
      #pragma unroll
      for (int n_of_groups = n/2; n_of_groups >= 1; n_of_groups /= 2) 
      {
          GSBasedINTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK>>>(d_a, q, mu, bit_length, psiinv_powers, n, n_inv, n_of_groups);
      }
#endif
    /*
    END
    Kernel Calls
    ****************************************************************/

    cudaMemcpy(a, d_a, size_array, cudaMemcpyDeviceToHost);  // do this in async 

    cudaDeviceSynchronize();  // CPU being a gentleman, and waiting for GPU to finish it's job

    for(long i=0; i<n; i++)
      res[i] = a[i];

}

void gpu_mulMod(NTL::zz_pX& x, unsigned long long x_dev[], unsigned long long gpu_powers_m_dev[], unsigned long long p, int n, cudaStream_t stream){
    long dx = deg(x);
    // memset(a,0, n*8);
    // memcpy(a, x.rep.data(), (dx+1)*sizeof(unsigned long long));
  HELIB_NTIMER_START(gpu_mulMod_cuMemSet);
    cudaMemsetAsync(x_dev, 0, n*sizeof(long int), stream);
  HELIB_NTIMER_STOP(gpu_mulMod_cuMemSet);

  HELIB_NTIMER_START(gpu_mulMod_cpyHD);
    CHECK(cudaMemcpyAsync(x_dev, x.rep.data(), (dx+1)*sizeof(unsigned long long), cudaMemcpyHostToDevice, stream));
  HELIB_NTIMER_STOP(gpu_mulMod_cpyHD);

  HELIB_NTIMER_START(gpu_mulMod_kernel);
    KernelMulMod<<<ceil(((double)dx+1)/THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, stream>>>(x_dev, gpu_powers_m_dev, p, dx+1);
  HELIB_NTIMER_STOP(gpu_mulMod_kernel);
}

__global__ void add_mod_kernel(unsigned long long x_dev[], long n, long l, unsigned long long p) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i + n < l) {
        // x[i - n].LoopHole() = NTL::AddMod(rep(x[i - n]), rep(x[i]), p);

        x_dev[i] = x_dev[i] + x_dev[i+n];
        if(x_dev[i] >= p)
          x_dev[i] -= p;
    }
}

void gpu_addMod(unsigned long long x_dev[], long n, long dx, unsigned long long p, cudaStream_t stream){
      int n_blocks= ceil(((double)(dx+1) - n + 1) / 512);
      add_mod_kernel<<<n_blocks, 512, 0, stream>>>(x_dev, n, dx, p);
}

void gpu_mulMod2(NTL::zz_pX& x, unsigned long long x_dev[], unsigned long long x_pinned[], 
  unsigned long long gpu_powers_m_dev[], unsigned long long p, int n, cudaStream_t stream){
    // long dx = deg(x);
    // memset(a,0, n*8);
    // memcpy(a, x.rep.data(), (dx+1)*sizeof(unsigned long long));
    // cudaMemset(x_dev, 0, n*8);
    // CHECK(cudaMemcpy(x_dev, x.rep.data(), (dx+1)*sizeof(unsigned long long), cudaMemcpyHostToDevice));
	HELIB_NTIMER_START(AfterPolyMul_mulMod_kernel);
    KernelMulMod<<<ceil((double)n/THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, stream>>>(x_dev, gpu_powers_m_dev, p, n);
	HELIB_NTIMER_STOP(AfterPolyMul_mulMod_kernel);


if(stream == 0){
#if 1 //Ardhi: when the iFFT is async we can disable this memory copy
  // x.SetLength(n);
	HELIB_NTIMER_START(AfterPolyMul_mulMod_cpyDH);
  CHECK(cudaMemcpy(x.rep.data(), x_dev, n*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  // CHECK(cudaMemcpy(x_pinned, x_dev, n*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    x.normalize();
	HELIB_NTIMER_STOP(AfterPolyMul_mulMod_cpyDH);
#endif
}
#if 0
	HELIB_NTIMER_START(AfterPolyMul_mulMod_buffer);
  memcpy(x.rep.data(), x_pinned, n*sizeof(unsigned long long));
	HELIB_NTIMER_STOP(AfterPolyMul_mulMod_buffer);
#endif
}

inline void checkCudaError(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        printf("%s\n", msg);
        printf("Error: %s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

__global__ void parallel_copy(long m, long x_dev_filtered[], unsigned long long x_dev[], long zMStar[], long target_dev[])
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < m && zMStar[i] != 0){
      x_dev_filtered[target_dev[i]] = x_dev[i];
    }
}

void gpu_parallel_copy(NTL::vec_long& y_h, long m, long p, bool inverse, unsigned long long x_dev[], long x_dev_filtered[], cudaStream_t stream)
{
    int numBlocks = (m + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GPU_Buffer myBuf = GetCModBuffer(p, inverse); //always false since only work for forward FFT

    parallel_copy<<<numBlocks, THREADS_PER_BLOCK, 0, stream>>>(m, x_dev_filtered, x_dev, myBuf.zMStar_dev, myBuf.target_dev);

    CHECK(cudaMemcpyAsync(y_h.data(), x_dev_filtered, y_h.length()*sizeof(long), cudaMemcpyDeviceToHost, stream));
}

//Ardhi: this is for blocking version of the bluestein
#if 1

//multistream
void gpu_fused_polymul(unsigned long long x_dev[], unsigned long long q, bool inverse, cudaStream_t stream)
{
  GPU_Buffer forwardBuffer = GetCModBuffer(q, false); //Forward buffer
  GPU_Buffer inverseBuffer = GetCModBuffer(q, true); //Inverse buffer

  unsigned long long *gpu_powers_dev;
  unsigned long long *gpu_ipowers_dev;
  long n = forwardBuffer.k2;
  long n_inv = forwardBuffer.k2_inv;

  //Assign the buffer
  if(inverse)
  {
    //caller: iFFT
    gpu_powers_dev = inverseBuffer.gpu_powers_dev;
    gpu_ipowers_dev = forwardBuffer.gpu_powers_dev;
  }
  else
  {
    //caller: FFT_aux
    gpu_powers_dev = forwardBuffer.gpu_powers_dev;
    gpu_ipowers_dev = inverseBuffer.gpu_powers_dev;
  }

  // int size_array = sizeof(unsigned long long) * n;
  unsigned int bit_length = forwardBuffer.bit_length; 

  //generate parameters for barrett
  unsigned long long mu = forwardBuffer.mu;

#if debug_blue
  unsigned long long *gpu_result0 = (unsigned long long*) malloc(sizeof(unsigned long long)*n);
  CHECK(cudaMemcpy(gpu_result0, x_dev, n*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  std::cout << "q " << q <<"\n";
  std::cout << "x_dev\n";
  for(long i = 0; i < 5; i++)
  {
    std::cout << gpu_result0[i] << " ";
  }
  std::cout<<std::endl;
#endif

	HELIB_NTIMER_START(KernelNTT);
  int num_blocks = n/(THREADS_PER_BLOCK*2);
  int n_of_groups=1;

  #if USE_GLOBAL_MUTEX
  void *kernelArgs[] = { (void *)&x_dev, (void *)&q, (void *)&mu, (void *)&bit_length, (void *)&gpu_powers_dev, (void *)&n};
  cudaLaunchCooperativeKernel((void *)CTBasedNTTInnerSingleGlobalMutex, num_blocks, THREADS_PER_BLOCK, kernelArgs, 0, stream);

  n_of_groups = n / 2048;
  #else
  #pragma unroll
  // for (int n_of_groups = 1; n_of_groups < n; n_of_groups *= 2)
  for (n_of_groups = 1; (n/n_of_groups>2048); n_of_groups *= 2) //Ardhi: do the ntt until the working size is 2048 elements
  {
      CTBasedNTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(x_dev, q, mu, bit_length, gpu_powers_dev, n, n_of_groups);
  }
  #endif
  
  CTBasedNTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK, 2048 * sizeof(unsigned long long), stream>>>(x_dev, q, mu, bit_length, gpu_powers_dev, n_of_groups);
	HELIB_NTIMER_STOP(KernelNTT);

#if debug_blue 
  unsigned long long *gpu_result = (unsigned long long*) malloc(sizeof(unsigned long long)*n);
  CHECK(cudaMemcpy(gpu_result, x_dev, n*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  std::cout << "ntt x_dev\n";
  for(long i = 0; i < 5; i++)
  {
    std::cout << gpu_result[i] << " ";
  }
  std::cout<<std::endl;

  unsigned long long *gpu_result2 = (unsigned long long*) malloc(sizeof(unsigned long long)*n);
  if(inverse){
    CHECK(cudaMemcpy((void *)gpu_result2, inverseBuffer.ntt_b, n*sizeof(unsigned long long), cudaMemcpyDeviceToHost));  
  }else{
    CHECK(cudaMemcpy((void *)gpu_result2, forwardBuffer.ntt_b, n*sizeof(unsigned long long), cudaMemcpyDeviceToHost));  
  }
  std::cout << "ntt_b\n";
  for(long i = 0; i < 5; i++)
  {
    std::cout << gpu_result2[i] << " ";
  }
  std::cout<<std::endl;
#endif

	HELIB_NTIMER_START(KernelMulMod);
  if(inverse)
    KernelMulMod<<<num_blocks*2, THREADS_PER_BLOCK, 0, stream>>>(x_dev, inverseBuffer.ntt_b, q, n); 
	else
    KernelMulMod<<<num_blocks*2, THREADS_PER_BLOCK, 0, stream>>>(x_dev, forwardBuffer.ntt_b, q, n);
  HELIB_NTIMER_STOP(KernelMulMod);

#if debug_blue
  unsigned long long *gpu_result25 = (unsigned long long*) malloc(sizeof(unsigned long long)*n);
  CHECK(cudaMemcpy((void *)gpu_result25, x_dev, n*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  std::cout << "ntt_x * ntt_b:\n";
  for(long i = 0; i < 5; i++)
  {
    std::cout << gpu_result25[i] << " ";
  }
  std::cout<<std::endl;
#endif

	HELIB_NTIMER_START(KernelNTT_inv);
  GSBasedINTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK, 2048 * sizeof(unsigned long long), stream>>>(x_dev, q, mu, bit_length, gpu_ipowers_dev, n, n/2);
  if(n>2048){
    #pragma unroll
    // for (int n_of_groups = n/2; n_of_groups >= 1; n_of_groups /= 2) 
    for (int n_of_groups = (n/2048)/2; n_of_groups >= 1; n_of_groups /= 2) 
    {
        GSBasedINTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(x_dev, q, mu, bit_length, gpu_ipowers_dev, n, n_inv, n_of_groups);
    }
  }
  HELIB_NTIMER_STOP(KernelNTT_inv);

#if debug_blue 
  unsigned long long *gpu_result3 = (unsigned long long*) malloc(sizeof(unsigned long long)*n);
  CHECK(cudaMemcpy((void *)gpu_result3, x_dev, n*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  std::cout << "result:\n";
  for(long i = 0; i < 5; i++)
  {
    std::cout << gpu_result3[i] << " ";
  }
  std::cout<<std::endl;
#endif
}

void gpu_fused_polymul(unsigned long long x_dev[], unsigned long long q, bool inverse)
{
  GPU_Buffer forwardBuffer = GetCModBuffer(q, false); //Forward buffer
  GPU_Buffer inverseBuffer = GetCModBuffer(q, true); //Inverse buffer

  unsigned long long *gpu_powers_dev;
  unsigned long long *gpu_ipowers_dev;
  long n = forwardBuffer.k2;
  long n_inv = forwardBuffer.k2_inv;

  //Assign the buffer
  if(inverse)
  {
    //caller: iFFT
    gpu_powers_dev = inverseBuffer.gpu_powers_dev;
    gpu_ipowers_dev = forwardBuffer.gpu_powers_dev;
  }
  else
  {
    //caller: FFT_aux
    gpu_powers_dev = forwardBuffer.gpu_powers_dev;
    gpu_ipowers_dev = inverseBuffer.gpu_powers_dev;

  }

  // int size_array = sizeof(unsigned long long) * n;
  unsigned int bit_length = forwardBuffer.bit_length;

  //generate parameters for barrett
  unsigned long long mu = forwardBuffer.mu;

#if debug_blue 
  unsigned long long *gpu_result0 = (unsigned long long*) malloc(sizeof(unsigned long long)*n);
  CHECK(cudaMemcpy(gpu_result0, x_dev, n*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  std::cout << "q " << q <<"\n";
  std::cout << "x_dev\n";
  for(long i = 0; i < 5; i++)
  {
    std::cout << gpu_result0[i] << " ";
  }
  std::cout<<std::endl;
#endif

	HELIB_NTIMER_START(KernelNTT);
  int num_blocks = n/(THREADS_PER_BLOCK*2);
  int n_of_groups=1;

  #if USE_GLOBAL_MUTEX
    void *kernelArgs[] = { (void *)&x_dev, (void *)&q, (void *)&mu, (void *)&bit_length, (void *)&gpu_powers_dev, (void *)&n};
    cudaLaunchCooperativeKernel((void *)CTBasedNTTInnerSingleGlobalMutex, num_blocks, THREADS_PER_BLOCK, kernelArgs);

    n_of_groups = n / 2048;
  #elif USE_LOCAL_SYNC
    //this code need to be compiled in debug mode to run without deadlock
    int *trigger_vectors;
    CHECK(cudaMalloc(&trigger_vectors, n * sizeof(int)));
    cudaMemset(trigger_vectors, 0, n * sizeof(int));
    CTBasedNTTInnerSingleKernel<<<num_blocks, THREADS_PER_BLOCK>>>(x_dev, q, mu, bit_length, gpu_powers_dev, n, trigger_vectors);
    CHECK(cudaFree(trigger_vectors));
    CHECK(cudaGetLastError());

    n_of_groups = n / 2048;
  #else
    #pragma unroll
    for (n_of_groups = 1; (n/n_of_groups>2048); n_of_groups *= 2) //Ardhi: do the ntt until the working size is 2048 elements
    {    
        // void *kernelArgs[] = { (void *)&x_dev, (void *)&q, (void *)&mu, (void *)&bit_length, (void *)&gpu_powers_dev, (void *)&n, (void *)&n_of_groups};
        // cudaLaunchCooperativeKernel((void *)CTBasedNTTInnerSingleK, num_blocks, THREADS_PER_BLOCK, kernelArgs);
        CTBasedNTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK>>>(x_dev, q, mu, bit_length, gpu_powers_dev, n, n_of_groups);
    }
  #endif

  CTBasedNTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK, 2048 * sizeof(unsigned long long)>>>(x_dev, q, mu, bit_length, gpu_powers_dev, n_of_groups);
	HELIB_NTIMER_STOP(KernelNTT);

#if debug_blue 
  unsigned long long *gpu_result = (unsigned long long*) malloc(sizeof(unsigned long long)*n);
  CHECK(cudaMemcpy(gpu_result, x_dev, n*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  std::cout << "ntt x_dev\n";
  for(long i = 0; i < 5; i++)
  {
    std::cout << gpu_result[i] << " ";
  }
  std::cout<<std::endl;

  unsigned long long *gpu_result2 = (unsigned long long*) malloc(sizeof(unsigned long long)*n);
  if(inverse){
    CHECK(cudaMemcpy((void *)gpu_result2, inverseBuffer.ntt_b, n*sizeof(unsigned long long), cudaMemcpyDeviceToHost));  
  }else{
    CHECK(cudaMemcpy((void *)gpu_result2, forwardBuffer.ntt_b, n*sizeof(unsigned long long), cudaMemcpyDeviceToHost));  
  }
  std::cout << "ntt_b\n";
  for(long i = 0; i < 5; i++)
  {
    std::cout << gpu_result2[i] << " ";
  }
  std::cout<<std::endl;
#endif

	HELIB_NTIMER_START(KernelMulMod);
  if(inverse)
    KernelMulMod<<<num_blocks*2, THREADS_PER_BLOCK>>>(x_dev, inverseBuffer.ntt_b, q, n); 
	else
    KernelMulMod<<<num_blocks*2, THREADS_PER_BLOCK>>>(x_dev, forwardBuffer.ntt_b, q, n);
  HELIB_NTIMER_STOP(KernelMulMod);

#if debug_blue 
  unsigned long long *gpu_result25 = (unsigned long long*) malloc(sizeof(unsigned long long)*n);
  CHECK(cudaMemcpy((void *)gpu_result25, x_dev, n*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  std::cout << "ntt_x * ntt_b:\n";
  for(long i = 0; i < 5; i++)
  {
    std::cout << gpu_result25[i] << " ";
  }
  std::cout<<std::endl;
#endif

	HELIB_NTIMER_START(KernelNTT_inv);
  GSBasedINTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK, 2048 * sizeof(unsigned long long)>>>(x_dev, q, mu, bit_length, gpu_ipowers_dev, n, n/2);
  if(n>2048){
    #pragma unroll
    // for (int n_of_groups = n/2; n_of_groups >= 1; n_of_groups /= 2) 
    for (int n_of_groups = (n/2048)/2; n_of_groups >= 1; n_of_groups /= 2) 
    {
        GSBasedINTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK>>>(x_dev, q, mu, bit_length, gpu_ipowers_dev, n, n_inv, n_of_groups);
    }
  }
  HELIB_NTIMER_STOP(KernelNTT_inv);

#if debug_blue 
  unsigned long long *gpu_result3 = (unsigned long long*) malloc(sizeof(unsigned long long)*n);
  CHECK(cudaMemcpy((void *)gpu_result3, x_dev, n*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  std::cout << "result:\n";
  for(long i = 0; i < 5; i++)
  {
    std::cout << gpu_result3[i] << " ";
  }
  std::cout<<std::endl;
#endif
}

void gpu_mulMod(NTL::zz_pX& x, unsigned long long p, bool inverse, unsigned long long x_dev[], unsigned long long gpu_powers_m_dev[], 
  long n)
{
    long dx = deg(x);

  HELIB_NTIMER_START(gpu_mulMod_cuMemSet);
    cudaMemset(x_dev, 0, n*8);
  HELIB_NTIMER_STOP(gpu_mulMod_cuMemSet);

  HELIB_NTIMER_START(gpu_mulMod_cpyHD);
    CHECK(cudaMemcpy(x_dev, x.rep.data(), (dx+1)*sizeof(unsigned long long), cudaMemcpyHostToDevice));
  HELIB_NTIMER_STOP(gpu_mulMod_cpyHD);

  HELIB_NTIMER_START(gpu_mulMod_kernel);
    KernelMulMod<<<ceil(((double)dx+1)/THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(x_dev, gpu_powers_m_dev, p, dx+1);
  HELIB_NTIMER_STOP(gpu_mulMod_kernel);
}

//multistream
void gpu_mulMod(NTL::zz_pX& x, unsigned long long p, bool inverse, unsigned long long x_dev[], unsigned long long gpu_powers_m_dev[], 
  long n, cudaStream_t stream)
{
  long dx = deg(x);

  HELIB_NTIMER_START(gpu_mulMod_cuMemSet);
    cudaMemsetAsync(x_dev, 0, n*8, stream);
  HELIB_NTIMER_STOP(gpu_mulMod_cuMemSet);

  HELIB_NTIMER_START(gpu_mulMod_cpyHD);
    CHECK(cudaMemcpyAsync(x_dev, x.rep.data(), (dx+1)*sizeof(unsigned long long), cudaMemcpyHostToDevice, stream));
  HELIB_NTIMER_STOP(gpu_mulMod_cpyHD);

  HELIB_NTIMER_START(gpu_mulMod_kernel);
    KernelMulMod<<<ceil(((double)dx+1)/THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, stream>>>(x_dev, gpu_powers_m_dev, p, dx+1);
  HELIB_NTIMER_STOP(gpu_mulMod_kernel);
}

void gpu_addMod(unsigned long long x_dev[], long n, unsigned long long p)
{
      long dx = 2*(n-1)+1;
      int n_blocks= ceil(((double)(dx+1) - n + 1) / 512);
      add_mod_kernel<<<n_blocks, 512>>>(x_dev, n, dx, p);
}

//multistream
void gpu_addMod(unsigned long long x_dev[], long n, unsigned long long p, cudaStream_t stream)
{
      long dx = 2*(n-1)+1;
      int n_blocks= ceil(((double)(dx+1) - n + 1) / 512);
      add_mod_kernel<<<n_blocks, 512, 0, stream>>>(x_dev, n, dx, p);
}

//multistream
void gpu_mulMod2(unsigned long long x_dev[], unsigned long long p, int n, unsigned long long gpu_powers_m_dev[], bool inverse, cudaStream_t stream)
{
    KernelMulMod<<<ceil((double)n/THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, stream>>>(x_dev, gpu_powers_m_dev, p, n);
}

void gpu_mulMod2(unsigned long long x_dev[], unsigned long long p, int n, unsigned long long gpu_powers_m_dev[], bool inverse)
{
    KernelMulMod<<<ceil((double)n/THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(x_dev, gpu_powers_m_dev, p, n);
}

#endif //Ardhi: End of blocking function

void initializeStreams(long n_streams, std::vector<cudaStream_t> &streams){
  for(int i=0; i<n_streams; i++){
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    streams.push_back(stream);
  }
}

void usecuFFT(std::vector<std::complex<double>>& input_buf, long m){
  int idx = getBufferIndex();

  CHECK(cudaMalloc(&buf[idx].buf_dev, m*sizeof(cufftDoubleComplex)));

  CHECK(cudaMemcpy(buf[idx].buf_dev, input_buf.data(), m*sizeof(std::complex<double>), cudaMemcpyHostToDevice));

  CHECK_CUFFT_ERRORS(cufftExecZ2Z(buf[idx].plan, buf[idx].buf_dev, buf[idx].buf_dev, CUFFT_FORWARD));

  CHECK(cudaMemcpy(input_buf.data(), buf[idx].buf_dev, m*sizeof(std::complex<double>), cudaMemcpyDeviceToHost));

}

void gpu_ntt_forward(unsigned long long res[], long n, NTL::zz_pX& x, unsigned long long q, 
  unsigned long long psi_powers[])
{
  unsigned int bit_length = ceil(std::log2(q));

  HELIB_NTIMER_START(CudaMemCpyHD);

  //generate parameters for barrett
  unsigned long long mu = computeMu(bit_length, q);

  long dx = deg(x);
  unsigned long long *a = (unsigned long long *) malloc(sizeof(unsigned long long) * n);
  // unsigned long long a[n];

  for(int i=0; i < n; i++)
    if(i<=dx)
      a[i] = NTL::rep(x.rep[i]);
    else
      a[i]=0;

  unsigned long long *tmp;
  //Ardhi: call cudamalloc for every CModulus creation maybe slow, but it will prevent race conditions
  CHECK(cudaMalloc(&tmp, n * sizeof(unsigned long long)));

  CHECK(cudaMemcpy(tmp, a, n * sizeof(unsigned long long), cudaMemcpyHostToDevice));

	HELIB_NTIMER_STOP(CudaMemCpyHD);

	HELIB_NTIMER_START(KernelNTT);
    long n_inv = NTL::InvMod(n, q);
    int num_blocks = n/(THREADS_PER_BLOCK*2);
    int n_of_groups=1;
    #pragma unroll
    // for (int n_of_groups = 1; n_of_groups < n; n_of_groups *= 2)
    for (n_of_groups = 1; (n/n_of_groups>2048); n_of_groups *= 2) //Ardhi: do the ntt until the working size is 2048 elements
    {
        CTBasedNTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK>>>(tmp, q, mu, bit_length, psi_powers, n, n_of_groups);
    }

    CTBasedNTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK, 2048 * sizeof(unsigned long long), 0>>>(tmp, q, mu, bit_length, psi_powers, n_of_groups);
    cudaDeviceSynchronize();  // CPU being a gentleman, and waiting for GPU to finish it's job
	HELIB_NTIMER_STOP(KernelNTT);

    CHECK(cudaMemcpy(res, tmp, n * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    cudaFree(tmp);
    free(a);
}


bool worksWithBlueAccel(long q, long k2_2)
{
  if(q % k2_2 != 1)
    return false;
  else
    return true;
}
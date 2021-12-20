#include <cuda_runtime.h>
#include "facegen.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <mpi.h>


extern int mpi_rank, mpi_size;
extern int num_to_gen;
extern int NETWORK_SIZE_IN_BYTES;

static float* myInput;
static float* myOutput;

static float* gpu_mem_input;
static float* gpu_mem_fm0;
static float* gpu_mem_fm1;
static float* gpu_mem_fm2;
static float* gpu_mem_fm3;
static float* gpu_mem_output;

static float* gpu_mem_proj_w;
static float* gpu_mem_proj_b; 
static float* gpu_mem_bn0_beta;
static float* gpu_mem_bn0_gamma;
static float* gpu_mem_bn0_mean;
static float* gpu_mem_bn0_var;
static float* gpu_mem_tconv1_w; 
static float* gpu_mem_tconv1_b;
static float* gpu_mem_bn1_beta;
static float* gpu_mem_bn1_gamma;
static float* gpu_mem_bn1_mean;
static float* gpu_mem_bn1_var;
static float* gpu_mem_tconv2_w; 
static float* gpu_mem_tconv2_b;
static float* gpu_mem_bn2_beta;
static float* gpu_mem_bn2_gamma;
static float* gpu_mem_bn2_mean;
static float* gpu_mem_bn2_var;
static float* gpu_mem_tconv3_w; 
static float* gpu_mem_tconv3_b;
static float* gpu_mem_bn3_beta;
static float* gpu_mem_bn3_gamma;
static float* gpu_mem_bn3_mean;
static float* gpu_mem_bn3_var;
static float* gpu_mem_tconv4_w; 
static float* gpu_mem_tconv4_b;

__global__ void proj(float *in, float *out, float *weight, float *bias, int C, int K) {
  int k = blockDim.x * blockIdx.x + threadIdx.x;
  if (k >= K) return;
  float s = 0;
  for (int c = 0; c < C; ++c) {
    s += in[c] * weight[c * K + k];
  }
  s += bias[k];
  out[k] = s;
}

__global__ void batch_norm(float *inout, float *beta, float *gamma, float *mean, float *var, int HW, int C) {
  int hw = blockDim.x * blockIdx.x + threadIdx.x;
  int c = blockDim.y * blockIdx.y + threadIdx.y;
  if (hw >= HW || c >= C) return;
  float scaled_gamma = gamma[c] / sqrtf(var[c] + 1e-5);
  inout[hw * C + c] = scaled_gamma * inout[hw * C + c] + (beta[c] - scaled_gamma * mean[c]);
}

__global__ void relu(float *inout, int HWC) {
  int hwc = blockDim.x * blockIdx.x + threadIdx.x;
  if (hwc >= HWC) return;
  inout[hwc] = fmaxf(inout[hwc], 0);
}

__global__ void init_c(float *out,int H_IN, int W_IN, int K) {
  int H_OUT = H_IN * 2, W_OUT = W_IN * 2;
  int h_out = blockDim.x * blockIdx.x + threadIdx.x;
  int w_out = blockDim.y * blockIdx.y + threadIdx.y;
  int k = blockDim.z * blockIdx.z + threadIdx.z;
  if (h_out >= H_OUT || w_out >= W_OUT || k >= K) return;
  out[(h_out * W_OUT + w_out) * K + k] = 0;
}

__global__ void tconv(float *in, float *out, float *weight, float *bias, int H_IN, int W_IN, int C, int K) {
  int H_OUT = H_IN * 2, W_OUT = W_IN * 2;
  int h_out = blockDim.x * blockIdx.x + threadIdx.x;
  int c = blockDim.y * blockIdx.y + threadIdx.y;
  int k = blockDim.z * blockIdx.z + threadIdx.z;
  if (h_out >= H_OUT || c >= C  || k >= K) return;
  for (int w_out = 0; w_out < W_OUT; ++w_out) {
    float ss = 0;
    for (int r = 0; r < 5; ++r) {
      for (int s = 0; s < 5; ++s) {
        int h_in = h_out - 3 + r;
        int w_in = w_out - 3 + s;
        if (h_in % 2 == 0 && w_in % 2 == 0) {
          h_in /= 2;
          w_in /= 2;
          if (0 <= h_in && h_in < H_IN && 0 <= w_in && w_in < W_IN) {
            ss += in[(h_in * W_IN + w_in) * C + c] * weight[(((4 - r) * 5 + (4 - s)) * K + k) * C + c];
          }
        }
      }
    }
    if (c == 0) ss += bias[k];
    atomicAdd(&out[(h_out * W_OUT + w_out) * K + k], ss);
  }
}

__global__ void tanh_layer(float *inout, int HWC) {
  int hwc = blockDim.x * blockIdx.x + threadIdx.x;
  if (hwc >= HWC) return;
  inout[hwc] = tanhf(inout[hwc]);
}

void facegen_init() {

  int division = num_to_gen / mpi_size;
  if (num_to_gen % mpi_size > mpi_rank) division++;
  
  if (mpi_rank != 0) {
    myInput = (float *) aligned_alloc(64, division * 100 * sizeof(float));
    myOutput = (float *) aligned_alloc(64, division * 64 * 64 * 3 * sizeof(float));
  }

  cudaMalloc(&gpu_mem_input, division * 100 * sizeof(float));
  cudaMalloc(&gpu_mem_fm0, 4 * 4 * 512 * sizeof(float));
  cudaMalloc(&gpu_mem_fm1, 8 * 8 * 256 * sizeof(float));
  cudaMalloc(&gpu_mem_fm2, 16 * 16 * 128 * sizeof(float));
  cudaMalloc(&gpu_mem_fm3, 32 * 32 * 64 * sizeof(float));
  cudaMalloc(&gpu_mem_output, division * 64 * 64 * 3 * sizeof(float));

  cudaMalloc(&gpu_mem_proj_w, 100 * 8192 * sizeof(float));
  cudaMalloc(&gpu_mem_proj_b, 8192 * sizeof(float));
  cudaMalloc(&gpu_mem_bn0_beta, 512 * sizeof(float));
  cudaMalloc(&gpu_mem_bn0_gamma, 512 * sizeof(float));
  cudaMalloc(&gpu_mem_bn0_mean, 512 * sizeof(float));
  cudaMalloc(&gpu_mem_bn0_var, 512 * sizeof(float));
  cudaMalloc(&gpu_mem_tconv1_w, 5 * 5 * 256 * 512 * sizeof(float));
  cudaMalloc(&gpu_mem_tconv1_b, 256 * sizeof(float));
  cudaMalloc(&gpu_mem_bn1_beta, 256 * sizeof(float));
  cudaMalloc(&gpu_mem_bn1_gamma, 256 * sizeof(float));
  cudaMalloc(&gpu_mem_bn1_mean, 256 * sizeof(float));
  cudaMalloc(&gpu_mem_bn1_var, 256 * sizeof(float));
  cudaMalloc(&gpu_mem_tconv2_w, 5 * 5 * 128 * 256 * sizeof(float));
  cudaMalloc(&gpu_mem_tconv2_b, 128 * sizeof(float));
  cudaMalloc(&gpu_mem_bn2_beta, 128 * sizeof(float));
  cudaMalloc(&gpu_mem_bn2_gamma, 128 * sizeof(float));
  cudaMalloc(&gpu_mem_bn2_mean, 128 * sizeof(float));
  cudaMalloc(&gpu_mem_bn2_var, 128 * sizeof(float));
  cudaMalloc(&gpu_mem_tconv3_w, 5 * 5 * 64 * 128 * sizeof(float));
  cudaMalloc(&gpu_mem_tconv3_b, 64 * sizeof(float));
  cudaMalloc(&gpu_mem_bn3_beta, 64 * sizeof(float));
  cudaMalloc(&gpu_mem_bn3_gamma, 64 * sizeof(float));
  cudaMalloc(&gpu_mem_bn3_mean, 64 * sizeof(float));
  cudaMalloc(&gpu_mem_bn3_var, 64 * sizeof(float));
  cudaMalloc(&gpu_mem_tconv4_w, 5 * 5 * 3 * 64 * sizeof(float));
  cudaMalloc(&gpu_mem_tconv4_b, 3 * sizeof(float));

}

void facegen(int num_to_gen, float *network, float *inputs, float *outputs) {

  int division = num_to_gen / mpi_size;
  if (num_to_gen % mpi_size > mpi_rank) division++;

  if (mpi_rank == 0) {
    myInput = inputs;
    myOutput = outputs;
    int idx = division;
    for (int i = 1; i < mpi_size; ++i) {
      int div = num_to_gen / mpi_size;
      if (num_to_gen % mpi_size > i) div++;
      MPI_Send(inputs + idx * 100, div * 100, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
      idx += div;
      MPI_Send(network, NETWORK_SIZE_IN_BYTES / sizeof(float), MPI_INT, i, 0, MPI_COMM_WORLD);
    }
  } else {
    network = (float*)malloc(NETWORK_SIZE_IN_BYTES);
    MPI_Recv(myInput, division * 100, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, NULL);
    MPI_Recv(network, NETWORK_SIZE_IN_BYTES / sizeof(float), MPI_INT, 0, 0, MPI_COMM_WORLD, NULL);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  float *proj_w = network; network += 100 * 8192;
  float *proj_b = network; network += 8192;
  float *bn0_beta = network; network += 512;
  float *bn0_gamma = network; network += 512;
  float *bn0_mean = network; network += 512;
  float *bn0_var = network; network += 512;
  float *tconv1_w = network; network += 5 * 5 * 256 * 512;
  float *tconv1_b = network; network += 256;
  float *bn1_beta = network; network += 256;
  float *bn1_gamma = network; network += 256;
  float *bn1_mean = network; network += 256;
  float *bn1_var = network; network += 256;
  float *tconv2_w = network; network += 5 * 5 * 128 * 256;
  float *tconv2_b = network; network += 128;
  float *bn2_beta = network; network += 128;
  float *bn2_gamma = network; network += 128;
  float *bn2_mean = network; network += 128;
  float *bn2_var = network; network += 128;
  float *tconv3_w = network; network += 5 * 5 * 64 * 128;
  float *tconv3_b = network; network += 64;
  float *bn3_beta = network; network += 64;
  float *bn3_gamma = network; network += 64;
  float *bn3_mean = network; network += 64;
  float *bn3_var = network; network += 64;
  float *tconv4_w = network; network += 5 * 5 * 3 * 64;
  float *tconv4_b = network; network += 3;
  
  
  cudaMemcpy(gpu_mem_input, myInput, division * 100 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mem_proj_w, proj_w, 100 * 8192 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mem_proj_b, proj_b, 8192 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mem_bn0_beta, bn0_beta, 512 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mem_bn0_gamma, bn0_gamma, 512 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mem_bn0_mean, bn0_mean, 512 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mem_bn0_var, bn0_var, 512 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mem_tconv1_w, tconv1_w, 5 * 5 * 256 * 512 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mem_tconv1_b, tconv1_b, 256 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mem_bn1_beta, bn1_beta, 256 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mem_bn1_gamma, bn1_gamma, 256 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mem_bn1_mean, bn1_mean, 256 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mem_bn1_var, bn1_var, 256 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mem_tconv2_w, tconv2_w, 5 * 5 * 128 * 256 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mem_tconv2_b, tconv2_b, 128 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mem_bn2_beta, bn2_beta, 128 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mem_bn2_gamma, bn2_gamma, 128 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mem_bn2_mean, bn2_mean, 128 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mem_bn2_var, bn2_var, 128 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mem_tconv3_w, tconv3_w, 5 * 5 * 64 * 128 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mem_tconv3_b, tconv3_b, 64 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mem_bn3_beta, bn3_beta, 64 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mem_bn3_gamma, bn3_gamma, 64 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mem_bn3_mean, bn3_mean, 64 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mem_bn3_var, bn3_var, 64 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mem_tconv4_w, tconv4_w, 5 * 5 * 3 * 64 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_mem_tconv4_b, tconv4_b, 3 * sizeof(float), cudaMemcpyHostToDevice);
  
  cudaDeviceSynchronize();

  dim3 blockDim(1, 16, 16); // number of blocks in a grid
  dim3 normDim(16, 64);

  #pragma omp parallel for num_threads(32) nowait
  for (int n = 0; n < division; ++n) {

    proj<<<32, 256>>>(gpu_mem_input + n * 100, gpu_mem_fm0, gpu_mem_proj_w, gpu_mem_proj_b, 100, 8192);
    dim3 norm1(1, 8);
    batch_norm<<<norm1, normDim>>>(gpu_mem_fm0, gpu_mem_bn0_beta, gpu_mem_bn0_gamma, gpu_mem_bn0_mean, gpu_mem_bn0_var, 4 * 4, 512);
    relu<<<32,256>>>(gpu_mem_fm0, 4 * 4 * 512);

    dim3 gridDim1(8, 8, 512);
    init_c<<<gridDim1, blockDim>>>(gpu_mem_fm1, 4, 4, 256);
    dim3 gridDim11(8, 256, 128);
    tconv<<<gridDim11, blockDim>>>(gpu_mem_fm0, gpu_mem_fm1, gpu_mem_tconv1_w, gpu_mem_tconv1_b, 4, 4, 512, 256);

    dim3 norm2(4, 4);
    batch_norm<<<norm2, normDim>>>(gpu_mem_fm1, gpu_mem_bn1_beta, gpu_mem_bn1_gamma, gpu_mem_bn1_mean, gpu_mem_bn1_var, 8 * 8, 256);
    relu<<<64,256>>>(gpu_mem_fm1, 8 * 8 * 256);

    dim3 gridDim2(16, 16, 256);
    init_c<<<gridDim2, blockDim>>>(gpu_mem_fm2, 8, 8, 128);
    dim3 gridDim22(16, 256, 128);
    tconv<<<gridDim22, blockDim>>>(gpu_mem_fm1, gpu_mem_fm2, gpu_mem_tconv2_w, gpu_mem_tconv2_b, 8, 8, 256, 128);

    dim3 norm3(16, 2);
    batch_norm<<<norm3,normDim>>>(gpu_mem_fm2, gpu_mem_bn2_beta, gpu_mem_bn2_gamma, gpu_mem_bn2_mean, gpu_mem_bn2_var, 16 * 16, 128);
    relu<<<128,256>>>(gpu_mem_fm2, 16 * 16 * 128);

    dim3 gridDim3(32, 32, 128);
    init_c<<<gridDim3, blockDim>>>(gpu_mem_fm3, 16, 16, 64);
    dim3 gridDim33(32, 128, 64);
    tconv<<<gridDim33, blockDim>>>(gpu_mem_fm2, gpu_mem_fm3, gpu_mem_tconv3_w, gpu_mem_tconv3_b, 16, 16, 128, 64);

    dim3 norm4(64, 1);
    batch_norm<<<norm4, normDim>>>(gpu_mem_fm3, gpu_mem_bn3_beta, gpu_mem_bn3_gamma, gpu_mem_bn3_mean, gpu_mem_bn3_var, 32 * 32, 64);
    relu<<<256,256>>>(gpu_mem_fm3, 32 * 32 * 64);

    dim3 gridDim4(64, 64, 64);
    init_c<<<gridDim4, blockDim>>>(gpu_mem_output + n * 64 * 64 * 3, 32, 32, 3);
    dim3 gridDim44(64, 64, 3);
    tconv<<<gridDim44, blockDim>>>(gpu_mem_fm3, gpu_mem_output + n * 64 * 64 * 3, gpu_mem_tconv4_w, gpu_mem_tconv4_b, 32, 32, 64, 3);

    tanh_layer<<<48,256>>>(gpu_mem_output + n * 64 * 64 * 3, 64 * 64 * 3);
  }

  cudaMemcpy(myOutput, gpu_mem_output, division * 64 * 64 * 3 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  if (mpi_rank == 0) {
    int idx = division;
    for (int i = 1; i < mpi_size; ++i) {
      int div = num_to_gen / mpi_size;
      if (num_to_gen % mpi_size > i) div++;
      MPI_Recv(myOutput + idx * 64 * 64 * 3, div * 64 * 64 * 3, MPI_FLOAT, i, 0, MPI_COMM_WORLD, NULL);
      idx += div;
    }
  } else {
    MPI_Send(myOutput, division * 64 * 64 * 3, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

void facegen_fin() {

  cudaFree(gpu_mem_input);
  cudaFree(gpu_mem_fm0);
  cudaFree(gpu_mem_fm1);
  cudaFree(gpu_mem_fm2);
  cudaFree(gpu_mem_fm3);
  cudaFree(gpu_mem_output);
  cudaFree(gpu_mem_proj_w);
  cudaFree(gpu_mem_proj_b);
  cudaFree(gpu_mem_bn0_beta);
  cudaFree(gpu_mem_bn0_gamma);
  cudaFree(gpu_mem_bn0_mean);
  cudaFree(gpu_mem_bn0_var);
  cudaFree(gpu_mem_tconv1_w);
  cudaFree(gpu_mem_tconv1_b);
  cudaFree(gpu_mem_bn1_beta);
  cudaFree(gpu_mem_bn1_gamma);
  cudaFree(gpu_mem_bn1_mean);
  cudaFree(gpu_mem_bn1_var);
  cudaFree(gpu_mem_tconv2_w);
  cudaFree(gpu_mem_tconv2_b);
  cudaFree(gpu_mem_bn2_beta);
  cudaFree(gpu_mem_bn2_gamma);
  cudaFree(gpu_mem_bn2_mean);
  cudaFree(gpu_mem_bn2_var);
  cudaFree(gpu_mem_tconv3_w);
  cudaFree(gpu_mem_tconv3_b);
  cudaFree(gpu_mem_bn3_beta);
  cudaFree(gpu_mem_bn3_gamma);
  cudaFree(gpu_mem_bn3_mean);
  cudaFree(gpu_mem_bn3_var);
  cudaFree(gpu_mem_tconv4_w);
  cudaFree(gpu_mem_tconv4_b);
}


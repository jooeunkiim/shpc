#include <cuda_runtime.h>
#include "facegen.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <mpi.h>

#include "cuda_util.h"

/*
 * TODO
 * Define global variables here.
 */

extern int mpi_rank, mpi_size;
extern int num_to_gen;

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
  for (int k = 0; k < K; ++k) {
    float s = 0;
    for (int c = 0; c < C; ++c) {
      s += in[c] * weight[c * K + k];
    }
    s += bias[k];
    out[k] = s;
  }
}

__global__ void batch_norm(float *inout, float *beta, float *gamma, float *mean, float *var, int HW, int C) {
  for (int hw = 0; hw < HW; ++hw) {
    for (int c = 0; c < C; ++c) {
      float scaled_gamma = gamma[c] / sqrtf(var[c] + 1e-5);
      inout[hw * C + c] = scaled_gamma * inout[hw * C + c] + (beta[c] - scaled_gamma * mean[c]);
    }
  }
}

__global__ void relu(float *inout, int HWC) {
  for (int hwc = 0; hwc < HWC; ++hwc) {
    inout[hwc] = fmaxf(inout[hwc], 0);
  }
}

__global__ void org_tconv(float *in, float *out, float *weight, float *bias, int H_IN, int W_IN, int C, int K) {
  int H_OUT = H_IN * 2, W_OUT = W_IN * 2;
  for (int h_out = 0; h_out < H_OUT; ++h_out) {
    for (int w_out = 0; w_out < W_OUT; ++w_out) {
      for (int k = 0; k < K; ++k) {
        float ss = 0;
        for (int r = 0; r < 5; ++r) {
          for (int s = 0; s < 5; ++s) {
            // top and left side has padding 3, bottom and right side has padding 2
            // so subtract 3
            int h_in = h_out - 3 + r;
            int w_in = w_out - 3 + s;
            // stride is 2, so check coordinates fall into input element or empty space
            if (h_in % 2 == 0 && w_in % 2 == 0) {
              h_in /= 2;
              w_in /= 2;
              // boundary check
              if (0 <= h_in && h_in < H_IN && 0 <= w_in && w_in < W_IN) {
                for (int c = 0; c < C; ++c) {
                  // filter is stored in reverse; so use [4 - r][4 - s] instead of [r][s]
                  // ss += in[h_in][w_in][c] * weight[4 - r][4 - s][k][c];
                  ss += in[(h_in * W_IN + w_in) * C + c] * weight[(((4 - r) * 5 + (4 - s)) * K + k) * C + c];
                }
              }
            }
          }
        }
        ss += bias[k];
        // out[h_out][w_out][k] = ss;
        out[(h_out * W_OUT + w_out) * K + k] = ss;
      }
    }
  }
}

// C를 병렬화해보기..!
__global__ void tconv(float *in, float *out, float *weight, float *bias, int H_IN, int W_IN, int C, int K) {
  int H_OUT = H_IN * 2, W_OUT = W_IN * 2;
  int h_out = blockDim.x * blockIdx.x + threadIdx.x;
  int w_out = blockDim.y * blockIdx.y + threadIdx.y;
  int k = blockDim.z * blockIdx.z + threadIdx.z;
  if (h_out >= H_OUT || w_out >= W_OUT || k >= K) return;
  // for (int k = 0; k < K; ++k) {
  float ss = 0;
  for (int r = 0; r < 5; ++r) {
    for (int s = 0; s < 5; ++s) {
      // top and left side has padding 3, bottom and right side has padding 2
      // so subtract 3
      int h_in = h_out - 3 + r;
      int w_in = w_out - 3 + s;
      // stride is 2, so check coordinates fall into input element or empty space
      if (h_in % 2 == 0 && w_in % 2 == 0) {
        h_in /= 2;
        w_in /= 2;
        // boundary check
        if (0 <= h_in && h_in < H_IN && 0 <= w_in && w_in < W_IN) {
          for (int c = 0; c < C; ++c) {
            // filter is stored in reverse; so use [4 - r][4 - s] instead of [r][s]
            // ss += in[h_in][w_in][c] * weight[4 - r][4 - s][k][c];
            ss += in[(h_in * W_IN + w_in) * C + c] * weight[(((4 - r) * 5 + (4 - s)) * K + k) * C + c];
          }
        }
      }
    }
  }
  ss += bias[k];
  // out[h_out][w_out][k] = ss;
  out[(h_out * W_OUT + w_out) * K + k] = ss;
  // }
}

__global__ void tanh_layer(float *inout, int HWC) {
  for (int hwc = 0; hwc < HWC; ++hwc) {
    inout[hwc] = tanhf(inout[hwc]);
  }
}

void facegen_init() {
  /*
   * TODO
   * Initialize required CUDA objects. For example,
   * cudaMalloc(...)
   */

  cudaMalloc(&gpu_mem_input, num_to_gen * 100 * sizeof(float));
  cudaMalloc(&gpu_mem_fm0, 4 * 4 * 512 * sizeof(float));
  cudaMalloc(&gpu_mem_fm1, 8 * 8 * 256 * sizeof(float));
  cudaMalloc(&gpu_mem_fm2, 16 * 16 * 128 * sizeof(float));
  cudaMalloc(&gpu_mem_fm3, 32 * 32 * 64 * sizeof(float));
  cudaMalloc(&gpu_mem_output, num_to_gen * 64 * 64 * 3 * sizeof(float));

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

  cudaDeviceSynchronize();
}

void facegen(int num_to_gen, float *network, float *inputs, float *outputs) {
  /*
   * TODO
   * Implement facegen computation here.
   * See "facegen_seq.c" if you don't know what to do.
   *
   * Below functions should be implemented in here:
   * Host-to-devie memory copy,
   * CUDA kernel launch,
   * Device-to-host memory copy
   */

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
  
  cudaMemcpy(gpu_mem_input, inputs, num_to_gen * 100 * sizeof(float), cudaMemcpyHostToDevice);
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

  dim3 blockDim(1, 8, 8); // number of blocks in a grid
  // dim3 gridDim(64, 64, 64); // number of threads in a block

  #pragma omp parallel num_threads(64) shared(gpu_mem_input, gpu_mem_output)
  #pragma omp for nowait
  for (int n = 0; n < num_to_gen; ++n) {

    proj<<<1,1>>>(gpu_mem_input + n * 100, gpu_mem_fm0, gpu_mem_proj_w, gpu_mem_proj_b, 100, 8192);
    batch_norm<<<1,1>>>(gpu_mem_fm0, gpu_mem_bn0_beta, gpu_mem_bn0_gamma, gpu_mem_bn0_mean, gpu_mem_bn0_var, 4 * 4, 512);
    relu<<<1,1>>>(gpu_mem_fm0, 4 * 4 * 512);

    dim3 gridDim1(8, 8, 512);
    tconv<<<gridDim1, blockDim>>>(gpu_mem_fm0, gpu_mem_fm1, gpu_mem_tconv1_w, gpu_mem_tconv1_b, 4, 4, 512, 256);

    batch_norm<<<1,1>>>(gpu_mem_fm1, gpu_mem_bn1_beta, gpu_mem_bn1_gamma, gpu_mem_bn1_mean, gpu_mem_bn1_var, 8 * 8, 256);
    relu<<<1,1>>>(gpu_mem_fm1, 8 * 8 * 256);

    dim3 gridDim2(16, 16, 256);
    tconv<<<gridDim2, blockDim>>>(gpu_mem_fm1, gpu_mem_fm2, gpu_mem_tconv2_w, gpu_mem_tconv2_b, 8, 8, 256, 128);

    batch_norm<<<1,1>>>(gpu_mem_fm2, gpu_mem_bn2_beta, gpu_mem_bn2_gamma, gpu_mem_bn2_mean, gpu_mem_bn2_var, 16 * 16, 128);
    relu<<<1,1>>>(gpu_mem_fm2, 16 * 16 * 128);

    dim3 gridDim3(32, 32, 128);
    tconv<<<gridDim3, blockDim>>>(gpu_mem_fm2, gpu_mem_fm3, gpu_mem_tconv3_w, gpu_mem_tconv3_b, 16, 16, 128, 64);

    batch_norm<<<1,1>>>(gpu_mem_fm3, gpu_mem_bn3_beta, gpu_mem_bn3_gamma, gpu_mem_bn3_mean, gpu_mem_bn3_var, 32 * 32, 64);
    relu<<<1,1>>>(gpu_mem_fm3, 32 * 32 * 64);

    dim3 gridDim4(64, 64, 64);
    tconv<<<gridDim4, blockDim>>>(gpu_mem_fm3, gpu_mem_output + n * 64 * 64 * 3, gpu_mem_tconv4_w, gpu_mem_tconv4_b, 32, 32, 64, 3);

    tanh_layer<<<1,1>>>(gpu_mem_output + n * 64 * 64 * 3, 64 * 64 * 3);
  }

  // CHECK_CUDA(cudaDeviceSynchronize());
  cudaDeviceSynchronize();

  cudaMemcpy(outputs, gpu_mem_output, num_to_gen * 64 * 64 * 3 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

}

void facegen_fin() {
  /*
   * TODO
   * Finalize required CUDA objects. For example,
   * cudaFree(...)
   */
  // cudaFree(gpu_mem_input);
  cudaFree(gpu_mem_fm0);
  cudaFree(gpu_mem_fm1);
  cudaFree(gpu_mem_fm2);
  cudaFree(gpu_mem_fm3);
  cudaFree(gpu_mem_output);
}

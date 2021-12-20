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

const int count = 4;
cudaStream_t streams[count];
int num_per_device[count];

static float* myInput;
static float* myOutput;

static float* gpu_mem_input[count];
static float* gpu_mem_fm0[count];
static float* gpu_mem_fm1[count];
static float* gpu_mem_fm2[count];
static float* gpu_mem_fm3[count];
static float* gpu_mem_output[count];

static float* gpu_mem_proj_w[count];
static float* gpu_mem_proj_b[count];
static float* gpu_mem_bn0_beta[count];
static float* gpu_mem_bn0_gamma[count];
static float* gpu_mem_bn0_mean[count];
static float* gpu_mem_bn0_var[count];
static float* gpu_mem_tconv1_w[count];
static float* gpu_mem_tconv1_b[count];
static float* gpu_mem_bn1_beta[count];
static float* gpu_mem_bn1_gamma[count];
static float* gpu_mem_bn1_mean[count];
static float* gpu_mem_bn1_var[count];
static float* gpu_mem_tconv2_w[count];
static float* gpu_mem_tconv2_b[count];
static float* gpu_mem_bn2_beta[count];
static float* gpu_mem_bn2_gamma[count];
static float* gpu_mem_bn2_mean[count];
static float* gpu_mem_bn2_var[count];
static float* gpu_mem_tconv3_w[count];
static float* gpu_mem_tconv3_b[count];
static float* gpu_mem_bn3_beta[count];
static float* gpu_mem_bn3_gamma[count];
static float* gpu_mem_bn3_mean[count];
static float* gpu_mem_bn3_var[count];
static float* gpu_mem_tconv4_w[count];
static float* gpu_mem_tconv4_b[count];


__global__ void proj(float *in, float *out, float *weight, float *bias, int C, int K) {
  printf("######in[0]=%f#####\n",in[0]);
  for (int k = 0; k < K; ++k) {
    float s = 0;
    for (int c = 0; c < C; ++c) {
      s += in[c] * weight[c * K + k];
    }
    s += bias[k];
    out[k] = s;
  }
  printf("######out[0]=%f#####",out[0]);
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

  int division = num_to_gen / mpi_size;
  if (num_to_gen % mpi_size > mpi_rank) division++;
  
  
  if (mpi_rank != 0) {
    // myInput = (float *) aligned_alloc(64, division * 100 * sizeof(float));
    cudaMallocHost(&myInput, division * 100 * sizeof(float));
    myOutput = (float *) aligned_alloc(64, division * 64 * 64 * 3 * sizeof(float));
  } else {
    cudaMallocHost(&myInput, num_to_gen * 100 * sizeof(float));
  }
  int device_count;
  cudaGetDeviceCount(&device_count);

  for (int device_id=0; device_id < device_count; device_id++) {
    int gen_per_gpu = division / count;
    if (division % count > device_id) gen_per_gpu++;
    num_per_device[device_id] = gen_per_gpu;

    cudaMalloc(&gpu_mem_input[device_id], gen_per_gpu * 100 * sizeof(float));
    cudaMalloc(&gpu_mem_fm0[device_id], 4 * 4 * 512 * sizeof(float));
    cudaMalloc(&gpu_mem_fm1[device_id], 8 * 8 * 256 * sizeof(float));
    cudaMalloc(&gpu_mem_fm2[device_id], 16 * 16 * 128 * sizeof(float));
    cudaMalloc(&gpu_mem_fm3[device_id], 32 * 32 * 64 * sizeof(float));
    cudaMalloc(&gpu_mem_output[device_id], gen_per_gpu * 64 * 64 * 3 * sizeof(float));

    cudaMalloc(&gpu_mem_proj_w[device_id], 100 * 8192 * sizeof(float));
    cudaMalloc(&gpu_mem_proj_b[device_id], 8192 * sizeof(float));
    cudaMalloc(&gpu_mem_bn0_beta[device_id], 512 * sizeof(float));
    cudaMalloc(&gpu_mem_bn0_gamma[device_id], 512 * sizeof(float));
    cudaMalloc(&gpu_mem_bn0_mean[device_id], 512 * sizeof(float));
    cudaMalloc(&gpu_mem_bn0_var[device_id], 512 * sizeof(float));
    cudaMalloc(&gpu_mem_tconv1_w[device_id], 5 * 5 * 256 * 512 * sizeof(float));
    cudaMalloc(&gpu_mem_tconv1_b[device_id], 256 * sizeof(float));
    cudaMalloc(&gpu_mem_bn1_beta[device_id], 256 * sizeof(float));
    cudaMalloc(&gpu_mem_bn1_gamma[device_id], 256 * sizeof(float));
    cudaMalloc(&gpu_mem_bn1_mean[device_id], 256 * sizeof(float));
    cudaMalloc(&gpu_mem_bn1_var[device_id], 256 * sizeof(float));
    cudaMalloc(&gpu_mem_tconv2_w[device_id], 5 * 5 * 128 * 256 * sizeof(float));
    cudaMalloc(&gpu_mem_tconv2_b[device_id], 128 * sizeof(float));
    cudaMalloc(&gpu_mem_bn2_beta[device_id], 128 * sizeof(float));
    cudaMalloc(&gpu_mem_bn2_gamma[device_id], 128 * sizeof(float));
    cudaMalloc(&gpu_mem_bn2_mean[device_id], 128 * sizeof(float));
    cudaMalloc(&gpu_mem_bn2_var[device_id], 128 * sizeof(float));
    cudaMalloc(&gpu_mem_tconv3_w[device_id], 5 * 5 * 64 * 128 * sizeof(float));
    cudaMalloc(&gpu_mem_tconv3_b[device_id], 64 * sizeof(float));
    cudaMalloc(&gpu_mem_bn3_beta[device_id], 64 * sizeof(float));
    cudaMalloc(&gpu_mem_bn3_gamma[device_id], 64 * sizeof(float));
    cudaMalloc(&gpu_mem_bn3_mean[device_id], 64 * sizeof(float));
    cudaMalloc(&gpu_mem_bn3_var[device_id], 64 * sizeof(float));
    cudaMalloc(&gpu_mem_tconv4_w[device_id], 5 * 5 * 3 * 64 * sizeof(float));
    cudaMalloc(&gpu_mem_tconv4_b[device_id], 3 * sizeof(float));
  }
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

  int division = num_to_gen / mpi_size;
  if (num_to_gen % mpi_size > mpi_rank) {
    division++;
  }

  if (mpi_rank == 0) {
    myInput = inputs;
    myOutput = outputs;
    int idx = division;
    for (int i = 1; i < mpi_size; ++i) {
      int div = num_to_gen / mpi_size;
      if (num_to_gen % mpi_size > i) div++;
      MPI_Send(inputs + idx * 100, div * 100, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
      idx += div;
    }
  } else {
    MPI_Recv(myInput, division * 100, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, NULL);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  float *proj_w;
  float *proj_b;
  float *bn0_beta;
  float *bn0_gamma;
  float *bn0_mean;
  float *bn0_var;
  float *tconv1_w;
  float *tconv1_b;
  float *bn1_beta;
  float *bn1_gamma;
  float *bn1_mean;
  float *bn1_var;
  float *tconv2_w;
  float *tconv2_b;
  float *bn2_beta;
  float *bn2_gamma;
  float *bn2_mean;
  float *bn2_var;
  float *tconv3_w;
  float *tconv3_b;
  float *bn3_beta;
  float *bn3_gamma;
  float *bn3_mean;
  float *bn3_var;
  float *tconv4_w;
  float *tconv4_b;
  cudaMallocHost(&proj_w, 100 * 8192 * sizeof(float));
  cudaMallocHost(&proj_b, 8192 * sizeof(float));
  cudaMallocHost(&bn0_beta, 512 * sizeof(float));
  cudaMallocHost(&bn0_gamma, 512 * sizeof(float));
  cudaMallocHost(&bn0_mean, 512 * sizeof(float));
  cudaMallocHost(&bn0_var, 512 * sizeof(float));
  cudaMallocHost(&tconv1_w, 5 * 5 * 256 * 512 * sizeof(float));
  cudaMallocHost(&tconv1_b, 256 * sizeof(float));
  cudaMallocHost(&bn1_beta, 256 * sizeof(float));
  cudaMallocHost(&bn1_gamma, 256 * sizeof(float));
  cudaMallocHost(&bn1_mean, 256 * sizeof(float));
  cudaMallocHost(&bn1_var, 256 * sizeof(float));
  cudaMallocHost(&tconv2_w, 5 * 5 * 128 * 256 * sizeof(float));
  cudaMallocHost(&tconv2_b, 128 * sizeof(float));
  cudaMallocHost(&bn2_beta, 128 * sizeof(float));
  cudaMallocHost(&bn2_gamma, 128 * sizeof(float));
  cudaMallocHost(&bn2_mean, 128 * sizeof(float));
  cudaMallocHost(&bn2_var, 128 * sizeof(float));
  cudaMallocHost(&tconv3_w, 5 * 5 * 64 * 128 * sizeof(float));
  cudaMallocHost(&tconv3_b, 64 * sizeof(float));
  cudaMallocHost(&bn3_beta, 64 * sizeof(float));
  cudaMallocHost(&bn3_gamma, 64 * sizeof(float));
  cudaMallocHost(&bn3_mean, 64 * sizeof(float));
  cudaMallocHost(&bn3_var, 64 * sizeof(float));
  cudaMallocHost(&tconv4_w, 5 * 5 * 3 * 64 * sizeof(float));
  cudaMallocHost(&tconv4_b, 3 * sizeof(float));

  proj_w = network; network += 100 * 8192;
  proj_b = network; network += 8192;
  bn0_beta = network; network += 512;
  bn0_gamma = network; network += 512;
  bn0_mean = network; network += 512;
  bn0_var = network; network += 512;
  tconv1_w = network; network += 5 * 5 * 256 * 512;
  tconv1_b = network; network += 256;
  bn1_beta = network; network += 256;
  bn1_gamma = network; network += 256;
  bn1_mean = network; network += 256;
  bn1_var = network; network += 256;
  tconv2_w = network; network += 5 * 5 * 128 * 256;
  tconv2_b = network; network += 128;
  bn2_beta = network; network += 128;
  bn2_gamma = network; network += 128;
  bn2_mean = network; network += 128;
  bn2_var = network; network += 128;
  tconv3_w = network; network += 5 * 5 * 64 * 128;
  tconv3_b = network; network += 64;
  bn3_beta = network; network += 64;
  bn3_gamma = network; network += 64;
  bn3_mean = network; network += 64;
  bn3_var = network; network += 64;
  tconv4_w = network; network += 5 * 5 * 3 * 64;
  tconv4_b = network; network += 3;
  
  cudaDeviceSynchronize();

  dim3 blockDim(8, 8, 8); // number of blocks in a grid
  // dim3 gridDim(1, 1, 64); // number of threads in a block

  int device_count;
  cudaGetDeviceCount(&device_count);

  for (int device_id=0; device_id < device_count; device_id++) {
    // cudaSetDevice(device_id);  
    cudaStreamCreate(&streams[device_id]);
    // cudaStreamCreateWithFlags(&streams[device_id], cudaStreamNonBlocking);
  }

  for (int device_id=0; device_id < device_count; device_id++) {

    int start_idx = 0;
    for (int prv=0; prv < device_id; prv++) start_idx += num_per_device[prv];

    cudaSetDevice(device_id);
    // CHECK_CUDA(cudaStreamCreate(&streams[device_id]));
    // cudaStreamCreateWithFlags(&streams[device_id], cudaStreamNonBlocking);
    
    // cudaMallocHost(&(myInput + start_idx * 100), num_per_device[device_id] * 100 * sizeof(float));
    cudaMemcpyAsync(gpu_mem_input[device_id], myInput + start_idx * 100, num_per_device[device_id] * 100 * sizeof(float), cudaMemcpyHostToDevice, streams[device_id]);
    cudaMemcpyAsync(gpu_mem_proj_w[device_id], proj_w, 100 * 8192 * sizeof(float), cudaMemcpyHostToDevice, streams[device_id]);
    cudaMemcpyAsync(gpu_mem_proj_b[device_id], proj_b, 8192 * sizeof(float), cudaMemcpyHostToDevice, streams[device_id]);
    cudaMemcpyAsync(gpu_mem_bn0_beta[device_id], bn0_beta, 512 * sizeof(float), cudaMemcpyHostToDevice, streams[device_id]);
    cudaMemcpyAsync(gpu_mem_bn0_gamma[device_id], bn0_gamma, 512 * sizeof(float), cudaMemcpyHostToDevice, streams[device_id]);
    cudaMemcpyAsync(gpu_mem_bn0_mean[device_id], bn0_mean, 512 * sizeof(float), cudaMemcpyHostToDevice, streams[device_id]);
    cudaMemcpyAsync(gpu_mem_bn0_var[device_id], bn0_var, 512 * sizeof(float), cudaMemcpyHostToDevice, streams[device_id]);
    cudaMemcpyAsync(gpu_mem_tconv1_w[device_id], tconv1_w, 5 * 5 * 256 * 512 * sizeof(float), cudaMemcpyHostToDevice, streams[device_id]);
    cudaMemcpyAsync(gpu_mem_tconv1_b[device_id], tconv1_b, 256 * sizeof(float), cudaMemcpyHostToDevice, streams[device_id]);
    cudaMemcpyAsync(gpu_mem_bn1_beta[device_id], bn1_beta, 256 * sizeof(float), cudaMemcpyHostToDevice, streams[device_id]);
    cudaMemcpyAsync(gpu_mem_bn1_gamma[device_id], bn1_gamma, 256 * sizeof(float), cudaMemcpyHostToDevice, streams[device_id]);
    cudaMemcpyAsync(gpu_mem_bn1_mean[device_id], bn1_mean, 256 * sizeof(float), cudaMemcpyHostToDevice, streams[device_id]);
    cudaMemcpyAsync(gpu_mem_bn1_var[device_id], bn1_var, 256 * sizeof(float), cudaMemcpyHostToDevice, streams[device_id]);
    cudaMemcpyAsync(gpu_mem_tconv2_w[device_id], tconv2_w, 5 * 5 * 128 * 256 * sizeof(float), cudaMemcpyHostToDevice, streams[device_id]);
    cudaMemcpyAsync(gpu_mem_tconv2_b[device_id], tconv2_b, 128 * sizeof(float), cudaMemcpyHostToDevice, streams[device_id]);
    cudaMemcpyAsync(gpu_mem_bn2_beta[device_id], bn2_beta, 128 * sizeof(float), cudaMemcpyHostToDevice, streams[device_id]);
    cudaMemcpyAsync(gpu_mem_bn2_gamma[device_id], bn2_gamma, 128 * sizeof(float), cudaMemcpyHostToDevice, streams[device_id]);
    cudaMemcpyAsync(gpu_mem_bn2_mean[device_id], bn2_mean, 128 * sizeof(float), cudaMemcpyHostToDevice, streams[device_id]);
    cudaMemcpyAsync(gpu_mem_bn2_var[device_id], bn2_var, 128 * sizeof(float), cudaMemcpyHostToDevice, streams[device_id]);
    cudaMemcpyAsync(gpu_mem_tconv3_w[device_id], tconv3_w, 5 * 5 * 64 * 128 * sizeof(float), cudaMemcpyHostToDevice, streams[device_id]);
    cudaMemcpyAsync(gpu_mem_tconv3_b[device_id], tconv3_b, 64 * sizeof(float), cudaMemcpyHostToDevice, streams[device_id]);
    cudaMemcpyAsync(gpu_mem_bn3_beta[device_id], bn3_beta, 64 * sizeof(float), cudaMemcpyHostToDevice, streams[device_id]);
    cudaMemcpyAsync(gpu_mem_bn3_gamma[device_id], bn3_gamma, 64 * sizeof(float), cudaMemcpyHostToDevice, streams[device_id]);
    cudaMemcpyAsync(gpu_mem_bn3_mean[device_id], bn3_mean, 64 * sizeof(float), cudaMemcpyHostToDevice, streams[device_id]);
    cudaMemcpyAsync(gpu_mem_bn3_var[device_id], bn3_var, 64 * sizeof(float), cudaMemcpyHostToDevice, streams[device_id]);
    cudaMemcpyAsync(gpu_mem_tconv4_w[device_id], tconv4_w, 5 * 5 * 3 * 64 * sizeof(float), cudaMemcpyHostToDevice, streams[device_id]);
    cudaMemcpyAsync(gpu_mem_tconv4_b[device_id], tconv4_b, 3 * sizeof(float), cudaMemcpyHostToDevice, streams[device_id]);
    CHECK_CUDA(cudaDeviceSynchronize());

    #pragma omp parallel num_threads(32)
    #pragma omp for nowait
    for (int n = 0; n < num_per_device[device_id]; ++n) {
      cudaSetDevice(device_id);  
      float* tmp;
      // tmp = (float *) aligned_alloc(64,  100 * sizeof(float));
      // cudaMemcpyAsync(tmp, gpu_mem_input[device_id] + n * 100, 100 * sizeof(float), cudaMemcpyDeviceToHost, streams[device_id]);
      // printf("\ngpu#%d, %dth image: gpu_mem_input[0]=%f\n", device_id, n, tmp[0]);

      // tmp = (float *) aligned_alloc(64, 100 * 8192 * sizeof(float));
      // cudaMemcpyAsync(tmp, gpu_mem_proj_w[device_id], 100 * 8192 * sizeof(float), cudaMemcpyDeviceToHost, streams[device_id]);
      // printf("\ngpu#%d, %dth image: gpu_mem_proj_w[0]=%f\n", device_id, n, tmp[0]);

      tmp = (float *) aligned_alloc(64, 8192 * sizeof(float));
      cudaMemcpyAsync(tmp, gpu_mem_proj_b[device_id], 8192 * sizeof(float), cudaMemcpyDeviceToHost, streams[device_id]);
      printf("\ngpu#%d, %dth image: gpu_mem_proj_b[0]=%f\n", device_id, n, tmp[0]);

      proj<<<1, 1, 0, streams[device_id]>>>(gpu_mem_input[device_id] + n * 100, gpu_mem_fm0[device_id], gpu_mem_proj_w[device_id], gpu_mem_proj_b[device_id], 100, 8192);
      cudaDeviceSynchronize();
      // proj<<<1,1>>>(gpu_mem_input[device_id] + n * 100, gpu_mem_fm0[device_id], gpu_mem_proj_w[device_id], gpu_mem_proj_b[device_id], 100, 8192);
      // CHECK_CUDA(cudaDeviceSynchronize());

      tmp = (float *) aligned_alloc(64, 4 * 4 * 512 * sizeof(float));
      cudaMemcpyAsync(tmp, gpu_mem_fm0[device_id], 4 * 4 * 512 * sizeof(float), cudaMemcpyDeviceToHost, streams[device_id]);
      printf("\ngpu#%d, %dth image: gpu_mem_fm0[0]=%f\n", device_id, n, tmp[0]);
      // cudaMemcpy(tmp, gpu_mem_fm0[device_id], 4 * 4 * 512 * sizeof(float), cudaMemcpyDeviceToHost);

      // tmp = (float *) aligned_alloc(64, 8192 * sizeof(float));
      // cudaMemcpyAsync(tmp, gpu_mem_proj_b[device_id], 8192 * sizeof(float), cudaMemcpyDeviceToHost, streams[device_id]);
      // printf("\ngpu#%d, %dth image: gpu_mem_proj_b[-1]=%f\n", device_id, n, tmp[8191]);
      batch_norm<<<1,1, 0, streams[device_id]>>>(gpu_mem_fm0[device_id], gpu_mem_bn0_beta[device_id], gpu_mem_bn0_gamma[device_id], gpu_mem_bn0_mean[device_id], gpu_mem_bn0_var[device_id], 4 * 4, 512);
      relu<<<1,1, 0, streams[device_id]>>>(gpu_mem_fm0[device_id], 4 * 4 * 512);

      dim3 gridDim1(8, 8, 256);
      tconv<<<gridDim1, blockDim, 0, streams[device_id]>>>(gpu_mem_fm0[device_id], gpu_mem_fm1[device_id], gpu_mem_tconv1_w[device_id], gpu_mem_tconv1_b[device_id], 4, 4, 512, 256);
      // dim3 blockDim1(8, 8, 256);
      // tconv<<<gridDim, blockDim1, streams[device_id]>>>(gpu_mem_fm0[device_id], gpu_mem_fm1[device_id], gpu_mem_tconv1_w[device_id], gpu_mem_tconv1_b[device_id], 4, 4, 512, 256);

      batch_norm<<<1,1, 0, streams[device_id]>>>(gpu_mem_fm1[device_id], gpu_mem_bn1_beta[device_id], gpu_mem_bn1_gamma[device_id], gpu_mem_bn1_mean[device_id], gpu_mem_bn1_var[device_id], 8 * 8, 256);
      relu<<<1,1, 0, streams[device_id]>>>(gpu_mem_fm1[device_id], 8 * 8 * 256);

      dim3 gridDim2(16, 16, 128);
      tconv<<<gridDim2, blockDim, 0, streams[device_id]>>>(gpu_mem_fm1[device_id], gpu_mem_fm2[device_id], gpu_mem_tconv2_w[device_id], gpu_mem_tconv2_b[device_id], 8, 8, 256, 128);
      // dim3 blockDim2(16, 16, 128);
      // tconv<<<gridDim, blockDim2, streams[device_id]>>>(gpu_mem_fm1[device_id], gpu_mem_fm2[device_id], gpu_mem_tconv2_w[device_id], gpu_mem_tconv2_b[device_id], 8, 8, 256, 128);

      batch_norm<<<1,1, 0, streams[device_id]>>>(gpu_mem_fm2[device_id], gpu_mem_bn2_beta[device_id], gpu_mem_bn2_gamma[device_id], gpu_mem_bn2_mean[device_id], gpu_mem_bn2_var[device_id], 16 * 16, 128);
      relu<<<1,1, 0, streams[device_id]>>>(gpu_mem_fm2[device_id], 16 * 16 * 128);

      dim3 gridDim3(32, 32, 64);
      tconv<<<gridDim3, blockDim, 0, streams[device_id]>>>(gpu_mem_fm2[device_id], gpu_mem_fm3[device_id], gpu_mem_tconv3_w[device_id], gpu_mem_tconv3_b[device_id], 16, 16, 128, 64);
      // dim3 blockDim3(32, 32, 64);
      // tconv<<<gridDim, blockDim3, streams[device_id]>>>(gpu_mem_fm2[device_id], gpu_mem_fm3[device_id], gpu_mem_tconv3_w[device_id], gpu_mem_tconv3_b[device_id], 16, 16, 128, 64);

      batch_norm<<<1,1, 0, streams[device_id]>>>(gpu_mem_fm3[device_id], gpu_mem_bn3_beta[device_id], gpu_mem_bn3_gamma[device_id], gpu_mem_bn3_mean[device_id], gpu_mem_bn3_var[device_id], 32 * 32, 64);
      relu<<<1,1, 0, streams[device_id]>>>(gpu_mem_fm3[device_id], 32 * 32 * 64);

      dim3 gridDim4(64, 64, 3);
      tconv<<<gridDim4, blockDim, 0, streams[device_id]>>>(gpu_mem_fm3[device_id], gpu_mem_output[device_id] + n * 64 * 64 * 3, gpu_mem_tconv4_w[device_id], gpu_mem_tconv4_b[device_id], 32, 32, 64, 3);
      // dim3 blockDim4(64, 64, 3);
      // tconv<<<gridDim, blockDim4, streams[device_id]>>>(gpu_mem_fm3[device_id], gpu_mem_output[device_id] + n * 64 * 64 * 3, gpu_mem_tconv4_w[device_id], gpu_mem_tconv4_b[device_id], 32, 32, 64, 3);

      tanh_layer<<<1,1, 0, streams[device_id]>>>(gpu_mem_output[device_id] + n * 64 * 64 * 3, 64 * 64 * 3);

      // tmp = (float *) aligned_alloc(64, 64 * 64 * 3 * sizeof(float));
      // cudaMemcpyAsync(tmp, gpu_mem_output[device_id] + n * 64 * 64 * 3, 64 * 64 * 3 * sizeof(float), cudaMemcpyDeviceToHost, streams[device_id]);
      // printf("\ngpu#%d, %dth image: output[0]=%f\n", device_id, n, tmp[0]);

    }
    // CHECK_CUDA(cudaDeviceSynchronize());
    cudaStreamSynchronize(streams[device_id]);
    cudaDeviceSynchronize ();
    // CHECK_CUDA(cudaDeviceSynchronize());
  }

  for (int device_id=0; device_id < device_count; device_id++) {
  
    int start_idx = 0;
    for (int prv=0; prv < device_id; prv++) start_idx += num_per_device[prv];

    CHECK_CUDA(cudaMemcpyAsync(myOutput + start_idx * 64 * 64 * 3, gpu_mem_output[device_id], num_per_device[device_id] * 64 * 64 * 3 * sizeof(float), cudaMemcpyDeviceToHost, streams[device_id]));
    cudaStreamSynchronize(streams[device_id]);
    cudaStreamDestroy(streams[device_id]);
  }


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
  cudaDeviceSynchronize();
}

void facegen_fin() {
  /*
   * TODO
   * Finalize required CUDA objects. For example,
   * cudaFree(...)
   */
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


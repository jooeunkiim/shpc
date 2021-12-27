# 2021-2 SHPC

# Term-project Report

2021-22861 김주은 (shpc095)



### 병렬화 방법


1. **openMP**

   CUDA를 이용하여 병렬화를 할 때 kernel 안에서 kernel을 호출할 수 없기 때문에 tconv를 병렬화 하면서 num_to_gen을 병렬화하는 것은 불가능하다. 따라서 tconv 함수는 GPU에서 병렬화를 하고, num_to_gen은 CPU에서 병렬화하기로 했다.

   opemmp 라이브러리를 이용하여 `#pragma omp parallel for num_threads(32) nowait`을 통해서 for문의 iteration을 나눠서 멀티쓰레딩으로 실행하게 해주었더니, 아무런 병렬화를 하지 않았을 때를 기준으로 실행 시간이 2배정도 단축되었다.



2. **MPI**

   1개의 노드, 1개의 코어만 사용하는 것보다 여러 개의 노드와 코어를 사용하여 병렬화를 하도록 프로세스 간의 통신을 MPI를 이용하여 구현하였다. 먼저 rank 0 프로세스에서 모든 file IO가 이루어지므로 rank 0 프로세스가 input, network를 읽고, 이 과정에서 읽은 num_to_gen을 먼저 initialize 과정에서 다른 rank 의 프로세스들에게 보내주었다. 

   이후에 face_gen 함수를 실행하면 가장 먼저 각 프로세스가 처리할 데이터를 num_to_gen을 기준으로 개수에 맞추어 나눠서 division을 만들어주었다. rank 0 프로세스에서 다른 rank 프로세스들에 대해서 해당 rank 프로세스가 처리해야하는 전체 input의 일부만 전송하고, 다른 프로세스들이 각자의 division에 대한 input을 받도록 한다. 이 때 network 또한 함께 전송해준다. MPI communication이 모든 프로세스에 대해서 잘 종료되었는지 동기화를 하기 위해서 또한, 다시 전송된 일부의 output을 rank 0 프로세스로 보내서 최종 결과를 출력하였다. 




3. **CUDA**

   먼저 CPU에 있는 메모리를 GPU device로 올려주기 위해서 `facegen_init()` 내부에서 cudaMalloc으로 GPU 메모리를 모두 초기화하고, 이후에 `facegen()` 이 실행되면 처음에 받는 input, network를 기반으로 cudaMemcpy를 하여 호스트에서 디바이스로 데이터를 모두 올려주었다.

   sequential한 코드를 GPU device를 이용하여 빠르게 확인한 결과, 각 for loop을 cuda thread block으로 병렬화하여 실행하도록 하였다. 그리고 `tconv, batch_norm` 등 시간이 오래 걸리는 레이어만 병렬화하는 것보다도 모든 레이어의 for loop을 최적화해주는 것이 실행시간 단축에 도움이 되어서 모든 레이어의 for loop을 병렬화하였다. 그 중 시간이 가장 오래 걸렸던 `tconv` 레이어의 코드는 다음과 같이 구성하였다. 

   h_out, c, k 세 가지 변수를 thread block의 3가지 차원으로 병렬화하였고, 크기가 가장 큰 channel의 병렬화를 하는 과정에서 race condition이 발생하여 output에 정확한 값이 출력되지 않는 문제가 있었다. 따라서 atomicAdd를 이용하여 atomicity가 지켜지도록 output을 write 했더니 정확한 결과를 얻을 수 있었다. 그리고 이전 이미지에서 출력되었던 feature map으로 output이 초기화되어있는 문제도 있었다. 따라서 init_c라는 커널 함수를 추가적으로 만들어서 tconv를 하기 전마다 실행하여 output map의 모든 값을 0으로 초기화해주어서 해결했다.
   

   또한, 각 노드에 달려있는 4개의 GPU를 모두 활용화여 병렬화를 하기 위해서 multi-gpu cuda programming을 하게 되면 현재 실행 시간과 비교하여 4배 가량 실행시간을 단축할 수 있을 것 같았다. 따라서 `cudaStream, cudaMallocHost, cudaMemcpyAsync` 등의 API로 pinned memory와 asynchronous streams를 이용한 최적화를 하려고 시도하였으나 default stream 이외의 stream 에서 쿠다 커널이 아예 실행이 되지 않는 이슈로 인해서 구현에 성공하지는 못했다.



### 측정 성능

`salloc --nodes=2 --ntasks-per-node=1 --cpus-per-task=64 --partition=shpc --gres=gpu:4 mpirun ./facegen_parallel network.bin input3.txt output3.txt output3.bmp`

위 명령어로 서버에서 실행했을 경우 1.12~3초 정도가 측정되었다.



### References

[https://on-demand.gputechconf.com/gtc/2014/presentations/S4158-cuda-streams-best-practices-common-pitfalls.pdf](https://on-demand.gputechconf.com/gtc/2014/presentations/S4158-cuda-streams-best-practices-common-pitfalls.pdf)

[https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf](https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf)

[https://www.nvidia.com/docs/IO/116711/sc11-multi-gpu.pdf](https://www.nvidia.com/docs/IO/116711/sc11-multi-gpu.pdf)

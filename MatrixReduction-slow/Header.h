#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <helper_cuda.h>

#include <stdio.h>

typedef float DATA_TYPE;

#include <cstdlib>
#define CHECK(call)												\
do {															\
	const cudaError_t error_code = call;						\
	if (error_code != cudaSuccess)								\
	{															\
		printf("CUDA Error:\n");								\
		printf("\tFile:\t%s\n", __FILE__);						\
		printf("\tLine:\t%d\n", __LINE__);						\
		printf("\tError code:%d\n", error_code);				\
		printf("\tError info:%s\n", cudaGetErrorString(error_code));			\
		exit(1);																\
	}																			\
}while(0)	



extern "C" void cudaReduce(DATA_TYPE *h_A, DATA_TYPE *h_B, unsigned int M, unsigned int N);

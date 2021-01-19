#include <stdio.h>
#include <stdlib.h>
#include <string.h> 

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <windows.h>

typedef float DATA_TYPE;
#define BLOCK_SIZE 256
#define BLOCK_SIZE_3 128

#define M 10
#define N 1000000

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


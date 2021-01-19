#include <stdio.h>
#include <string.h>  //memset
#include <cuda.h> //malloc
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <windows.h>

#define N 10000000
#define BLOCKSIZE 256
#define BLOCKSIZE_3 256

typedef float DATA_TYPE;

#ifdef _WIN32
#pragma warning (disable : 4710)
#endif

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


#include "Header.h"
#define BLOCKSIZE 512

void __global__ kernel_1(DATA_TYPE *d_A, DATA_TYPE *d_B, unsigned int M, unsigned int N)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int stride = blockDim.x;

	for (int j = stride; j > 0; j /= 2)
	{
		if (tx + j >= N)
		{
			d_A[bx*blockDim.x + tx] = d_A[bx*blockDim.x + tx];
		}
		else{
			d_A[bx*blockDim.x + tx] += d_A[bx*blockDim.x + tx + j];
		}
			
		__syncthreads();
	}
	
	d_B[bx] = d_A[bx*blockDim.x];
	

}

void __global__ kernel_2_1_global(DATA_TYPE *d_A, DATA_TYPE *d_temp, unsigned int M, unsigned int N, unsigned int blockofRow)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int x = bx * blockDim.x + tx; //col
	int y = by * blockDim.y + ty; //row

	int stride = blockDim.x/2;
	for (int j = stride; j > 0; j /= 2)
	{
		if( tx < j && x + j < N)
		{
			//printf("(x,y)=(%d,%d), j=%d, y*N + x + j=%d  , (tx,ty)=(%d,%d), (bx,by)=(%d,%d), \n", x, y,j, (y*N + x + j), tx, ty, bx, by);
			d_A[y*N + x] += d_A[y*N + x + j];
		}
		__syncthreads();
	}
	
	if(tx==0 && ty==0)
		d_temp[by*blockofRow + bx] = d_A[y*N + x];

	//if (bx == 156 && by == 190 && tx == 0)
	//	printf("d_temp[by*blockofRow + bx]=%f\n", d_temp[by*blockofRow + bx]);


}

void __global__ kernel_2_2(DATA_TYPE *d_temp, DATA_TYPE *d_B, unsigned int M, unsigned int blockofRow)
{
	int tx = threadIdx.x;

	if(tx<M)
	{ 
		double temp = 0.0;
		for (int i = 0; i < blockofRow; i++)
		{

			temp += d_temp[tx*blockofRow + i];

		}
		d_B[tx] = temp;
	}
}

void __global__ kernel_3(DATA_TYPE *d_A, DATA_TYPE *d_temp, unsigned int M, unsigned int N, unsigned int S_NUM)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y; // ty=0
	int bx = blockIdx.x;
	int by = blockIdx.y; 
	int x = tx + bx * blockDim.x;

	//__shared__ float d_s[TILE_WIDTH];
	extern __shared__ DATA_TYPE d_s[];
	
	int n = by * N + bx * blockDim.x + tx;

	d_s[tx] = x < N ? d_A[n] : 0.0;
	__syncthreads();
	
	int stride = blockDim.x / 2;

	for (int j = stride; j > 0; j /= 2)
	{
		if (tx < j && tx + j < blockDim.x)
		{
			d_s[tx] += d_s[tx + j];
		}
		__syncthreads();

	}
	if (tx == 0)
	{
		d_temp[by*S_NUM + bx] = d_s[0];
	}
		

}

__device__ void warpReduce( volatile DATA_TYPE *d_s, unsigned int tx)
{
	d_s[tx] += d_s[tx + 32];
	d_s[tx] += d_s[tx + 16];
	d_s[tx] += d_s[tx + 8];
	d_s[tx] += d_s[tx + 4];
	d_s[tx] += d_s[tx + 2];
	d_s[tx] += d_s[tx + 1];
}
 __global__ void kernel_3_1(DATA_TYPE *d_A, DATA_TYPE *d_temp, unsigned int M, unsigned int N, unsigned int S_NUM)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y; // ty=0
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int x = tx + bx * blockDim.x*2;
	int y = by;

	//__shared__ float d_s[BLOCKSIZE];
	extern __shared__ DATA_TYPE d_s[];


	d_s[tx] = x + blockDim.x < N ? (d_A[y * N + x] + d_A[y * N + x + blockDim.x]) : 0.0;
	__syncthreads();

	if (BLOCKSIZE >= 512)
	{
		if(tx<256) d_s[tx] += d_s[tx + 256];
		__syncthreads();
	}
	if (BLOCKSIZE >= 256)
	{
		if (tx < 128) d_s[tx] += d_s[tx + 128];
		__syncthreads();
	}
	if (BLOCKSIZE >= 128)
	{
		if (tx < 64) d_s[tx] += d_s[tx + 64];
		__syncthreads();
	}

	if (tx < 32)
		warpReduce(d_s, tx);
	
	if (tx == 0)
		d_temp[by*S_NUM + bx] = d_s[0];

}


void kernelLaunch_1(DATA_TYPE *d_A, DATA_TYPE *d_B, unsigned int M, unsigned int N);
void kernelLaunch_2(DATA_TYPE *d_A, DATA_TYPE *d_B, unsigned int M, unsigned int N);
void kernelLaunch_3(DATA_TYPE *d_A, DATA_TYPE *d_B, unsigned int M, unsigned int N);
//void kernelLaunch_4(float *d_A, float *d_B, int M, int N);

extern "C" void cudaReduce(DATA_TYPE *h_A, DATA_TYPE *h_B, unsigned int M, unsigned int N)
{
	////printf("=======================kernelLaunch_1:reduce a row in one block(N<=2048)================\n");
	cudaEvent_t start, stop;
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));
	CHECK(cudaEventRecord(start, 0));
	cudaEventQuery(start);

	unsigned int SIZEA = M * N * sizeof(DATA_TYPE);
	unsigned int SIZEB = M * sizeof(DATA_TYPE);

	DATA_TYPE *d_A;
	DATA_TYPE *d_B;

	checkCudaErrors(cudaMalloc((void**)&d_A, SIZEA));
	checkCudaErrors(cudaMalloc((void**)&d_B, SIZEB));

	checkCudaErrors(cudaMemset(d_B, 0, SIZEB));
	checkCudaErrors(cudaMemcpy(d_A, h_A, SIZEA, cudaMemcpyDefault));

	////reduce a row in one block(N<=2048):  
	//kernelLaunch_1(d_A, d_B, M, N);

	//printf("=======================kernelLaunch_2: reduce a row in multiple blocks(Global memory) and reduce again at CPU/GPU================\n");
	//reduce a row in multiple blocks of Global memory and reduce again at CPU
	//kernelLaunch_2(d_A, d_B, M, N);

	//printf("=======================kernelLaunch_3: reduce in shared memory ================\n");
	kernelLaunch_3(d_A, d_B, M, N);


	checkCudaErrors(cudaMemcpy(h_B, d_B, SIZEB, cudaMemcpyDefault));

	CHECK(cudaDeviceSynchronize());
	//time end
	CHECK(cudaEventRecord(stop, 0));
	CHECK(cudaEventSynchronize(stop));
	float elapsedTime_cuda;
	CHECK(cudaEventElapsedTime(&elapsedTime_cuda, start, stop));
	printf("GPU time=%f ms\n\n", elapsedTime_cuda);
	CHECK(cudaEventDestroy(start));
	CHECK(cudaEventDestroy(stop));

	CHECK(cudaFree(d_A));
	CHECK(cudaFree(d_B));
}

void kernelLaunch_3(DATA_TYPE *d_A, DATA_TYPE *d_B, unsigned int M, unsigned int N)
{
	dim3 block(BLOCKSIZE);
	dim3 grid((N + block.x - 1) / block.x, M);

	//shared memory
	unsigned int S_SIZE = block.x * sizeof(DATA_TYPE);

	//temp
	DATA_TYPE * d_temp;
	unsigned int S_NUM = (N + block.x - 1) / block.x;
	unsigned int SIZE_T = S_NUM * M * sizeof(DATA_TYPE);

	CHECK(cudaMalloc((void**)&d_temp, SIZE_T));
	CHECK(cudaMemset(d_temp, 0, SIZE_T));
	
	cudaEvent_t start, stop;
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));
	CHECK(cudaEventRecord(start, 0));
	cudaEventQuery(start);

	//shared memory
	//kernel_3 <<<grid,block, S_SIZE >>> (d_A, d_temp,M,N, S_NUM);  

	// Fully expand warp
	kernel_3_1 << <grid, block, S_SIZE >> > (d_A, d_temp, M, N, S_NUM);

	CHECK(cudaDeviceSynchronize()); //cudaThreadSynchronize() is deprecated
	//time end
	CHECK(cudaEventRecord(stop, 0));
	CHECK(cudaEventSynchronize(stop));
	float elapsedTime_cuda;
	CHECK(cudaEventElapsedTime(&elapsedTime_cuda, start, stop));
	printf("kernel time=%f ms\n\n", elapsedTime_cuda);
	CHECK(cudaEventDestroy(start));
	CHECK(cudaEventDestroy(stop));

	//second reduce on CPU
	DATA_TYPE *h_temp = (DATA_TYPE*)malloc(SIZE_T);
	checkCudaErrors(cudaMemcpy(h_temp, d_temp, SIZE_T, cudaMemcpyDefault));

	DATA_TYPE *h_cpureduce= (DATA_TYPE*)malloc(M* sizeof(DATA_TYPE));

	for (int i = 0; i < M; i++)
	{
		double temp = 0.0f;

		for (int j = 0; j < S_NUM; j++)
		{
			temp +=h_temp[i*S_NUM +j];
		}

		h_cpureduce[i] = temp;

		//printf("h_cpureduce[%d]:%f\n", i, h_cpureduce[i]);
	}

}

void kernelLaunch_2(DATA_TYPE *d_A, DATA_TYPE *d_B, unsigned int M, unsigned int N)
{
	//first level reduce d_A to d_temp
	DATA_TYPE *d_temp;
	
	dim3 block(1024);
	int blockofRow = (N + block.x - 1) / block.x;
	dim3 grid(blockofRow,M);


	CHECK(cudaMalloc((void**)&d_temp, M*blockofRow * sizeof(DATA_TYPE)));

	cudaEvent_t start, stop;
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));
	CHECK(cudaEventRecord(start, 0));
	cudaEventQuery(start);


	kernel_2_1_global << <grid, block >> > (d_A, d_temp, M, N, blockofRow);  //use global memory to reduce 


	CHECK(cudaDeviceSynchronize());
	//time end
	CHECK(cudaEventRecord(stop, 0));
	CHECK(cudaEventSynchronize(stop));
	float elapsedTime_cuda;
	CHECK(cudaEventElapsedTime(&elapsedTime_cuda, start, stop));
	printf("kernel_2_1_global time=%f ms\n\n", elapsedTime_cuda);
	CHECK(cudaEventDestroy(start));
	CHECK(cudaEventDestroy(stop));


	////second reduce d_temp to  d_B, used for M<=1024
	int block_size_tmp = 32*((M+31)/32);
	int block_size = block_size_tmp > 1024 ? 1024 : block_size_tmp;

	dim3 block_num(block_size);

	kernel_2_2 << <1, block_num >> > (d_temp, d_B, M, blockofRow);

	//CHECK(cudaGetLastError());

	//second reduce on CPU
	//float *h_temp = (float*)malloc(M*blockofRow * sizeof(float));
	//checkCudaErrors(cudaMemcpy(h_temp, d_temp, M*blockofRow * sizeof(float), cudaMemcpyDefault));
	////printf("h_temp result: \n");
	//for (int i = 0; i < M; i++)
	//{
	//	for (int j = 0; j < blockofRow; j++)
	//	{
	//		printf("h_temp[%d]=%f\t", i*blockofRow +j, h_temp[i*blockofRow +j]);
	//	}
	//	printf("\n");
	//}	
	//float *h_cpureduce= (float*)malloc(M* sizeof(float));
	//for (int i = 0; i < M; i++)
	//{
	//	float temp = 0.0f;
	//	for (int j = 0; j < blockofRow; j++)
	//	{
	//		temp +=h_temp[i*blockofRow+j];
	//	}
	//	h_cpureduce[i] = temp;
	//	//printf("h_cpureduce[%d]:%f\n", i, h_cpureduce[i]);
	//}

	CHECK(cudaFree(d_temp));
}

void kernelLaunch_1(DATA_TYPE *d_A, DATA_TYPE *d_B, unsigned int M, unsigned int N)
{

	dim3 block(1024);
	dim3 grid(M);

	kernel_1 << <grid, block >> > (d_A, d_B, M, N);
	CHECK(cudaGetLastError());



}



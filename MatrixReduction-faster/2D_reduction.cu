#include "2D_Reduction_Head.cuh"


void InitializeArray(DATA_TYPE *array_a, unsigned int m, unsigned int n);
void ReductHost(DATA_TYPE *array_a, DATA_TYPE *array_b, unsigned int m, unsigned int n);


__device__ void warpReduce(volatile DATA_TYPE *d_s, unsigned int tx)
{
	if (tx < 32) d_s[tx] += d_s[tx + 32];
	if (tx < 16) d_s[tx] += d_s[tx + 16];
	if (tx < 8) d_s[tx] += d_s[tx + 8];
	if (tx < 4) d_s[tx] += d_s[tx + 4];
	if (tx < 2) d_s[tx] += d_s[tx + 2];
	if (tx < 1) d_s[tx] += d_s[tx + 1];
}

__global__ void kernal(DATA_TYPE *src, DATA_TYPE *dst, unsigned int m, unsigned int n, unsigned int n1)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int x = blockDim.x * bx + tx;

	__shared__ DATA_TYPE d_s[BLOCK_SIZE];

	d_s[tx] = (x<n && by<m)? src[by*n+x]:0.0;
	__syncthreads();

	//if(tx==0)
	//	printf("(bx,by):(%d,%d),  (tx,ty):(%d,%d),   d_s=%f)\n", bx, by,tx,ty, d_s[tx]);

	if (BLOCK_SIZE >= 512)
	{
		if (tx < 256) d_s[tx] += d_s[tx + 256];
		__syncthreads();
	}

	if (BLOCK_SIZE >= 256)
	{
		if (tx < 128) d_s[tx] += d_s[tx + 128];
		__syncthreads();
	}
	if (BLOCK_SIZE >= 128)
	{
		if (tx < 64) d_s[tx] += d_s[tx + 64];
		__syncthreads();
	}

	if(tx<32) warpReduce(d_s, tx);


	if (tx == 0)
	{
		dst[by*n1+bx] = d_s[0];
		//printf("(bx,by)=(%d,%d), dst=%f, d_s=%f\n", bx, by, dst[by*n1 + bx], d_s[0]);

	}
}


__global__ void kernel_3(DATA_TYPE *src, DATA_TYPE *dst, unsigned int size_s)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;

	int x = blockDim.x * size_s + tx;

	__shared__ DATA_TYPE d_s[BLOCK_SIZE_3];

	d_s[tx] = 0.0;

	d_s[tx] = (tx < size_s && bx < M) ? src[x] : 0.0;
	__syncthreads();


	if (BLOCK_SIZE >= 128)
	{
		if (tx < 64) d_s[tx] += d_s[tx + 64];
		__syncthreads();
	}

	if (tx < 32) warpReduce(d_s, tx);

	if (tx == 0 && bx<M)
	{
		dst[bx] = d_s[0];
		//printf("bx=%d, dst=%f, d_s=%f\n",bx, dst[bx], d_s[0]);
	}
		
	

}
void kernel_launch(DATA_TYPE *d_A, DATA_TYPE *d_R, unsigned int m, unsigned int n)
{
	unsigned int COLS_C = ((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
	unsigned int COLS_D = ((COLS_C + BLOCK_SIZE - 1) / BLOCK_SIZE);

	unsigned int SIZEC = M * COLS_C * sizeof(DATA_TYPE);  //stage 1 result
	unsigned int SIZED = M * COLS_D * sizeof(DATA_TYPE);  //stage 2 result

	DATA_TYPE *d_C; //stage 1 result
	DATA_TYPE *d_D; //stage 2 result

	CHECK(cudaMalloc((void**)&d_C, SIZEC)); //stage 1 result MEMORY
	CHECK(cudaMalloc((void**)&d_D, SIZED)); //stage 2 result MEMORY

	dim3 block(BLOCK_SIZE,1);
	dim3 grid((N + block.x - 1) / block.x, M);

	dim3 grid2((COLS_C + block.x - 1) / block.x, M);

	cudaEvent_t start, stop;
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));
	CHECK(cudaEventRecord(start, 0));
	cudaEventQuery(start);

	//reduce d_A(M*N) to d_C(M*COLS_C)
	kernal << <grid, block >> > (d_A, d_C, M, N, COLS_C);
	kernal << <grid2, block >> > (d_C, d_D, M, COLS_C, COLS_D);
	kernel_3 << <M, BLOCK_SIZE_3 >> > (d_D, d_R, COLS_D);

	CHECK(cudaEventRecord(stop, 0));
	CHECK(cudaEventSynchronize(stop));
	float elapsedTime_cuda;
	CHECK(cudaEventElapsedTime(&elapsedTime_cuda, start, stop));
	printf("kernel time=%f ms\n\n", elapsedTime_cuda);
	CHECK(cudaEventDestroy(start));
	CHECK(cudaEventDestroy(stop));

	cudaFree(d_C);
	cudaFree(d_D);
}

int main(int argc, char **argv)
{
	unsigned int SIZEA = M * N * sizeof(DATA_TYPE);
	unsigned int SIZER = M * sizeof(DATA_TYPE);           //stage 3 and also be final result

	printf("Allocate %e MB on CPU\n", SIZEA/ (1024.f*1024.f));

	DATA_TYPE *h_A = (DATA_TYPE*)malloc(SIZEA);
	DATA_TYPE *h_R = (DATA_TYPE*)malloc(SIZER);

	if (h_A == NULL || h_R == NULL)
		printf("Failed to allocate h_A on CPU memory\n");

	memset(h_A, 0, SIZEA);
	memset(h_R, 0, SIZER);

	InitializeArray(h_A, M, N);

	//check CPU Result
	DATA_TYPE *r = (DATA_TYPE*)malloc(SIZER);
	memset(r, 0, SIZER);

	//timer
	LARGE_INTEGER nFreq;
	LARGE_INTEGER nBeginTime;
	LARGE_INTEGER nEndTime;
	double host_time;
	QueryPerformanceFrequency(&nFreq);
	QueryPerformanceCounter(&nBeginTime);

	ReductHost(h_A,r,M,N);

	QueryPerformanceCounter(&nEndTime);
	host_time = (double)(nEndTime.QuadPart - nBeginTime.QuadPart) / (double)nFreq.QuadPart;
	//for (int i = 0; i < M; i++)
	//	printf("CPU result:%f\n", r[i]);
	printf("CPU time: %f ms\n", host_time * 1000);

	//GPU
	DATA_TYPE *d_A; 
	DATA_TYPE *d_R; //stage 3 and also be final result

	CHECK(cudaMalloc((void**)&d_A, SIZEA));
	CHECK(cudaMalloc((void**)&d_R, SIZER));
	
	CHECK(cudaMemset(d_R, 0, SIZER));
	CHECK(cudaMemset(d_A, 0, SIZEA));

	CHECK(cudaMemcpy(d_A, h_A, SIZEA, cudaMemcpyDefault));

	cudaEvent_t start, stop;
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));
	CHECK(cudaEventRecord(start, 0));
	cudaEventQuery(start);

	kernel_launch(d_A, d_R, M, N);
	CHECK(cudaDeviceSynchronize());

	//time end
	CHECK(cudaEventRecord(stop, 0));
	CHECK(cudaEventSynchronize(stop));
	float elapsedTime_cuda;
	CHECK(cudaEventElapsedTime(&elapsedTime_cuda, start, stop));
	printf("GPU time=%f ms\n", elapsedTime_cuda);
	CHECK(cudaEventDestroy(start));
	CHECK(cudaEventDestroy(stop));

	CHECK(cudaMemcpy(h_R, d_R, SIZER, cudaMemcpyDefault));
	//for (int i = 0; i < M; i++)
	//	printf("GPU result:%f\n", h_R[i]);
	printf("Speed up=%f \n", (host_time * 1000) / elapsedTime_cuda);

	//free
	cudaFree(d_A);
	cudaFree(d_R);
	free(h_A);
	free(h_R);
	free(r);
	return 0;
}


void InitializeArray(DATA_TYPE *array_a, unsigned int m, unsigned int n )
{
	int i;
	for (int i = 0; i < m*n; i++)
	{
		array_a[i] = 1.0f;
	}
}

void ReductHost(DATA_TYPE *array_a, DATA_TYPE *array_b, unsigned int m, unsigned int n)
{
	unsigned int i,j;
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			array_b[i] += array_a[i*n + j];
		}
	}
		 

}
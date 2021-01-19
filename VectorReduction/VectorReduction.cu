#include "VectorReductHead.cuh"

void InitializeArray(DATA_TYPE *array, unsigned int size);
DATA_TYPE ReductHost(DATA_TYPE *array, unsigned int size);
void reduce_2_cpu(DATA_TYPE *array, unsigned int size)
{
	double temp=0.0;
	int i;
	for ( i = 0; i < size; i++)
	{
		temp += array[i];
	}

	printf("GPU result:%f\t", temp);

}

__global__ void kernel(DATA_TYPE *src, DATA_TYPE *dst, unsigned int src_s)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int x = tx + bx * blockDim.x;

	__shared__ DATA_TYPE d_s[BLOCKSIZE];
	
	d_s[tx] = x < src_s ? src[x] : 0.0;
	__syncthreads();

	int stride = blockDim.x;
	int j;
	for(j = stride; j > 0; j /= 2)
	{ 
		if (tx < j && tx + j < blockDim.x)
		{
			d_s[tx] += d_s[tx + j];
		}
		__syncthreads();
	}

	if (tx == 0)
	{
		dst[bx] = d_s[0];
	}
}


__device__ void warpReduce(volatile DATA_TYPE *d_s, unsigned int tx)
{
	if (tx < 32) d_s[tx] += d_s[tx + 32];
	if (tx < 16) d_s[tx] += d_s[tx + 16];
	if (tx < 8) d_s[tx] += d_s[tx + 8];
	if (tx < 4) d_s[tx] += d_s[tx + 4];
	if (tx < 2) d_s[tx] += d_s[tx + 2];
	if (tx < 1) d_s[tx] += d_s[tx + 1];
}

__global__ void kernel_2(const DATA_TYPE *src, DATA_TYPE *dst, unsigned int src_s)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x + blockIdx.y * gridDim.x;
	int x = tx + bx * blockDim.x;

	__shared__ DATA_TYPE d_s[BLOCKSIZE];

	d_s[tx] = x < src_s ? src[x] : 0.0;
	__syncthreads();

	if (BLOCKSIZE >= 512)
	{
		if (tx < 256) d_s[tx] += d_s[tx + 256];
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

	if (tx < 32) warpReduce(d_s, tx);

	if (tx == 0)
	{
		dst[bx] = d_s[0];
	}


	//if (tx == 0 && bx < 153)
	//	printf("%f\n", dst[bx]);
}


__global__ void kernel_3(DATA_TYPE *d_D, DATA_TYPE *d_B, unsigned int size_d)
{
	
	int tx = threadIdx.x;

	//printf("d_D: %d,  %f\n", tx, d_D[tx]);

	__shared__ DATA_TYPE d_s[BLOCKSIZE_3];
	d_s[tx] = 0;

	if (tx< size_d)
		d_s[tx] = d_D[tx];

	__syncthreads();

	//printf("d_s: %d:  %f\n",tx, d_s[tx]);
	
	if (tx < 128) d_s[tx] += d_s[tx + 128];
	__syncthreads();

	if (tx < 64) d_s[tx] += d_s[tx + 64];
	__syncthreads();

	if (tx < 32) warpReduce(d_s, tx);
	
	if (tx == 0)
	{
		d_B[0] = d_s[0];
		//printf("%f\n", d_s[0]);
		//printf("%f\n", d_B[0]);
	}
	
}


void kernel_call(DATA_TYPE *d_A, DATA_TYPE *d_B, unsigned int size_A)
{
	DATA_TYPE *d_C;
	DATA_TYPE *d_D;

	dim3 block(BLOCKSIZE);
	dim3 grid((N + block.x - 1) / block.x);

	unsigned int size_C = grid.x;
	unsigned int SIZEC = size_C * sizeof(DATA_TYPE);

	dim3 block_2(BLOCKSIZE);
	dim3 grid_2((size_C + block_2.x-1) / block_2.x);

	unsigned int size_D = grid_2.x;
	unsigned int SIZED = size_D * sizeof(DATA_TYPE);

	CHECK(cudaMalloc((void**)&d_C, SIZEC));
	CHECK(cudaMemset(d_C, 0, SIZEC));

	CHECK(cudaMalloc((void**)&d_D, SIZED));
	CHECK(cudaMemset(d_D, 0, SIZED));
	
	cudaEvent_t start, stop;
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));
	CHECK(cudaEventRecord(start, 0));
	cudaEventQuery(start);

	//printf("size_A=%d\t ", size_A); 1000 0000
	//printf("size_C=%d\t ", size_C); 39063
	//printf("size_D=%d\t ", size_D); 153

	//kernel<<<grid,block>>>(d_A, d_C, size_A);
	kernel_2 << <grid, block >> > (d_A, d_C, N);
	CHECK(cudaGetLastError());
	kernel_2 << <grid_2, block_2 >> > (d_C, d_D, size_C);
	CHECK(cudaGetLastError());
	kernel_3 << <1, BLOCKSIZE_3 >> > (d_D, d_B, size_D);
	CHECK(cudaGetLastError());

	CHECK(cudaDeviceSynchronize()); //cudaThreadSynchronize() is deprecated
	//time end
	CHECK(cudaEventRecord(stop, 0));
	CHECK(cudaEventSynchronize(stop));
	float elapsedTime_cuda;
	CHECK(cudaEventElapsedTime(&elapsedTime_cuda, start, stop));
	printf("kernel time=%f ms\n", elapsedTime_cuda);
	CHECK(cudaEventDestroy(start));
	CHECK(cudaEventDestroy(stop));

	//DATA_TYPE *h_C = (DATA_TYPE*)malloc(SIZEC);
	//CHECK(cudaMemcpy(h_C, d_C, SIZEC, cudaMemcpyDefault));

	//reduce_2_cpu(h_C, size_C);

	cudaFree(d_C);
	cudaFree(d_D);
}

int main()
{
	unsigned int SIZEA = N * sizeof(DATA_TYPE);
			 int SIZEB = 1 * sizeof(DATA_TYPE);

	printf("Allocate %e MB on CPU\n", SIZEA / (1024.f*1024.f));

	DATA_TYPE *h_A = (DATA_TYPE*)malloc(SIZEA);
	DATA_TYPE *h_B = (DATA_TYPE*)malloc(SIZEB);

	if (h_A == NULL)
		printf("Failed to allocate CPU memory - h_A\n");

	memset(h_A, 0, SIZEA);
	memset(h_B, 0, SIZEB);

	InitializeArray(h_A, N);

	//timer
	LARGE_INTEGER nFreq;
	LARGE_INTEGER nBeginTime;
	LARGE_INTEGER nEndTime;
	double host_time;
	QueryPerformanceFrequency(&nFreq);
	QueryPerformanceCounter(&nBeginTime);

	DATA_TYPE h_sum =ReductHost(h_A, N);
	printf("CPU result:%f\t", h_sum);
	
	QueryPerformanceCounter(&nEndTime);
	host_time = (double)(nEndTime.QuadPart - nBeginTime.QuadPart) / (double)nFreq.QuadPart;
	printf("CPU time: %f ms\n", host_time*1000);
	
	
	//GPU
	DATA_TYPE *d_A;
	DATA_TYPE *d_B;
	CHECK(cudaMalloc((void**)&d_A, SIZEA));
	CHECK(cudaMalloc((void**)&d_B, SIZEB));
	CHECK(cudaMemcpy(d_A, h_A, SIZEA, cudaMemcpyDefault));

	//timer
	cudaEvent_t start, stop;
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));
	CHECK(cudaEventRecord(start, 0));
	cudaEventQuery(start);

	kernel_call(d_A, d_B, N);

	//time end
	CHECK(cudaEventRecord(stop, 0));
	CHECK(cudaEventSynchronize(stop));
	float elapsedTime;
	CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
	CHECK(cudaEventDestroy(start));
	CHECK(cudaEventDestroy(stop));

	CHECK(cudaMemcpy(h_B, d_B, SIZEB, cudaMemcpyDefault));

	printf("GPU result:%f\t", h_B[0]);
	printf("GPU time=%f ms\n", elapsedTime);
	printf("Speed up=%f \n", (host_time * 1000)/elapsedTime);

	free(h_A);
	free(h_B);
	cudaFree(d_A);
	cudaFree(d_B);


	return 0;

}



void InitializeArray(DATA_TYPE *array, unsigned int size)
{
	int i;
	for (i = 0; i < size; i++)
	{
		array[i] = 1.0;
	}
}

DATA_TYPE ReductHost(DATA_TYPE *array, unsigned int size)
{
	unsigned int i;
	double result = 0;
	for (i = 0; i < size; i++)
		result += array[i];

	DATA_TYPE result_1 = (DATA_TYPE)result;

	return result_1;

}

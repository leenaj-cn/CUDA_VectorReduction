#include "Header.h"
#define M 10
#define N 1000000

void initArray(DATA_TYPE *array, int m, int n)
{
	for (int i = 0; i < m*n; i++)
	{
		array[i] = 1.0f;
	}
}

void reduceArray(DATA_TYPE *arrayA, DATA_TYPE *arrayB, int m, int n)
{
	for (int i = 0; i < m; i++)
	{
		double *temp;
		for (int j = 0; j < n; j++)
		{
			arrayB[i] += arrayA[i*N + j];
		}
	}
}

int main()
{
	unsigned int SIZEA = M * N * sizeof(DATA_TYPE);
	unsigned int SIZEB = M * sizeof(DATA_TYPE);

	DATA_TYPE *h_A = NULL;
	DATA_TYPE *h_B = NULL;

	printf("allocate %e MB on GPU\n", SIZEA / (1024.f*1024.f));

	h_A = (DATA_TYPE*)malloc(SIZEA);
	h_B = (DATA_TYPE*)malloc(SIZEB);

	if (h_A == NULL || h_B == NULL)
	{
		printf("failed to allocate memory\n");
		return -1;
	}

	memset(h_A, 0, SIZEA);
	memset(h_B, 0, SIZEB);

	initArray(h_A, M, N);

	//cpu reduce
	//reduceArray(h_A, h_B, M, N);

	//cuda reduce
	cudaReduce(h_A, h_B, M, N);

	//for (int i = 0; i < M; i++)
	//	printf("h_B[%d]=%f\n", i, h_B[i]);



	//getchar();
	return 0;
}
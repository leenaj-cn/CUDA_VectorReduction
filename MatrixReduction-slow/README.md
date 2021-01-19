# Performance

## DATA SIZE
2-D data reduction

    3.814697e+01 MB 

```
#define M 10
#define N 1000000
M * N * sizeof(float) -> M *  sizeof(float)

```


# kernel_1: reduce one block per row
### Note: 

1. the GPU time depend on the number of row rather than the cols. so there is not any meaning to do time compare for kernel_1.

2. only suit for N<=2048 
```
	dim3 block(1024);
	dim3 grid(M);

    kernel_1 << <grid, block >> >
```

```
allocate 1.525879e+00 MB on GPU (200*2000)
GPU time=1.910784 ms
```

# kernel_2_1_global + CPU reduce

###Note:

1. reduce a row by multiple blocks
2. used for M<=1024

only used Global memory


```
	dim3 block(1024);
	int blockofRow = (N + block.x - 1) / block.x;
	dim3 grid(blockofRow,M);

    kernel_2_1_global << <grid, block >> >
```
```
allocate 3.814697e+01 MB on GPU
kernel_2_1_global time=33.036255 ms

GPU time=40.581665 ms

```


# kernel_2_1_global + kernel_2_2 
###Note:

1. reduce again on kernel_2_2 rather than CPU
2. used for M<=1024

```
	int block_size_tmp = 32*((M+31)/32);
	int block_size = block_size_tmp > 1024 ? 1024 : block_size_tmp;

    dim3 block_num(block_size);

    kernel_2_2 << <1, block_num >> >
```


```
allocate 3.814697e+01 MB on GPU
kernel_2_1_global time=33.002369 ms

GPU time=40.606880 ms

```


# kernel 3 shared memory + CPU reduce
```
allocate 3.814697e+01 MB on GPU
kernel time=28.920832 ms

GPU time=35.698914 ms

```
```
    dim3 block(BLOCKSIZE);
	dim3 grid((N + block.x - 1) / block.x, M);

    unsigned int S_SIZE = block.x * sizeof(DATA_TYPE);

    kernel_3 <<<grid,block, S_SIZE >>> 

 ```

# kernel 3_1 shared memory + Fully expand warp + CPU reduce


```
	dim3 block(BLOCKSIZE);
	dim3 grid((N + block.x - 1) / block.x, M);
    kernel_3_1 << <grid, block, S_SIZE >> > 
```

```
allocate 3.814697e+01 MB on GPU
kernel time=5.101568 ms

GPU time=12.632800 ms
```

# NVIDIA GPU:
## GeForce GTX 1060 6GB

```
   Device 0: "GeForce GTX 1060 6GB"
    CUDA Capability Major/Minor version number:    6.1
    Total amount of global memory:                 6144 MBytes (6442450944 bytes)
    Total amount of constant memory:               65536 bytes
    Total amount of shared memory per block:       49152 bytes
    Total number of registers available per block: 65536
    Warp size:                                     32
    Maximum number of threads per multiprocessor:  2048
    Maximum number of threads per block:           1024
    Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
    Max dimension size of a grid size (x,y,z):    (2147483647, 65535, 65535)
    Texture alignment:                             512 bytes
    Maximum memory pitch:                          2147483647 bytes
    Memory Bus Width:                              192-bit
    L2 Cache Size:                                 1572864 bytes
    Device has ECC support:                        Disabled
    CUDA Device Driver Mode (TCC or WDDM):         WDDM (Windows Display Driver Model)
```
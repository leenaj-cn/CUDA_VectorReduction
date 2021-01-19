# Performance

## DATA SIZE
2-D data reduction

    3.814697e+01 MB 

```
#define M 10
#define N 1000000
M * N * sizeof(float) -> M *  sizeof(float)

```
# kernal+kernal+kernel_3
```
 	dim3 block(BLOCK_SIZE,1);
	dim3 grid((N + block.x - 1) / block.x, M);

	dim3 grid2((COLS_C + block.x - 1) / block.x, M);


	kernal << <grid, block >> > (d_A, d_C, M, N, COLS_C);
	kernal << <grid2, block >> > (d_C, d_D, M, COLS_C, COLS_D);
	kernel_3 << <M, BLOCK_SIZE_3 >> > (d_D, d_R, COLS_D);
```

```
CPU time: 11.437500 ms
kernel time=0.609536 ms

GPU time=0.861664 ms

Speed up=13.273736

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
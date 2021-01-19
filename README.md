# Performance

1-D data reduction

## DATA SIZE

    3.814697e+01 MB 

```
#define N 10000000
N * sizeof(float) -> sizeof(float)

```

# kernel

```
	dim3 block(BLOCKSIZE);
	dim3 grid((N + block.x - 1) / block.x);

	dim3 block_2(BLOCKSIZE);
	dim3 grid_2((size_C + block_2.x-1) / block_2.x);

 	kernel_2 << <grid, block >> > (d_A, d_C, N);

	kernel_2 << <grid_2, block_2 >> > (d_C, d_D, size_C);

	kernel_3 << <1, BLOCKSIZE_3 >> > (d_D, d_B, size_D);
   

```


```

Allocate 3.814697e+01 MB on CPU

CPU result:10000000.000000      CPU time: 12.505200 ms

kernel time=0.642848 ms

GPU result:10000000.000000      GPU time=0.894208 ms

Speed up=13.984666


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
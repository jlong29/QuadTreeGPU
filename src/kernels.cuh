#ifndef __KERNELS_H__
#define __KERNELS_H__
//CUDA
#include <cuda_runtime.h>

//BASIC LAYOUT
#define NTHREADS 256    //reference block size
#define WARPSIZE 32
#define NWARPS ((NTHREADS)/(WARPSIZE))

typedef unsigned char uchar;

// Image Processing //
//Black Image: device and host code
__global__ void d_setBlackImag(uchar4* dst, const int w, const int h);
//Write 2D noise data to image buffer
__global__ void d_writeData2Image(uchar4* dst, const int* __restrict noiseX, const int* __restrict noiseY,const int w, const int h, const int n);

// Random Number Generators //
// Generate 2D uniform random values
__global__ void generate_uniform2D_kernel(int* noiseX, int* noiseY, int seed, const int w, const int h, const int n);

// Quad Tree Routines //
__global__ void reset_arrays_kernel(int *mutex, float *x, float *y, float *mass, int *count, int *start, int *sorted, int *child, int *index, float *left, float *right, float *bottom, float *top, int n, int m);
__global__ void compute_bounding_box_kernel(int *mutex, float *x, float *y, volatile float *left, volatile float *right, volatile float *bottom, volatile float *top, int n);

__global__ void build_tree_kernel(volatile float *x, volatile float *y, volatile float *mass, volatile int *count,
									int *start, volatile int *child, int *index,
									const float *left, const float *right, const float *bottom, const float *top,
									const int n, const int m);

#endif

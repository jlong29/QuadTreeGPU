#ifndef __QUADTREEKERNELS_H__
#define __QUADTREEKERNELS_H__
//CUDA
#include <cuda_runtime.h>

//BASIC LAYOUT
#define NTHREADS 256    //reference block size
#define WARPSIZE 32
#define NWARPS ((NTHREADS)/(WARPSIZE))

#define NUM_CELLS 4

typedef unsigned char uchar;

namespace quadTreeGPU{ namespace quadTreeKernels
{

// Image Processing //
//Black Image: device and host code
__global__ void d_setBlackImag(uchar4* dst, const int w, const int h);
//Gray Image: device and host code
__global__ void d_setGrayScaleImag(uchar4* dst, const int w, const int h);
//Write 2D noise data to image buffer
__global__ void d_writeData2Image(uchar4* dst, const float* __restrict noiseX, const float* __restrict noiseY,const int w, const int h, const int n);
//Write 2D filtered noise data to the image buffer
__global__ void d_writeFilter2Image(uchar4* dst, const float* __restrict filterX, const float* __restrict filterY,const int w, const int h, const int n);
//Draw internal edges of cells
__global__ void d_drawCellInnerEdges(uchar4* dst, int* index, const float* __restrict x, const float* __restrict y, const float* __restrict rx, const float* __restrict ry,
										const int w, const int h, const int n, const int m);

// Random Number Generators //
// Generate 2D uniform random values
__global__ void generate_uniform2D_kernel(float* noiseX, float* noiseY, int seed, const int w, const int h, const int n);
// Generate 2D uniform random coordinate and filter values
__global__ void generate_uniform2Dfilter_kernel(float* noiseX, float* noiseY, float* score, int seed, const int w, const int h, const int n);

// Quad Tree Routines //
__global__ void reset_arrays_kernel(int* mutex, float* x, float* y, float* rx, float* ry, int* child, int* index,
											float* left, float* right, float* bottom, float* top, const int w, const int h, int n, int m);

__global__ void reset_filter_arrays_kernel(int* mutex, float* x, float* y, float* score, float* xf, float* yf, float* scoref,
											float* rx, float* ry, int* child, int* index, float* left, float* right, float* bottom, float* top,
											const int q, const int w, const int h, int n, int m);

__global__ void d_setData(float* x, float* y, float* score, unsigned int* pd, float* xn, float* yn, float* scoren, const unsigned int* pdn);

__global__ void build_tree_kernel(volatile float* x, volatile float* y, float* rx, float* ry, volatile int* child, int* index,
									const float* left, const float* right, const float* bottom, const float* top,
									const int n, const int m);

__global__ void filter_tree_kernel(volatile float* x, volatile float* y, volatile float* score,
									float* rx, float* ry, volatile int* child, int* index,
									const float* left, const float* right, const float* bottom, const float* top,
									const int n, const int m, const int d, const int f);

__global__ void filter_treeDev_kernel(volatile float* x, volatile float* y, volatile float* score,
									float* rx, float* ry, volatile int* child, int* index,
									const float* left, const float* right, const float* bottom, const float* top,
									const int n, const int m, unsigned int* pd, const int f);

__global__ void pack_filtered_data_kernel(float* xf, float* yf, float* scoref,
											float* x, float* y, float* score,
											int* child, const int n, const int d, const int q);

__global__ void pack_filteredDev_data_kernel(float* xf, float* yf, float* scoref,
											float* x, float* y, float* score,
											int* child, const int n, unsigned int* pd, const int q);

} // namespace quadTreeKernels
} // namespace quadTreeGPU

#endif	//__QUADTREEKERNELS_H__

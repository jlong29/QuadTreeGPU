//StL
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//CUDA
#include "QuadTreeBuilder.h"
#include <helper_cuda.h>         // helper functions for CUDA error check

#include "quadTreeKernels.h"

using namespace quadTreeKernels;

QuadTreeBuilder::QuadTreeBuilder()
{
	width    = -1;
	height   = -1;
	numData  = -1;

	step     = 0;

	d_left   = NULL;
	d_right  = NULL;
	d_bottom = NULL;
	d_top    = NULL;

	d_x      = NULL;
	d_y      = NULL;

	d_child  = NULL;

	d_index  = NULL;
	d_mutex  = NULL;

	h_x      = NULL;
	h_y      = NULL;
	d_img    = NULL;

	timersCreated = false;
}

QuadTreeBuilder::QuadTreeBuilder(int n, int w, int h):
	numData(n),
	width(w),
	height(h)
{
	step     = 0;
	numNodes = 2*n+12000;	// A magic large function of n

	// allocate host data
	dataSz   = numData*sizeof(float);
	nodeSz   = numNodes*sizeof(float);
	imgSz    = width*height*sizeof(uchar4);

	//GPU Launch Configurations
	//1D
	threads = NTHREADS;
	blocks  = divUp(numData, NTHREADS);

	//2D
	blockDim.x = WARPSIZE;
	blockDim.y = NWARPS;

	gridDim.x  = divUp(width, blockDim.x);
	gridDim.y  = divUp(height, blockDim.y);

	// allocate device data
	checkCudaErrors(cudaMalloc((void**)&d_left, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_right, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_bottom, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_top, sizeof(float)));
	checkCudaErrors(cudaMemset(d_left, 0, sizeof(float)));
	checkCudaErrors(cudaMemset(d_right, 0, sizeof(float)));
	checkCudaErrors(cudaMemset(d_bottom, 0, sizeof(float)));
	checkCudaErrors(cudaMemset(d_top, 0, sizeof(float)));

	checkCudaErrors(cudaMalloc((void**)&d_x, nodeSz));
	checkCudaErrors(cudaMalloc((void**)&d_y, nodeSz));
	checkCudaErrors(cudaMalloc((void**)&d_child, 4*numNodes*sizeof(int)));

	checkCudaErrors(cudaMalloc((void**)&d_index, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&d_mutex, sizeof(int)));
	
	h_x      = new float[numNodes];
	h_y      = new float[numNodes];
	checkCudaErrors(cudaMalloc((void**)&d_img, imgSz));

	//Create Timers
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	timersCreated = true;
}

QuadTreeBuilder::~QuadTreeBuilder()
{
	deallocate();
}

int QuadTreeBuilder::allocate()
{
	if (numData < 0)
	{
		fprintf(stderr, "QuadTreeBuilder::allocate(): numData < 0, must initialize prior to allocation\n");
		return -1;
	}

	dataSz   = numData*sizeof(float);

	numNodes = 2*numData+12000;	// A magic large function of n
	nodeSz   = numNodes*sizeof(float);

	//GPU Launch Configurations
	//1D
	threads = NTHREADS;
	blocks  = divUp(numData, NTHREADS);

	//2D
	blockDim.x = WARPSIZE;
	blockDim.y = NWARPS;

	gridDim.x  = divUp(width, blockDim.x);
	gridDim.y  = divUp(height, blockDim.y);

	// allocate device data
	if (d_left==NULL)
	{
		checkCudaErrors(cudaMalloc((void**)&d_left, sizeof(float)));
		checkCudaErrors(cudaMemset(d_left, 0, sizeof(float)));
	}
	if (d_left==NULL)
	{
		checkCudaErrors(cudaMalloc((void**)&d_right, sizeof(float)));
		checkCudaErrors(cudaMemset(d_right, 0, sizeof(float)));
	}
	if (d_left==NULL)
	{
		checkCudaErrors(cudaMalloc((void**)&d_bottom, sizeof(float)));
		checkCudaErrors(cudaMemset(d_bottom, 0, sizeof(float)));
	}
	if (d_left==NULL)
	{
		checkCudaErrors(cudaMalloc((void**)&d_top, sizeof(float)));
		checkCudaErrors(cudaMemset(d_top, 0, sizeof(float)));
	}
	if (d_x==NULL)
	{
		checkCudaErrors(cudaMalloc((void**)&d_x, nodeSz));
	}
	if (d_y==NULL)
	{
		checkCudaErrors(cudaMalloc((void**)&d_y, nodeSz));
	}
	if (d_child==NULL)
	{
		checkCudaErrors(cudaMalloc((void**)&d_child, 4*numNodes*sizeof(int)));
	}
	if (d_index==NULL)
	{
		checkCudaErrors(cudaMalloc((void**)&d_index, sizeof(int)));
	}
	if (d_mutex==NULL)
	{
		checkCudaErrors(cudaMalloc((void**)&d_mutex, sizeof(int)));
	}
	if (h_x==NULL)
		h_x      = new float[numNodes];
	if (h_y==NULL)
		h_y      = new float[numNodes];
	if ((d_img==NULL) && (width>0) && (height>0))
	{
		imgSz    = width*height*sizeof(uchar4);
		checkCudaErrors(cudaMalloc((void**)&d_img, imgSz));
	}

	//Create Timers
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	timersCreated = true;

	return 0;
}
void QuadTreeBuilder::deallocate()
{
	if (d_left!=NULL)
	{
		checkCudaErrors(cudaFree(d_left));
		d_left = NULL;
	}
	if (d_right!=NULL)
	{
		checkCudaErrors(cudaFree(d_right));
		d_right = NULL;
	}
	if (d_bottom!=NULL)
	{
		checkCudaErrors(cudaFree(d_bottom));
		d_bottom = NULL;
	}
	if (d_top!=NULL)
	{
		checkCudaErrors(cudaFree(d_top));
		d_top = NULL;
	}

	if (d_x!=NULL)
	{
		checkCudaErrors(cudaFree(d_x));
		d_x = NULL;
	}
	if (d_y!=NULL)
	{
		checkCudaErrors(cudaFree(d_y));
		d_y = NULL;
	}
	if (d_child!=NULL)
	{
		checkCudaErrors(cudaFree(d_child));
		d_child = NULL;
	}

	if (d_index!=NULL)
	{
		checkCudaErrors(cudaFree(d_index));
		d_index = NULL;	
	}
	if (d_mutex!=NULL)
	{
		checkCudaErrors(cudaFree(d_mutex));
		d_mutex = NULL;
	}
	if (h_x!=NULL)
	{
		delete [] h_x;
		h_x = NULL;
	}
	if (h_y!=NULL)
	{
		delete [] h_y;
		h_y = NULL;
	}
	if (d_img!=NULL)
	{
		checkCudaErrors(cudaFree(d_img));
		d_img = NULL;
	}

	if (timersCreated)
	{
		checkCudaErrors(cudaEventDestroy(start));
		checkCudaErrors(cudaEventDestroy(stop));
		timersCreated = false;
	}

	cudaDeviceSynchronize();
}

//Set parameters: n is required for functioning. w and h are for visualization
void QuadTreeBuilder::setParameters(int n, int w, int h)
{
	//Set object
	width   = w;
	height  = h;
	numData = n;
}

//Set data generated by another GPU process
void QuadTreeBuilder::setData(const float* x, const float* y)
{
	checkCudaErrors(cudaMemcpy(d_x, x, dataSz, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(d_y, y, dataSz, cudaMemcpyDeviceToDevice));
}

//build the quad tree
void QuadTreeBuilder::build()
{
	checkCudaErrors(cudaEventRecord(start,0));

	if ((width < 0) || (height < 0))
	{
		ResetArrays(d_mutex, d_x, d_y, d_child, d_index, d_left, d_right, d_bottom, d_top, numData, numNodes);
		ComputeBoundingBox(d_mutex, d_index, d_x, d_y, d_left, d_right, d_bottom, d_top, numData);
	} else
	{
		ResetArrays(d_mutex, d_x, d_y, d_child, d_index, d_left, d_right, d_bottom, d_top, width, height, numData, numNodes);
	}
	BuildQuadTree(d_x, d_y, d_child, d_index, d_left, d_right, d_bottom, d_top, numData, numNodes);

	checkCudaErrors(cudaEventRecord(stop,0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elpsTime, start, stop));
	printf("\n\nElapsed time for QuadTree Build:  %9.6f ms \n", elpsTime);

	step++;
}

//Write visualization
int QuadTreeBuilder::createViz()
{
	if ((width<0) || (height<0))
	{
		fprintf(stderr, "QuadTreeBuilder::createViz(): width or height < 0, must initialize prior to vizualization\n");
		return -1;
	}


	//Write Random data onto image buffer
	checkCudaErrors(cudaEventRecord(start, 0));

	d_setBlackImag<<<gridDim, blockDim>>>(d_img, width, height);
	d_writeData2Image<<<blocks, threads>>>(d_img, d_x, d_y, width, height, numData);

	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elpsTime, start, stop));
	printf("\n\nElapsed time for creating viz:    %9.6f ms \n", elpsTime);
	return 0;
}

int QuadTreeBuilder::downloadData()
{
	if ((width<0) || (height<0))
	{
		fprintf(stderr, "QuadTreeBuilder::downloadData(): width or height < 0, must initialize prior to data download\n");
		return -1;
	}

	//Copy back to host for checking    
	checkCudaErrors(cudaMemcpy(h_x, d_x, dataSz, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_y, d_y, dataSz, cudaMemcpyDeviceToHost));

	return 0;
}

int QuadTreeBuilder::resetData()
{
	if ((width<0) || (height<0))
	{
		fprintf(stderr, "QuadTreeBuilder::resetData(): width or height < 0, must initialize prior to data generation\n");
		return -1;
	}

	//Generate Random Data
	checkCudaErrors(cudaEventRecord(start, 0));

	int seed = (int)time(0);

	generate_uniform2D_kernel<<<blocks, threads>>>(d_x, d_y, seed, width, height, numData);

	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elpsTime, start, stop));
	printf("\n\nElapsed time for Random Data Generation:  %9.6f ms \n", elpsTime);

	return 0;
}

//Resets arrays used in constructing the quad tree
void QuadTreeBuilder::ResetArrays(int* mutex, float* x, float* y, int* child, int* index, float* left, float* right, float* bottom, float* top, int n, int m)
{
	reset_arrays_kernel<<<gridSize, blockSize>>>(mutex, x, y, child, index, left, right, bottom, top, n, m);
}
void QuadTreeBuilder::ResetArrays(int* mutex, float* x, float* y, int* child, int* index, float* left, float* right, float* bottom, float* top, const int w, const int h, int n, int m)
{
	reset_arrays_kernel<<<gridSize, blockSize>>>(mutex, x, y, child, index, left, right, bottom, top, w, h, n, m);	
}

//Computes a bounding box around user input data
void QuadTreeBuilder::ComputeBoundingBox(int* mutex, int* index, float* x, float* y, float* left, float* right, float* bottom, float* top, int n)
{
	compute_bounding_box_kernel<<<gridSize, blockSize>>>(mutex, index, x, y, left, right, bottom, top, n);
}

//Builds a quad tree
void QuadTreeBuilder::BuildQuadTree(float* x, float* y, int* child, int* index, float* left, float* right, float* bottom, float* top, int n, int m)
{
	build_tree_kernel<<<gridSize, blockSize>>>(x, y, child, index, left, right, bottom, top, n, m);
}

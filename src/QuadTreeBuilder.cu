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

namespace quadTreeGPU
{

// #define TIMINGDEBUG

using namespace quadTreeKernels;

QuadTreeBuilder::QuadTreeBuilder()
{
	width    = -1;
	height   = -1;
	numData  = -1;

	numTestData      = -1;
	numFilteredData  = -1;
	
	d_left   = NULL;
	d_right  = NULL;
	d_bottom = NULL;
	d_top    = NULL;

	d_x      = NULL;
	d_y      = NULL;
	d_score  = NULL;
	d_rx     = NULL;
	d_ry     = NULL;

	d_xf     = NULL;
	d_yf     = NULL;
	d_scoref = NULL;

	d_child  = NULL;

	d_index  = NULL;
	d_mutex  = NULL;

	h_x      = NULL;
	h_y      = NULL;

	h_xf	 = NULL;
	h_yf     = NULL;

	d_img    = NULL;

	timersCreated = false;
}

QuadTreeBuilder::QuadTreeBuilder(int n, int w, int h, int q, int d):
	numData(n),
	width(w),
	height(h),
	numFilteredData(q),
	numTestData(d)
{
	numNodes   = 2*n+12000;	// A magic large function of n
	cellMargin = 2.0f;

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
	checkCudaErrors(cudaMalloc((void**)&d_score, nodeSz));
	checkCudaErrors(cudaMalloc((void**)&d_rx, nodeSz-dataSz));	//NOTE; -dataSz
	checkCudaErrors(cudaMalloc((void**)&d_ry, nodeSz-dataSz));
	
	checkCudaErrors(cudaMalloc((void**)&d_child, 4*numNodes*sizeof(int)));

	checkCudaErrors(cudaMalloc((void**)&d_index, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&d_mutex, sizeof(int)));
	
	h_x      = new float[numNodes];
	h_y      = new float[numNodes];
	checkCudaErrors(cudaMalloc((void**)&d_img, imgSz));

	if (numFilteredData > 0)
	{	
		checkCudaErrors(cudaMalloc((void**)&d_xf, numFilteredData*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&d_yf, numFilteredData*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&d_scoref, numFilteredData*sizeof(float)));

		h_xf = new float[numFilteredData];
		h_yf = new float[numFilteredData];
	}

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

	cellMargin = 2.0f;

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
	if (d_right==NULL)
	{
		checkCudaErrors(cudaMalloc((void**)&d_right, sizeof(float)));
		checkCudaErrors(cudaMemset(d_right, 0, sizeof(float)));
	}
	if (d_bottom==NULL)
	{
		checkCudaErrors(cudaMalloc((void**)&d_bottom, sizeof(float)));
		checkCudaErrors(cudaMemset(d_bottom, 0, sizeof(float)));
	}
	if (d_top==NULL)
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
	if (d_score==NULL)
	{
		checkCudaErrors(cudaMalloc((void**)&d_score, nodeSz));
	}
	if (d_rx==NULL)
	{
		checkCudaErrors(cudaMalloc((void**)&d_rx, nodeSz-dataSz));	//NOTE: -dataSz
	}
	if (d_ry==NULL)
	{
		checkCudaErrors(cudaMalloc((void**)&d_ry, nodeSz-dataSz));
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
	if (d_img==NULL)
	{
		imgSz    = width*height*sizeof(uchar4);
		checkCudaErrors(cudaMalloc((void**)&d_img, imgSz));
	}

	if (numFilteredData > 0)
	{
		if (d_xf==NULL)
		{
			checkCudaErrors(cudaMalloc((void**)&d_xf, numFilteredData*sizeof(float)));
		}
		if (d_yf==NULL)
		{
			checkCudaErrors(cudaMalloc((void**)&d_yf, numFilteredData*sizeof(float)));
		}
		if (d_scoref==NULL)
		{
			checkCudaErrors(cudaMalloc((void**)&d_scoref, numFilteredData*sizeof(float)));
		}
		if (h_xf==NULL)
			h_xf = new float[numFilteredData];
		if (h_yf==NULL)
			h_yf = new float[numFilteredData];
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
	if (d_score!=NULL)
	{
		checkCudaErrors(cudaFree(d_score));
		d_score = NULL;
	}
	if (d_rx!=NULL)
	{
		checkCudaErrors(cudaFree(d_rx));
		d_rx = NULL;
	}
	if (d_ry!=NULL)
	{
		checkCudaErrors(cudaFree(d_ry));
		d_ry = NULL;
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
	if (d_xf!=NULL)
	{
		checkCudaErrors(cudaFree(d_xf));
		d_xf = NULL;
	}
	if (d_yf!=NULL)
	{
		checkCudaErrors(cudaFree(d_yf));
		d_yf = NULL;
	}
	if (d_scoref!=NULL)
	{
		checkCudaErrors(cudaFree(d_scoref));
		d_scoref = NULL;
	}
	if (h_xf!=NULL)
	{
		delete [] h_xf;
		h_xf = NULL;
	}
	if (h_yf!=NULL)
	{
		delete [] h_yf;
		h_yf = NULL;
	}

	if (timersCreated)
	{
		checkCudaErrors(cudaEventDestroy(start));
		checkCudaErrors(cudaEventDestroy(stop));
		timersCreated = false;
	}
}

//Set parameters: n is required for functioning. w and h are for visualization
void QuadTreeBuilder::setParameters(int n, int w, int h, int q, int d)
{
	//Set object
	width           = w;
	height          = h;
	numData         = n;
	numTestData     = d;
	numFilteredData = q;
}

//Set data generated by another GPU process
void QuadTreeBuilder::setData(const float* x, const float* y)
{
	checkCudaErrors(cudaMemcpy(d_x, x, dataSz, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(d_y, y, dataSz, cudaMemcpyDeviceToDevice));
}
void QuadTreeBuilder::setData(const float* x, const float* y, const float* score, const int d)
{
	checkCudaErrors(cudaMemcpy(d_x, x, d*sizeof(float), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(d_y, y, d*sizeof(float), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(d_score, score, d*sizeof(float), cudaMemcpyDeviceToDevice));
}
void QuadTreeBuilder::setData(float* x, float* y, float* score, const unsigned int* d)
{
	d_setData<<<blocks, threads>>>(d_x, d_y, d_score, x, y, score, d);
}

void QuadTreeBuilder::setCellMargin(const float cm)
{
	cellMargin = cm;
}

//build the quad tree
int QuadTreeBuilder::build()
{
	if (numData < 0)
	{
		fprintf(stderr, "QuadTreeBuilder::build(): numData < 0, must initialize prior to building\n");
		return -1;
	}

	#ifdef TIMINGDEBUG
	checkCudaErrors(cudaEventRecord(start,0));
	#endif

	ResetArrays(width, height);
	BuildQuadTree();

	#ifdef TIMINGDEBUG
	checkCudaErrors(cudaEventRecord(stop,0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elpsTime, start, stop));
	printf("\n\nElapsed time for QuadTree Build:  %9.6f ms \n", elpsTime);
	#endif

	return 0;
}

//filter the data from d > f points to f points according to highest score
//Takes in host side counter
int QuadTreeBuilder::filter(float* x, float* y, float* score, const int d, const int q)
{
	if (numData < 0)
	{
		fprintf(stderr, "QuadTreeBuilder::filter(): numData < 0, must initialize prior to filtering\n");
		return -1;
	}
	if ((d > numData) || (d <= q))
	{
		fprintf(stderr, "QuadTreeBuilder::filter(): d must be <= numData and > q\n");
		return -1;
	}

	//This a tight upperbound upon cells for a target set size of filtered data
	int f = (int)ceil(((float)q*cellMargin - 1.0f)/3.0f);

	ResetFilterArrays(q, width, height);
	FilterQuadTree(d, q, f);

	return 0;
}
//Operates upon internal state
int QuadTreeBuilder::filter()
{
	return filter(d_x, d_y, d_score, numTestData, numFilteredData);
}

//Takes in device side counter
int QuadTreeBuilder::filter(float* x, float* y, float* score, unsigned int* d, const int q)
{
	if (numData < 0)
	{
		fprintf(stderr, "QuadTreeBuilder::filter(): numData < 0, must initialize prior to filtering\n");
		return -1;
	}

	//This a tight upperbound upon cells for a target set size of filtered data
	int f = (int)ceil(((float)q*cellMargin - 1.0f)/3.0f);

	setData(x, y, score, d);
	ResetFilterArrays(q, width, height);
	FilterQuadTreeDev(d, q, f);

	return 0;
}

//Operates upon internal state and external device data intput
int QuadTreeBuilder::filter(unsigned int* d)
{
	return filter(d_x, d_y, d_score, d, numFilteredData);
}
//Write visualization
int QuadTreeBuilder::createBuildViz()
{
	if ((width<0) || (height<0))
	{
		fprintf(stderr, "QuadTreeBuilder::createBuildViz(): width or height < 0, must initialize prior to vizualization\n");
		return -1;
	}

	//Write Random data onto image buffer
	#ifdef TIMINGDEBUG
	checkCudaErrors(cudaEventRecord(start, 0));
	#endif

	d_setBlackImag<<<gridDim, blockDim>>>(d_img, width, height);

	int blocksD = divUp(numNodes - numData, threads);
	std::cout << "BlocksD is " << blocksD << std::endl;
	d_drawCellInnerEdges<<<blocksD, threads>>>(d_img, d_index, d_x, d_y, d_rx, d_ry, width, height, numData, numNodes);

	//Write point last to avoid occulsion by lines (no alpha blending)
	d_writeData2Image<<<blocks, threads>>>(d_img, d_x, d_y, width, height, numData);

	#ifdef TIMINGDEBUG
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elpsTime, start, stop));
	printf("\n\nElapsed time for creating build viz:      %9.6f ms \n", elpsTime);
	#endif

	return 0;
}

//Create filter visualization
int QuadTreeBuilder::createFilterViz()
{
	if ((numTestData < 0) || (numFilteredData < 0) || (numFilteredData > numTestData))
	{
		fprintf(stderr, "QuadTreeBuilder::createFilterViz(): filter parameters not configured correctly\n");
		return -1;
	}

	//Write Random data onto image buffer
	#ifdef TIMINGDEBUG
	checkCudaErrors(cudaEventRecord(start, 0));
	#endif

	d_setBlackImag<<<gridDim, blockDim>>>(d_img, width, height);

	int blocksD = divUp(numNodes - numData, threads);
	std::cout << "BlocksD is " << blocksD << std::endl;
	d_drawCellInnerEdges<<<blocksD, threads>>>(d_img, d_index, d_x, d_y, d_rx, d_ry, width, height, numData, numNodes);

	//Write point last to avoid occulsion by lines (no alpha blending)
	d_writeData2Image<<<blocks, threads>>>(d_img, d_x, d_y, width, height, numTestData);
	
	blocksD = divUp(numFilteredData, threads);
	d_writeFilter2Image<<<blocksD, threads>>>(d_img, d_xf, d_yf, width, height, numFilteredData);

	#ifdef TIMINGDEBUG
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elpsTime, start, stop));
	printf("\n\nElapsed time for creating filter viz:     %9.6f ms \n", elpsTime);
	#endif

	return 0;
}

int QuadTreeBuilder::downloadData()
{
	//Copy back to host for checking    
	checkCudaErrors(cudaMemcpy(h_x, d_x, dataSz, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_y, d_y, dataSz, cudaMemcpyDeviceToHost));

	return 0;
}

int QuadTreeBuilder::downloadFilterData()
{
	//Copy back to host for checking    
	checkCudaErrors(cudaMemcpy(h_xf, d_xf, numFilteredData*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_yf, d_yf, numFilteredData*sizeof(float), cudaMemcpyDeviceToHost));

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
	#ifdef TIMINGDEBUG
	checkCudaErrors(cudaEventRecord(start, 0));
	#endif

	int seed = (int)time(0);

	generate_uniform2D_kernel<<<blocks, threads>>>(d_x, d_y, seed, width, height, numData);

	#ifdef TIMINGDEBUG
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elpsTime, start, stop));
	printf("\n\nElapsed time for Random Data Generation:  %9.6f ms \n", elpsTime);
	#endif

	return 0;
}

int QuadTreeBuilder::resetFilterData()
{
	if ((width<0) || (height<0))
	{
		fprintf(stderr, "QuadTreeBuilder::resetFilterData(): width or height < 0, must initialize prior to data generation\n");
		return -1;
	}

	//Generate Random Data
	#ifdef TIMINGDEBUG
	checkCudaErrors(cudaEventRecord(start, 0));
	#endif

	int seed = (int)time(0);

	generate_uniform2Dfilter_kernel<<<blocks, threads>>>(d_x, d_y, d_score, seed, width, height, numTestData);

	#ifdef TIMINGDEBUG
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elpsTime, start, stop));
	printf("\n\nElapsed time make Random Filter Data:     %9.6f ms \n", elpsTime);
	#endif

	return 0;
}

//Resets arrays used in constructing the quad tree
void QuadTreeBuilder::ResetArrays(const int w, const int h)
{
	reset_arrays_kernel<<<blocks, threads>>>(d_mutex, d_x, d_y, d_rx, d_ry, d_child, d_index, d_left, d_right, d_bottom, d_top, w, h, numData, numNodes);
}
void QuadTreeBuilder::ResetFilterArrays(const int q, const int w, const int h)
{
	reset_filter_arrays_kernel<<<blocks, threads>>>(d_mutex, d_x, d_y, d_score, d_xf, d_yf, d_scoref, d_rx, d_ry, d_child, d_index,
												d_left, d_right, d_bottom, d_top, q, w, h, numData, numNodes);
}

//Builds a quad tree
void QuadTreeBuilder::BuildQuadTree()
{
	build_tree_kernel<<<blocks, threads>>>(d_x, d_y, d_rx, d_ry, d_child, d_index, d_left, d_right, d_bottom, d_top, numData, numNodes);
}

//Filter with quad tree
void QuadTreeBuilder::FilterQuadTree(const int d, const int q, const int f)
{
	#ifdef TIMINGDEBUG
	checkCudaErrors(cudaEventRecord(start,0));
	#endif

	filter_tree_kernel<<<blocks, threads>>>(d_x, d_y, d_score, d_rx, d_ry, d_child, d_index, d_left, d_right, d_bottom, d_top, numData, numNodes, d, f);

	#ifdef TIMINGDEBUG
	checkCudaErrors(cudaEventRecord(stop,0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elpsTime, start, stop));
	printf("\n\nElapsed time for QuadTree Filter:  %9.6f ms \n", elpsTime);

	checkCudaErrors(cudaEventRecord(start,0));
	#endif

	pack_filtered_data_kernel<<<blocks, threads>>>(d_xf, d_yf, d_scoref, d_x, d_y, d_score, d_child, numData, d, q);

	#ifdef TIMINGDEBUG
	checkCudaErrors(cudaEventRecord(stop,0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elpsTime, start, stop));
	printf("\n\nElapsed time for QuadTree Pack:  %9.6f ms \n", elpsTime);
	#endif
}

void QuadTreeBuilder::FilterQuadTreeDev(unsigned int* d, const int q, const int f)
{
	#ifdef TIMINGDEBUG
	checkCudaErrors(cudaEventRecord(start,0));
	#endif

	filter_treeDev_kernel<<<blocks, threads>>>(d_x, d_y, d_score, d_rx, d_ry, d_child, d_index, d_left, d_right, d_bottom, d_top, numData, numNodes, d, f);

	#ifdef TIMINGDEBUG
	checkCudaErrors(cudaEventRecord(stop,0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elpsTime, start, stop));
	printf("\n\nElapsed time for QuadTree Filter:  %9.6f ms \n", elpsTime);

	checkCudaErrors(cudaEventRecord(start,0));
	#endif

	pack_filteredDev_data_kernel<<<blocks, threads>>>(d_xf, d_yf, d_scoref, d_x, d_y, d_score, d_child, numData, d, q);

	#ifdef TIMINGDEBUG
	checkCudaErrors(cudaEventRecord(stop,0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elpsTime, start, stop));
	printf("\n\nElapsed time for QuadTree Pack:  %9.6f ms \n", elpsTime);
	#endif
}

}	//namespace QuadTree

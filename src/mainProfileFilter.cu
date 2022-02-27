//STL
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <time.h>

//CUDA
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_functions.h>    // sdkCreateTimer: contains definitions in header

#include "QuadTreeBuilder.h"

using namespace quadTreeGPU;

//The star of the show
QuadTreeBuilder quadTree;

//Help Information
static void show_usage(std::string name)
{
	std::cerr << "Usage: " << name << " <options(s)>"
			  << "Options:\n"
			  << "\t-i,--help\tShow this help message\n"
			  << "\t-n,\t\tset the number of data points to generate\n"
			  << "\t-m,\t\tset the number of iterations to run\n"
			  << "\t-d,\t\tset the number of data points to generate then filtered\n"
			  << "\t-q,\t\tset the number of filtered data points to generate\n"
			  << "\t-w,\t\tset the width of the image plane\n"
			  << "\t-h,\t\tset the height of the image plane\n"
			  << std::endl;
}

//Forward Declarations
void cleanup();
int runFilter();

int main(int argc, char** argv)
{
	//Parameters
	int D;
	int Q;
	int N;
	int M;
	int W;
	int H;

	//Default Parameters
	D = 16;
	Q = D/2;
	N = 2*D;
	M = 100;
	W = 640;
	H = 480;

	for (int i = 1; i < argc; ++i)
	{
		std::string arg = argv[i];
		if ((arg == "-i") || (arg == "--help"))
		{
			show_usage(argv[0]);
			return 0;  
		} else if (arg == "-n")
		{
			if (i + 1 < argc)
			{
				N = (size_t)atoi(argv[++i]);
			} else
			{
				fprintf(stderr, "-n option requires one argument indicating a sample size.\n");
				return -1;
			}
		}  else if (arg == "-m")
		{
			if (i + 1 < argc)
			{
				M = (size_t)atoi(argv[++i]);
			} else
			{
				fprintf(stderr, "-m option requires one argument indicating a sample size.\n");
				return -1;
			}
		} else if (arg == "-d")
		{
			if (i + 1 < argc)
			{
				D = (size_t)atoi(argv[++i]);
			} else
			{
				fprintf(stderr, "-n option requires one argument indicating a sample size.\n");
				return -1;
			}
		} else if (arg == "-q")
		{
			if (i + 1 < argc)
			{
				Q = (size_t)atoi(argv[++i]);
			} else
			{
				fprintf(stderr, "-n option requires one argument indicating a sample size.\n");
				return -1;
			}
		} else if (arg == "-w")
		{
			if (i + 1 < argc)
			{
				W = (size_t)atoi(argv[++i]);
			} else
			{
				fprintf(stderr, "-w option requires one argument indicating an image width.\n");
				return -1;
			}
		} else if (arg == "-h")
		{
			if (i + 1 < argc)
			{
				H = (size_t)atoi(argv[++i]);
			} else
			{
				fprintf(stderr, "-h option requires one argument indicating an image height\n");
				return -1;
			}
		}
	}

	//Check input constraints
	if (N < D)
	{
		N = 2*D;
		fprintf(stdout, "Setting N to 2*D because inputs had N < D\n");
	}
	if (Q >= D)
	{
		fprintf(stderr, "The number of filtered data points must be less than the number of generated data points\n");
		show_usage(argv[0]);
		return -1;
	}

	//Set QuadTreeBuilder parameters
	quadTree.setParameters(N, W, H, Q, D);

	/* Set Up */
	//Set Device
	int deviceCount;
	checkCudaErrors(cudaGetDeviceCount(&deviceCount));
	if (deviceCount == 0) {
		printf("There is no device supporting CUDA\n");
		return EXIT_FAILURE;
	}

	int dev = 0;
	if (dev >= deviceCount){
		printf("Input error: dev >= deviceCount\n");
		return EXIT_FAILURE;
	}

	cudaSetDevice(dev);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);

	printf("\tDevice %d: \"%s\"\n", dev, deviceProp.name);

	//Allocate Memory
	if (quadTree.allocate() < 0)
	{
		cleanup();
		return -1;
	}

	if (runFilter() < 0)
	{
		cleanup();
		return -1;
	}

	for (int i=0; i< M; i++)
		runFilter();

	cleanup();
	return 0;
}

void cleanup()
{
	// CUDA/OPENGL
	fprintf(stdout,"\tCUDA:\n");
	//Deallocate device memory and destory timers
	quadTree.deallocate();

	fprintf(stdout,"\t\tAll Cuda resources cleaned\n");
}

int runFilter()
{
	//Set random data
	if (quadTree.resetFilterData() < 0)
	{
		return -1;
	}

	// FILTER WITH QUAD TREE
	quadTree.filter();

	return 0;
}

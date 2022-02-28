#ifndef __QUADTREEBUILDER_H__
#define __QUADTREEBUILDER_H__

/* QuadTreeBuilder
This class is for demonstrating how to build a QuadTree on a GPU using CUDA.

It also demonstrates how to filter a data set of size D down to a dataset of
size Q with Q < D, using a quad tree.

It assumes the data and quadtree are defined over an integer grid of size
[W, H] with W and H > 0 aka an Image.

Evaluation applications are built to validate the code through vizualization.
See the following defines to compile with alternative behaviors:

QuadTreeBuild.cu
	TIMINGDEBUG: for wrapping kernels in timers
quadTreeKernels.cu
	BUILDDEBUG: for outputting logic that traces threads
	FILTERDEBUG:  for outputting logic that traces threads
	PACKDEBUG:  for outputting logic that traces threads

author: John D. Long, II PhD	email: jlong29@gmail.com
*/

#include <cuda_runtime.h>

namespace quadTreeGPU
{

class QuadTreeBuilder 
{
	private:
		//General Quad Tree Builder parameters
		int numData;			//Max Possible
		int width;
		int height;

		//Optional Quad Tree Filter parameters
		int numFilteredData;	// < numTestData
		int numTestData;		// <= numData

		int numNodes;			//Max Possible

		//A constant multipler on the number of filtered data
		//to create enough cells to return the requested number
		//of filtered data (this is a fudge factor for handling random insertion order)
		float cellMargin;

		unsigned int* d_numTestData;

		float* d_left;
		float* d_right;
		float* d_bottom;
		float* d_top;

		float* d_x;
		float* d_y;
		float* d_score;
		float* d_rx;	//horizontal cell radius
		float* d_ry;	//vertical cell radius
		size_t dataSz;
		size_t nodeSz;

		int*   d_child;

		int*   d_index;
		int*   d_mutex;  //used for locking 

		float elpsTime;
		bool timersCreated;
		cudaEvent_t start, stop; // used for timing

	public:
		//Packed arrays from filtered data
		float* d_xf;
		float* d_yf;
		float* d_scoref;

		float* h_x;
		float* h_y;

		float* h_xf;
		float* h_yf;

		uchar4* d_img;
		size_t imgSz;

		QuadTreeBuilder();
		QuadTreeBuilder(int n, int w, int h, int q = -1, int d = -1);
		~QuadTreeBuilder();

		int allocate();
		void deallocate();

		//Set parameters: n is required for functioning. w and h are for visualization
		void setParameters(int n, int w, int h, int q = -1, int d = -1);

		//Set data generated by another GPU process
		void setData(const float* x, const float* y);
		void setData(const float* x, const float* y, const float* score, const int d);
		void setData(float* x, float* y, float* score, const unsigned int* d);
		void setData(float* x, float* y, float* score, const unsigned int* d, const int q);

		void setCellMargin(const float cm);

		int getNumData();
		int resetData();
		int resetFilterData();

		//build the quad tree
		int build();

		//filter the data from m > n points to n points according to highest score
		//Takes in host side counter
		int filter(float* x, float* y, float* score, const int d, const int q);
		//Operates upon internal state
		int filter();
		//Takes in device side counter
		int filter(float* x, float* y, float* score, unsigned int* d, const int q);
		//Operates upon internal state and external device data intput
		int filter(unsigned int* d);
		//Takes in device side counter
		int filter_async(float* x, float* y, float* score, unsigned int* d, const int q);
		//Operates upon internal state and external device data intput
		int filter_async(unsigned int* d);
		//Operates upon internal state
		int filter_async();

		//Create build visualization
		int createBuildViz();
		//Create filter visualization
		int createFilterViz();

		int downloadData();
		int downloadFilterData();

		//Join Cuda Stream
		void join();

	private:
		//Root GPU Launch Optimization
		//1D
		int threads;
		int blocks;

		//2D
		dim3 blockDim, gridDim;

		cudaStream_t stream;
		bool streamCreated;

		//Resets arrays used in constructing the quad tree
		void ResetArrays(const int w, const int h);
		void ResetFilterArrays(const int q, const int w, const int h);
		void BuildQuadTree();
		void FilterQuadTree(const int d, const int q, const int f);
		void FilterQuadTreeDev(unsigned int* d, const int q, const int f);

		static inline int divUp(int x, int y)
		{
			return (x + y - 1) / y;
		}
};

}	//namespace quadTreeGPU

#endif

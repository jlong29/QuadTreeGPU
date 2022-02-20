#ifndef __QUADTREEBUILDER_H__
#define __QUADTREEBUILDER_H__

/* QuadTreeBuilder
This class is for demonstrating how to build a QuadTree on a GPU using CUDA.
It assumes the output data and quadtree will be vizualized, and as such the
toy data that is created is generated to fit within an image. The underlying
routines for generating the QuadTree do not require this constraint

author: John D. Long, II PhD	email: jlong29@gmail.com
*/

#include <cuda_runtime.h>

class QuadTreeBuilder 
{
	private:
		int width;
		int height;
		int numData;

		int numNodes;

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
		float* h_x;
		float* h_y;

		uchar4* d_img;
		size_t imgSz;

		QuadTreeBuilder();
		QuadTreeBuilder(int n, int w = -1, int h = -1);
		~QuadTreeBuilder();

		int allocate();
		void deallocate();

		//Set parameters: n is required for functioning. w and h are for visualization
		void setParameters(int n, int w = -1, int h = -1);

		//Set data generated by another GPU process
		void setData(const float* x, const float* y);
		void setData(const float* x, const float* y, const float* score, const int d);

		int getNumData();
		int resetData();
		int resetFilterData();

		//build the quad tree
		int build();

		//filter the data from m > n points to n points according to highest score
		int filter(const int d, const int f);

		//Create build visualization
		int createBuildViz();

		int downloadData();

	private:
		//Root GPU Launch Optimization
		//1D
		int threads;
		int blocks;

		//2D
		dim3 blockDim, gridDim;

		//Resets arrays used in constructing the quad tree
		void ResetArrays();
		void ResetArrays(const int w, const int h);
		void ComputeBoundingBox();
		void BuildQuadTree();
		void FilterQuadTree(const int d, const int f);

		static inline int divUp(int x, int y)
		{
			return (x + y - 1) / y;
		}
};

#endif

#include <stdio.h>
#include <math_constants.h>
#include <curand_kernel.h>

#include "quadTreeKernels.h"

namespace quadTreeGPU{ namespace quadTreeKernels
{

// #define BUILDDEBUG
// #define FILTERDEBUG
// #define PACKDEBUG

// Image Processing //
//Image bounds check
__device__ __forceinline__ int in_img(int x, int y, int w, int h)
{
	return x >= 0 && x < w && y >= 0 && y < h;
}
//pixel coloring
__device__ __inline__ void setRedHue(const uchar& hue, uchar4& RGBA)
{
	RGBA.x = hue;
	RGBA.y = 0;
	RGBA.z = 0;
	RGBA.w = 255;
}
__device__ __inline__ void setGreenHue(const uchar& hue, uchar4& RGBA)
{
	RGBA.x = 0;
	RGBA.y = hue;
	RGBA.z = 0;
	RGBA.w = 255;

}
__device__ __inline__ void setBlueHue(const uchar& hue, uchar4& RGBA)
{
	RGBA.x = 0;
	RGBA.y = 0;
	RGBA.z = hue;
	RGBA.w = 255;
}
__device__ __inline__ void setGrayHue(const uchar& hue, uchar4& RGBA)
{
	RGBA.x = hue;
	RGBA.y = hue;
	RGBA.z = hue;
	RGBA.w = 255;
}
__device__ __inline__ void setBlack(uchar4& RGBA)
{
	// points as black(transparent)
	RGBA.x = 0;
	RGBA.y = 0;
	RGBA.z = 0;
	RGBA.w = 0;
}
__device__ __inline__ void setYellow(uchar4& RGBA)
{
	// points as black(transparent)
	RGBA.x = 255;
	RGBA.y = 234;
	RGBA.z = 0;
	RGBA.w = 255;
}

//Black Image: device and host code
__global__ void d_setBlackImag(uchar4* dst, const int w, const int h)
{
	//Position of the thread in the image
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	//Early exit if the thread is not in the image
	if (!in_img(x, y, w, h))
		return;

	//Set to RGBA black
	setBlack(dst[y*w + x]);
}

//GrayScale Image: device and host code
__global__ void d_setGrayScaleImag(uchar4* dst, const int w, const int h)
{
	// Position of the thread in the image.
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	// Early exit if the thread is not in the image.
	if (!in_img(x, y, w, h))
		return;

	//Set to RGBA grayscale
	setGrayHue(127, dst[y*w + x]);
}

//Write Random data onto image buffer
__global__ void d_writeData2Image(uchar4* dst, const float* __restrict noiseX, const float* __restrict noiseY,const int w, const int h, const int n)
{
	int idx        = threadIdx.x + blockIdx.x * blockDim.x;
	int numThreads = blockDim.x*gridDim.x;

	for(int i = idx; i < n; i+=numThreads)
	{
		int shotX = (int)noiseX[i];
		int shotY = (int)noiseY[i];
		if (in_img(shotX, shotY, w, h))
		{
			//Mark 2x2 pixel region
			setGreenHue(255, dst[shotY*w + shotX]);
			setGreenHue(255, dst[shotY*w + shotX+1]);
			setGreenHue(255, dst[(shotY+1)*w + shotX]);
			setGreenHue(255, dst[(shotY+1)*w + shotX+1]);
		}
	}
}

__global__ void d_writeFilter2Image(uchar4* dst, const float* __restrict filterX, const float* __restrict filterY,const int w, const int h, const int n)
{
	int idx        = threadIdx.x + blockIdx.x * blockDim.x;
	int numThreads = blockDim.x*gridDim.x;

	for(int i = idx; i < n; i+=numThreads)
	{
		int shotX = (int)filterX[i];
		int shotY = (int)filterY[i];
		if (in_img(shotX, shotY, w, h))
		{
			//Mark 2x2 pixel region
			setBlueHue(255, dst[shotY*w + shotX]);
			setBlueHue(255, dst[shotY*w + shotX+1]);
			setBlueHue(255, dst[(shotY+1)*w + shotX]);
			setBlueHue(255, dst[(shotY+1)*w + shotX+1]);
		}
	}
}

//Draw internal edges of cells
__global__ void d_drawCellInnerEdges(uchar4* dst, int* index, const float* __restrict x, const float* __restrict y, const float* __restrict rx, const float* __restrict ry,
										const int w, const int h, const int n, const int m)
{
	//Global WarpID
	int wid    = (threadIdx.x + blockDim.x*blockIdx.x) / WARPSIZE;
	//Block level WarpID
	int widB   = threadIdx.x / WARPSIZE;
	//Lane within warp
	int lane   = threadIdx.x % WARPSIZE;
	//Global Warp Stride
	int stride = blockDim.x*gridDim.x / WARPSIZE;

	//Stores per block, per warp data
	static __shared__ float4 shared[32];

	// Process cell one warp at a time
	while(wid < (m-n))
	{
		//First lane checks for valid cell
		if (lane == 0)
		{
			if (!isnan(rx[wid]))
			{
				shared[widB] = make_float4(x[wid+n], y[wid+n], rx[wid], ry[wid]);
				int old = atomicSub(index, 1);
			} else
			{
				shared[widB] = make_float4(CUDART_NAN_F, CUDART_NAN_F, CUDART_NAN_F, CUDART_NAN_F);
			}
		}
		__syncthreads();

		//All threads load from shared into registers
		float4 cell = shared[widB];
		if (!isnan(cell.x))
		{
			//Draw cell
			int xC  = (int)cell.x;
			int yC  = (int)cell.y;
			int rxC = (int)cell.z;
			int ryC = (int)cell.w;

			//Horizontal Edge through yC
			for (int ii=xC-rxC + lane; ii < xC+rxC; ii+=WARPSIZE)
				setRedHue(255, dst[yC*w + ii]);

			//Vertical Edge Through xC
			for (int ii=yC-ryC + lane; ii < yC+ryC; ii+=WARPSIZE)
				setRedHue(255, dst[ii*w + xC]);
		}
		wid += stride;
	}

}
// Random Number Generators //
// Generate 2D uniform random values
__global__ void generate_uniform2D_kernel(float* noiseX, float* noiseY, int seed, const int w, const int h, const int n)
{
	int idx        = threadIdx.x + blockIdx.x * blockDim.x;
	int numThreads = blockDim.x*gridDim.x;

	curandState localState;

	/* Each thread gets different seed, a different sequence number, no offset */
	curand_init(seed, idx, 0, &localState);

	/* Generate pseudo-random normals */
	for(int i = idx; i < n; i+=numThreads)
	{
		// Generate and store
		noiseX[i] = (float)w*curand_uniform(&localState);
		noiseY[i] = (float)h*curand_uniform(&localState);
	}
}
// Generate 2D uniform random coordinate and filter values
__global__ void generate_uniform2Dfilter_kernel(float* noiseX, float* noiseY, float* score, int seed, const int w, const int h, const int n)
{
	//Scores are random uniform [0 1)
	int idx        = threadIdx.x + blockIdx.x * blockDim.x;
	int numThreads = blockDim.x*gridDim.x;

	curandState localState;

	/* Each thread gets different seed, a different sequence number, no offset */
	curand_init(seed, idx, 0, &localState);

	/* Generate pseudo-random normals */
	for(int i = idx; i < n; i+=numThreads)
	{
		// Generate and store
		noiseX[i] = (float)w*curand_uniform(&localState);
		noiseY[i] = (float)h*curand_uniform(&localState);
		score[i]  = curand_uniform(&localState);
	}
}

// Quad Tree Routines //
__global__ void reset_arrays_kernel(int* mutex, float* x, float* y, float* rx, float* ry, int* child, int* index,
										float* left, float* right, float* bottom, float* top, const int w, const int h, int n, int m)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = blockDim.x*gridDim.x;
	int offset = 0;

	// reset quadtree arrays
	while(idx + offset < m)
	{  
		#pragma unroll 4
		for(int i=0;i<4;i++)
		{
			child[(idx + offset)*4 + i] = -1;
		}
		if(idx + offset >= n)
		{
			x[idx + offset] = CUDART_NAN_F;
			y[idx + offset] = CUDART_NAN_F;
			rx[idx + offset - n] = CUDART_NAN_F;
			ry[idx + offset - n] = CUDART_NAN_F;
		}
		offset += stride;
	}

	//To ensure the write below doesn't get overwritten from above
	__threadfence();

	//Set bounds to image bounds
	if(idx == 0)
	{
		*mutex = 0;
		*index = n+1;	//Set to n + 1 to allow for root
		*left = 0.0f;
		*right = (float)w;
		*bottom = 0.0f;
		*top = (float)h;
		//set root coordinates
		//Create a new cell, starting at index n
		x[n]  = 0.5f*(float)w;
		y[n]  = 0.5f*(float)h;
		rx[0] = 0.5f*(float)w;
		ry[0] = 0.5f*(float)h;
	}
}

__global__ void reset_filter_arrays_kernel(int* mutex, float* x, float* y, float* score, float* xf, float* yf, float* scoref,
											float* rx, float* ry, int* child, int* index, float* left, float* right, float* bottom, float* top,
											const int q, const int w, const int h, int n, int m)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = blockDim.x*gridDim.x;
	int offset = 0;

	// reset quadtree arrays
	while(idx + offset < m)
	{  
		#pragma unroll 4
		for(int i=0;i<4;i++)
		{
			child[(idx + offset)*4 + i] = -1;
		}
		if (idx + offset < q)
		{
			xf[idx + offset] = CUDART_NAN_F;
			yf[idx + offset] = CUDART_NAN_F;
			scoref[idx + offset]  = CUDART_NAN_F;
		}
		if(idx + offset >= n)
		{
			x[idx + offset] = CUDART_NAN_F;
			y[idx + offset] = CUDART_NAN_F;
			rx[idx + offset - n] = CUDART_NAN_F;
			ry[idx + offset - n] = CUDART_NAN_F;
		}
		offset += stride;
	}

	//To ensure the write below doesn't get overwritten from above
	__threadfence();

	//Set bounds to image bounds
	if(idx == 0)
	{
		*mutex  = 0;
		*index  = n+1;	//Set to n + 1 to allow for root
		*left   = 0.0f;
		*right  = (float)w;
		*bottom = 0.0f;
		*top    = (float)h;
		//set root coordinates
		//Create a new cell, starting at index n
		x[n]  = 0.5f*(float)w;
		y[n]  = 0.5f*(float)h;
		rx[0] = 0.5f*(float)w;
		ry[0] = 0.5f*(float)h;
	}
}

__global__ void d_setData(float* x, float* y, float* score, unsigned int* pd, float* xn, float* yn, float* scoren, const unsigned int* pdn)
{
	int idx    = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = blockDim.x*gridDim.x;

	//Read d into shared memory and broadcast
	static __shared__ int shared[1];
	if (threadIdx.x == 0)
		shared[0] = (int)(*pdn);
	__syncthreads();

	//Write to local register
	int d = shared[0];

	//First thread first out to device counter
	if (idx == 0)
		pd[0] = pdn[0];

	// reset quadtree arrays
	while(idx < d)
	{
		//Copy over data
		x[idx]     = xn[idx];
		y[idx]     = yn[idx];
		score[idx] = scoren[idx];

		idx += stride;
	}
}

__global__ void build_tree_kernel(volatile float *x, volatile float *y, float* rx, float* ry, volatile int *child, int *index,
									const float *left, const float *right, const float *bottom, const float *top,
									const int n, const int m)
{
	/*
	This routine combines building the Quad Tree with summarizing internal node information
	index:	a global index start at n
	n:		the number of data
	m:		the number of possible nodes
	*/

	int idx    = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x*gridDim.x;

	// build quadtree
	float l;
	float r;
	float b;
	float t;
	int childPath;
	int node;

	bool newBody  = true;
	float posX, posY;
	while(idx < n){

		if(newBody){
			newBody = false;
			//Top/Down Traversal: All particles start in one of the top 4 quads
			l = *left;
			r = *right;
			b = *bottom;
			t = *top;

			node      = n;
			childPath = 0;
			posX      = x[idx];
			posY      = y[idx];

			//Check body location within the top 4 nodes
			if(posX < 0.5*(l+r)){
				childPath += 1;
				r = 0.5*(l+r);
			}
			else{
				l = 0.5*(l+r);
			}
			if(posY < 0.5*(b+t)){
				childPath += 2;
				t = 0.5*(t+b);
			}
			else{
				b = 0.5*(t+b);
			}
		}

		//Set childIndex, which could be after mutliple loops
		int childIndex = child[node*4 + childPath];

		// traverse tree until we hit leaf node (could be allocated or not)

		//NOTE: childIndex >= n means we are in a cell not a leaf
		// You could also land in an unallocated (-1) or locked (-2) node
		while(childIndex >= n){
			//Check body location within the 4 quads of this node
			node = childIndex;
			childPath = 0;
			if(posX < 0.5*(l+r)){
				childPath += 1;
				r = 0.5*(l+r);
			}
			else{
				l = 0.5*(l+r);
			}
			if(posY < 0.5*(b+t)){
				childPath += 2;
				t = 0.5*(t+b);
			}
			else{
				b = 0.5*(t+b);
			}

			//Advance to child of this cell
			childIndex = child[4*node + childPath];
		}

		//At this point childIndex: [-1 n]

		// Check if child is already locked i.e. childIndex == -2
		if(childIndex != -2){
			//Acquire lock, which is only possible if child[locked]: [-1 n]
			int locked = node*4 + childPath;
			if(atomicCAS((int*)&child[locked], childIndex, -2) == childIndex){
				//If unallocated, insert body and unlock
				if(childIndex == -1){
					// Insert body and release lock
					child[locked] = idx;
				}
				else{
					//Sets max on number of cells
					int patch = 4*n;
					while(childIndex >= 0){

						// childIndex should always be -1, unallocated, or >=0, allocated

						//Create a new cell, starting at index n
						int cell = atomicAdd(index,1);

						//Compare against maximum allowable cells
						patch = min(patch, cell);

						//If the maximum number of cells have been reached:
						// It prunes away the node above
						if(patch != cell){
							child[4*node + childPath] = cell;
						}

						// insert old particle into new cell
						childPath = 0;
						if(x[childIndex] < 0.5*(l+r)){
							childPath += 1;
						}
						if(y[childIndex] < 0.5*(b+t)){
							childPath += 2;
						}

						#ifdef BUILDDEBUG
							// if(cell >= 2*n){
							if(cell >= m){
								printf("%s\n", "error cell index is too large!!");
								printf("cell: %d\n", cell);
							}
						#endif

						//Assign old particle to subtree leaf
						child[4*cell + childPath] = childIndex;

						//SET ROOT OF NEW CELL AND LENGTH OF SIDES
						x[cell]    = 0.5*(l+r);
						y[cell]    = 0.5*(b+t);
						rx[cell-n] = 0.5*(r-l);
						ry[cell-n] = 0.5*(t-b);

						// insert new particle
						node = cell;
						childPath = 0;
						if(posX < 0.5*(l+r)){
							childPath += 1;
							r = 0.5*(l+r);
						}
						else{
							l = 0.5*(l+r);
						}
						if(posY < 0.5*(b+t)){
							childPath += 2;
							t = 0.5*(t+b);
						}
						else{
							b = 0.5*(t+b);
						}

						//Set to value of child at this entry, which could be:
						// -1 == break
						// > n if new data landed in the same part of the sub-tree as old data
						childIndex = child[4*node + childPath];
					}

					//This means childIndex is set to -1, unallocated, so allocated as body Index
					child[4*node + childPath] = idx;

					__threadfence();  // Ensures all writes to global memory are complete before lock is released

					//Release lock and replace leaf with this cell
					child[locked] = patch;
				}	// if(childIndex == -1): first assignment to body or not

				//Advance to next body
				idx += stride;
				newBody    = true;
			}	//if(atomicCAS((int*)&child[locked], childIndex, -2) == childIndex)

		}	//if(childIndex != -2): locked already or not. If locked, go around again

		// Wait for threads in block to release locks to reduce memory pressure
		__syncthreads(); // not needed for correctness
	}
}

__global__ void filter_tree_kernel(volatile float* x, volatile float* y, volatile float* score,
									float* rx, float* ry, volatile int* child, int* index,
									const float* left, const float* right, const float* bottom, const float* top,
									const int n, const int m, const int d, const int f)
{
	/*
	This routine combines building the Quad Tree with spatial filtering and summarizing internal node information
	index:	a global index start at n
	n:		the number of possible data
	m:		the number of possible nodes
	d:		the number of current data
	f:		the maximum number of cells to be created, which limits data by single occupancy filter
	*/

	#ifdef FILTERDEBUG
	if (d > 32)
	{
		//Don't break your device with printf if running in FILTERDEBUG mode
		printf("FILTERDEBUG mode has a max of 32 on the value of d\n");
		return;
	}
	#endif

	int idx    = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x*gridDim.x;

	// build quadtree
	float l;
	float r;
	float b;
	float t;
	int childPath;
	int node;

	bool newBody  = true;
	float posX, posY;
	while(idx < d){

		if(newBody){
			newBody = false;
			//Top/Down Traversal: All particles start in one of the top 4 quads
			l = *left;
			r = *right;
			b = *bottom;
			t = *top;

			node      = n;
			childPath = 0;
			posX      = x[idx];
			posY      = y[idx];

			//Check body location within the top 4 nodes
			if(posX < 0.5*(l+r)){
				childPath += 1;
				r = 0.5*(l+r);
			}
			else{
				l = 0.5*(l+r);
			}
			if(posY < 0.5*(b+t)){
				childPath += 2;
				t = 0.5*(t+b);
			}
			else{
				b = 0.5*(t+b);
			}
		}

		//Set childIndex, which could be after mutliple loops
		int childIndex = child[node*4 + childPath];

		// traverse tree until we hit leaf node (could be allocated or not)

		//NOTE: childIndex >= d means we are in a cell not a leaf
		// You could also land in an unallocated (-1) or locked (-2) node
		while(childIndex >= d){
			//Check body location within the 4 quads of this node
			node = childIndex;
			childPath = 0;
			if(posX < 0.5*(l+r)){
				childPath += 1;
				r = 0.5*(l+r);
			}
			else{
				l = 0.5*(l+r);
			}
			if(posY < 0.5*(b+t)){
				childPath += 2;
				t = 0.5*(t+b);
			}
			else{
				b = 0.5*(t+b);
			}

			//Advance to child of this cell
			childIndex = child[4*node + childPath];
		}

		//At this point childIndex: [-1 d]

		// Check if child is already locked i.e. childIndex == -2
		if(childIndex != -2){
			//Acquire lock, which is only possible if child[locked]: [-1 d]
			int locked = 4*node + childPath;
			if(atomicCAS((int*)&child[locked], childIndex, -2) == childIndex){
				//If unallocated, insert body and unlock
				if(childIndex == -1){
					// Insert body and release lock
					child[locked] = idx;
					#ifdef FILTERDEBUG
						printf("Initializing with %02d at [%f, %f]\n", idx, x[idx], y[idx]);
					#endif
				}
				else{
					//Sets max on number of cells
					int patch = 4*n;
					bool bMoreCells = true;

					//for handling the case of new and old data landing in same node
					int parentCell  = -1;
					int tmpIdx;
					while(childIndex >= 0)
					{
						// childIndex should always be -1, unallocated, or >=0, allocated

						//Create a new cell, starting at index n
						int cell = atomicAdd(index,1);

						//Compare against maximum allowable cells
						patch = min(patch, cell);

						//If f cells already created, filter by response
						if (cell - n >= f)
						{
							int keeper = idx;
							if (score[childIndex] < score[idx])
							{
								// Replace data and release lock
								keeper = idx;
								#ifdef FILTERDEBUG
									printf("\tSwapping %d with %d\n", childIndex, idx);
								#endif
							} else
							{
								//... or put it back to unlock
								keeper = childIndex;
								#ifdef FILTERDEBUG
									printf("\tKeeping %d over %d\n", childIndex, idx);
								#endif
							}
							//Check for the case of new and old data landing in same node
							if (parentCell > 0)
							{
								child[tmpIdx] = keeper;
								__threadfence();
								child[locked] = parentCell;
							} else
							{
								child[locked] = keeper;
							}

							bMoreCells = false;
							break;
						}

						//If the maximum number of cells have been reached:
						// It prunes away the node above
						if(patch != cell){
							child[4*node + childPath] = cell;
						}

						// insert old particle into new cell
						childPath = 0;
						if(x[childIndex] < 0.5*(l+r)){
							childPath += 1;
						}
						if(y[childIndex] < 0.5*(b+t)){
							childPath += 2;
						}

						#ifdef FILTERDEBUG
							// if(cell >= 2*n){
							if(cell >= m){
								printf("%s\n", "error cell index is too large!!");
								printf("cell: %d\n", cell);
							}
						#endif

						//Assign old particle to subtree leaf
						child[4*cell + childPath] = childIndex;

						//SET ROOT OF NEW CELL AND LENGTH OF SIDES
						x[cell]    = 0.5*(l+r);
						y[cell]    = 0.5*(b+t);
						rx[cell-n] = 0.5*(r-l);
						ry[cell-n] = 0.5*(t-b);

						// insert new particle
						parentCell = cell;
						node       = cell;
						childPath = 0;
						if(posX < 0.5*(l+r)){
							childPath += 1;
							r = 0.5*(l+r);
						}
						else{
							l = 0.5*(l+r);
						}
						if(posY < 0.5*(b+t)){
							childPath += 2;
							t = 0.5*(t+b);
						}
						else{
							b = 0.5*(t+b);
						}

						//Set to value of child at this entry, which could be:
						// -1 == break
						// > n if new data landed in the same part of the sub-tree as old data
						tmpIdx     = 4*node + childPath;
						childIndex = child[tmpIdx];
					}

					if (bMoreCells)
					{
						//This means childIndex is set to -1, unallocated, so allocated as body Index
						child[4*node + childPath] = idx;
						#ifdef FILTERDEBUG
							printf("Initializing NEW with %02d at [%f, %f]\n", idx, x[idx], y[idx]);
						#endif

						__threadfence();  // Ensures all writes to global memory are complete before lock is released

						//Release lock and replace leaf with this cell
						child[locked] = patch;
						#ifdef FILTERDEBUG
							printf("Releasing lock as %d\n", patch);
						#endif
					}
				}	// if(childIndex == -1): first assignment to body or not

				//Advance to next body
				idx += stride;
				newBody    = true;
			}	//if(atomicCAS((int*)&child[locked], childIndex, -2) == childIndex)

		}	//if(childIndex != -2): locked already or not. If locked, go around again

		// Wait for threads in block to release locks to reduce memory pressure
		__syncthreads(); // not needed for correctness
	}

	#ifdef FILTERDEBUG
		__syncthreads();
		if (threadIdx.x + blockIdx.x*blockDim.x == 0)
		{
			for (int i = 0; i < 16; i++){
				printf("%d, ", child[4*n+i]);
			}
			printf("\n");
		}
	#endif
}

__global__ void filter_treeDev_kernel(volatile float* x, volatile float* y, volatile float* score,
									float* rx, float* ry, volatile int* child, int* index,
									const float* left, const float* right, const float* bottom, const float* top,
									const int n, const int m, unsigned int* pd, const int f)
{
	/*
	This routine combines building the Quad Tree with spatial filtering and summarizing internal node information
	index:	a global index start at n
	n:		the number of possible data
	m:		the number of possible nodes
	d:		the number of current data
	f:		the maximum number of cells to be created, which limits data by single occupancy filter
	*/

	//Read d into shared memory and broadcast
	static __shared__ int shared[1];
	if (threadIdx.x == 0)
		shared[0] = (int)(*pd);
	__syncthreads();

	//Write to local register
	int d = shared[0];

	#ifdef FILTERDEBUG
	if (d > 32)
	{
		//Don't break your device with printf if running in FILTERDEBUG mode
		printf("FILTERDEBUG mode has a max of 32 on the value of d\n");
		return;
	}
	#endif

	int idx    = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x*gridDim.x;

	// build quadtree
	float l;
	float r;
	float b;
	float t;
	int childPath;
	int node;

	bool newBody  = true;
	float posX, posY;
	while(idx < d){

		if(newBody){
			newBody = false;
			//Top/Down Traversal: All particles start in one of the top 4 quads
			l = *left;
			r = *right;
			b = *bottom;
			t = *top;

			node      = n;
			childPath = 0;
			posX      = x[idx];
			posY      = y[idx];

			//Check body location within the top 4 nodes
			if(posX < 0.5f*(l+r)){
				childPath += 1;
				r = 0.5f*(l+r);
			}
			else{
				l = 0.5f*(l+r);
			}
			if(posY < 0.5f*(b+t)){
				childPath += 2;
				t = 0.5f*(t+b);
			}
			else{
				b = 0.5f*(t+b);
			}
		}

		//Set childIndex, which could be after mutliple loops
		int childIndex = child[node*4 + childPath];

		// traverse tree until we hit leaf node (could be allocated or not)

		//NOTE: childIndex >= d means we are in a cell not a leaf
		// You could also land in an unallocated (-1) or locked (-2) node
		while(childIndex >= d){
			//Check body location within the 4 quads of this node
			node = childIndex;
			childPath = 0;
			if(posX < 0.5f*(l+r)){
				childPath += 1;
				r = 0.5f*(l+r);
			}
			else{
				l = 0.5f*(l+r);
			}
			if(posY < 0.5f*(b+t)){
				childPath += 2;
				t = 0.5f*(t+b);
			}
			else{
				b = 0.5f*(t+b);
			}

			//Advance to child of this cell
			childIndex = child[4*node + childPath];
		}

		//At this point childIndex: [-1 d]

		// Check if child is already locked i.e. childIndex == -2
		if(childIndex != -2){
			//Acquire lock, which is only possible if child[locked]: [-1 d]
			int locked = 4*node + childPath;
			if(atomicCAS((int*)&child[locked], childIndex, -2) == childIndex){
				//If unallocated, insert body and unlock
				if(childIndex == -1){
					// Insert body and release lock
					child[locked] = idx;
					#ifdef FILTERDEBUG
						printf("Initializing with %02d at [%f, %f]\n", idx, x[idx], y[idx]);
					#endif
				}
				else{
					//Sets max on number of cells
					int patch = 4*n;
					bool bMoreCells = true;

					//for handling the case of new and old data landing in same node
					int parentCell  = -1;
					int tmpIdx;
					while(childIndex >= 0)
					{
						// childIndex should always be -1, unallocated, or >=0, allocated

						//Create a new cell, starting at index n
						int cell = atomicAdd(index,1);

						//Compare against maximum allowable cells
						patch = min(patch, cell);

						//If f cells already created, filter by response
						if (cell - n >= f)
						{
							int keeper = idx;
							if (score[childIndex] < score[idx])
							{
								// Replace data and release lock
								keeper = idx;
								#ifdef FILTERDEBUG
									printf("\tSwapping %d with %d\n", childIndex, idx);
								#endif
							} else
							{
								//... or put it back to unlock
								keeper = childIndex;
								#ifdef FILTERDEBUG
									printf("\tKeeping %d over %d\n", childIndex, idx);
								#endif
							}
							//Check for the case of new and old data landing in same node
							if (parentCell > 0)
							{
								child[tmpIdx] = keeper;
								__threadfence();
								child[locked] = parentCell;
							} else
							{
								child[locked] = keeper;
							}

							bMoreCells = false;
							break;
						}

						//If the maximum number of cells have been reached:
						// It prunes away the node above
						if(patch != cell){
							child[4*node + childPath] = cell;
						}

						// insert old particle into new cell
						childPath = 0;
						if(x[childIndex] < 0.5f*(l+r)){
							childPath += 1;
						}
						if(y[childIndex] < 0.5f*(b+t)){
							childPath += 2;
						}

						#ifdef FILTERDEBUG
							// if(cell >= 2*n){
							if(cell >= m){
								printf("%s\n", "error cell index is too large!!");
								printf("cell: %d\n", cell);
							}
						#endif

						//Assign old particle to subtree leaf
						child[4*cell + childPath] = childIndex;

						//SET ROOT OF NEW CELL AND LENGTH OF SIDES
						x[cell]    = 0.5*(l+r);
						y[cell]    = 0.5*(b+t);
						rx[cell-n] = 0.5*(r-l);
						ry[cell-n] = 0.5*(t-b);

						// insert new particle
						parentCell = cell;
						node       = cell;
						childPath = 0;
						if(posX < 0.5f*(l+r)){
							childPath += 1;
							r = 0.5f*(l+r);
						}
						else{
							l = 0.5f*(l+r);
						}
						if(posY < 0.5f*(b+t)){
							childPath += 2;
							t = 0.5f*(t+b);
						}
						else{
							b = 0.5f*(t+b);
						}

						//Set to value of child at this entry, which could be:
						// -1 == break
						// > n if new data landed in the same part of the sub-tree as old data
						tmpIdx     = 4*node + childPath;
						childIndex = child[tmpIdx];
					}

					if (bMoreCells)
					{
						//This means childIndex is set to -1, unallocated, so allocated as body Index
						child[4*node + childPath] = idx;
						#ifdef FILTERDEBUG
							printf("Initializing NEW with %02d at [%f, %f]\n", idx, x[idx], y[idx]);
						#endif

						__threadfence();  // Ensures all writes to global memory are complete before lock is released

						//Release lock and replace leaf with this cell
						child[locked] = patch;
						#ifdef FILTERDEBUG
							printf("Releasing lock as %d\n", patch);
						#endif
					}
				}	// if(childIndex == -1): first assignment to body or not

				//Advance to next body
				idx += stride;
				newBody    = true;
			}	//if(atomicCAS((int*)&child[locked], childIndex, -2) == childIndex)

		}	//if(childIndex != -2): locked already or not. If locked, go around again

		// Wait for threads in block to release locks to reduce memory pressure
		__syncthreads(); // not needed for correctness
	}

	#ifdef FILTERDEBUG
		__syncthreads();
		if (threadIdx.x + blockIdx.x*blockDim.x == 0)
		{
			for (int i = 0; i < 16; i++){
				printf("%d, ", child[4*n+i]);
			}
			printf("\n");
		}
	#endif
}

__global__ void pack_filtered_data_kernel(float* xf, float* yf, float* scoref,
											float* x, float* y, float* score,
											int* child, const int n, const int d, const int q)
{
	/*
	Data filtered through the Quad Tree are scattered across the child array. We need to pack
	them into xf, yf, and scoref.

	It uses depth-first search.

	Marking leaves or internal cells as -1 means they have been fully processed (unallocated again)

	index:	a global index start at n
	n:		root of tree node index
	d:		the number of current data
	q:		the number of filtered data
	*/

	#ifdef PACKDEBUG
	if (d > 32)
	{
		//Don't break your device with printf if running in PACKDEBUG mode
		printf("PACKDEBUG mode has a max of 32 on the value of d\n");
		return;
	}
	#endif

	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if (idx >= q)
		return;
	int lane = threadIdx.x % 4;

	//Shift indices beyond 4 to break up synchrony
	int shift = threadIdx.x % 5;
	int offsets[] = { 0, 1, 2, 3, 0, 1, 2, 3, 0 };

	int parentIndex;
	int parentNode;

	//Every thread will hit a leaf
	int  childIndex;
	bool notAtTop = false;

	int iterations = 0;
	while(true)
	{
		iterations++;

		//Start at the top of the tree
		if (!notAtTop)
		{
			//Start at Root Node
			parentNode = n;

			//Inspect children of root for initial parentIndex
			// - At least one of them should always be active
			//Returns the NEXT parent node when indexing child array
			parentIndex = 4*parentNode + lane;

			childIndex  = child[parentIndex];
			if (childIndex >= 0)
			{
				//Leaf check
				if ((childIndex < d) && (childIndex >=0))
				{
					//We're at a leaf, so set to unallocated = -1
					if(atomicCAS((int*)&child[parentIndex], childIndex, -1) == childIndex)
					{
						//This thread is the first here, so the childIndex goes with it
						#ifdef PACKDEBUG
							printf("\tThread %d will write to childIndex %d\n", idx, childIndex);
						#endif
						break;
					} else
					{
						#ifdef PACKDEBUG
							printf("\tThread %d was too slow\n", idx);
						#endif
					}
				}
				//Advance down the tree and assign new parent node
				parentNode = childIndex;
				#ifdef PACKDEBUG
					printf("TOP: Thread %d hit childIndex %d\n", idx, childIndex);
				#endif
			}
		}

		//Inspect children of this parent cell
		notAtTop         = false;	//assume all children are done
		bool writeOutput = false;
		for (int i = 0; i< 4; i++)
		{
			int chk = offsets[shift + i];

			int tmpIdx = 4*parentNode + chk;
			childIndex = child[tmpIdx];
			if (childIndex >= 0)
			{
				//Leaf check
				if ((childIndex < d) && (childIndex >=0))
				{
					//We're at a leaf, so set to unallocated = -1
					if(atomicCAS((int*)&child[tmpIdx], childIndex, -1) == childIndex)
					{
						//This thread is the first here, so the childIndex goes with it
						writeOutput = true;
						#ifdef PACKDEBUG
							printf("\tThread %d will write to NEW childIndex %d\n", idx, childIndex);
						#endif
						break;
					} else
					{
						#ifdef PACKDEBUG
							printf("\tThread %d was too slow\n", idx);
						#endif
						continue;
					}
				} else
				{
					//Advance to next level down tree
					notAtTop    = true;
					parentIndex = tmpIdx;
					parentNode  = childIndex;
					break;
				}
			}
		}
		if (writeOutput)
			break;

		//If all children are done, then mark parent as done and go back to the top
		if (!notAtTop)
		{
			//It doesn't matter which thread gets here first
			atomicExch((int*)&child[parentIndex], -1);
			#ifdef PACKDEBUG
				printf("\tThread %d is closing out parent [node, index]: [%d, %d]\n", idx, parentNode, parentIndex);
			#endif
		}
		//Return if no more children
		if (parentNode == n)
		{
			#ifdef PACKDEBUG
				printf("Thread %d has found no children. Returning...\n", idx, iterations);
			#endif
			xf[idx]     = -1.0f;
			yf[idx]     = -1.0f;
			scoref[idx] = -1.0f;
			return;
		}

		//DEBUGGING
		if (iterations > q)
		{
			#ifdef PACKDEBUG
				printf("Thread %d has gone around %d times. Returning...\n", idx, iterations);
			#endif
			xf[idx]     = -1.0f;
			yf[idx]     = -1.0f;
			scoref[idx] = -1.0f;
			return;
		}
	}

	#ifdef PACKDEBUG
		printf("BOTTOM: Thread %d writing out childIndex %d\n", idx, childIndex);
	#endif

	//Write out into packed array
	xf[idx]     = x[childIndex];
	yf[idx]     = y[childIndex];
	scoref[idx] = score[childIndex];
}

__global__ void pack_filteredDev_data_kernel(float* xf, float* yf, float* scoref,
											float* x, float* y, float* score,
											int* child, const int n, unsigned int* pd, const int q)
{
	/*
	Data filtered through the Quad Tree are scattered across the child array. We need to pack
	them into xf, yf, and scoref.

	It uses depth-first search.

	Marking leaves or internal cells as -1 means they have been fully processed (unallocated again)

	index:	a global index start at n
	n:		root of tree node index
	d:		the number of current data
	q:		the number of filtered data
	*/

	//Read d into shared memory and broadcast
	static __shared__ int shared[1];
	if (threadIdx.x == 0)
		shared[0] = (int)(*pd);
	__syncthreads();

	//Write to local register
	int d = shared[0];

	#ifdef PACKDEBUG
	if (d > 32)
	{
		//Don't break your device with printf if running in PACKDEBUG mode
		printf("PACKDEBUG mode has a max of 32 on the value of d\n");
		return;
	}
	#endif

	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if (idx >= q)
		return;
	int lane = threadIdx.x % 4;

	//Shift indices beyond 4 to break up synchrony
	int shift = threadIdx.x % 5;
	int offsets[] = { 0, 1, 2, 3, 0, 1, 2, 3, 0 };

	int parentIndex;
	int parentNode;

	//Every thread will hit a leaf
	int  childIndex;
	bool notAtTop = false;

	int iterations = 0;
	while(true)
	{
		iterations++;

		//Start at the top of the tree
		if (!notAtTop)
		{
			//Start at Root Node
			parentNode = n;

			//Inspect children of root for initial parentIndex
			// - At least one of them should always be active
			//Returns the NEXT parent node when indexing child array
			parentIndex = 4*parentNode + lane;

			childIndex  = child[parentIndex];
			if (childIndex >= 0)
			{
				//Leaf check
				if ((childIndex < d) && (childIndex >=0))
				{
					//We're at a leaf, so set to unallocated = -1
					if(atomicCAS((int*)&child[parentIndex], childIndex, -1) == childIndex)
					{
						//This thread is the first here, so the childIndex goes with it
						#ifdef PACKDEBUG
							printf("\tThread %d will write to childIndex %d\n", idx, childIndex);
						#endif
						break;
					} else
					{
						#ifdef PACKDEBUG
							printf("\tThread %d was too slow\n", idx);
						#endif
					}
				}
				//Advance down the tree and assign new parent node
				parentNode = childIndex;
				#ifdef PACKDEBUG
					printf("TOP: Thread %d hit childIndex %d\n", idx, childIndex);
				#endif
			}
		}

		//Inspect children of this parent cell
		notAtTop         = false;	//assume all children are done
		bool writeOutput = false;
		for (int i = 0; i< 4; i++)
		{
			int chk = offsets[shift + i];

			int tmpIdx = 4*parentNode + chk;
			childIndex = child[tmpIdx];
			if (childIndex >= 0)
			{
				//Leaf check
				if ((childIndex < d) && (childIndex >=0))
				{
					//We're at a leaf, so set to unallocated = -1
					if(atomicCAS((int*)&child[tmpIdx], childIndex, -1) == childIndex)
					{
						//This thread is the first here, so the childIndex goes with it
						writeOutput = true;
						#ifdef PACKDEBUG
							printf("\tThread %d will write to NEW childIndex %d\n", idx, childIndex);
						#endif
						break;
					} else
					{
						#ifdef PACKDEBUG
							printf("\tThread %d was too slow\n", idx);
						#endif
						continue;
					}
				} else
				{
					//Advance to next level down tree
					notAtTop    = true;
					parentIndex = tmpIdx;
					parentNode  = childIndex;
					break;
				}
			}
		}
		if (writeOutput)
			break;

		//If all children are done, then mark parent as done and go back to the top
		if (!notAtTop)
		{
			//It doesn't matter which thread gets here first
			atomicExch((int*)&child[parentIndex], -1);
			#ifdef PACKDEBUG
				printf("\tThread %d is closing out parent [node, index]: [%d, %d]\n", idx, parentNode, parentIndex);
			#endif
		}
		//Return if no more children
		if (parentNode == n)
		{
			#ifdef PACKDEBUG
				printf("Thread %d has found no children. Returning...\n", idx, iterations);
			#endif
			xf[idx]     = -1.0f;
			yf[idx]     = -1.0f;
			scoref[idx] = -1.0f;
			return;
		}

		//DEBUGGING
		if (iterations > q)
		{
			#ifdef PACKDEBUG
				printf("Thread %d has gone around %d times. Returning...\n", idx, iterations);
			#endif
			xf[idx]     = -1.0f;
			yf[idx]     = -1.0f;
			scoref[idx] = -1.0f;
			return;
		}
	}

	#ifdef PACKDEBUG
		printf("BOTTOM: Thread %d writing out childIndex %d\n", idx, childIndex);
	#endif

	//Write out into packed array
	xf[idx]     = x[childIndex];
	yf[idx]     = y[childIndex];
	scoref[idx] = score[childIndex];
}

}	// namespace quadTreeKernels
}	// namespace quadTreeGPU

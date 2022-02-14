#include <stdio.h>
#include <math_constants.h>
#include <curand_kernel.h>

#include "kernels.cuh"

// #define DEBUG

__device__ const int blockSize = 256;
__device__ const int warp = 32;
__device__ const int stackSize = 64;
__device__ const float eps2 = 0.025;
__device__ const float theta = 0.5;

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
__device__ __inline__ void setBlack(uchar4& RGBA)
{
	// points as black(transparent)
	RGBA.x = 0;
	RGBA.y = 0;
	RGBA.z = 0;
	RGBA.w = 0;
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
	// setGreenHue(255, dst[y*w + x]);
	setBlack(dst[y*w + x]);
}

//Write Random data onto image buffer
__global__ void d_writeData2Image(uchar4* dst, const int* __restrict noiseX, const int* __restrict noiseY,const int w, const int h, const int n)
{
	int idx        = threadIdx.x + blockIdx.x * blockDim.x;
	int numThreads = blockDim.x*gridDim.x;

	for(int i = idx; i < n; i+=numThreads)
	{
		int shotX = noiseX[i];
		int shotY = noiseY[i];
		if (in_img(shotX, shotY, w, h))
			setGreenHue(255, dst[shotY*w + shotX]);
	}
}

// Random Number Generators //
// Generate 2D uniform random values
__global__ void generate_uniform2D_kernel(int* noiseX, int* noiseY, int seed, const int w, const int h, const int n)
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
		noiseX[i] = (int)((float)w*curand_uniform(&localState));
		noiseY[i] = (int)((float)h*curand_uniform(&localState));
	}
}

// Quad Tree Routines //
__global__ void reset_arrays_kernel(int *mutex, float *x, float *y, float *mass, int *count, int *start, int *sorted, int *child, int *index, float *left, float *right, float *bottom, float *top, int n, int m)
{
	int bodyIndex = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = blockDim.x*gridDim.x;
	int offset = 0;

	// reset quadtree arrays
	while(bodyIndex + offset < m){  
#pragma unroll 4
		for(int i=0;i<4;i++){
			child[(bodyIndex + offset)*4 + i] = -1;
		}
		if(bodyIndex + offset < n){
			count[bodyIndex + offset] = 1;
		}
		else{
			x[bodyIndex + offset] = 0;
			y[bodyIndex + offset] = 0;
			mass[bodyIndex + offset] = 0;
			count[bodyIndex + offset] = 0;
		}
		start[bodyIndex + offset] = -1;
		sorted[bodyIndex + offset] = 0;
		offset += stride;
	}

	if(bodyIndex == 0){
		*mutex = 0;
		*index = n;
		*left = CUDART_INF_F;
		*right = -CUDART_INF_F;
		*bottom = CUDART_INF_F;
		*top = -CUDART_INF_F;
	}
}

  
__global__ void compute_bounding_box_kernel(int *mutex, float *x, float *y, volatile float *left, volatile float *right, volatile float *bottom, volatile float *top, int n)
{
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = blockDim.x*gridDim.x;
	float x_min = x[index];
	float x_max = x[index];
	float y_min = y[index];
	float y_max = y[index];
	
	__shared__ float left_cache[blockSize];
	__shared__ float right_cache[blockSize];
	__shared__ float bottom_cache[blockSize];
	__shared__ float top_cache[blockSize];


	int offset = stride;
	while(index + offset < n){
		x_min = fminf(x_min, x[index + offset]);
		x_max = fmaxf(x_max, x[index + offset]);
		y_min = fminf(y_min, y[index + offset]);
		y_max = fmaxf(y_max, y[index + offset]);
		offset += stride;
	}

	left_cache[threadIdx.x] = x_min;
	right_cache[threadIdx.x] = x_max;
	bottom_cache[threadIdx.x] = y_min;
	top_cache[threadIdx.x] = y_max;

	__syncthreads();

	//////////////////////////
	// BLOCK-WISE REDUCTION //
	//////////////////////////

	// NOTE: This could be done by warps

	// assumes blockDim.x is a power of 2!
	int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			left_cache[threadIdx.x]   = fminf(left_cache[threadIdx.x], left_cache[threadIdx.x + i]);
			right_cache[threadIdx.x]  = fmaxf(right_cache[threadIdx.x], right_cache[threadIdx.x + i]);
			bottom_cache[threadIdx.x] = fminf(bottom_cache[threadIdx.x], bottom_cache[threadIdx.x + i]);
			top_cache[threadIdx.x]    = fmaxf(top_cache[threadIdx.x], top_cache[threadIdx.x + i]);
		}
		__syncthreads();
		i /= 2;
	}

	/////////////////////
	// FINAL REDUCTION //
	/////////////////////

	//NOTE: threadIdx.x == 0 in each block performs final reduction using atomics

	// How the lock works
	// -If a thread has the lock, the mutex will be 1, and the thread loops (spin lock)
	// -If a thread does not have the lock, it takes the lock and is done

	if(threadIdx.x == 0){
		while (atomicCAS(mutex, 0 ,1) != 0); // lock
		*left = fminf(*left, left_cache[0]);
		*right = fmaxf(*right, right_cache[0]);
		*bottom = fminf(*bottom, bottom_cache[0]);
		*top = fmaxf(*top, top_cache[0]);
		atomicExch(mutex, 0); // unlock
	}
}


__global__ void build_tree_kernel(volatile float *x, volatile float *y, volatile float *mass, volatile int *count,
									int *start, volatile int *child, int *index,
									const float *left, const float *right, const float *bottom, const float *top,
									const int n, const int m)
{
	/*
	This routine combines building the Quad Tree with summarizing internal node information
	index:	a global index start at n
	n:		the number of bodies
	m:		the number of possible nodes
	*/

	int bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
	int stride    = blockDim.x*gridDim.x;

	// build quadtree
	float l; 
	float r; 
	float b; 
	float t;
	int childPath;
	int node;

	bool newBody  = true;
	float posX, posY;
	while((bodyIndex) < n){

		if(newBody){
			newBody = false;
			//Top/Down Traversal: All particles start in one of the top 4 quads
			l = *left;
			r = *right;
			b = *bottom;
			t = *top;

			node      = 0;
			childPath = 0;
			posX      = x[bodyIndex];
			posY      = y[bodyIndex];

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

			//Update the Centroid in this cell
			atomicAdd((float*)&x[node], mass[bodyIndex]*posX);
			atomicAdd((float*)&y[node], mass[bodyIndex]*posY);
			//Increment total mass in this cell
			atomicAdd((float*)&mass[node], mass[bodyIndex]);
			//Increment body count within this cell
			atomicAdd((int*)&count[node], 1);

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
					child[locked] = bodyIndex;
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

						#ifdef DEBUG
							// if(cell >= 2*n){
							if(cell >= m){
								printf("%s\n", "error cell index is too large!!");
								printf("cell: %d\n", cell);
							}
						#endif

						//Update the Centroid in this new cell with old particle
						x[cell] += mass[childIndex]*x[childIndex];
						y[cell] += mass[childIndex]*y[childIndex];
						//Increment total mass in this new cell with old particle
						mass[cell] += mass[childIndex];
						//Increments body count within this cell with old particle
						count[cell] += count[childIndex];

						//Assign old particle to subtree leaf
						child[4*cell + childPath] = childIndex;

						start[cell] = -1;

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
						//Update the Centroid in this new cell with new particle
						x[cell] += mass[bodyIndex]*posX;
						y[cell] += mass[bodyIndex]*posY;
						//Increment total mass in this new cell with new particle
						mass[cell] += mass[bodyIndex];
						//Increments body count within this cell with new particle
						count[cell] += count[bodyIndex];

						//Set to value of child at this entry, which could be:
						// -1 == break
						childIndex = child[4*node + childPath]; 
					}

					//This means childIndex is set to -1, unallocated, so allocated as body Index
					child[4*node + childPath] = bodyIndex;

					__threadfence();  // Ensures all writes to global memory are complete before lock is released

					//Release lock and replace leaf with this cell
					child[locked] = patch;
				}	// if(childIndex == -1): first assignment to body or not

				//Advance to next body
				bodyIndex += stride;
				newBody    = true;
			}	//if(atomicCAS((int*)&child[locked], childIndex, -2) == childIndex)

		}	//if(childIndex != -2): locked already or not. If locked, go around again

		// Wait for threads in block to release locks to reduce memory pressure
		__syncthreads(); // not needed for correctness
	}
}

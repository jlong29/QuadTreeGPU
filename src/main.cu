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

//OPENGL
#include <helper_gl.h>
#include <GL/freeglut.h>

//CUDA OpenGL interoperability
#include <cuda_gl_interop.h>
#define REFRESH_DELAY     10 //ms

#include "kernels.cuh"

//Reset Quad Tree state with new data
static bool bReset      = false;

//////////////////////
// GLOBAL RESOURCES //
//////////////////////
struct GlobalResources
{
	//Problem Size
	//Parameters
	int N;
	int W;
	int H;

	int window_width;
	int window_height;
	float aspRat;

	//CUDA
	int threads;
	int blocks;

	float elpsTime;
	cudaEvent_t start, stop;

	//2D
	dim3 gridDim, blockDim;

	int2*   h_noise;
	int2*   d_noise;
	uchar4* d_img;

	//OPENGL
	//Texture variables
	GLuint imageTex;
	struct cudaGraphicsResource *pcuImageRes;

	// Timing Code
	//OpenGL loop = GPU + host
	StopWatchInterface *timer;
	int fpsCount;        // FPS count for averaging
	int fpsLimit;        // FPS limit for sampling
	float avgFPS;
	uint frameCount;
};
static GlobalResources g_res;

/////////////////////////////////
// OPENGL FORWARD DECLARATIONS //
/////////////////////////////////
bool initGL(int *argc, char **argv);

//IMAGE DATA
void display();
void keyboard(unsigned char key, int x, int y);
void cleanup();
void timerEvent(int value);

#include "kernels.cuh"

static inline int divUp(int x, int y)
{
	return (x + y - 1) / y;
}

int main(int argc, char** argv)
{
	//Parameters
	int N = 16*1;
	int W = 640;
	int H = 480;

	/////////////////
	// GLOBAL DATA //
	/////////////////
	memset(&g_res, 0, sizeof(g_res));

	//initialize timers
	g_res.fpsCount   = 0;
	g_res.fpsLimit   = 1;
	g_res.avgFPS     = 0.0f;
	g_res.frameCount = 0;

	//Set buffer pointers to NULL
	g_res.h_noise=NULL;
	g_res.d_noise=NULL;
	g_res.d_img=NULL;
	
	//Set Global data	
	g_res.N             = N;
	g_res.W             = W;
	g_res.H             = H;
	g_res.window_width  = W;
	g_res.window_height = H;
	g_res.aspRat        = (float)g_res.window_width/(float)g_res.window_height;

	//Root GPU Launch Optimization
	//1D
	g_res.threads = NTHREADS;
	g_res.blocks  = divUp(g_res.N, NTHREADS);

	//2D
	g_res.blockDim.x = WARPSIZE;
	g_res.blockDim.y = NWARPS;

	g_res.gridDim.x  = divUp(g_res.W, g_res.blockDim.x);
	g_res.gridDim.y  = divUp(g_res.H, g_res.blockDim.y);

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

	// OpenGL: initialize on this device and set up windows
	if (false == initGL(&argc, argv))
	{
		return -1;
	}

	// Create the CUTIL timer
	sdkCreateTimer(&g_res.timer);

	//Allocate Memory
	checkCudaErrors(cudaMallocHost((void **)&g_res.h_noise, sizeof(int2)*g_res.N));
	checkCudaErrors(cudaMalloc((void **)&g_res.d_noise, sizeof(int2)*g_res.N));
	checkCudaErrors(cudaMalloc((void **)&g_res.d_img, sizeof(uchar4)*g_res.W*g_res.H));

	//Set Timers
	checkCudaErrors(cudaEventCreate(&g_res.start));
	checkCudaErrors(cudaEventCreate(&g_res.stop));
	
	//Generate Random Data
	checkCudaErrors(cudaEventRecord(g_res.start, 0));

	int seed = (int)time(0);

	generate_uniform2D_kernel<<<g_res.blocks, g_res.threads>>>(g_res.d_noise, seed, g_res.W, g_res.H, g_res.N);

	checkCudaErrors(cudaEventRecord(g_res.stop, 0));
	checkCudaErrors(cudaEventSynchronize(g_res.stop));
	checkCudaErrors(cudaEventElapsedTime(&g_res.elpsTime, g_res.start, g_res.stop));
	printf("\n\nElapsed time for random number generation:  %9.6f ms \n", g_res.elpsTime);
	
	//Set Image to Black
	// d_setBlackImag<<<g_res.gridDim, g_res.blockDim>>>(g_res.d_img, g_res.W, g_res.H);
	checkCudaErrors(cudaMemset(g_res.d_img, 0, sizeof(uchar4)*g_res.W*g_res.H));

	//Write Random data onto image buffer
	checkCudaErrors(cudaEventRecord(g_res.start, 0));

	d_writeData2Image<<<g_res.blocks, g_res.threads>>>(g_res.d_img, g_res.d_noise, g_res.W, g_res.H, g_res.N);

	checkCudaErrors(cudaEventRecord(g_res.stop, 0));
	checkCudaErrors(cudaEventSynchronize(g_res.stop));
	checkCudaErrors(cudaEventElapsedTime(&g_res.elpsTime, g_res.start, g_res.stop));
	printf("\n\nElapsed time for writing noise data:        %9.6f ms \n", g_res.elpsTime);

	//Copy back to host for checking    
	checkCudaErrors(cudaMemcpy(g_res.h_noise, g_res.d_noise, sizeof(int2)*g_res.N, cudaMemcpyDeviceToHost));

	printf("Printing out 2D Noise:\n\t");
	for (int ii = 0; ii < min(100, g_res.N); ii++)
	{
		printf("[%d, %d], ", g_res.h_noise[ii].x, g_res.h_noise[ii].y);
	}
	printf("\n");

	//convert device memory to texture
	cudaArray_t ArrIm;
	cudaGraphicsMapResources(1, &g_res.pcuImageRes, 0);
	cudaGraphicsSubResourceGetMappedArray(&ArrIm, g_res.pcuImageRes, 0, 0);

	checkCudaErrors(cudaMemcpyToArray(ArrIm, 0, 0, g_res.d_img, g_res.W*g_res.H*sizeof(uchar4), cudaMemcpyDeviceToDevice));
	cudaGraphicsUnmapResources(1, &g_res.pcuImageRes, 0);

	////////////////////////////////
	// LAUNCH OPENGL DISPLAY LOOP //
	////////////////////////////////
	glutMainLoop();
	return 0;
}

//OpenGL function definitions
void computeFPS()
{
	g_res.frameCount++;
	g_res.fpsCount++;

	if (g_res.fpsCount == g_res.fpsLimit)
	{
		g_res.avgFPS = 1.f / (sdkGetAverageTimerValue(&g_res.timer) / 1000.f);
		g_res.fpsCount = 0;
		g_res.fpsLimit = (int)MAX(g_res.avgFPS, 1.f);

		sdkResetTimer(&g_res.timer);
	}
	
	char fps[256];
	sprintf(fps, "Quad Tree: %3.1f fps", g_res.avgFPS);
	glutSetWindowTitle(fps);
}

bool initGL(int *argc, char **argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

	//Create Image Data Window
	glutInitWindowSize(g_res.window_width, g_res.window_height);
	glutCreateWindow("Quad Tree");
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glutReportErrors();

	// NOTE: These calls don't work until first window is created

	//initialize necessary OpenGL extensions
	if (!isGLVersionSupported(2,0))
	{
		fprintf(stderr,"ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}
	fprintf(stdout,"OpenGL version supported by this platform: (%s)\n", glGetString(GL_VERSION));

	// Create and Register OpenGL Texture for Image(1 channel)
	cudaError_t err1;
	glGenTextures(1, &g_res.imageTex);
	glBindTexture(GL_TEXTURE_2D, g_res.imageTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, g_res.window_width, g_res.window_height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);

	err1 = cudaGraphicsGLRegisterImage(&g_res.pcuImageRes, g_res.imageTex, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);

	if (err1 != 0)
	{
		fprintf(stderr,"ERROR: Registering openGL texture failed\n");
		return false;
	}
	glutReportErrors();

	// Image Data: register callbacks
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	glutCloseFunc(cleanup);

	return true;
}

void display()
{
	//Time it
	sdkStartTimer(&g_res.timer);

	if (bReset)
	{
		bReset = false;
		//Generate Random Data
		checkCudaErrors(cudaEventRecord(g_res.start, 0));

		int seed = (int)time(0);

		generate_uniform2D_kernel<<<g_res.blocks, g_res.threads>>>(g_res.d_noise, seed, g_res.W, g_res.H, g_res.N);

		checkCudaErrors(cudaEventRecord(g_res.stop, 0));
		checkCudaErrors(cudaEventSynchronize(g_res.stop));
		checkCudaErrors(cudaEventElapsedTime(&g_res.elpsTime, g_res.start, g_res.stop));
		printf("\n\nElapsed time for random number generation:  %9.6f ms \n", g_res.elpsTime);
		
		//Set Image to Black
		// d_setBlackImag<<<g_res.gridDim, g_res.blockDim>>>(g_res.d_img, g_res.W, g_res.H);
		checkCudaErrors(cudaMemset(g_res.d_img, 0, sizeof(uchar4)*g_res.W*g_res.H));

		//Write Random data onto image buffer
		checkCudaErrors(cudaEventRecord(g_res.start, 0));

		d_writeData2Image<<<g_res.blocks, g_res.threads>>>(g_res.d_img, g_res.d_noise, g_res.W, g_res.H, g_res.N);

		checkCudaErrors(cudaEventRecord(g_res.stop, 0));
		checkCudaErrors(cudaEventSynchronize(g_res.stop));
		checkCudaErrors(cudaEventElapsedTime(&g_res.elpsTime, g_res.start, g_res.stop));
		printf("\n\nElapsed time for writing noise data:        %9.6f ms \n", g_res.elpsTime);

		//Copy back to host for checking    
		checkCudaErrors(cudaMemcpy(g_res.h_noise, g_res.d_noise, sizeof(int2)*g_res.N, cudaMemcpyDeviceToHost));

		printf("Printing out 2D Noise:\n\t");
		for (int ii = 0; ii < min(100, g_res.N); ii++)
		{
			printf("[%d, %d], ", g_res.h_noise[ii].x, g_res.h_noise[ii].y);
		}
		printf("\n");

		//convert device memory to texture
		cudaArray_t ArrIm;
		cudaGraphicsMapResources(1, &g_res.pcuImageRes, 0);
		cudaGraphicsSubResourceGetMappedArray(&ArrIm, g_res.pcuImageRes, 0, 0);

		//NOTE: DISTORTION MODEL: 3 * d_dstReSz
		checkCudaErrors(cudaMemcpyToArray(ArrIm, 0, 0, g_res.d_img, g_res.W*g_res.H*sizeof(uchar4), cudaMemcpyDeviceToDevice));
		cudaGraphicsUnmapResources(1, &g_res.pcuImageRes, 0);
	}

	/////////////////////////
	// DISPLAY WITH OPENGL //
	/////////////////////////

	//OpenGL Part
	glClear(GL_COLOR_BUFFER_BIT);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	glBindTexture(GL_TEXTURE_2D, g_res.imageTex);
	glEnable(GL_TEXTURE_2D);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	//Note: texCoords go clockwise from top left and verts go counter-clockwise from lower left
	glBegin(GL_QUADS);
	glTexCoord2f(0.0, 1.0);
	glVertex3f(-g_res.aspRat, -1.0, 0.0);
	glTexCoord2f(1.0, 1.0);
	glVertex3f(g_res.aspRat, -1.0, 0.0);
	glTexCoord2f(1.0, 0.0);
	glVertex3f(g_res.aspRat, 1.0, 0.0);
	glTexCoord2f(0.0, 0.0);
	glVertex3f(-g_res.aspRat, 1.0, 0.0);
	glEnd();
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);

	glutReportErrors();

	cudaDeviceSynchronize();

	// Updating timing information
	sdkStopTimer(&g_res.timer);
	computeFPS();

	//swap
	glutSwapBuffers();
}

//Keyboard callback
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	fprintf(stdout,"\tKey Press: %u\n", (uint)key);
	switch (key)
	{
	case (27) :
	{
		glutDestroyWindow(glutGetWindow());
		return;
	}
	case 'r':
	{
		//r: Reset Quad Tree with new data
		bReset = true;
		return;
	}
	}
}

void timerEvent(int value)
{
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, timerEvent,0);
	}
}

void cleanup()
{
	sdkDeleteTimer(&g_res.timer);

	// CUDA/OPENGL
	fprintf(stdout,"\tCUDA:\n");
	cudaGraphicsUnregisterResource(g_res.pcuImageRes);
	glDeleteTextures(1, &g_res.imageTex);
	g_res.imageTex = 0;
	fprintf(stdout,"\t\tAll openGL resources cleaned\n");

	//Deallocate memory
	checkCudaErrors(cudaFreeHost(g_res.h_noise));
	checkCudaErrors(cudaFree(g_res.d_noise));
	checkCudaErrors(cudaFree(g_res.d_img));

	checkCudaErrors(cudaEventDestroy(g_res.start));
	checkCudaErrors(cudaEventDestroy(g_res.stop));
	fprintf(stdout,"\t\tAll Cuda resources cleaned\n");
}

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

#include "QuadTreeBuilder.h"

//Reset Quad Tree state with new data
static bool bReset = false;

// GLOBAL Variables //
//Problem Size
//Parameters
int N;
int W;
int H;

int window_width;
int window_height;
float aspRat;

//The star of the show
QuadTreeBuilder quadTree;

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

/////////////////////////////////
// OPENGL FORWARD DECLARATIONS //
/////////////////////////////////
bool initGL(int *argc, char **argv);

//IMAGE DATA
void display();
void keyboard(unsigned char key, int x, int y);
void cleanup();
void timerEvent(int value);

int main(int argc, char** argv)
{
	//Input Parameters
	N = 16;
	W = 640;
	H = 480;

	window_width  = W;
	window_height = H;

	//Set QuadTreeBuilder parameters
	quadTree.setParameters(N, W, H);

	//initialize timers
	fpsCount   = 0;
	fpsLimit   = 1;
	avgFPS     = 0.0f;
	frameCount = 0;

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
	sdkCreateTimer(&timer);

	//Allocate Memory
	if (quadTree.allocate() < 0)
	{
		cleanup();
		return -1;
	}

	//Set random data
	if (quadTree.resetData() < 0)
	{
		cleanup();
		return -1;
	}
	
	//Visualize output
	if (quadTree.createViz() < 0)
	{
		cleanup();
		return -1;
	}

	//Download data
	if (quadTree.downloadData() < 0)
	{
		cleanup();
		return -1;
	}

	printf("Printing out 2D Noise:\n\t");
	for (int ii = 0; ii < min(100, N); ii++)
	{
		printf("[%d, %d], ", (int)quadTree.h_x[ii], (int)quadTree.h_y[ii]);
	}
	printf("\n");

	//convert device memory to texture
	cudaArray_t ArrIm;
	cudaGraphicsMapResources(1, &pcuImageRes, 0);
	cudaGraphicsSubResourceGetMappedArray(&ArrIm, pcuImageRes, 0, 0);

	checkCudaErrors(cudaMemcpyToArray(ArrIm, 0, 0, quadTree.d_img, quadTree.imgSz, cudaMemcpyDeviceToDevice));
	cudaGraphicsUnmapResources(1, &pcuImageRes, 0);

	////////////////////////////////
	// LAUNCH OPENGL DISPLAY LOOP //
	////////////////////////////////
	glutMainLoop();
	return 0;
}

//OpenGL function definitions
void computeFPS()
{
	frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit)
	{
		avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		fpsCount = 0;
		fpsLimit = (int)MAX(avgFPS, 1.f);

		sdkResetTimer(&timer);
	}
	
	char fps[256];
	sprintf(fps, "Quad Tree: %3.1f fps", avgFPS);
	glutSetWindowTitle(fps);
}

bool initGL(int *argc, char **argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

	//Create Image Data Window
	glutInitWindowSize(window_width, window_height);
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
	glGenTextures(1, &imageTex);
	glBindTexture(GL_TEXTURE_2D, imageTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, window_width, window_height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);

	err1 = cudaGraphicsGLRegisterImage(&pcuImageRes, imageTex, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);

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
	sdkStartTimer(&timer);

	if (bReset)
	{
		bReset = false;

		//Set random data
		if (quadTree.resetData() < 0)
		{
			cleanup();
			return;
		}
		
		//Visualize output
		if (quadTree.createViz() < 0)
		{
			cleanup();
			return;
		}

		//Download data
		if (quadTree.downloadData() < 0)
		{
			cleanup();
			return;
		}

		printf("Printing out 2D Noise:\n\t");
		for (int ii = 0; ii < min(100, N); ii++)
		{
			printf("[%d, %d], ", (int)quadTree.h_x[ii], (int)quadTree.h_y[ii]);
		}
		printf("\n");

		//convert device memory to texture
		cudaArray_t ArrIm;
		cudaGraphicsMapResources(1, &pcuImageRes, 0);
		cudaGraphicsSubResourceGetMappedArray(&ArrIm, pcuImageRes, 0, 0);

		//NOTE: DISTORTION MODEL: 3 * d_dstReSz
		checkCudaErrors(cudaMemcpyToArray(ArrIm, 0, 0, quadTree.d_img, quadTree.imgSz, cudaMemcpyDeviceToDevice));
		cudaGraphicsUnmapResources(1, &pcuImageRes, 0);
	}

	/////////////////////////
	// DISPLAY WITH OPENGL //
	/////////////////////////

	//OpenGL Part
	glClear(GL_COLOR_BUFFER_BIT);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	glBindTexture(GL_TEXTURE_2D, imageTex);
	glEnable(GL_TEXTURE_2D);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	//Note: texCoords go clockwise from top left and verts go counter-clockwise from lower left
	glBegin(GL_QUADS);
	glTexCoord2f(0.0, 1.0);
	glVertex3f(-1, -1.0, 0.0);
	glTexCoord2f(1.0, 1.0);
	glVertex3f(1, -1.0, 0.0);
	glTexCoord2f(1.0, 0.0);
	glVertex3f(1, 1.0, 0.0);
	glTexCoord2f(0.0, 0.0);
	glVertex3f(-1, 1.0, 0.0);
	glEnd();
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);

	glutReportErrors();

	cudaDeviceSynchronize();

	// Updating timing information
	sdkStopTimer(&timer);
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
	sdkDeleteTimer(&timer);

	// CUDA/OPENGL
	fprintf(stdout,"\tCUDA:\n");
	cudaGraphicsUnregisterResource(pcuImageRes);
	glDeleteTextures(1, &imageTex);
	imageTex = 0;
	fprintf(stdout,"\t\tAll openGL resources cleaned\n");

	//Deallocate device memory and destory timers
	quadTree.deallocate();

	fprintf(stdout,"\t\tAll Cuda resources cleaned\n");
}

#include <windows.h>
#include <GL/glew.h> //Wrangler For PP , add additional headers and lib path
#include <gl/GL.h>
#include <stdio.h>
#include <math.h>
#include "vmath.h"

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "kernel32.lib")

#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")

#pragma comment(lib, "cudart.lib")

//global namespace

using namespace vmath;

enum
{
	AMC_ATTRIBUTE_POSITION = 0,
	AMC_ATTRIBUTE_COLOR,
	AMC_ATTRIBUTE_NORMAL,
	AMC_ATTRIBUTE_TEXCOORD0
};

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

bool gbFullScreen = false;
DWORD dwStyle;
WINDOWPLACEMENT wpPrev = {sizeof(WINDOWPLACEMENT)};
HWND ghWnd = NULL;
HDC ghdc = NULL;
HGLRC ghrc = NULL;
bool gbActiveWindow = false;
FILE *gpFile = NULL;

void ToggleFullScreen(void);
int initialize(void);
void resize(int, int);
void display(void);
void update(void);
void uninitialize(void);

GLuint gShaderProgramObject;
GLuint vao;		   // vertex array object
GLuint vbo;		   // vertex buffer object
GLuint mvpUniform; //model view projection uniform
mat4 perspectiveProjectionMatrix;

//cuda
const int gMesh_width = 1024;
const int gMesh_height = 1024;

#define MY_ARRAY_SIZE gMesh_width *gMesh_height * 4

float pos[gMesh_width][gMesh_height][4];

struct cudaGraphicsResource *graphicsResource = NULL;

GLfloat animationTime = 0.0f;
GLuint vbo_gpu;
bool bOnGPU = false;
cudaError_t error;

void launchCPUKernel(unsigned int, unsigned int, float);
void launchCudaKernel(float4 *, unsigned int, unsigned int, float);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szAppName[] = TEXT("MY OGL WINDOW");
	bool bDone = false;
	int iRet = 0;

	if (fopen_s(&gpFile, "Log.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Log file can not be created"), TEXT("Error"), MB_OK);
		exit(0);
	}
	else
	{
		fprintf(gpFile, "Log file created successfully...\n");
		fflush(gpFile);
	}

	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.lpfnWndProc = WndProc;
	wndclass.hInstance = hInstance;
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass.lpszClassName = szAppName;
	wndclass.lpszMenuName = NULL;
	wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);

	RegisterClassEx(&wndclass);

	hwnd = CreateWindowEx(WS_EX_APPWINDOW,
						  szAppName,
						  TEXT("My Double buffer Window - Jayshree"),
						  WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
						  100,
						  100,
						  WIN_WIDTH,
						  WIN_HEIGHT,
						  NULL,
						  NULL,
						  hInstance,
						  NULL);

	ghWnd = hwnd;

	iRet = initialize();
	if (iRet == -1)
	{
		fprintf(gpFile, "ChoosePixelFormat() Failed\n");
		fflush(gpFile);
		DestroyWindow(hwnd);
	}
	else if (iRet == -2)
	{
		fprintf(gpFile, "SetPixelFormat() Failed\n");
		fflush(gpFile);
		DestroyWindow(hwnd);
	}
	else if (iRet == -3)
	{
		fprintf(gpFile, "wglCreateContext() Failed\n");
		fflush(gpFile);
		DestroyWindow(hwnd);
	}
	else if (iRet == -4)
	{
		fprintf(gpFile, "wglMakeCurrent() Failed\n");
		fflush(gpFile);
		DestroyWindow(hwnd);
	}
	else
	{
		fprintf(gpFile, "Initialization succeded\n");
		fflush(gpFile);
	}

	ToggleFullScreen();
	ShowWindow(hwnd, iCmdShow);
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);

	while (bDone == false)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
			{
				bDone = true;
			}
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		else
		{
			//Play game here
			if (gbActiveWindow == true)
			{
				//code
				//here call update
				update();
			}
			display();
		}
	}

	return ((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{

	switch (iMsg)
	{
	case WM_SETFOCUS:
		gbActiveWindow = true;
		break;

	case WM_KILLFOCUS:
		gbActiveWindow = false;
		break;

	case WM_SIZE:
		resize(LOWORD(lParam), HIWORD(lParam));
		break;

	case WM_ERASEBKGND:
		return (0);
		break;

	case WM_CLOSE:
		DestroyWindow(hwnd);
		break;

	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			DestroyWindow(hwnd);
			break;

		case 'c':
		case 'C':
			bOnGPU = false;
			break;

		case 'g':
		case 'G':
			bOnGPU = true;
			break;
		}
		break;

	case WM_DESTROY:
		uninitialize();
		PostQuitMessage(0);
		break;
	}

	return (DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void ToggleFullScreen(void)
{
	MONITORINFO mi;

	if (gbFullScreen == false)
	{
		dwStyle = GetWindowLong(ghWnd, GWL_STYLE);

		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{
			mi = {sizeof(MONITORINFO)};

			if (GetWindowPlacement(ghWnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghWnd, MONITORINFOF_PRIMARY), &mi))
			{
				SetWindowLong(ghWnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);

				SetWindowPos(ghWnd,
							 HWND_TOP,
							 mi.rcMonitor.left,
							 mi.rcMonitor.top,
							 mi.rcMonitor.right - mi.rcMonitor.left,
							 mi.rcMonitor.bottom - mi.rcMonitor.top,
							 SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}

		ShowCursor(FALSE);
		gbFullScreen = true;
	}
}

int initialize(void)
{
	//cuda
	int deviceCount = 0;
	error = cudaGetDeviceCount(&deviceCount);
	if (error != cudaSuccess)
	{
		fprintf(gpFile, "cudaGetDeviceCount() failed\n");
		fflush(gpFile);
		uninitialize();
		exit(0);
	}
	else if (deviceCount == 0)
	{
		fprintf(gpFile, "No CUDA device\n");
		fflush(gpFile);
		uninitialize();
		exit(0);
	}
	else
	{
		cudaSetDevice(0);
	}

	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;
	GLenum result;

	ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));

	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 32;
	pfd.cRedBits = 8;
	pfd.cGreenBits = 8;
	pfd.cBlueBits = 8;
	pfd.cAlphaBits = 8;
	pfd.cDepthBits = 32;

	ghdc = GetDC(ghWnd);

	iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);
	if (iPixelFormatIndex == 0)
	{
		return (-1);
	}

	if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
	{
		return (-2);
	}

	ghrc = wglCreateContext(ghdc);
	if (ghrc == NULL)
	{
		return (-3);
	}

	if (wglMakeCurrent(ghdc, ghrc) == FALSE)
	{
		return (-4);
	}

	//On the extensions requred for PP
	result = glewInit();
	if (result != GLEW_OK)
	{
		fprintf(gpFile, "\nglewInit() failed...\n");
		fflush(gpFile);
		uninitialize();
		DestroyWindow(ghWnd);
	}

	GLuint vertexShaderObject;
	GLuint fragmentShaderObject;

	//***************** 1. VERTEX SHADER ************************************
	//define vertex shader object
	//create vertex shader object
	vertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	//Write vertex shader code
	const GLchar *vertexShaderSourceCode =
		"#version 450 core"
		"\n"
		"in vec4 vPosition;"
		"uniform mat4 u_mvp_matrix;"
		"out vec4 out_color;"
		"void main(void)"
		"{"
		"gl_Position = u_mvp_matrix * vPosition;"
		"out_color = vec4(vPosition.xy / 2.0 + 0.5, 1, 1);"
		"}";

	//specify above source code to vertex shader object
	glShaderSource(vertexShaderObject,						 //to whom?
				   1,										 //how many strings
				   (const GLchar **)&vertexShaderSourceCode, //address of string
				   NULL);									 // NULL specifes that there is only one string with fixed length

	//Compile the vertex shader
	glCompileShader(vertexShaderObject);

	//Error checking for compilation:
	GLint iShaderCompileStatus = 0;
	GLint iInfoLogLength = 0;
	GLchar *szInfoLog = NULL;

	//Step 1 : Call glGetShaderiv() to get comiple status of particular shader
	glGetShaderiv(vertexShaderObject,	  // whose?
				  GL_COMPILE_STATUS,	  //what to get?
				  &iShaderCompileStatus); //in what?

	//Step 2 : Check shader compile status for GL_FALSE
	if (iShaderCompileStatus == GL_FALSE)
	{
		//Step 3 : If GL_FALSE , call glGetShaderiv() again , but this time to get info log length
		glGetShaderiv(vertexShaderObject,
					  GL_INFO_LOG_LENGTH,
					  &iInfoLogLength);

		//Step 4 : if info log length > 0 , call glGetShaderInfoLog()
		if (iInfoLogLength > 0)
		{
			//allocate memory to pointer
			szInfoLog = (GLchar *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;

				glGetShaderInfoLog(vertexShaderObject, //whose?
								   iInfoLogLength,	   //length?
								   &written,		   //might have not used all, give that much only which have been used in what?
								   szInfoLog);		   //store in what?

				fprintf(gpFile, "\nVertex Shader Compilation Log : %s\n", szInfoLog);
				fflush(gpFile);

				//free the memory
				free(szInfoLog);

				uninitialize();
				DestroyWindow(ghWnd);
				exit(0);
			}
		}
	}

	//************************** 2. FRAGMENT SHADER ********************************
	//define fragment shader object
	//create fragment shader object
	fragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	//write fragment shader code
	const GLchar *fragmentShaderSourceCode =
		"#version 450 core"
		"\n"
		"in vec4 out_color;"
		"out vec4 fragColor;"
		"void main(void)"
		"{"
		"fragColor = out_color;"
		"}";

	//specify the above source code to fragment shader object
	glShaderSource(fragmentShaderObject,
				   1,
				   (const GLchar **)&fragmentShaderSourceCode,
				   NULL);

	//compile the fragment shader
	glCompileShader(fragmentShaderObject);

	//Error checking for compilation
	iShaderCompileStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	//Step 1 : Call glGetShaderiv() to get comiple status of particular shader
	glGetShaderiv(fragmentShaderObject,	  // whose?
				  GL_COMPILE_STATUS,	  //what to get?
				  &iShaderCompileStatus); //in what?

	//Step 2 : Check shader compile status for GL_FALSE
	if (iShaderCompileStatus == GL_FALSE)
	{
		//Step 3 : If GL_FALSE , call glGetShaderiv() again , but this time to get info log length
		glGetShaderiv(fragmentShaderObject,
					  GL_INFO_LOG_LENGTH,
					  &iInfoLogLength);

		//Step 4 : if info log length > 0 , call glGetShaderInfoLog()
		if (iInfoLogLength > 0)
		{
			//allocate memory to pointer
			szInfoLog = (GLchar *)malloc(iInfoLogLength);

			if (szInfoLog != NULL)
			{
				GLsizei written;

				glGetShaderInfoLog(fragmentShaderObject, //whose?
								   iInfoLogLength,		 //length?
								   &written,			 //might have not used all, give that much only which have been used in what?
								   szInfoLog);			 //store in what?

				fprintf(gpFile, "\nFragment Shader Compilation Log : %s\n", szInfoLog);
				fflush(gpFile);

				//free the memory
				free(szInfoLog);

				uninitialize();
				DestroyWindow(ghWnd);
				exit(0);
			}
		}
	}

	//create shader program object
	gShaderProgramObject = glCreateProgram();

	//Attach vertex shader to shader program
	glAttachShader(gShaderProgramObject, //to whom?
				   vertexShaderObject);	 //what to attach?

	//Attach fragment shader to shader program
	glAttachShader(gShaderProgramObject,
				   fragmentShaderObject);

	//Pre-Linking binding to vertex attribute
	glBindAttribLocation(gShaderProgramObject,
						 AMC_ATTRIBUTE_POSITION,
						 "vPosition");

	//Link the shader program
	glLinkProgram(gShaderProgramObject); //link to whom?

	//Error checking for linking
	GLint iProgramLinkStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	//Step 1 : Call glGetShaderiv() to get comiple status of particular shader
	glGetProgramiv(gShaderProgramObject, // whose?
				   GL_LINK_STATUS,		 //what to get?
				   &iProgramLinkStatus); //in what?

	//Step 2 : Check shader compile status for GL_FALSE
	if (iProgramLinkStatus == GL_FALSE)
	{
		//Step 3 : If GL_FALSE , call glGetShaderiv() again , but this time to get info log length
		glGetShaderiv(gShaderProgramObject,
					  GL_INFO_LOG_LENGTH,
					  &iInfoLogLength);

		//Step 4 : if info log length > 0 , call glGetShaderInfoLog()
		if (iInfoLogLength > 0)
		{
			//allocate memory to pointer
			szInfoLog = (GLchar *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;

				glGetShaderInfoLog(gShaderProgramObject, //whose?
								   iInfoLogLength,		 //length?
								   &written,			 //might have not used all, give that much only which have been used in what?
								   szInfoLog);			 //store in what?

				fprintf(gpFile, "\nShader Program Linking Log : %s\n", szInfoLog);
				fflush(gpFile);

				//free the memory
				free(szInfoLog);

				uninitialize();
				DestroyWindow(ghWnd);
				exit(0);
			}
		}
	}

	//Post-Linking reteriving uniform location
	mvpUniform = glGetUniformLocation(gShaderProgramObject,
									  "u_mvp_matrix");

	//above is the preparation of data transfer from CPU to GPU
	//i.e glBindAttribLocation() & glGetUniformLocation()

	//array initialization (glBegin() and glEnd())
	for (int i = 0; i < gMesh_width; i++)
	{
		for (int j = 0; j < gMesh_height; j++)
		{
			for (int k = 0; k < 4; k++)
			{
				pos[i][j][k] = 0;
			}
		}
	}

	//create vao (vertex array object)
	glGenVertexArrays(1, &vao);

	//Bind vao
	glBindVertexArray(vao);

	//generate vertex buffers
	glGenBuffers(1, &vbo);

	//bind buffer
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	//transfer vertex data(CPU) to GPU buffer
	glBufferData(GL_ARRAY_BUFFER,
				 MY_ARRAY_SIZE * sizeof(float),
				 NULL,
				 GL_DYNAMIC_DRAW);

	//attach or map attribute pointer to vbo's buffer
	/*glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0,
		NULL);

	//enable vertex attribute array
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	*/

	//unbind vbo
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &vbo_gpu);

	//bind buffer
	glBindBuffer(GL_ARRAY_BUFFER, vbo_gpu);

	//transfer vertex data(CPU) to GPU buffer
	glBufferData(GL_ARRAY_BUFFER,
				 MY_ARRAY_SIZE * sizeof(float),
				 NULL,
				 GL_DYNAMIC_DRAW);

	//unbind vbo
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//register our vbo to cudaGraphicsResource
	error = cudaGraphicsGLRegisterBuffer(&graphicsResource, vbo_gpu, cudaGraphicsMapFlagsWriteDiscard);
	if (error != cudaSuccess)
	{
		fprintf(gpFile, "cudaGraphicsGLRegisterBuffer() failed\n");
		fflush(gpFile);
		uninitialize();
		exit(0);
	}

	//unbind vao
	glBindVertexArray(0);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	//make identity
	perspectiveProjectionMatrix = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);

	return (0);
}

void resize(int width, int height)
{
	if (height == 0)
	{
		height = 1;
	}

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	perspectiveProjectionMatrix = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}

void display(void)
{
	float4 *pPos = NULL;
	size_t byteCount;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//Binding Opengl code to shader program object
	glUseProgram(gShaderProgramObject);

	//matrices
	mat4 modelViewMatrix;
	mat4 modelViewProjectionMatrix;
	mat4 translationMatrix;

	//make identity
	modelViewMatrix = mat4::identity();
	modelViewProjectionMatrix = mat4::identity();
	translationMatrix = mat4::identity();

	//do necessary transformation
	translationMatrix = translate(0.0f, 0.0f, -3.0f);

	//do necessary matrix multiplication
	//this was internally done by gluOrtho() in ffp
	modelViewMatrix = modelViewMatrix * translationMatrix;
	modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

	//send necessary matrices to shader in respective uniforms
	glUniformMatrix4fv(mvpUniform,				   //which uniform?
					   1,						   //how many matrices
					   GL_FALSE,				   //have to transpose?
					   modelViewProjectionMatrix); //actual matrix

	//bind with vao
	glBindVertexArray(vao);

	if (bOnGPU == true)
	{
		//1.map with the resource
		error = cudaGraphicsMapResources(1, &graphicsResource, 0);
		if (error != cudaSuccess)
		{
			fprintf(gpFile, "cudaGraphicsMapResources() failed\n");
			fflush(gpFile);
			uninitialize();
			exit(0);
		}

		//2.Get permission to fill data
		error = cudaGraphicsResourceGetMappedPointer((void **)&pPos, &byteCount, graphicsResource);
		if (error != cudaSuccess)
		{
			fprintf(gpFile, "cudaGraphicsResourceGetMappedPointer() failed\n");
			fflush(gpFile);
			uninitialize();
			exit(0);
		}

		//3.Fill data
		launchCudaKernel(pPos, gMesh_width, gMesh_height, animationTime);

		//glBindBuffer(GL_ARRAY_BUFFER, vbo_gpu);
		//glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(float), pos, GL_DYNAMIC_DRAW);
		//glBindBuffer(GL_ARRAY_BUFFER, 0);

		//4.unmap resource
		error = cudaGraphicsUnmapResources(1, &graphicsResource, 0);
		if (error != cudaSuccess)
		{
			fprintf(gpFile, "cudaGraphicsUnmapResources() failed\n");
			fflush(gpFile);
			uninitialize();
			exit(0);
		}
	}
	else
	{
		launchCPUKernel(gMesh_width, gMesh_height, animationTime);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(float), pos, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	if (bOnGPU == true)
	{
		glBindBuffer(GL_ARRAY_BUFFER, vbo_gpu);
	}
	else
	{
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
						  4,
						  GL_FLOAT,
						  GL_FALSE,
						  0,
						  NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//now draw the necessary scene
	glDrawArrays(GL_POINTS,
				 0,
				 gMesh_width * gMesh_height);

	//unbind vao
	glBindVertexArray(0);

	//unbinding program
	glUseProgram(0);

	SwapBuffers(ghdc);
}

void update(void)
{
	animationTime = animationTime + 0.01f;
}

void uninitialize(void)
{
	cudaGraphicsUnregisterResource(graphicsResource);

	if (vbo)
	{
		glDeleteBuffers(1, &vbo);
		vbo = 0;
	}

	if (vao)
	{
		glDeleteVertexArrays(1, &vao);
		vao = 0;
	}

	if (gShaderProgramObject)
	{
		GLsizei shaderCount;
		GLsizei shaderNumber;

		glUseProgram(gShaderProgramObject);

		//ask the program how many shaders are attached to you?
		glGetProgramiv(gShaderProgramObject,
					   GL_ATTACHED_SHADERS,
					   &shaderCount);

		GLuint *pShaders = (GLuint *)malloc(sizeof(GLuint) * shaderCount);

		if (pShaders)
		{
			//fprintf(gpFile, "\npshaders sucessful\n");

			//get shaders
			glGetAttachedShaders(gShaderProgramObject,
								 shaderCount,
								 &shaderCount,
								 pShaders);

			for (shaderNumber = 0; shaderNumber < shaderCount; shaderNumber++)
			{
				//detach
				glDetachShader(gShaderProgramObject,
							   pShaders[shaderNumber]);

				//delete
				glDeleteShader(pShaders[shaderNumber]);

				//explicit 0
				pShaders[shaderNumber] = 0;
			}

			free(pShaders);
		}

		glDeleteProgram(gShaderProgramObject);
		gShaderProgramObject = 0;

		glUseProgram(0);
	}

	if (gbFullScreen == true)
	{
		SetWindowLong(ghWnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);

		SetWindowPlacement(ghWnd, &wpPrev);

		SetWindowPos(ghWnd,
					 HWND_TOP,
					 0,
					 0,
					 0,
					 0,
					 SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER);

		ShowCursor(TRUE);
	}

	if (wglGetCurrentContext() == ghrc)
	{
		wglMakeCurrent(NULL, NULL);
	}

	if (ghrc)
	{
		wglDeleteContext(ghrc);
		ghrc = NULL;
	}

	if (ghdc)
	{
		ReleaseDC(ghWnd, ghdc);
		ghdc = NULL;
	}

	if (gpFile)
	{
		fprintf(gpFile, "Log file closed successfully\n");
		fflush(gpFile);
		fclose(gpFile);
		gpFile = NULL;
	}
}

void launchCPUKernel(unsigned int mesh_width, unsigned int mesh_height, float time)
{
	for (int i = 0; i < mesh_width; i++)
	{
		for (int j = 0; j < mesh_height; j++)
		{
			for (int k = 0; k < 4; k++)
			{
				float u = i / (float)mesh_width;
				float v = j / (float)mesh_height;

				u = u * 2.0f - 1.0f;
				v = v * 2.0f - 1.0f;

				float frequency = 4.0f;
				float w = sinf(frequency * u + time) * cosf(frequency * v + time) * 0.5f;

				if (k == 0)
					pos[i][j][k] = u;

				if (k == 1)
					pos[i][j][k] = w;

				if (k == 2)
					pos[i][j][k] = v;

				if (k == 3)
					pos[i][j][k] = 1.0f;
			}
		}
	}
}

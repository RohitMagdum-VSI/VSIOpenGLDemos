
#include<windows.h>
#include<GL\glew.h>
#include<gl/GL.h>
#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
//#include <cuda_runtime_api.h>

#include<cuda_gl_interop.h>
#include"helper_timer.h"
#include<vector_types.h>
#include"vmath.h"
#include <device_functions.h>
#include <device_launch_parameters.h>

#pragma comment(lib,"User32.lib")
#pragma comment(lib,"GDI32.lib")
#pragma comment(lib,"glew32.lib")
#pragma comment(lib,"opengl32.lib")

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

using namespace vmath;

enum
{
	HAD_ATTRIBUTE_POSITION = 0,
	HAD_ATTRIBUTE_COLOR,
	HAD_ATTRIBUTE_NORMAL,
	HAD_ATTRIBUTE_TEXTURE0,
};

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

FILE* gpFile;
HWND ghwnd;
HDC ghdc;
HGLRC ghrc;
DWORD dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };
bool gbActiveWindow = false;
bool gbFullscreen = false;
bool gbIsEscapeKeyPressed = false;

GLuint gVertexShaderObject;
GLuint gFragmentShaderObject;
GLuint gShaderProgramObject;

GLuint gVao;
GLuint gVbo;
GLuint gMVPUniform, gColorUniform;
cudaGraphicsResource_t cuda_vbo_resource = 0;

mat4 gPerspectiveProjectionMatrix;

float gfAnimate = 0.0f;

cudaError_t err = cudaSuccess;

char str[256];

int MESH_WIDTH = 64;
int MESH_HEIGHT = 64;

GLfloat color[] = { 1.0f,1.0f,1.0f,1.0f };
GLfloat gfTranslateFactor = -1.0f;

StopWatchInterface* timer = NULL;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling

float avgFPS = 0.0f;
unsigned int frameCount = 0;

__global__ void calculate_vertices(float4* pos, unsigned int width, unsigned int height, float time)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float u = x / (float)width;
	__syncthreads();
	float v = y / (float)height;
	__syncthreads();
	u = u * 2.0f - 1.0f;
	v = v * 2.0f - 1.0f;

	float freq = 4.0f;
	float w = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;

	pos[y * width + x] = make_float4(u, w, v, 1.0f);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	void initialize(void);
	void display(void);
	void uninitialize(int);
	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szClassName[] = TEXT("My App");
	bool bDone = false;

	if (fopen_s(&gpFile, "Log.txt", "w") != NULL)
	{
		MessageBox(NULL, TEXT("Cannot Create Log File !!!"), TEXT("Error"), MB_OK);
		exit(EXIT_FAILURE);
	}
	else
		fprintf(gpFile, "Log File Created Successfully...\n");

	fclose(gpFile);

	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.hInstance = hInstance;
	wndclass.lpszClassName = szClassName;
	wndclass.lpszMenuName = NULL;
	wndclass.lpfnWndProc = WndProc;
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);

	RegisterClassEx(&wndclass);

	hwnd = CreateWindowEx(WS_EX_APPWINDOW, szClassName, TEXT("OpenGLPP : OpenGL - CUDA Interoperability"), WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE, 100, 100, WIN_WIDTH, WIN_HEIGHT, NULL, NULL, hInstance, NULL);
	if (hwnd == NULL)
	{
		fprintf(gpFile, "Cannot Create Window...\n");
		uninitialize(1);
	}

	ghwnd = hwnd;

	ShowWindow(hwnd, iCmdShow);
	SetFocus(hwnd);
	SetForegroundWindow(hwnd);

	initialize();

	while (bDone == false)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
				bDone = true;
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		else
		{
			if (gbActiveWindow == true)
			{
				if (gbIsEscapeKeyPressed == true)
					bDone = true;
				display();
			}
		}
	}

	uninitialize(0);
	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	void update(void);
	void resize(int, int);
	void ToggleFullscreen(void);
	switch (iMsg)
	{
	case WM_ACTIVATE:
		if (HIWORD(wParam) == 0)
			gbActiveWindow = true;
		else
			gbActiveWindow = false;
		break;
	case WM_CREATE:
		break;
	case WM_SIZE:
		resize(LOWORD(lParam), HIWORD(lParam));
		break;
	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			gbIsEscapeKeyPressed = true;
			break;

		case 0x31: //1
			MESH_WIDTH = 64;
			MESH_HEIGHT = 64;
			update();
			break;

		case 0x32: //2
			MESH_WIDTH = 128;
			MESH_HEIGHT = 128;
			update();
			break;

		case 0x33: //3
			MESH_WIDTH = 256;
			MESH_HEIGHT = 256;
			update();
			break;

		case 0x34: //4
			MESH_WIDTH = 512;
			MESH_HEIGHT = 512;
			update();
			break;

		case 0x35: //5
			MESH_WIDTH = 1024;
			MESH_HEIGHT = 1024;
			update();
			break;

		case 0x52: //R
			color[0] = 1.0f;
			color[1] = 0.0f;
			color[2] = 0.0f;
			glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
			break;

		case 0x47: //G
			color[0] = 0.0f;
			color[1] = 1.0f;
			color[2] = 0.0f;
			glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
			break;

		case 0x42: //B
			color[0] = 0.0f;
			color[1] = 0.0f;
			color[2] = 1.0f;
			glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
			break;

		case 0x43: //C
			color[0] = 0.0f;
			color[1] = 1.0f;
			color[2] = 1.0f;
			glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
			break;

		case 0x4D: //M
			color[0] = 1.0f;
			color[1] = 0.0f;
			color[2] = 1.0f;
			glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
			break;

		case 0x59: //Y
			color[0] = 1.0f;
			color[1] = 1.0f;
			color[2] = 0.0f;
			glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
			break;

		case 0x4B: //K
			color[0] = 0.0f;
			color[1] = 0.0f;
			color[2] = 0.0f;
			glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
			break;

		case VK_UP:
			gfTranslateFactor -= 0.05f;
			break;

		case VK_DOWN:
			gfTranslateFactor += 0.05f;
			break;

		case 0x46:
			if (gbFullscreen == false)
			{
				ToggleFullscreen();
				gbFullscreen = true;
			}
			else
			{
				ToggleFullscreen();
				gbFullscreen = false;
			}
			break;
		default:
			color[0] = 1.0f;
			color[1] = 1.0f;
			color[2] = 1.0f;
			glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
			break;
		}
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	}
	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void initialize(void)
{
	void resize(int, int);
	void uninitialize(int);
	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;

	ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));

	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 24;
	pfd.cRedBits = 8;
	pfd.cGreenBits = 8;
	pfd.cBlueBits = 8;
	pfd.cAlphaBits = 8;
	pfd.cDepthBits = 32;

	ghdc = GetDC(ghwnd);
	if (ghdc == NULL)
	{
		fprintf(gpFile, "GetDC() Failed.\n");
		uninitialize(1);
	}

	iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);
	if (iPixelFormatIndex == 0)
	{
		fprintf(gpFile, "ChoosePixelFormat() Failed.\n");
		uninitialize(1);
	}

	if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
	{
		fprintf(gpFile, "SetPixelFormat() Failed.\n");
		uninitialize(1);
	}

	ghrc = wglCreateContext(ghdc);
	if (ghrc == NULL)
	{
		fprintf(gpFile, "wglCreateContext() Failed.\n");
		uninitialize(1);
	}

	if (wglMakeCurrent(ghdc, ghrc) == FALSE)
	{
		fprintf(gpFile, "wglMakeCurrent() Failed");
		uninitialize(1);
	}

	GLenum glew_error = glewInit();
	if (glew_error != GLEW_OK)
	{
		wglDeleteContext(ghrc);
		ghrc = NULL;
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	//Vertex Shader
	gVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar* vertexShaderSourceCode =
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"uniform mat4 u_mvp_matrix;" \
		"void main(void)" \
		"{" \
		"gl_Position = u_mvp_matrix * vPosition;" \
		"}";

	glShaderSource(gVertexShaderObject, 1, (const GLchar**)&vertexShaderSourceCode, NULL);

	glCompileShader(gVertexShaderObject);
	GLint iInfoLogLength = 0;
	GLint iShaderCompiledStatus = 0;
	char* szInfoLog = NULL;

	glGetShaderiv(gVertexShaderObject, GL_COMPILE_STATUS, &iShaderCompiledStatus);
	if (iShaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(gVertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(gVertexShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Vertex Shader Compilation Log : %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize(1);
				exit(0);
			}
		}
	}

	//Fragment Shader
	gFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar* fragmentShaderSourceCode =
		"#version 450 core"\
		"\n"\
		"out vec4 FragColor;"\
		"uniform vec4 color;" \
		"void main(void)"\
		"{"\
		"FragColor=color;"\
		"}";

	glShaderSource(gFragmentShaderObject, 1, (const GLchar**)&fragmentShaderSourceCode, NULL);

	glCompileShader(gFragmentShaderObject);

	glGetShaderiv(gFragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompiledStatus);
	if (iShaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(gFragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(gFragmentShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Fragment Shader Compilation Log : %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize(1);
				exit(0);
			}
		}
	}

	//Shader Program
	gShaderProgramObject = glCreateProgram();

	glAttachShader(gShaderProgramObject, gVertexShaderObject);

	glAttachShader(gShaderProgramObject, gFragmentShaderObject);

	glBindAttribLocation(gShaderProgramObject, HAD_ATTRIBUTE_POSITION, "vPosition");

	glLinkProgram(gShaderProgramObject);

	GLint iShaderProgramLinkStatus = 0;

	glGetProgramiv(gShaderProgramObject, GL_LINK_STATUS, &iShaderProgramLinkStatus);
	if (iShaderProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(gShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetProgramInfoLog(gShaderProgramObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Shader Program Link Log : %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize(1);
				exit(0);
			}
		}
	}

	gMVPUniform = glGetUniformLocation(gShaderProgramObject, "u_mvp_matrix");

	gColorUniform = glGetUniformLocation(gShaderProgramObject, "color");

	glGenVertexArrays(1, &gVao);
	glBindVertexArray(gVao);

	glGenBuffers(1, &gVbo);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo);
	glBufferData(GL_ARRAY_BUFFER, MESH_WIDTH * MESH_HEIGHT * 4 * sizeof(float), 0, GL_DYNAMIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(HAD_ATTRIBUTE_POSITION);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	err = cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, gVbo, cudaGraphicsMapFlagsWriteDiscard);
	if (err != cudaSuccess)
	{
		sprintf(str, "GPU Memory Fatal Error = %s In File Name %s at Line No.%d\nExitting...\n", cudaGetErrorString(err), __FILE__, __LINE__);
		MessageBox(NULL, str, TEXT("MSG"), MB_OK);
		//cleanup();
		exit(EXIT_FAILURE);
	}

	glBindVertexArray(0);

	glClearDepth(1.0f);
	glDisable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glShadeModel(GL_SMOOTH);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glEnable(GL_CULL_FACE);


	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	gPerspectiveProjectionMatrix = mat4::identity();

	sdkCreateTimer(&timer);
	resize(WIN_WIDTH, WIN_HEIGHT);
}

void runCuda(void)
{
	void uninitialize(int);

	// MessageBox(NULL,TEXT("In RunCuda"),TEXT("MSG"),MB_OK);
	float4* dptr;
	size_t num_bytes;

	err = cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
	if (err != cudaSuccess)
	{
		sprintf(str, "GPU Memory Fatal Error = %s In File Name %s at Line No.%d\nExitting...\n", cudaGetErrorString(err), __FILE__, __LINE__);
		MessageBox(NULL, str, TEXT("MSG"), MB_OK);
		//cleanup();
		exit(EXIT_FAILURE);
	}

	err = cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, cuda_vbo_resource);
	if (err != cudaSuccess)
	{
		sprintf(str, "GPU Memory Fatal Error = %s In File Name %s at Line No.%d\nExitting...\n", cudaGetErrorString(err), __FILE__, __LINE__);
		MessageBox(NULL, str, TEXT("MSG"), MB_OK);
		//cleanup();
		exit(EXIT_FAILURE);
	}

	dim3 block(8, 8, 1);
	dim3 grid(MESH_WIDTH / block.x, MESH_HEIGHT / block.y, 1);
	//MessageBox(NULL,TEXT("RunCuda Before Kernel"),TEXT("MSG"),MB_OK);
	calculate_vertices << < grid, block >> > (dptr, MESH_WIDTH, MESH_HEIGHT, gfAnimate);

	// err=cudaMemcpy(waveVerticesHost,waveVerticesDevice,MESH_WIDTH*MESH_HEIGHT*sizeof(float4),cudaMemcpyDeviceToHost);
	// if(err!=cudaSuccess)
	// {
	// 	printf("GPU Memory Fatal Error = %s In File Name %s at Line No.%d\nExitting...\n",cudaGetErrorString(err),__FILE__,__LINE__);
	//     uninitialize(1);
	//     exit(EXIT_FAILURE);
	// }

	cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
	// MessageBox(NULL,TEXT("Leaving RunCuda"),TEXT("MSG"),MB_OK);
}

void display(void)
{
	void computeFPS(void);

	runCuda();
	sdkStartTimer(&timer);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//Use Shader Program Object
	glUseProgram(gShaderProgramObject);

	mat4 modelViewMatrix = mat4::identity();
	mat4 modelViewProjectionMatrix = mat4::identity();

	modelViewMatrix = translate(0.0f, 0.0f, gfTranslateFactor);

	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(gMVPUniform, 1, GL_FALSE, modelViewProjectionMatrix);
	glUniform4fv(gColorUniform, 1, color);

	glBindVertexArray(gVao);

	glBindBuffer(GL_ARRAY_BUFFER, gVbo);
	glDrawArrays(GL_POINTS, 0, MESH_WIDTH * MESH_HEIGHT);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	glUseProgram(0);

	gfAnimate += 0.01f;
	sdkStopTimer(&timer);
	//computeFPS();
	SwapBuffers(ghdc);
}

void update(void)
{
	void uninitialize(int);

	glBindVertexArray(gVao);

	glGenBuffers(1, &gVbo);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo);
	glBufferData(GL_ARRAY_BUFFER, MESH_WIDTH * MESH_HEIGHT * 4 * sizeof(float), 0, GL_DYNAMIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(HAD_ATTRIBUTE_POSITION);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	err = cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, gVbo, cudaGraphicsMapFlagsWriteDiscard);
	if (err != cudaSuccess)
	{
		sprintf(str, "GPU Memory Fatal Error = %s In File Name %s at Line No.%d\nExitting...\n", cudaGetErrorString(err), __FILE__, __LINE__);
		MessageBox(NULL, str, TEXT("MSG"), MB_OK);
		uninitialize(1);
		exit(EXIT_FAILURE);
	}

	glBindVertexArray(0);
}

void computeFPS(void)
{
	//float max(float, float);

	frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit)
	{
		avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		fpsCount = 0;
		fpsLimit = (int)max(avgFPS, 1.f);

		sdkResetTimer(&timer);
	}

	char fps[256];
	sprintf(fps, "OpenGL CUDA Interoperability %3.1f fps", avgFPS);
	SetWindowText(ghwnd, fps);
}

// float max(float a, float b)
// {
// 	if(a>b)
// 		return a;
// 	else
// 		return b;
// }

void resize(int width, int height)
{
	if (height == 0)
		height = 1;
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	gPerspectiveProjectionMatrix = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}

void ToggleFullscreen(void)
{
	MONITORINFO mi = { sizeof(MONITORINFO) };
	if (gbFullscreen == false)
	{
		dwStyle = GetWindowLong(ghwnd, GWL_STYLE);
		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{
			if (GetWindowPlacement(ghwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &mi))
			{
				SetWindowLong(ghwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}
		ShowCursor(FALSE);
	}

	else
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
	}
}

void uninitialize(int i_Exit_Flag)
{
	if (gbFullscreen == false)
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
	}

	cudaGraphicsUnregisterResource(cuda_vbo_resource);

	if (gVao)
	{
		glDeleteVertexArrays(1, &gVao);
		gVao = 0;
	}

	if (gVbo)
	{
		glDeleteBuffers(1, &gVbo);
		gVbo = 0;
	}

	//Detach Shader 
	glDetachShader(gShaderProgramObject, gVertexShaderObject);
	glDetachShader(gShaderProgramObject, gFragmentShaderObject);

	//Delete Shader
	glDeleteShader(gVertexShaderObject);
	gVertexShaderObject = 0;

	glDeleteShader(gFragmentShaderObject);
	gFragmentShaderObject = 0;

	//Delete Program
	glDeleteProgram(gShaderProgramObject);
	gShaderProgramObject = 0;

	//Stray call to glUseProgram(0)
	glUseProgram(0);

	wglMakeCurrent(NULL, NULL);

	if (ghrc != NULL)
	{
		wglDeleteContext(ghrc);
		ghrc = NULL;
	}

	if (ghdc != NULL)
	{
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	if (i_Exit_Flag == 0)
	{
		fopen_s(&gpFile, "Log.txt", "a");
		fprintf(gpFile, "Log File Closed Successfully");
		fclose(gpFile);
	}
	else if (i_Exit_Flag == 1)
	{
		fopen_s(&gpFile, "Log.txt", "a");
		fprintf(gpFile, "Log File Closed Erroniously");
		fclose(gpFile);
	}

	gpFile = NULL;

	DestroyWindow(ghwnd);
}
#include<Windows.h>
#include<stdio.h>
#include<gl/glew.h>
#include<GL/GL.h>
#include"vmath.h"


#include<cuda_gl_interop.h>
#include<cuda_runtime.h>
#include<cuda.h>

#include"12-Interop.h"


#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")

#pragma comment(lib, "cudart.lib")

enum {
	AMC_ATTRIBUTE_POSITION = 0,
	AMC_ATTRIBUTE_COLOR,
	AMC_ATTRIBUTE_NORMAL,
	AMC_ATTRIBUTE_TEXCOORD0
};

using namespace vmath;

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

//For FullScreen
bool bIsFullScreen_RRJ = false;
HWND ghwnd_RRJ = NULL;
WINDOWPLACEMENT wpPrev_RRJ = { sizeof(WINDOWPLACEMENT) };
DWORD dwStyle_RRJ;

//For SuperMan
bool bActiveWindow_RRJ = false;
HDC ghdc_RRJ = NULL;
HGLRC ghrc_RRJ = NULL;

//For Error
FILE *gbFile_RRJ = NULL;

//For Shader Program Object;
GLint gShaderProgramObject_RRJ;

//For Perspective Matric
mat4 gPerspectiveProjectionMatrix_RRJ;

//For Triangle
GLuint vao_Points_RRJ;
GLuint vbo_Points_Position_RRJ;

GLuint vao_Points_RRJ_GPU_RRJ;
GLuint vbo_Points_GPU_Pos_RRJ;


//For Uniform
GLuint mvpUniform_RRJ;


//For Interop
const int GMESH_WIDTH = 1024;		//265
const int GMESH_HEIGHT = 1024;		//265
#define MYARRAY_SIZE GMESH_HEIGHT * GMESH_WIDTH * 4

float cpuPos_RRJ[GMESH_HEIGHT][GMESH_WIDTH][4];

struct cudaGraphicsResource *graphicsResource_RRJ = NULL;
float animationTime_RRJ = 0.0f;
bool bOnGPU_RRJ = false;
cudaError_t error_RRJ;




LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow) {
	
	if (fopen_s(&gbFile_RRJ, "Log.txt", "w") != 0) {
		MessageBox(NULL, TEXT("Log Creation Failed!!\n"), TEXT("Error"), MB_OK);
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "Log Created!!\n");

	int initialize(void);
	void display(void);
	void ToggleFullScreen(void);

	int iRet_RRJ;
	bool bDone_RRJ = false;

	WNDCLASSEX wndclass_RRJ;
	HWND hwnd_RRJ;
	MSG msg_RRJ;
	TCHAR szName_RRJ[] = TEXT("RohitRJadhav-PP-PerspectiveTriangle");

	wndclass_RRJ.lpszClassName = szName_RRJ;
	wndclass_RRJ.lpszMenuName = NULL;
	wndclass_RRJ.lpfnWndProc = WndProc;

	wndclass_RRJ.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass_RRJ.cbSize = sizeof(WNDCLASSEX);
	wndclass_RRJ.cbWndExtra = 0;
	wndclass_RRJ.cbClsExtra = 0;

	wndclass_RRJ.hInstance = hInstance;
	wndclass_RRJ.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass_RRJ.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass_RRJ.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wndclass_RRJ.hCursor = LoadCursor(NULL, IDC_ARROW);

	RegisterClassEx(&wndclass_RRJ);

	hwnd_RRJ = CreateWindowEx(WS_EX_APPWINDOW,
		szName_RRJ,
		TEXT("RohitRJadhav-PP-PerspectiveTriangle"),
		WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_VISIBLE,
		100, 100,
		WIN_WIDTH, WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL);

	ghwnd_RRJ = hwnd_RRJ;

	SetForegroundWindow(hwnd_RRJ);
	SetFocus(hwnd_RRJ);

	iRet_RRJ = initialize();
	if (iRet_RRJ == -1) {
		fprintf(gbFile_RRJ, "ChoosePixelFormat() Failed!!\n");
		DestroyWindow(hwnd_RRJ);
	}
	else if (iRet_RRJ == -2) {
		fprintf(gbFile_RRJ, "SetPixelFormat() Failed!!\n");
		DestroyWindow(hwnd_RRJ);
	}
	else if (iRet_RRJ == -3) {
		fprintf(gbFile_RRJ, "wglMakeCurrent() Failed!!\n");
		DestroyWindow(hwnd_RRJ);
	}
	else if (iRet_RRJ == -4) {
		fprintf(gbFile_RRJ, "wglMakeCurrent() Failed!!\n");
		DestroyWindow(hwnd_RRJ);
	}
	else
		fprintf(gbFile_RRJ, "initialize() done!!\n");

	

	ShowWindow(hwnd_RRJ, iCmdShow);
	ToggleFullScreen();

	while (bDone_RRJ == false) {
		if (PeekMessage(&msg_RRJ, NULL, 0, 0, PM_REMOVE)) {
			if (msg_RRJ.message == WM_QUIT)
				bDone_RRJ = true;
			else {
				TranslateMessage(&msg_RRJ);
				DispatchMessage(&msg_RRJ);
			}
		}
		else {
			if (bActiveWindow_RRJ == true) {
				//update();
			}
			display();
		}
	}
	return((int)msg_RRJ.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam) {
	
	void uninitialize(void);
	void ToggleFullScreen(void);
	void resize(int, int);

	switch (iMsg) {
	case WM_SETFOCUS:
		bActiveWindow_RRJ = true;
		break;
	case WM_KILLFOCUS:
		bActiveWindow_RRJ = false;
		break;
	case WM_SIZE:
		resize(LOWORD(lParam), HIWORD(lParam));
		break;

	case WM_CHAR:
		switch (wParam) {
		case VK_ESCAPE:
			DestroyWindow(hwnd);
			break;

		case 'F':
		case 'f':
			ToggleFullScreen();
			break;

		case 'G':
		case 'g':
			bOnGPU_RRJ = true;
			break;

		case 'C':
		case 'c':
			bOnGPU_RRJ = false;
			break;
		}
		break;

	case WM_ERASEBKGND:
		return(0);

	case WM_DESTROY:
		uninitialize();
		PostQuitMessage(0);
		break;
	}
	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void ToggleFullScreen(void) {
	
	MONITORINFO mi_RRJ;

	if (bIsFullScreen_RRJ == false) {
		dwStyle_RRJ = GetWindowLong(ghwnd_RRJ, GWL_STYLE);
		mi_RRJ = { sizeof(MONITORINFO) };
		if (dwStyle_RRJ & WS_OVERLAPPEDWINDOW) {
			if (GetWindowPlacement(ghwnd_RRJ, &wpPrev_RRJ) && GetMonitorInfo(MonitorFromWindow(ghwnd_RRJ, MONITORINFOF_PRIMARY), &mi_RRJ)) {
				SetWindowLong(ghwnd_RRJ, GWL_STYLE, dwStyle_RRJ & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghwnd_RRJ,
					HWND_TOP,
					mi_RRJ.rcMonitor.left,
					mi_RRJ.rcMonitor.top,
					(mi_RRJ.rcMonitor.right - mi_RRJ.rcMonitor.left),
					(mi_RRJ.rcMonitor.bottom - mi_RRJ.rcMonitor.top),
					SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}
		ShowCursor(FALSE);
		bIsFullScreen_RRJ = true;
	}
	else {
		SetWindowLong(ghwnd_RRJ, GWL_STYLE, dwStyle_RRJ | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd_RRJ, &wpPrev_RRJ);
		SetWindowPos(ghwnd_RRJ,
			HWND_TOP,
			0, 0, 0, 0,
			SWP_NOZORDER | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
		bIsFullScreen_RRJ = false;
	}
}

int initialize(void) {

	void resize(int, int);
	void uninitialize(void);


	int devCount_RRJ = 0;
	error_RRJ = cudaGetDeviceCount(&devCount_RRJ);
	if(error_RRJ != cudaSuccess){
		fprintf(gbFile_RRJ, "cudaGetDeviceCount() Failed!!\n");
		uninitialize();
		exit(0);
	}
	else if(devCount_RRJ == 0){
		fprintf(gbFile_RRJ, "devCount_RRJ == 0\n");
		uninitialize();
		exit(0);
	}
	else{
		fprintf(gbFile_RRJ, "DevCount: %d\n", devCount_RRJ);
		cudaSetDevice(0);
	}




	PIXELFORMATDESCRIPTOR pfd_RRJ;
	int iPixelFormatIndex_RRJ;
	GLenum Result_RRJ;

	//Shader Object;
	GLint iVertexShaderObject_RRJ;
	GLint iFragmentShaderObject_RRJ;


	memset(&pfd_RRJ, NULL, sizeof(PIXELFORMATDESCRIPTOR));

	ghdc_RRJ = GetDC(ghwnd_RRJ);

	pfd_RRJ.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd_RRJ.nVersion = 1;
	pfd_RRJ.dwFlags = PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER | PFD_DRAW_TO_WINDOW;
	pfd_RRJ.iPixelType = PFD_TYPE_RGBA;

	pfd_RRJ.cColorBits = 32;
	pfd_RRJ.cRedBits = 8;
	pfd_RRJ.cGreenBits = 8;
	pfd_RRJ.cBlueBits = 8;
	pfd_RRJ.cAlphaBits = 8;

	pfd_RRJ.cDepthBits = 32;

	iPixelFormatIndex_RRJ = ChoosePixelFormat(ghdc_RRJ, &pfd_RRJ);
	if (iPixelFormatIndex_RRJ == 0)
		return(-1);

	if (SetPixelFormat(ghdc_RRJ, iPixelFormatIndex_RRJ, &pfd_RRJ) == FALSE)
		return(-2);

	ghrc_RRJ = wglCreateContext(ghdc_RRJ);
	if (ghrc_RRJ == NULL)
		return(-3);

	if (wglMakeCurrent(ghdc_RRJ, ghrc_RRJ) == FALSE)
		return(-4);

	Result_RRJ = glewInit();
	if (Result_RRJ != GLEW_OK) {
		fprintf(gbFile_RRJ, "glewInit() Failed!!\n");
		uninitialize();
		DestroyWindow(ghwnd_RRJ);
		exit(1);
	}

	/********** Vertex Shader **********/
	iVertexShaderObject_RRJ = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *szVertexShaderSourceCode_RRJ =
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"uniform mat4 u_mvp_matrix;" \
		"void main(void)" \
		"{" \
		"gl_Position = u_mvp_matrix * vPosition;" \
		"}";

	glShaderSource(iVertexShaderObject_RRJ, 1,
		(const GLchar**)&szVertexShaderSourceCode_RRJ, NULL);

	glCompileShader(iVertexShaderObject_RRJ);

	GLint iShaderCompileStatus_RRJ;
	GLint iInfoLogLength_RRJ;
	GLchar *szInfoLog_RRJ = NULL;
	glGetShaderiv(iVertexShaderObject_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(iVertexShaderObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetShaderInfoLog(iVertexShaderObject_RRJ, iInfoLogLength_RRJ,
					&written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Vertex Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
				exit(0);
			}
		}
	}

	/********** Fragment Shader **********/
	iFragmentShaderObject_RRJ = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *szFragmentShaderSourceCode_RRJ =
		"#version 450 core" \
		"\n" \
		"out vec4 FragColor;" \
		"void main(void)" \
		"{" \
		"FragColor = vec4(1.0, 1.0, 0.0, 1.0);" \
		"}";

	glShaderSource(iFragmentShaderObject_RRJ, 1,
		(const GLchar**)&szFragmentShaderSourceCode_RRJ, NULL);

	glCompileShader(iFragmentShaderObject_RRJ);

	iShaderCompileStatus_RRJ = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;

	glGetShaderiv(iFragmentShaderObject_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(iFragmentShaderObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetShaderInfoLog(iFragmentShaderObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Fragment Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
				exit(0);
			}
		}
	}

	/********** Shader Program Object **********/
	gShaderProgramObject_RRJ = glCreateProgram();

	glAttachShader(gShaderProgramObject_RRJ, iVertexShaderObject_RRJ);
	glAttachShader(gShaderProgramObject_RRJ, iFragmentShaderObject_RRJ);

	glBindAttribLocation(gShaderProgramObject_RRJ, AMC_ATTRIBUTE_POSITION, "vPosition");

	glLinkProgram(gShaderProgramObject_RRJ);

	GLint iProgramLinkingStatus_RRJ = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;
	glGetProgramiv(gShaderProgramObject_RRJ, GL_LINK_STATUS, &iProgramLinkingStatus_RRJ);
	if (iProgramLinkingStatus_RRJ == GL_FALSE) {
		glGetProgramiv(gShaderProgramObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetProgramInfoLog(gShaderProgramObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Shader Program Object Linking Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
				exit(0);
			}
		}
	}

	mvpUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_mvp_matrix");



	/********** Position and Vao Vbo **********/
	for(int i = 0; i < GMESH_HEIGHT; i++){
		for(int j = 0; j < GMESH_WIDTH; j++){
			for(int k = 0; k < 4; k++)
				cpuPos_RRJ[i][j][k] = 0.0f;
				
		}
	}


	glGenVertexArrays(1, &vao_Points_RRJ);
	glBindVertexArray(vao_Points_RRJ);

		/********** Position **********/
		glGenBuffers(1, &vbo_Points_Position_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Points_Position_RRJ);

		glBufferData(GL_ARRAY_BUFFER,
			sizeof(float) * MYARRAY_SIZE,
			NULL,
			GL_DYNAMIC_DRAW);

		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
			4,
			GL_FLOAT,
			GL_FALSE,
			0,
			NULL);

		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);

		glBindBuffer(GL_ARRAY_BUFFER, 0);


	glBindVertexArray(0);




	//GPU
	glGenVertexArrays(1, &vao_Points_RRJ_GPU_RRJ);
	glBindVertexArray(vao_Points_RRJ_GPU_RRJ);



		/********** GPU Position **********/
		glGenBuffers(1, &vbo_Points_GPU_Pos_RRJ);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Points_GPU_Pos_RRJ);

		glBufferData(GL_ARRAY_BUFFER,
			sizeof(float) * MYARRAY_SIZE,
			NULL,
			GL_DYNAMIC_DRAW);

		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
			4,
			GL_FLOAT,
			GL_FALSE,
			0,
			NULL);

		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);

		glBindBuffer(GL_ARRAY_BUFFER, 0);


	glBindVertexArray(0);



	//1. Creating Bond b/w graphicsResource_RRJ and vbo_Points_GPU_Pos_RRJ
	error_RRJ = cudaGraphicsGLRegisterBuffer(&graphicsResource_RRJ, vbo_Points_GPU_Pos_RRJ, cudaGraphicsMapFlagsWriteDiscard);
	if(error_RRJ != cudaSuccess){
		fprintf(gbFile_RRJ, "cudaGraphicsGLRegisterBuffer() Failed!!\n");
		uninitialize();
		exit(0);
	}






	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);
	
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	gPerspectiveProjectionMatrix_RRJ = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);
	return(0);
}

void uninitialize(void) {

	cudaGraphicsUnregisterResource(graphicsResource_RRJ);

	

	if (vbo_Points_GPU_Pos_RRJ) {
		glDeleteBuffers(1, &vbo_Points_GPU_Pos_RRJ);
		vbo_Points_GPU_Pos_RRJ = 0;
	}	

	if(vao_Points_RRJ_GPU_RRJ){
		glDeleteVertexArrays(1, &vao_Points_RRJ_GPU_RRJ);
		vao_Points_RRJ_GPU_RRJ = 0;
	}
	

	if (vbo_Points_Position_RRJ) {
		glDeleteBuffers(1, &vbo_Points_Position_RRJ);
		vbo_Points_Position_RRJ = 0;
	}

	if (vao_Points_RRJ) {
		glDeleteVertexArrays(1, &vao_Points_RRJ);
		vao_Points_RRJ = 0;
	}

	GLsizei ShaderCount_RRJ;
	GLsizei ShaderNumber_RRJ;

	if (gShaderProgramObject_RRJ) {
		glUseProgram(gShaderProgramObject_RRJ);

		glGetProgramiv(gShaderProgramObject_RRJ, GL_ATTACHED_SHADERS, &ShaderCount_RRJ);

		GLuint *pShader_RRJ = (GLuint*)malloc(sizeof(GLuint) * ShaderCount_RRJ);
		if (pShader_RRJ) {
			glGetAttachedShaders(gShaderProgramObject_RRJ, ShaderCount_RRJ,
				&ShaderCount_RRJ, pShader_RRJ);
			for (ShaderNumber_RRJ = 0; ShaderNumber_RRJ < ShaderCount_RRJ; ShaderNumber_RRJ++) {
				glDetachShader(gShaderProgramObject_RRJ, pShader_RRJ[ShaderNumber_RRJ]);
				glDeleteShader(pShader_RRJ[ShaderNumber_RRJ]);
				pShader_RRJ[ShaderNumber_RRJ] = 0;
			}
			free(pShader_RRJ);
			pShader_RRJ = NULL;
		}
		glDeleteProgram(gShaderProgramObject_RRJ);
		gShaderProgramObject_RRJ = 0;
		glUseProgram(0);
	}

	if (bIsFullScreen_RRJ == true) {
		SetWindowLong(ghwnd_RRJ, GWL_STYLE, dwStyle_RRJ | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd_RRJ, &wpPrev_RRJ);
		SetWindowPos(ghwnd_RRJ,
			HWND_TOP,
			0, 0, 0, 0,
			SWP_NOZORDER | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
		bIsFullScreen_RRJ = false;
	}

	if (wglGetCurrentContext() == ghrc_RRJ) {
		wglMakeCurrent(NULL, NULL);
	}

	if (ghrc_RRJ) {
		wglDeleteContext(ghrc_RRJ);
		ghrc_RRJ = NULL;
	}

	if (ghdc_RRJ) {
		ReleaseDC(ghwnd_RRJ, ghdc_RRJ);
		ghdc_RRJ = NULL;
	}

	if (gbFile_RRJ) {
		fprintf(gbFile_RRJ, "Log Close!!\n");
		fclose(gbFile_RRJ);
		gbFile_RRJ = NULL;
	}
}

void resize(int width, int height) {
	if (height == 0)
		height = 1;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	gPerspectiveProjectionMatrix_RRJ = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}


void display(void) {

	void launchCpuKernel(unsigned int, unsigned int, float);

	mat4 TranslateMatrix_RRJ;
	mat4 ModelViewMatrix_RRJ;
	mat4 ModelViewProjectionMatrix_RRJ;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(gShaderProgramObject_RRJ);

	TranslateMatrix_RRJ = mat4::identity();
	ModelViewMatrix_RRJ = mat4::identity();
	ModelViewProjectionMatrix_RRJ = mat4::identity();

	TranslateMatrix_RRJ = translate(0.0f, 0.0f, -3.0f);
	ModelViewMatrix_RRJ = ModelViewMatrix_RRJ * TranslateMatrix_RRJ;
	ModelViewProjectionMatrix_RRJ = gPerspectiveProjectionMatrix_RRJ * ModelViewMatrix_RRJ;
	glUniformMatrix4fv(mvpUniform_RRJ, 1, GL_FALSE, ModelViewProjectionMatrix_RRJ);
	



		if(bOnGPU_RRJ == true){
			
			//GPU
			error_RRJ = cudaGraphicsMapResources(1, &graphicsResource_RRJ, 0);
			if(error_RRJ != cudaSuccess){
				fprintf(gbFile_RRJ, "cudaGraphicsMapResources() Failed!!\n");
				uninitialize();
				exit(0);
			}
		

			float4 *gpuPos = NULL;
			size_t byteCount;
			error_RRJ = cudaGraphicsResourceGetMappedPointer((void**)&gpuPos, &byteCount, graphicsResource_RRJ);
			if(error_RRJ != cudaSuccess){
				fprintf(gbFile_RRJ, "cudaGraphicsResourceGetMappedPointer() Failed!!\n");
				uninitialize();
				exit(0);
			}

			
			launchCudaKernel(gpuPos, GMESH_WIDTH, GMESH_HEIGHT, animationTime_RRJ);
			

			error_RRJ = cudaGraphicsUnmapResources(1, &graphicsResource_RRJ, 0);
			if(error_RRJ != cudaSuccess){
				fprintf(gbFile_RRJ, "cudaGraphicsUnmapResources() Failed!!\n");
				uninitialize();
				exit(0);
			}



			glBindVertexArray(vao_Points_RRJ_GPU_RRJ);
				/*glBindBuffer(GL_ARRAY_BUFFER, vbo_Points_GPU_Pos_RRJ);
				//glBufferData(GL_ARRAY_BUFFER, sizeof(float) * MYARRAY_SIZE, gpuPos, GL_DYNAMIC_DRAW);
				glBindBuffer(GL_ARRAY_BUFFER, 0);*/

				glDrawArrays(GL_POINTS, 0, GMESH_WIDTH * GMESH_HEIGHT);
			glBindVertexArray(0);



		}
		else{
			//CPU

			launchCpuKernel(GMESH_WIDTH, GMESH_HEIGHT, animationTime_RRJ);
			
			glBindVertexArray(vao_Points_RRJ);
				glBindBuffer(GL_ARRAY_BUFFER, vbo_Points_Position_RRJ);
				glBufferData(GL_ARRAY_BUFFER, sizeof(float) * MYARRAY_SIZE, cpuPos_RRJ, GL_DYNAMIC_DRAW);
				glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
				glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
				glBindBuffer(GL_ARRAY_BUFFER, 0);

				glDrawArrays(GL_POINTS, 0, GMESH_HEIGHT * GMESH_WIDTH);

			glBindVertexArray(0);

		}




	glUseProgram(0);

	SwapBuffers(ghdc_RRJ);


	//animationTime_RRJ = 1.0;
	animationTime_RRJ = animationTime_RRJ + 0.01;

}




void launchCpuKernel(unsigned int width, unsigned int height, float time){


	for(int i = 0; i < width; i++){
		for(int j = 0; j < height ; j++){
			for(int k = 0; k < 4; k++){

				float freq = 4.0f;
				float u = i / float(width);
				float v = j / float(height);

				u = u * 2.0f - 1.0f;
				v = v * 2.0f - 1.0f;

				float w = sin(freq * u + time) * cos(freq * v + time) * 0.5f;
				
				
				if(k == 0)
					cpuPos_RRJ[i][j][k] = u;
				else if(k == 1)
					cpuPos_RRJ[i][j][k] = w;
				else if(k == 2)
					cpuPos_RRJ[i][j][k] = v;
				else if(k == 3)
					cpuPos_RRJ[i][j][k] = 1.0f;

				//fprintf(gbFile_RRJ, "%f/%f/%f/%f\n", cpuPos_RRJ[i][j][0], cpuPos_RRJ[i][j][1], cpuPos_RRJ[i][j][2], cpuPos_RRJ[i][j][3]);
			}
		}
	}
}



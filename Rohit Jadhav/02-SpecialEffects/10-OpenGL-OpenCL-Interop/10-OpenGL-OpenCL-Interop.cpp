#include<Windows.h>
#include<stdio.h>
#include<gl/glew.h>
#include<GL/GL.h>
#include"vmath.h"

#include<CL/opencl.h>


#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")

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
bool bIsFullScreen = false;
HWND ghwnd = NULL;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };
DWORD dwStyle;

//For SuperMan
bool bActiveWindow = false;
HDC ghdc = NULL;
HGLRC ghrc = NULL;

//For Error
FILE *gbFile = NULL;

//For Shader Program Object;
GLint gShaderProgramObject;

//For Perspective Matric
mat4 gPerspectiveProjectionMatrix;

//For Triangle
GLuint vao_Grid;
GLuint vbo_Grid_Pos_CPU;
GLuint vbo_Grid_Pos_GPU;


//For Interop
const int GMESH_WIDTH = 2048;		//265
const int GMESH_HEIGHT = 2048;		//265
#define MYARRAY_SIZE GMESH_HEIGHT * GMESH_WIDTH * 4

float cpuPos_RRJ[GMESH_HEIGHT][GMESH_WIDTH][4];
float animationTime_RRJ = 0.0f;


//For Toggle
#define ON_CPU 1
#define ON_GPU 2
GLint iWhichDevice = ON_CPU;


//For Uniform
GLuint mvpUniform;


//For OpenCL
cl_int ret_ocl;

cl_uint numOfPlatformIDs;
cl_platform_id *oclPlatformID;



struct PlatformInfo{
	cl_platform_id iPlatformID;
	int iNumOfDevice;
	cl_device_id *pDeviceID;
};

typedef struct PlatformInfo PLATFORM_INFO;
PLATFORM_INFO *gpPlatform = NULL;

cl_context oclContext;
cl_command_queue oclCommandQueue;
cl_program oclProgram;
cl_kernel oclKernel;

char *szKernelSourceCode = NULL;
size_t sizeKernelSourceCodeLength;

cl_mem deviceOutputBuffer = NULL;



LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow) {
	
	if (fopen_s(&gbFile, "Log.txt", "w") != 0) {
		MessageBox(NULL, TEXT("Log Creation Failed!!\n"), TEXT("Error"), MB_OK);
		exit(0);
	}
	else
		fprintf(gbFile, "Log Created!!\n");

	int initialize(void);
	void display(void);
	void ToggleFullScreen(void);

	int iRet;
	bool bDone = false;

	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szName[] = TEXT("RohitRJadhav-PP-OpenGL-OpenCL-Interop");

	wndclass.lpszClassName = szName;
	wndclass.lpszMenuName = NULL;
	wndclass.lpfnWndProc = WndProc;

	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.cbWndExtra = 0;
	wndclass.cbClsExtra = 0;

	wndclass.hInstance = hInstance;
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);

	RegisterClassEx(&wndclass);

	hwnd = CreateWindowEx(WS_EX_APPWINDOW,
		szName,
		TEXT("RohitRJadhav-PP-OpenGL-OpenCL-Interop"),
		WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_VISIBLE,
		100, 100,
		WIN_WIDTH, WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL);

	ghwnd = hwnd;

	SetForegroundWindow(hwnd);
	SetFocus(hwnd);

	iRet = initialize();
	if (iRet == -1) {
		fprintf(gbFile, "ChoosePixelFormat() Failed!!\n");
		DestroyWindow(hwnd);
	}
	else if (iRet == -2) {
		fprintf(gbFile, "SetPixelFormat() Failed!!\n");
		DestroyWindow(hwnd);
	}
	else if (iRet == -3) {
		fprintf(gbFile, "wglMakeCurrent() Failed!!\n");
		DestroyWindow(hwnd);
	}
	else if (iRet == -4) {
		fprintf(gbFile, "wglMakeCurrent() Failed!!\n");
		DestroyWindow(hwnd);
	}
	else
		fprintf(gbFile, "initialize() done!!\n");

	

	ShowWindow(hwnd, iCmdShow);
	ToggleFullScreen();

	while (bDone == false) {
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
			if (msg.message == WM_QUIT)
				bDone = true;
			else {
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		else {
			if (bActiveWindow == true) {
				//update();
			}
			display();
		}
	}
	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam) {
	
	void uninitialize(void);
	void ToggleFullScreen(void);
	void resize(int, int);

	switch (iMsg) {
	case WM_SETFOCUS:
		bActiveWindow = true;
		break;
	case WM_KILLFOCUS:
		bActiveWindow = false;
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
			iWhichDevice = ON_GPU;
			break;

		case 'C':
		case 'c':
			iWhichDevice = ON_CPU;
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
	
	MONITORINFO mi;

	if (bIsFullScreen == false) {
		dwStyle = GetWindowLong(ghwnd, GWL_STYLE);
		mi = { sizeof(MONITORINFO) };
		if (dwStyle & WS_OVERLAPPEDWINDOW) {
			if (GetWindowPlacement(ghwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &mi)) {
				SetWindowLong(ghwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghwnd,
					HWND_TOP,
					mi.rcMonitor.left,
					mi.rcMonitor.top,
					(mi.rcMonitor.right - mi.rcMonitor.left),
					(mi.rcMonitor.bottom - mi.rcMonitor.top),
					SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}
		ShowCursor(FALSE);
		bIsFullScreen = true;
	}
	else {
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd,
			HWND_TOP,
			0, 0, 0, 0,
			SWP_NOZORDER | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
		bIsFullScreen = false;
	}
}

int initialize(void) {

	void resize(int, int);
	void uninitialize(void);
	void initializeOpenCL_DeviceAndPlatform(void);
	void getOpenCLContextFromDeviceIDAndPlatformID(cl_platform_id, cl_device_id);
	char* loadOclProgramSource(const char*, const char*, size_t*);



	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;
	GLenum Result;

	//Shader Object;
	GLint iVertexShaderObject;
	GLint iFragmentShaderObject;


	memset(&pfd, NULL, sizeof(PIXELFORMATDESCRIPTOR));

	ghdc = GetDC(ghwnd);

	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER | PFD_DRAW_TO_WINDOW;
	pfd.iPixelType = PFD_TYPE_RGBA;

	pfd.cColorBits = 32;
	pfd.cRedBits = 8;
	pfd.cGreenBits = 8;
	pfd.cBlueBits = 8;
	pfd.cAlphaBits = 8;

	pfd.cDepthBits = 32;

	iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);
	if (iPixelFormatIndex == 0)
		return(-1);

	if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
		return(-2);

	ghrc = wglCreateContext(ghdc);
	if (ghrc == NULL)
		return(-3);

	if (wglMakeCurrent(ghdc, ghrc) == FALSE)
		return(-4);

	Result = glewInit();
	if (Result != GLEW_OK) {
		fprintf(gbFile, "glewInit() Failed!!\n");
		uninitialize();
		DestroyWindow(ghwnd);
		exit(1);
	}

	/********** Vertex Shader **********/
	iVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *szVertexShaderSourceCode =
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"uniform mat4 u_mvp_matrix;" \
		"void main(void)" \
		"{" \
		"gl_Position = u_mvp_matrix * vPosition;" \
		"}";

	glShaderSource(iVertexShaderObject, 1,
		(const GLchar**)&szVertexShaderSourceCode, NULL);

	glCompileShader(iVertexShaderObject);

	GLint iShaderCompileStatus;
	GLint iInfoLogLength;
	GLchar *szInfoLog = NULL;
	glGetShaderiv(iVertexShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE) {
		glGetShaderiv(iVertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (GLchar*)malloc(sizeof(GLchar) * iInfoLogLength);
			if (szInfoLog != NULL) {
				GLsizei written;
				glGetShaderInfoLog(iVertexShaderObject, iInfoLogLength,
					&written, szInfoLog);
				fprintf(gbFile, "Vertex Shader Compilation Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				DestroyWindow(ghwnd);
				exit(0);
			}
		}
	}

	/********** Fragment Shader **********/
	iFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *szFragmentShaderSourceCode =
		"#version 450 core" \
		"\n" \
		"out vec4 FragColor;" \
		"void main(void)" \
		"{" \
		"FragColor = vec4(0.0, 0.0, 0.50, 1.0);" \
		"}";

	glShaderSource(iFragmentShaderObject, 1,
		(const GLchar**)&szFragmentShaderSourceCode, NULL);

	glCompileShader(iFragmentShaderObject);

	iShaderCompileStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetShaderiv(iFragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE) {
		glGetShaderiv(iFragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength);
			if (szInfoLog != NULL) {
				GLsizei written;
				glGetShaderInfoLog(iFragmentShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gbFile, "Fragment Shader Compilation Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				DestroyWindow(ghwnd);
				exit(0);
			}
		}
	}

	/********** Shader Program Object **********/
	gShaderProgramObject = glCreateProgram();

	glAttachShader(gShaderProgramObject, iVertexShaderObject);
	glAttachShader(gShaderProgramObject, iFragmentShaderObject);

	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_POSITION, "vPosition");

	glLinkProgram(gShaderProgramObject);

	GLint iProgramLinkingStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;
	glGetProgramiv(gShaderProgramObject, GL_LINK_STATUS, &iProgramLinkingStatus);
	if (iProgramLinkingStatus == GL_FALSE) {
		glGetProgramiv(gShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0) {
			szInfoLog = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength);
			if (szInfoLog != NULL) {
				GLsizei written;
				glGetProgramInfoLog(gShaderProgramObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gbFile, "Shader Program Object Linking Error: %s\n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				uninitialize();
				DestroyWindow(ghwnd);
				exit(0);
			}
		}
	}

	mvpUniform = glGetUniformLocation(gShaderProgramObject, "u_mvp_matrix");

	



	/********** Position and Vao Vbo **********/
	for(int i = 0; i < GMESH_HEIGHT; i++){
		for(int j = 0; j < GMESH_WIDTH; j++){
			for(int k = 0; k < 4; k++)
				cpuPos_RRJ[i][j][k] = 0.0f;
				
		}
	}




	/********** Grid **********/
	glGenVertexArrays(1, &vao_Grid);
	glBindVertexArray(vao_Grid);

		/********** For CPU POSITION *********/
		glGenBuffers(1, &vbo_Grid_Pos_CPU);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Grid_Pos_CPU);
		glBufferData(GL_ARRAY_BUFFER, sizeof(cpuPos_RRJ), NULL, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0,NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		/********** For GPU Position **********/
		glGenBuffers(1, &vbo_Grid_Pos_GPU);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Grid_Pos_GPU);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * MYARRAY_SIZE, NULL, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


	glBindVertexArray(0);



	/********** For OpenCL **********/
	initializeOpenCL_DeviceAndPlatform();
	getOpenCLContextFromDeviceIDAndPlatformID(gpPlatform[0].iPlatformID, gpPlatform[0].pDeviceID[0]);



	/********** Creating OpenCL Buffer From OpenGL Buffer **********/
	deviceOutputBuffer = clCreateFromGLBuffer(oclContext, CL_MEM_WRITE_ONLY, vbo_Grid_Pos_GPU, &ret_ocl);
	if(ret_ocl != CL_SUCCESS){
		fprintf(gbFile, "ERROR: clCreateFromGLBuffer()\n");
		uninitialize();
		exit(1);
	}
	else
		fprintf(gbFile, "SUCCESS: clCreateFromGLBuffer()\n");



	/********** OpenCL Program ************/
	szKernelSourceCode = loadOclProgramSource("10-OpenCL-Kernel.cl", "", &sizeKernelSourceCodeLength);

	oclProgram = clCreateProgramWithSource(oclContext, 1, (const char**)&szKernelSourceCode, &sizeKernelSourceCodeLength, &ret_ocl);
	if(ret_ocl != CL_SUCCESS){
		fprintf(gbFile, "ERROR: clCreateProgramWithSource()\n");
		uninitialize();
		exit(1);
	}
	else
		fprintf(gbFile, "SUCCESS: Program Created\n");



	/********** OpenCL Program Compilation **********/
	ret_ocl = clBuildProgram(oclProgram, 0, NULL, NULL, NULL, NULL);
	if(ret_ocl != CL_SUCCESS){
		fprintf(gbFile, "ERROR: clBuildProgram()\n");
		
		size_t len;
		char buffer[1024];
		clGetProgramBuildInfo(oclProgram, gpPlatform[0].pDeviceID[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		fprintf(gbFile, "%s\n", buffer);
		uninitialize();
		exit(1);
	}
	else
		fprintf(gbFile, "SUCCESS: Program Build\n");



	/********** Creating OpenCL Kernel **********/
	oclKernel = clCreateKernel(oclProgram, "interop", &ret_ocl);
	if(ret_ocl != CL_SUCCESS){
		fprintf(gbFile, "ERROR: clCreateKernel()\n");
		uninitialize();
		exit(1);
	}
	else
		fprintf(gbFile, "SUCCESS: clCreateKernel()\n");

	int width = GMESH_WIDTH;
	int height = GMESH_HEIGHT;


	/********** Setting Kernel Argument **********/
	ret_ocl = clSetKernelArg(oclKernel, 0, sizeof(cl_mem), (void*)&deviceOutputBuffer);
	if(ret_ocl != CL_SUCCESS){
		fprintf(gbFile, "ERROR: clSetKernelArg() For deviceOutputBuffer\n");
		uninitialize();
		exit(1);
	}


	ret_ocl = clSetKernelArg(oclKernel, 1, sizeof(int), (void*)&width);
	if(ret_ocl != CL_SUCCESS){
		fprintf(gbFile, "ERROR: clSetKernelArg() For GMESH_WIDTH\n");
		uninitialize();
		exit(1);
	}		


	ret_ocl = clSetKernelArg(oclKernel, 2, sizeof(int), (void*)&height);
	if(ret_ocl != CL_SUCCESS){
		fprintf(gbFile, "ERROR: clSetKernelArg() For GMESH_HEIGHT\n");
		uninitialize();
		exit(1);
	}



	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);
	
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	gPerspectiveProjectionMatrix = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);
	return(0);
}


void initializeOpenCL_DeviceAndPlatform(void){

	void uninitialize(void);

	char platformInfo[128];


	/********** Platform and Their Respective Devices **********/
	clGetPlatformIDs(0, NULL, &numOfPlatformIDs);
	oclPlatformID = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numOfPlatformIDs);

	ret_ocl = clGetPlatformIDs(numOfPlatformIDs, oclPlatformID, NULL);
	if(ret_ocl != CL_SUCCESS){
		fprintf(gbFile, "ERROR: clGetPlatformIDs()\n");
		uninitialize();
		exit(0);
	}
	else{

		//fprintf(gbFile, "SUCCESS: numOfPlatformIDs: %d\n", numOfPlatformIDs);


		/********** For Platform Detail ********/
		gpPlatform = (PLATFORM_INFO*)malloc(sizeof(PLATFORM_INFO) * numOfPlatformIDs);
		if(gpPlatform == NULL){
			fprintf(gbFile, "ERROR: gpTotalDevices malloc()\n");
			uninitialize();
			exit(1);
		}



		/********** Getting Device and info For Each Platform **********/
		for(int i = 0; i < numOfPlatformIDs; i++){
			
			
			/********** Platform Vendor Info **********/
			ret_ocl = clGetPlatformInfo(oclPlatformID[i], CL_PLATFORM_VENDOR, sizeof(char) * 128, platformInfo, NULL);
			if(ret_ocl != CL_SUCCESS){
				fprintf(gbFile, "ERROR: clGetPlatformInfo()\n");
				uninitialize();
				exit(0);
			}
			else
				fprintf(gbFile, "SUCCESS: OpenCL Supported Platform: %s\n", platformInfo);




			/********** OpenCL Supported Devices Per Platform **********/
			cl_uint temp;

			gpPlatform[i].iPlatformID = oclPlatformID[i];

			clGetDeviceIDs(gpPlatform[i].iPlatformID, CL_DEVICE_TYPE_ALL, 0, NULL, &temp);

			gpPlatform[i].iNumOfDevice = temp;
			gpPlatform[i].pDeviceID = (cl_device_id*)malloc(sizeof(cl_device_id) * gpPlatform[i].iNumOfDevice);

			clGetDeviceIDs(
				gpPlatform[i].iPlatformID, 
				CL_DEVICE_TYPE_ALL, 
				gpPlatform[i].iNumOfDevice, 
				gpPlatform[i].pDeviceID, NULL);





			/********** Displaying Per Device Info for Each OpenCL Supported Device **********/
			char deviceInfo[128];
			cl_bool yesNo;
			size_t grpSize;
			size_t itemSize[3];
			cl_device_type deviceType;
			for(int j = 0; j < gpPlatform[i].iNumOfDevice; j++){

				fprintf(gbFile, "\tSUCCESS: Device No: %d\n", j + 1);
				clGetDeviceInfo(gpPlatform[i].pDeviceID[j], CL_DEVICE_NAME, sizeof(deviceInfo), deviceInfo, NULL);
				fprintf(gbFile, "\tSUCCESS: Device Name: %s\n", deviceInfo);

				clGetDeviceInfo(gpPlatform[i].pDeviceID[j], CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, NULL);
				
				switch(deviceType){
					case CL_DEVICE_TYPE_CPU:
						fprintf(gbFile, "\tSUCCESS: Device Type: GL_DEVICE_TYPE_CPU\n");
						break;

					case CL_DEVICE_TYPE_GPU:
						fprintf(gbFile, "\tSUCCESS: Device Type: GL_DEVICE_TYPE_GPU\n");
						break;
				}



				clGetDeviceInfo(gpPlatform[i].pDeviceID[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(grpSize), &grpSize, NULL);
				fprintf(gbFile, "\tSUCCESS: Device Max Work Group Size: %zd\n", grpSize);

				clGetDeviceInfo(gpPlatform[i].pDeviceID[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(itemSize), itemSize, NULL);
				fprintf(gbFile, "\tSUCCESS: Device Max Work Group Item Sizes: %zd/ %zd/ %zd\n", itemSize[0], itemSize[1], itemSize[2]);

				clGetDeviceInfo(gpPlatform[i].pDeviceID[j], CL_DEVICE_AVAILABLE, sizeof(yesNo), &yesNo, NULL);
				fprintf(gbFile, "\tSUCCESS: Device Available(Yes = 1/ No = 0): %d\n", yesNo);

				fprintf(gbFile, "\n");

			}	

			

		}

	}

	fprintf(gbFile, "\n");


}


void getOpenCLContextFromDeviceIDAndPlatformID(cl_platform_id pid, cl_device_id did){

	void uninitialize(void);

	cl_context_properties property[] = {
		CL_CONTEXT_PLATFORM, (cl_context_properties)pid,
		CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(), //<--ghrc
		CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(), 		//<--ghdc
		0
	};


	/********** Finding How Many OpenCL Devices are Related to this OpenGL Context **********
	size_t bytes;

	clGetGLContextInfoKHR(property, CL_DEVICES_FOR_GL_CONTEXT_KHR, 0, NULL, &bytes);

	cl_uint numOfDev = bytes / sizeof(cl_device_id);
	
	fprintf(gbFile, "SUCCESS: OpenGL Context Supporting OpenCL Devices %d\n", numOfDev);*/






	/********** Checking OpenCL and OpenGL Sharering Ahe ka **********/
	size_t extention;
	clGetDeviceInfo(did, CL_DEVICE_EXTENSIONS, 0, NULL, &extention);
	char *ext = (char*)malloc(sizeof(char) * extention);
	clGetDeviceInfo(did, CL_DEVICE_EXTENSIONS, extention, ext, NULL);

	fprintf(gbFile, "SUCCESS: %s\n", ext);

	free((void*)ext);
	ext = NULL;



	fprintf(gbFile, "\n");
	fprintf(gbFile, "SUCCESS: Context is Created For: \n");

	char devName[128];
	cl_device_type devTyp;

	clGetDeviceInfo(did, CL_DEVICE_NAME, sizeof(devName), devName, NULL);
	clGetDeviceInfo(did, CL_DEVICE_TYPE, sizeof(devTyp), &devTyp, NULL);

	fprintf(gbFile, "\tSUCCESS: Device Name: %s\n", devName);
	switch(devTyp){
		case CL_DEVICE_TYPE_CPU:
			fprintf(gbFile, "\tSUCCESS: Device Type : CL_DEVICE_TYPE_CPU\n");
			break;

		case CL_DEVICE_TYPE_GPU:
			fprintf(gbFile, "\tSUCCESS: Device Type : CL_DEVICE_TYPE_GPU\n");
			break;
	}	


	/********** OpenCL Context **********/
	oclContext = clCreateContext(property, 1, (const cl_device_id*)&did, NULL, NULL, &ret_ocl);
	if(ret_ocl != CL_SUCCESS){
		fprintf(gbFile, "ERROR: clCreateContext()\n");
		uninitialize();
		exit(1);
	}
	else
		fprintf(gbFile, "SUCCESS: OpenCL Context Created\n");


	/********** OpenCL Command Queue **********/
	oclCommandQueue = clCreateCommandQueue(oclContext, did, 0, &ret_ocl);
	if(ret_ocl != CL_SUCCESS){
		fprintf(gbFile, "ERROR: clCreateCommandQueue()\n");
		uninitialize();
		exit(1);
	}
	else
		fprintf(gbFile, "SUCCESS: OpenCL Command Queue Created\n");

	fprintf(gbFile, "\n");

}


void uninitialize(void) {

	if(deviceOutputBuffer){
		clReleaseMemObject(deviceOutputBuffer);
		deviceOutputBuffer = NULL;
	}

	if(oclKernel){
		clReleaseKernel(oclKernel);
		oclKernel = 0;
	}


	if(oclProgram){
		clReleaseProgram(oclProgram);
		oclProgram = 0;

	}

	if(szKernelSourceCode){
		free((void*)szKernelSourceCode);
		szKernelSourceCode = NULL;
	}



	if(oclCommandQueue){
		clReleaseCommandQueue(oclCommandQueue);
		oclCommandQueue = 0;
	}

	if(oclContext){
		clReleaseContext(oclContext);
		oclContext = 0;
	}


	if(gpPlatform){

		for(int i = 0; i < numOfPlatformIDs; i++){

			if(gpPlatform[i].pDeviceID){
				free(gpPlatform[i].pDeviceID);
				gpPlatform[i].pDeviceID = NULL;
			}

		}

		free(gpPlatform);
		gpPlatform = NULL;
	}



	
	if (vbo_Grid_Pos_CPU) {
		glDeleteBuffers(1, &vbo_Grid_Pos_CPU);
		vbo_Grid_Pos_CPU = 0;
	}

	if (vao_Grid) {
		glDeleteVertexArrays(1, &vao_Grid);
		vao_Grid = 0;
	}

	GLsizei ShaderCount;
	GLsizei ShaderNumber;

	if (gShaderProgramObject) {
		glUseProgram(gShaderProgramObject);

		glGetProgramiv(gShaderProgramObject, GL_ATTACHED_SHADERS, &ShaderCount);

		GLuint *pShader = (GLuint*)malloc(sizeof(GLuint) * ShaderCount);
		if (pShader) {
			glGetAttachedShaders(gShaderProgramObject, ShaderCount,
				&ShaderCount, pShader);
			for (ShaderNumber = 0; ShaderNumber < ShaderCount; ShaderNumber++) {
				glDetachShader(gShaderProgramObject, pShader[ShaderNumber]);
				glDeleteShader(pShader[ShaderNumber]);
				pShader[ShaderNumber] = 0;
			}
			free(pShader);
			pShader = NULL;
		}
		glDeleteProgram(gShaderProgramObject);
		gShaderProgramObject = 0;
		glUseProgram(0);
	}

	if (bIsFullScreen == true) {
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd,
			HWND_TOP,
			0, 0, 0, 0,
			SWP_NOZORDER | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
		bIsFullScreen = false;
	}

	if (wglGetCurrentContext() == ghrc) {
		wglMakeCurrent(NULL, NULL);
	}

	if (ghrc) {
		wglDeleteContext(ghrc);
		ghrc = NULL;
	}

	if (ghdc) {
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	if (gbFile) {
		fprintf(gbFile, "Log Close!!\n");
		fclose(gbFile);
		gbFile = NULL;
	}
}

void resize(int width, int height) {
	if (height == 0)
		height = 1;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	gPerspectiveProjectionMatrix = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}

void display(void) {

	void launchCpuKernel(unsigned int , unsigned int , float);
	void uninitialize(void);



	mat4 TranslateMatrix;
	mat4 ModelViewMatrix;
	mat4 ModelViewProjectionMatrix;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(gShaderProgramObject);

	TranslateMatrix = mat4::identity();
	ModelViewMatrix = mat4::identity();
	ModelViewProjectionMatrix = mat4::identity();

	TranslateMatrix = translate(0.0f, 0.0f, -3.0f);
	ModelViewMatrix = ModelViewMatrix * TranslateMatrix;
	ModelViewProjectionMatrix = gPerspectiveProjectionMatrix * ModelViewMatrix;
	glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, ModelViewProjectionMatrix);
	

	if(iWhichDevice == ON_CPU){

		launchCpuKernel(GMESH_WIDTH, GMESH_HEIGHT, animationTime_RRJ);
		glBindVertexArray(vao_Grid);

			glBindBuffer(GL_ARRAY_BUFFER, vbo_Grid_Pos_CPU);
			glBufferData(GL_ARRAY_BUFFER, sizeof(cpuPos_RRJ), cpuPos_RRJ, GL_DYNAMIC_DRAW);
			glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
			glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);

			glDrawArrays(GL_POINTS, 0, GMESH_HEIGHT * GMESH_WIDTH);

			glBindBuffer(GL_ARRAY_BUFFER, 0);

			

		glBindVertexArray(0);
		glUseProgram(0);

	}
	else if(iWhichDevice == ON_GPU){


		 clEnqueueAcquireGLObjects(oclCommandQueue, 1, &deviceOutputBuffer, 0, NULL, NULL);


		 	ret_ocl = clSetKernelArg(oclKernel, 3, sizeof(float), (void*)&animationTime_RRJ);
			if(ret_ocl != CL_SUCCESS){
				fprintf(gbFile, "ERROR: clSetKernelArg() For animationTime_RRJ\n");
				uninitialize();
				exit(1);
			}

			const size_t localSize[] = {32, 32};
			const size_t globalSize[] = {GMESH_WIDTH, GMESH_HEIGHT};

			ret_ocl = clEnqueueNDRangeKernel(oclCommandQueue, oclKernel, 2, 0, globalSize, localSize, 0, NULL, NULL);
			if(ret_ocl != CL_SUCCESS){
				fprintf(gbFile, "ERROR: clEnqueueNDRangerKernel()\n");
				uninitialize();
				exit(1);
			}
			clFinish(oclCommandQueue);

		clEnqueueReleaseGLObjects(oclCommandQueue, 1, &deviceOutputBuffer, 0, NULL, NULL);


		glUseProgram(gShaderProgramObject);
		glBindVertexArray(vao_Grid);

			glBindBuffer(GL_ARRAY_BUFFER, vbo_Grid_Pos_GPU);
			glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
			glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
			glDrawArrays(GL_POINTS, 0, GMESH_HEIGHT * GMESH_WIDTH);
			glBindBuffer(GL_ARRAY_BUFFER, 0);


		glBindVertexArray(0);
		glUseProgram(0);


	}


	SwapBuffers(ghdc);


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



char *loadOclProgramSource(const char *filename, const char *preamble, size_t *sizeFinalLength) {

	FILE *pFile = NULL;
	size_t sizeSourceLength;

	pFile = fopen(filename, "rb");
	if (pFile == NULL)
		return(NULL);

	size_t sizePreambleLength = (size_t)strlen(preamble);

	fseek(pFile, 0, SEEK_END);
	sizeSourceLength = ftell(pFile);
	fseek(pFile, 0, SEEK_SET);

	char *sourceString = (char*)malloc(sizeSourceLength + sizePreambleLength + 1);
	memcpy(sourceString, preamble, sizePreambleLength);

	if (fread((sourceString)+sizePreambleLength, sizeSourceLength, 1, pFile) != 1) {
		fclose(pFile);
		free(sourceString);
		return(0);
	}

	fclose(pFile);
	if (sizeFinalLength != 0)
		*sizeFinalLength = sizeSourceLength + sizePreambleLength;

	sourceString[sizeSourceLength + sizePreambleLength] = '\0';
	return(sourceString);
}



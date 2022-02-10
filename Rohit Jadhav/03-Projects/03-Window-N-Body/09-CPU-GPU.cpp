#include<Windows.h>
#include<stdio.h>
#include<gl/glew.h>
#include<GL/GL.h>

#include<cuda_gl_interop.h>
#include<cuda_runtime.h>
#include<cuda.h>

#include"vmath.h"
#include"helper_timer.h"
#include"09-CPU-GPU-Kernel.h"
#include"09-Resource.h"
#include<Mmsystem.h>

#include<ft2build.h>
#include FT_FREETYPE_H


#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "freetype.lib")
#pragma comment(lib, "winmm.lib")


enum {
	AMC_ATTRIBUTE_POSITION = 0,
	AMC_ATTRIBUTE_OLD_POSITION,
	AMC_ATTRIBUTE_COLOR,
	AMC_ATTRIBUTE_NORMAL,
	AMC_ATTRIBUTE_TEXCOORD0
};

using namespace vmath;

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

typedef struct _NUM_PARTICALES{
	unsigned int  uiParticals;
}NUM_PARTICALES;


NUM_PARTICALES particals[] = {
	{1024 * 1},
	{1024 * 2},
	{1024 * 4},
	{1024 * 8},
	{1024 * 16},
	{1024 * 32}
};



int giParticalsCount = 1;



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
GLuint vao_Points;
GLuint vbo_Points_Position;
GLuint vbo_Points_Color;


GLfloat *points_Pos = NULL;
GLfloat *points_Velocity = NULL;
GLfloat *points_Force = NULL;
GLfloat *points_Color = NULL;

/*GLfloat points_Pos[PARTICALES * 4];
GLfloat points_Force[PARTICALES * 3];
GLfloat points_Velocity[PARTICALES * 4];
GLfloat points_Color[PARTICALES * 4];*/

/*GLfloat *points_Pos = NULL;
GLfloat *points_Acc = NULL;
GLfloat *points_Velocity = NULL;*/



//For Uniform
GLuint mvpUniform;

GLuint modelViewMatrixUniform;
GLuint projectionMatrixUniform;

GLuint timeUniform;


// For Points
typedef struct _DEMO_VERSIONS{
	GLfloat fTimeStep;
	GLfloat fClusterScale;
	GLfloat fVelocityScale;
	GLfloat fSoftening;
	GLfloat fDumping;
	GLfloat fPointSize;
	GLfloat fX, fY, fZ;

}DEMO_VERSIONS;


DEMO_VERSIONS demos[] = {

	{ 0.016f, 1.54f, 8.0f, 0.1f, 1.0f, 1.0f, 0, -2, -100},
	{ 0.016f, 0.68f, 20.0f, 0.1f, 1.0f, 0.8f, 0, -2, -30},
	{ 0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
	{ 0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
	{ 0.0019f, 0.32f, 276.0f, 1.0f, 1.0f, 0.07f, 0, 0, -5},
	{ 0.0016f, 0.32f, 272.0f, 0.145f, 1.0f, 0.08f, 0, 0, -5},
	{ 0.016000f, 6.040000f, 0.000000f, 1.000000f, 1.000000f, 0.760000f, 0, 0, -50},
};

GLint iDemoCount = 0;

#define RRJ_RANDOM 1
#define RRJ_SHELL 2
#define RRJ_EXPAND 3

GLint iFlags = RRJ_SHELL;



//For Camera
GLfloat gfCamTrans[] = {0.0f, -2.0f, 150.0f};
GLfloat gfCamTrans_Lag[] = {0.0f, -2.0f, 150.0f};

GLfloat gfCamRot[] = {0.0f, 0.0f, 0.0f};
GLfloat gfCamRot_Lag[] = {0.0f, 0.0f, 0.0f};

const GLfloat gfInertia = 1.5f;

GLfloat gfDemoTime = 8000.0f;
StopWatchInterface *gpDemoTimer = NULL;




// *** For CUDA ***
GLuint vbo_GPU_NewPos;
GLuint vbo_GPU_OldPos;
GLuint vbo_GPU_Color;

cudaError_t gCudaError;
#define RRJ_CPU 1
#define RRJ_GPU 2

GLuint iPlatform = RRJ_CPU;

float *pNewPos = NULL;
float *pOldPos = NULL;
float *pNewVel = NULL;
float *pOldVelo = NULL;


GLfloat *gpGpuPos = NULL;
GLfloat *gGpuColor = NULL;

/*GLfloat gpGpuPos[4 * PARTICALES];
GLfloat gGpuColor[4 * PARTICALES];*/


GLuint samplerUniform;


// *** For Font ***
GLuint vao_Rect;
GLuint vbo_Rect_Position;


struct Characters{
	GLuint textureId;
	vec2 size;
	vec2 bearing;
	unsigned int advance;
};


struct Characters TotalChar[128];

FT_Library gFt;
FT_Face gFace;

GLuint fontColorUniform;
GLuint toggleUniform;
GLuint gToggle = 1;

#define BLACKOUT 0

#define AMC 1
#define BLACKOUT1 2

#define GROUP 3
#define BLACKOUT2 4

#define NBODY 5
#define BLACKOUT3 6

#define DEMO 7

#define TECHNOLOGY_USED 8
#define BLACKOUT4 9

#define THANKYOU 10
#define END 11


GLuint giScene = -1;

GLfloat gfSceneTime = 5000.0f;
StopWatchInterface *gpSceneTimer = NULL;

StopWatchInterface *gpFpsCounter = NULL;
char gszFps[128];


// *** For Point Sprint ***
GLfloat gfPointSprint = 2.0f;
GLuint pointSpritUniform;
GLuint texture_PointSprite;




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
	TCHAR szName[] = TEXT("RohitRJadhav-PP-09-CPU-GPU-N-Body");

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
		TEXT("RohitRJadhav-PP-09-CPU-GPU-N-Body"),
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
	void ResetAllPointsData(int i);
	void ResetPointsData(void);

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
			iPlatform = RRJ_GPU;
			sdkResetTimer(&gpDemoTimer);
			//sdkResetTimer(&gpFpsCounter);
			//sdkStopTimer(&gpFpsCounter);
			//memset(gszFps, 0, sizeof(gszFps));
			break;

		case 'C':
		case 'c':
			iPlatform = RRJ_CPU;
			sdkResetTimer(&gpDemoTimer);
			//sdkResetTimer(&gpFpsCounter);
			//sdkStopTimer(&gpFpsCounter);
			//memset(gszFps, 0, sizeof(gszFps));
			break;

		case 'Q':
		case 'q':
			giScene = TECHNOLOGY_USED;
			gfSceneTime = 5000.0f;
			sdkResetTimer(&gpSceneTimer);
			break;


		case '1':
			giParticalsCount = 1;
			fprintf(gbFile, "1\n");
			ResetAllPointsData(giParticalsCount);
			fprintf(gbFile, "1 out\n");
			sdkResetTimer(&gpDemoTimer);
			//sdkResetTimer(&gpFpsCounter);
			//sdkStopTimer(&gpFpsCounter);
			//memset(gszFps, 0, sizeof(gszFps));
			break;


		case '2':
			giParticalsCount = 2;
			fprintf(gbFile, "2\n");
			ResetAllPointsData(giParticalsCount);
			fprintf(gbFile, "2 out\n");
			sdkResetTimer(&gpDemoTimer);
			//sdkResetTimer(&gpFpsCounter);
			//sdkStopTimer(&gpFpsCounter);
			//memset(gszFps, 0, sizeof(gszFps));
			break;

		case '3':
			giParticalsCount = 3;
			fprintf(gbFile, "3\n");
			ResetAllPointsData(giParticalsCount);
			fprintf(gbFile, "3 out\n");
			sdkResetTimer(&gpDemoTimer);
			//sdkResetTimer(&gpFpsCounter);
			//sdkStopTimer(&gpFpsCounter);
			//memset(gszFps, 0, sizeof(gszFps));
			break;



		case '4':
			giParticalsCount = 4;
			fprintf(gbFile, "4\n");
			ResetAllPointsData(giParticalsCount);
			fprintf(gbFile, "4 out\n");
			sdkResetTimer(&gpDemoTimer);
			//sdkResetTimer(&gpFpsCounter);
			//sdkStopTimer(&gpFpsCounter);
			//memset(gszFps, 0, sizeof(gszFps));
			break;


		case '5':
			giParticalsCount = 5;
			fprintf(gbFile, "5\n");
			ResetAllPointsData(giParticalsCount);
			fprintf(gbFile, "5 out\n");
			sdkResetTimer(&gpDemoTimer);
			//sdkResetTimer(&gpFpsCounter);
			//sdkStopTimer(&gpFpsCounter);
			//memset(gszFps, 0, sizeof(gszFps));
			break;

		case '6':
			giParticalsCount = 6;
			fprintf(gbFile, "6\n");
			ResetAllPointsData(giParticalsCount);
			fprintf(gbFile, "6 out\n");
			sdkResetTimer(&gpDemoTimer);
			//sdkResetTimer(&gpFpsCounter);
			//sdkStopTimer(&gpFpsCounter);
			//memset(gszFps, 0, sizeof(gszFps));
			break;


		case 'P':
		case 'p':
			giScene = BLACKOUT;
			gfSceneTime = 5000.0f;
			sdkResetTimer(&gpSceneTimer);
			PlaySound(MAKEINTRESOURCE(ID_SONG), GetModuleHandle(NULL), SND_RESOURCE | SND_ASYNC | SND_LOOP);
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
	void ResetPointsData(void);
	void loadAllFonts(void);


	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;
	GLenum Result;

	//Shader Object;
	GLint iVertexShaderObject;
	GLint iFragmentShaderObject;


	// *** Setting Cuda Device ***

	int iCudaDevCount = 0;
	gCudaError = cudaGetDeviceCount(&iCudaDevCount);
	if(gCudaError != cudaSuccess){
		fprintf(gbFile, "ERROR: cudaGetDeviceCount() Failed\n");
		uninitialize();
		DestroyWindow(ghwnd);
	}
	else if(iCudaDevCount == 0){
		fprintf(gbFile, "ERROR: iCudaDevCount == 0\n");
		uninitialize();
		DestroyWindow(ghwnd);
	}
	else{
		fprintf(gbFile, "SUCCESS: iCudaDevCount : %d\n", iCudaDevCount);
		cudaSetDevice(0);
	}



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
		"in vec4 vColor;" \

		"out vec4 outColor;" \
		"out vec2 outTex;" \

		"out float outIntensity;" \
		
		"uniform mat4 u_mvp_matrix;" \

		"uniform mat4 u_modelview_matrix;" \
		"uniform mat4 u_projection_matrix;" \

		"uniform int u_Toggle;" \


		"void main(void)" \
		"{" \

			"if (u_Toggle == 1) { " \

				"outTex = vPosition.zw;" \
				"gl_Position = u_mvp_matrix * vec4(vPosition.xy, 0.0f, 1.0f);" \

			"}" \

			"else { " \

				"gl_PointSize = 10.0f;" \

				"outColor = vColor;" \

				//"outTex = gl_PointCoord.xy;" \
				
				"gl_Position = u_projection_matrix * u_modelview_matrix * vPosition;" \



			"}" \

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

		"uniform int u_Toggle;" \

		"uniform sampler2D u_sampler;" \

		"in vec4 outColor;" \
		"in vec2 outTex;" \


		"out vec4 FragColor;" \

		"uniform vec3 u_texColor;" \

		"vec4 my_lerp(vec4 a, vec4 b, float w) { " \
			"return(a + w * (b - a));" \
		"}" \



		"void main(void)" \
		"{" \

			 "if(u_Toggle == 1) { " \

			 	"vec4 sampled = vec4(1.0f, 1.0f, 1.0f, texture(u_sampler, outTex));" \
				"FragColor = vec4(u_texColor, 1.0f) * sampled;" \

			 "}" \

			 "else {" \

			 	"vec4 color = (0.6f  + 0.4f * vec4(1.0f)) * texture2D(u_sampler, gl_PointCoord.xy);" \
    	
			 	"FragColor = color * my_lerp(vec4(0.1, 0.0, 0.0, color.w), vec4(1.0, 0.7, 0.3, color.w), color.w);" \
    			

    			"}" \
   			
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
	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_COLOR, "vColor");
	
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
	modelViewMatrixUniform = glGetUniformLocation(gShaderProgramObject, "u_modelview_matrix");
	projectionMatrixUniform = glGetUniformLocation(gShaderProgramObject, "u_projection_matrix");


	samplerUniform = glGetUniformLocation(gShaderProgramObject, "u_sampler");
	fontColorUniform = glGetUniformLocation(gShaderProgramObject, "u_texColor");
	toggleUniform = glGetUniformLocation(gShaderProgramObject, "u_Toggle");





	// *************** Fonts ***************


	loadAllFonts();


	/********** Vao Rect On Which We Apply Texture **********/
	glGenVertexArrays(1, &vao_Rect);
	glBindVertexArray(vao_Rect);

	/********** Position **********/
	glGenBuffers(1, &vbo_Rect_Position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Rect_Position);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(GLfloat) * 6 * 4,
		NULL,
		GL_DYNAMIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		4,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);





	

	/********* POSITIOn and Data **********/
	if(points_Pos == NULL){
		points_Pos = (GLfloat*)malloc(sizeof(GLfloat) * particals[giParticalsCount - 1].uiParticals * 4);
		if(points_Pos == NULL){
			fprintf(gbFile, "ERROR: malloc for position failed\n");
			uninitialize();
			DestroyWindow(ghwnd);
		}
	}

	

	if(points_Color == NULL){
		points_Color = (GLfloat*)malloc(sizeof(GLfloat) * particals[giParticalsCount - 1].uiParticals * 4);
		if(points_Color == NULL){
			fprintf(gbFile, "ERROR: malloc for Color failed\n");
			uninitialize();
			DestroyWindow(ghwnd);
		}
	}



	if(points_Velocity == NULL){
		points_Velocity = (GLfloat*)malloc(sizeof(GLfloat) * particals[giParticalsCount - 1].uiParticals * 4);
		if(points_Velocity == NULL){
			fprintf(gbFile, "ERROR: malloc for velocity failed\n");
			uninitialize();
			DestroyWindow(ghwnd);
		}
	}


	if(points_Force == NULL){
		points_Force = (GLfloat*)malloc(sizeof(GLfloat) * particals[giParticalsCount - 1].uiParticals * 3);
		if(points_Force == NULL){
			fprintf(gbFile, "ERROR: malloc for Force failed\n");
			uninitialize();
			DestroyWindow(ghwnd);
		}
	}


	if(gpGpuPos == NULL){
		gpGpuPos = (GLfloat*)malloc(sizeof(GLfloat) * particals[giParticalsCount - 1].uiParticals * 4);
		if(gpGpuPos == NULL){
			fprintf(gbFile, "ERROR: malloc for gpGpuPos failed\n");
			uninitialize();
			DestroyWindow(ghwnd);
		}
	}


	if(gGpuColor == NULL){
		gGpuColor = (GLfloat*)malloc(sizeof(GLfloat) * particals[giParticalsCount - 1].uiParticals * 4);
		if(gGpuColor == NULL){
			fprintf(gbFile, "ERROR: malloc for gGpuColor failed\n");
			uninitialize();
			DestroyWindow(ghwnd);
		}
	}


	
	fprintf(gbFile, "%d\n", particals[giParticalsCount - 1].uiParticals);

	if (particals[giParticalsCount - 1].uiParticals <= 1024)
	{
		demos[iDemoCount].fClusterScale = 1.52f;
		demos[iDemoCount].fVelocityScale = 2.0f;
	}
	else if (particals[giParticalsCount - 1].uiParticals <= 2048)
	{
		demos[iDemoCount].fClusterScale = 1.56f;
		demos[iDemoCount].fVelocityScale = 2.64f;
	}
	else if (particals[giParticalsCount - 1].uiParticals <= 4096)
	{
		demos[iDemoCount].fClusterScale = 1.68f;
		demos[iDemoCount].fVelocityScale = 2.98f;
	}
	else if (particals[giParticalsCount - 1].uiParticals <= 8192)
	{
		demos[iDemoCount].fClusterScale = 1.98f;
		demos[iDemoCount].fVelocityScale = 2.9f;
	}
	else if (particals[giParticalsCount - 1].uiParticals <= 16384)
	{
		demos[iDemoCount].fClusterScale = 1.54f;
		demos[iDemoCount].fVelocityScale = 8.0f;
	}
	else if (particals[giParticalsCount - 1].uiParticals <= 32768)
	{
		demos[iDemoCount].fClusterScale = 1.44f;
		demos[iDemoCount].fVelocityScale = 11.0f;
	}
	
	ResetPointsData();

	//memcpy(gpGpuPos, points_Pos, sizeof(GLfloat) * 4 * PARTICALES);

	// *** Memory For Velocity *** 
	gCudaError = cudaMalloc((void**)&pNewVel, sizeof(GLfloat) * 4 * particals[giParticalsCount - 1].uiParticals);
	if(gCudaError != cudaSuccess){
		fprintf(gbFile, "ERROR: cudaMalloc() Failed for pNewVel : %s\n", cudaGetErrorString(gCudaError));
		uninitialize();
		DestroyWindow(ghwnd);
	}

	gCudaError = cudaMalloc((void**)&pOldVelo, sizeof(GLfloat) * 4 * particals[giParticalsCount - 1].uiParticals);
	if(gCudaError != cudaSuccess){
		fprintf(gbFile, "ERROR: cudaMalloc() Failed for pOldVelo : %s\n", cudaGetErrorString(gCudaError));
		uninitialize();
		DestroyWindow(ghwnd);
	}


	gfCamTrans[0] = demos[iDemoCount].fX;
	gfCamTrans[1] = demos[iDemoCount].fY;
	gfCamTrans[2] = demos[iDemoCount].fZ;

	gfCamTrans_Lag[0] = demos[iDemoCount].fX;
	gfCamTrans_Lag[1] = demos[iDemoCount].fY;
	gfCamTrans_Lag[2] = demos[iDemoCount].fZ;


	// *** Points ***
	glGenVertexArrays(1, &vao_Points);
	glBindVertexArray(vao_Points);

		//Pos
		glGenBuffers(1, &vbo_Points_Position);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Points_Position);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * particals[giParticalsCount - 1].uiParticals * 4, points_Pos, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		//Color
		glGenBuffers(1, &vbo_Points_Color);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Points_Color);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * particals[giParticalsCount - 1].uiParticals * 4, points_Color, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		// *** GPU New Pos ***
		glGenBuffers(1, &vbo_GPU_NewPos);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_GPU_NewPos);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * particals[giParticalsCount - 1].uiParticals * 4, gpGpuPos, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		// *** GPU Old Pos ***
		glGenBuffers(1, &vbo_GPU_OldPos);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_GPU_OldPos);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * particals[giParticalsCount - 1].uiParticals * 4, gpGpuPos, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_OLD_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_OLD_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		// *** GPU Color ***
		glGenBuffers(1, &vbo_GPU_Color);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_GPU_Color);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * particals[giParticalsCount - 1].uiParticals * 4, gGpuColor, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);


	// *** Register VBO's ***
	gCudaError = cudaGLRegisterBufferObject(vbo_GPU_NewPos);
	if(gCudaError != cudaSuccess){
		fprintf(gbFile, "ERROR: cudaGLRegisterBufferObject() Failed for pNewPos : %s\n", cudaGetErrorString(gCudaError));
		uninitialize();
		DestroyWindow(ghwnd);
	}

	gCudaError = cudaGLRegisterBufferObject(vbo_GPU_OldPos);
	if(gCudaError != cudaSuccess){
		fprintf(gbFile, "ERROR: cudaGLRegisterBufferObject() Failed for pOldPos : %s\n", cudaGetErrorString(gCudaError));
		uninitialize();
		DestroyWindow(ghwnd);
	}



	void CreatePointSpriteTex(int);

	CreatePointSpriteTex(32);

	
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);

	// ********** For Blending **********
	//glEnable(GL_BLEND);
	//glBlendFunc(GL_ONE, GL_ONE);


	//glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	sdkCreateTimer(&gpDemoTimer);
	sdkStartTimer(&gpDemoTimer);


	sdkCreateTimer(&gpSceneTimer);
	sdkStartTimer(&gpSceneTimer);

	sdkCreateTimer(&gpFpsCounter);
	

	//glPointSize(1.20f);

	//glEnable(GL_POINT_SMOOTH);


	gPerspectiveProjectionMatrix = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);
	return(0);
}

void loadAllFonts(void){

	void uninitialize(void);


	if(FT_Init_FreeType(&gFt)){
		fprintf(gbFile, "loadAllFonts: FT_Init_FreeType() Failed\n");
		uninitialize();
		DestroyWindow(ghwnd);
	}


	if(FT_New_Face(gFt, "C:\\Windows\\Fonts\\Arial.ttf", 0, &gFace)){
		fprintf(gbFile, "loadAllFonts: FT_New_Face() failed\n");
		uninitialize();
		DestroyWindow(ghwnd);
	}
	else{


		FT_Set_Pixel_Sizes(gFace, 0, 48);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

		for(unsigned char i = 0; i < 128; i++){

			if(FT_Load_Char(gFace, i, FT_LOAD_RENDER)){
				fprintf(gbFile, "loadAllFonts: FT_Load_Face() failed\n");
				uninitialize();
				DestroyWindow(ghwnd);
			}

			GLuint texture;

			glGenTextures(1, &texture);
			glBindTexture(GL_TEXTURE_2D, texture);

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

			glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, 
				gFace->glyph->bitmap.width, gFace->glyph->bitmap.rows,
				0,
				GL_RED,
				GL_UNSIGNED_BYTE, gFace->glyph->bitmap.buffer);

			glBindTexture(GL_TEXTURE_2D, 0);



			TotalChar[i].textureId = texture;
			TotalChar[i].size = vec2(gFace->glyph->bitmap.width, gFace->glyph->bitmap.rows);
			TotalChar[i].bearing = vec2(gFace->glyph->bitmap_left, gFace->glyph->bitmap_top);
			TotalChar[i].advance = (unsigned int)gFace->glyph->advance.x;
		}

	}

	FT_Done_Face(gFace);
	FT_Done_FreeType(gFt);
}



GLfloat my_normalize(vec3 &v){

	float dist = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

	if(dist > 1e-6){
		v[0] = v[0] / dist;
		v[1] = v[1] / dist;
		v[2] = v[2] / dist;
	}

	return(dist);
}






void uninitialize(void) {



	if(points_Pos){
		free(points_Pos);
		points_Pos = NULL;	
	}

	if(points_Velocity){
		free(points_Velocity);
		points_Velocity = NULL;
	}

	if(points_Color){
		free(points_Color);
		points_Color = NULL;
	}

	if(points_Force){
		free(points_Force);
		points_Force = NULL;
	}


	if(gpGpuPos){
		free(gpGpuPos);
		gpGpuPos = NULL;
	}

	if(gGpuColor){
		free(gGpuColor);
		gGpuColor = NULL;
	}

	if(pNewVel){
		cudaFree(pNewVel);
		pNewVel = NULL;
	}

	if(pOldVelo){
		cudaFree(pOldVelo);
		pOldVelo = NULL;
	}


	cudaGLUnregisterBufferObject(vbo_GPU_NewPos);

	cudaGLUnregisterBufferObject(vbo_GPU_OldPos);


	sdkDeleteTimer(&gpDemoTimer);
	sdkDeleteTimer(&gpSceneTimer);
	sdkDeleteTimer(&gpFpsCounter);


	if(texture_PointSprite){
		glDeleteTextures(1, &texture_PointSprite);
		texture_PointSprite = 0;
	}


	if(pOldVelo){
		cudaFree((void**)&pOldVelo);
		pOldVelo = NULL;
	}

	if(pNewVel){
		cudaFree((void**)&pNewVel);
		pNewVel = NULL;
	}


	if(vbo_GPU_Color){
		glDeleteBuffers(1, &vbo_GPU_Color);
		vbo_GPU_Color = 0;
	}

	if(vbo_GPU_OldPos){
		glDeleteBuffers(1, &vbo_GPU_OldPos);
		vbo_GPU_OldPos = 0;
	}

	if(vbo_GPU_NewPos){
		glDeleteBuffers(1, &vbo_GPU_NewPos);
		vbo_GPU_NewPos = NULL;
	}

	if(vbo_Points_Color){
		glDeleteBuffers(1, &vbo_Points_Color);
		vbo_Points_Color = 0;
	}

	if (vbo_Points_Position) {
		glDeleteBuffers(1, &vbo_Points_Position);
		vbo_Points_Position = 0;
	}

	if (vao_Points) {
		glDeleteVertexArrays(1, &vao_Points);
		vao_Points = 0;
	}


	// ********** Fonts **********

	for(int i = 0; i < 128; i++){
		if(TotalChar[i].textureId){
			glDeleteTextures(1, &TotalChar[i].textureId);
			TotalChar[i].textureId = 0;
		}
	}



	if (vbo_Rect_Position) {
		glDeleteBuffers(1, &vbo_Rect_Position);
		vbo_Rect_Position = 0;
	}

	if (vao_Rect) {
		glDeleteVertexArrays(1, &vao_Rect);
		vao_Rect = 0;
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

	gPerspectiveProjectionMatrix = perspective(60.0f, (GLfloat)width / (GLfloat)height, 0.1f, 1000.0f);
}


void display(void) {

	void update(void);
	void ResetPointsData(void);
	void RenderText(char*, float, float, float, vec3);
	void CreatePointSpriteTex(int resolution);



	mat4 TranslateMatrix;
	mat4 ModelViewMatrix;
	mat4 rotateMatrix;
	mat4 ModelViewProjectionMatrix;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	sdkResetTimer(&gpFpsCounter);
	sdkStartTimer(&gpFpsCounter);


	glUseProgram(gShaderProgramObject);


	switch(giScene){

		case BLACKOUT:

			if(sdkGetTimerValue(&gpSceneTimer) > gfSceneTime){
				giScene = AMC;
				gfSceneTime = 5000.0f;
				sdkResetTimer(&gpSceneTimer);
			}

			break;

		case AMC:

			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 

			TranslateMatrix = mat4::identity();
			ModelViewMatrix = mat4::identity();
			rotateMatrix = mat4::identity();
			ModelViewProjectionMatrix = mat4::identity();

			TranslateMatrix = translate(0.0f, 0.0f, -100.0f);
			ModelViewMatrix = TranslateMatrix * rotateMatrix;

			ModelViewProjectionMatrix = gPerspectiveProjectionMatrix * ModelViewMatrix;
			glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, ModelViewProjectionMatrix);

			glUniform1i(toggleUniform, 1);	

			RenderText("ASTROMEDICOMP", -30.0f, 0.0f, 0.150f, vec3(1.0f, 1.0f, 0.0f));

			if(sdkGetTimerValue(&gpSceneTimer) > gfSceneTime){
				giScene = BLACKOUT1;
				gfSceneTime = 1000.0f;
				sdkResetTimer(&gpSceneTimer);
			}

			glDisable(GL_BLEND);

			break;

		case BLACKOUT1:

			if(sdkGetTimerValue(&gpSceneTimer) > gfSceneTime){
				giScene = GROUP;
				gfSceneTime = 5000.0f;
				sdkResetTimer(&gpSceneTimer);
			}
			break;


		case GROUP:

			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 

			TranslateMatrix = mat4::identity();
			ModelViewMatrix = mat4::identity();
			rotateMatrix = mat4::identity();
			ModelViewProjectionMatrix = mat4::identity();

			TranslateMatrix = translate(0.0f, 0.0f, -100.0f);
			ModelViewMatrix = TranslateMatrix * rotateMatrix;

			ModelViewProjectionMatrix = gPerspectiveProjectionMatrix * ModelViewMatrix;
			glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, ModelViewProjectionMatrix);	

			RenderText("COMPUTE GROUP", -30.0f, 0.0f, 0.150f, vec3(1.0f, 1.0f, 0.0f));
			RenderText("PRESENTS...", 0.0f, -10.0f, 0.10f, vec3(1.0f, 1.0f, 0.0f));

			if(sdkGetTimerValue(&gpSceneTimer) > gfSceneTime){
				giScene = BLACKOUT2;
				gfSceneTime = 1000.0f;
				sdkResetTimer(&gpSceneTimer);
			}

			glDisable(GL_BLEND);

			break;



		case BLACKOUT2:

			if(sdkGetTimerValue(&gpSceneTimer) > gfSceneTime){
				giScene = NBODY;
				gfSceneTime = 5000.0f;
				sdkResetTimer(&gpSceneTimer);
			}
			break;


		case NBODY:

			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 

			TranslateMatrix = mat4::identity();
			ModelViewMatrix = mat4::identity();
			rotateMatrix = mat4::identity();
			ModelViewProjectionMatrix = mat4::identity();

			TranslateMatrix = translate(0.0f, 0.0f, -100.0f);
			ModelViewMatrix = TranslateMatrix * rotateMatrix;

			ModelViewProjectionMatrix = gPerspectiveProjectionMatrix * ModelViewMatrix;
			glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, ModelViewProjectionMatrix);	

			RenderText("N-Body Simulation", -30.0f, 0.0f, 0.15f, vec3(1.0f, 1.0f, 0.0f));

			if(sdkGetTimerValue(&gpSceneTimer) > gfSceneTime){
				giScene = BLACKOUT3;
				gfSceneTime = 1000.0f;
				sdkResetTimer(&gpSceneTimer);
				sdkResetTimer(&gpDemoTimer);
			}

			glDisable(GL_BLEND);

			break;


		case BLACKOUT3:

			if(sdkGetTimerValue(&gpSceneTimer) > gfSceneTime){
				giScene = DEMO;
				gfSceneTime = 0.0f;
				sdkResetTimer(&gpSceneTimer);
				CreatePointSpriteTex(32);
			}
			break;



		case DEMO:

			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 

			TranslateMatrix = mat4::identity();
			ModelViewMatrix = mat4::identity();
			rotateMatrix = mat4::identity();
			ModelViewProjectionMatrix = mat4::identity();

			TranslateMatrix = translate(0.0f, 0.0f, -100.0f);
			ModelViewMatrix = TranslateMatrix * rotateMatrix;

			ModelViewProjectionMatrix = gPerspectiveProjectionMatrix * ModelViewMatrix;
			glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, ModelViewProjectionMatrix);

			glUniform1i(toggleUniform, 1);	

			RenderText("Press 1 : 1024 x 01 x 4   Particles", -100.0f, 50.0f, 0.050f, vec3(1.0f, 1.0f, 1.0f));
			RenderText("Press 2 : 1024 x 02 x 4   Particles", -100.0f, 45.0f, 0.050f, vec3(1.0f, 1.0f, 1.0f));
			RenderText("Press 3 : 1024 x 04 x 4   Particles", -100.0f, 40.0f, 0.050f, vec3(1.0f, 1.0f, 1.0f));
			RenderText("Press 4 : 1024 x 08 x 4   Particles", -100.0f, 35.0f, 0.050f, vec3(1.0f, 1.0f, 1.0f));
			RenderText("Press 5 : 1024 x 16 x 4   Particles", -100.0f, 30.0f, 0.050f, vec3(1.0f, 1.0f, 1.0f));
			RenderText("Press 6 : 1024 x 32 x 4   Particles", -100.0f, 25.0f, 0.050f, vec3(1.0f, 1.0f, 1.0f));

			switch(giParticalsCount){
				case 1:
					RenderText("Press 1 : 1024 x 01 x 4   Particles", -100.0f, 50.0f, 0.050f, vec3(0.0f, 1.0f, 0.0f));
					break;

				case 2:
					RenderText("Press 2 : 1024 x 02 x 4   Particles", -100.0f, 45.0f, 0.050f, vec3(0.0f, 1.0f, 0.0f));
					break;

				case 3:
					RenderText("Press 3 : 1024 x 04 x 4   Particles", -100.0f, 40.0f, 0.050f, vec3(0.0f, 1.0f, 0.0f));
					break;

				case 4:
					RenderText("Press 4 : 1024 x 08 x 4   Particles", -100.0f, 35.0f, 0.050f, vec3(0.0f, 1.0f, 0.0f));
					break;

				case 5:
					RenderText("Press 5 : 1024 x 16 x 4   Particles", -100.0f, 30.0f, 0.050f, vec3(0.0f, 1.0f, 0.0f));
					break;

				case 6:
					RenderText("Press 6 : 1024 x 32 x 4   Particles", -100.0f, 25.0f, 0.050f, vec3(0.0f, 1.0f, 0.0f));
					break;
			}


			if(iPlatform == RRJ_CPU)
				RenderText("RUNNING ON CPU", 50.0f, 50.0f, 0.10f, vec3(1.0f, 0.0f, 0.0f));
			else
				RenderText("RUNNING ON GPU", 50.0f, 50.0f, 0.10f, vec3(0.0f, 1.0f, 0.0f));


			RenderText(gszFps, 50.0f, 40.0f, 0.060f, vec3(1.0f, 1.0f, 0.0f));

			RenderText("Press Q to QUIT", -100.0f, -50.0f, 0.10f, vec3(1.0f, 1.0f, 0.0f));

			glDisable(GL_BLEND);


			if(iPlatform == RRJ_CPU){


				if(sdkGetTimerValue(&gpDemoTimer) > gfDemoTime){

					iDemoCount++;
					
					if(iDemoCount > 5)
						iDemoCount = 0;

					iFlags = RRJ_SHELL;
					sdkResetTimer(&gpDemoTimer);

					gfCamTrans[0] = demos[iDemoCount].fX;
					gfCamTrans[1] = demos[iDemoCount].fY;
					gfCamTrans[2] = demos[iDemoCount].fZ;

					gfCamTrans_Lag[0] = demos[iDemoCount].fX;
					gfCamTrans_Lag[1] = demos[iDemoCount].fY;
					gfCamTrans_Lag[2] = demos[iDemoCount].fZ;


					//memset(points_Acc, 0, sizeof(GLfloat) * 4 * PARTICALES);
					ResetPointsData();

					glBindBuffer(GL_ARRAY_BUFFER, vbo_Points_Position);
					glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * particals[giParticalsCount - 1].uiParticals * 4, points_Pos, GL_DYNAMIC_DRAW);
					glBindBuffer(GL_ARRAY_BUFFER, 0);

					glBindBuffer(GL_ARRAY_BUFFER, vbo_Points_Color);
					glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * particals[giParticalsCount - 1].uiParticals * 4, points_Color, GL_DYNAMIC_DRAW);
					glBindBuffer(GL_ARRAY_BUFFER, 0);

				}

				TranslateMatrix = mat4::identity();
				ModelViewMatrix = mat4::identity();
				rotateMatrix = mat4::identity();
				ModelViewProjectionMatrix = mat4::identity();


				gfCamTrans_Lag[0] = gfCamTrans_Lag[0] + (gfCamTrans[0] - gfCamTrans_Lag[0]) * gfInertia;
				gfCamTrans_Lag[1] = gfCamTrans_Lag[1] + (gfCamTrans[1] - gfCamTrans_Lag[1]) * gfInertia;
				gfCamTrans_Lag[2] = gfCamTrans_Lag[2] + (gfCamTrans[2] - gfCamTrans_Lag[2]) * gfInertia;

				gfCamRot_Lag[0] = gfCamRot_Lag[0] + (gfCamRot[0] - gfCamRot_Lag[0]) * gfInertia;
				gfCamRot_Lag[1] = gfCamRot_Lag[1] + (gfCamRot[1] - gfCamRot_Lag[1]) * gfInertia;
				gfCamRot_Lag[2] = gfCamRot_Lag[2] + (gfCamRot[2] - gfCamRot_Lag[2]) * gfInertia;


				TranslateMatrix = translate(gfCamTrans_Lag[0], gfCamTrans_Lag[1], gfCamTrans_Lag[2]);
				rotateMatrix = rotate(gfCamRot_Lag[0], 1.0f, 0.0f, 0.0f);
				rotateMatrix = rotateMatrix * rotate(gfCamRot_Lag[1], 0.0f, 1.0f, 0.0f);

				ModelViewMatrix = TranslateMatrix * rotateMatrix;

				//ModelViewProjectionMatrix = gPerspectiveProjectionMatrix * ModelViewMatrix;
				//glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, ModelViewProjectionMatrix);	

				glUniformMatrix4fv(modelViewMatrixUniform, 1, GL_FALSE, ModelViewMatrix);
				glUniformMatrix4fv(projectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

				glUniform1i(toggleUniform, 0);


				glEnable(GL_POINT_SPRITE_ARB);
				glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
				glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
				//glPointSize(gfPointSprint);
				glBlendFunc(GL_SRC_ALPHA, GL_ONE);
				glEnable(GL_BLEND);
				glDepthMask(GL_FALSE);


				glActiveTextureARB(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, texture_PointSprite);
				glUniform1i(samplerUniform, 0);

				update();

				glBindVertexArray(vao_Points);
			
				glBindBuffer(GL_ARRAY_BUFFER, vbo_Points_Position);
				glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * particals[giParticalsCount - 1].uiParticals * 4, points_Pos, GL_DYNAMIC_DRAW);
				glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
				glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
				glBindBuffer(GL_ARRAY_BUFFER, 0);

				glBindBuffer(GL_ARRAY_BUFFER, vbo_Points_Color);
				glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 4, GL_FLOAT, GL_FALSE, 0, NULL);
				glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
				glBindBuffer(GL_ARRAY_BUFFER, 0);

				glDrawArrays(GL_POINTS, 0, particals[giParticalsCount - 1].uiParticals);

				glBindVertexArray(0);
			
				glUseProgram(0);

				glDisable(GL_POINT_SPRITE_ARB);
			 	glDisable(GL_BLEND);
			 	glDepthMask(GL_TRUE);

			 	//fprintf(gbFile, "Platform : %d Demo : %d\n", iPlatform, iDemoCount);

			
			}
			else if(iPlatform == RRJ_GPU){


				if(sdkGetTimerValue(&gpDemoTimer) > gfDemoTime){

					iDemoCount++;
					
					if(iDemoCount > 5)
						iDemoCount = 0;

					iFlags = RRJ_SHELL;
					sdkResetTimer(&gpDemoTimer);

					gfCamTrans[0] = demos[iDemoCount].fX;
					gfCamTrans[1] = demos[iDemoCount].fY;
					gfCamTrans[2] = demos[iDemoCount].fZ;

					gfCamTrans_Lag[0] = demos[iDemoCount].fX;
					gfCamTrans_Lag[1] = demos[iDemoCount].fY;
					gfCamTrans_Lag[2] = demos[iDemoCount].fZ;
					
					//memset(points_Acc, 0, sizeof(GLfloat) * 4 * PARTICALES);
					ResetPointsData();

					glBindBuffer(GL_ARRAY_BUFFER, vbo_GPU_NewPos);
					glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * particals[giParticalsCount - 1].uiParticals * 4, gpGpuPos, GL_DYNAMIC_DRAW);
					glBindBuffer(GL_ARRAY_BUFFER, 0);

					glBindBuffer(GL_ARRAY_BUFFER, vbo_GPU_OldPos);
					glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * particals[giParticalsCount - 1].uiParticals * 4, gpGpuPos, GL_DYNAMIC_DRAW);
					glBindBuffer(GL_ARRAY_BUFFER, 0);

				}


				

				TranslateMatrix = mat4::identity();
				ModelViewMatrix = mat4::identity();
				rotateMatrix = mat4::identity();
				ModelViewProjectionMatrix = mat4::identity();

				gfCamTrans_Lag[0] = gfCamTrans_Lag[0] + (gfCamTrans[0] - gfCamTrans_Lag[0]) * gfInertia;
				gfCamTrans_Lag[1] = gfCamTrans_Lag[1] + (gfCamTrans[1] - gfCamTrans_Lag[1]) * gfInertia;
				gfCamTrans_Lag[2] = gfCamTrans_Lag[2] + (gfCamTrans[2] - gfCamTrans_Lag[2]) * gfInertia;

				gfCamRot_Lag[0] = gfCamRot_Lag[0] + (gfCamRot[0] - gfCamRot_Lag[0]) * gfInertia;
				gfCamRot_Lag[1] = gfCamRot_Lag[1] + (gfCamRot[1] - gfCamRot_Lag[1]) * gfInertia;
				gfCamRot_Lag[2] = gfCamRot_Lag[2] + (gfCamRot[2] - gfCamRot_Lag[2]) * gfInertia;

				TranslateMatrix = translate(gfCamTrans_Lag[0], gfCamTrans_Lag[1], gfCamTrans_Lag[2]);
				rotateMatrix = rotate(gfCamRot_Lag[0], 1.0f, 0.0f, 0.0f);
				rotateMatrix = rotateMatrix * rotate(gfCamRot_Lag[1], 0.0f, 1.0f, 0.0f);

				//ModelViewMatrix = rotateMatrix * TranslateMatrix;

				ModelViewMatrix = TranslateMatrix;

				//ModelViewProjectionMatrix = gPerspectiveProjectionMatrix * ModelViewMatrix;
				//glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, ModelViewProjectionMatrix);	

				glUniformMatrix4fv(modelViewMatrixUniform, 1, GL_FALSE, ModelViewMatrix);
				glUniformMatrix4fv(projectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

				glUniform1i(toggleUniform, 0);

				glEnable(GL_POINT_SPRITE_ARB);
				glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
				glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
				//glPointSize(gfPointSprint);
				glBlendFunc(GL_SRC_ALPHA, GL_ONE);
				glEnable(GL_BLEND);
				glDepthMask(GL_FALSE);


				glActiveTextureARB(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, texture_PointSprite);
				glUniform1i(samplerUniform, 0);



				static int iToggle = 1;

				if(iToggle == 1){

					//fprintf(gbFile, "Display: Toggle1\n");

					gCudaError = cudaGLMapBufferObject((void**)&pNewPos, vbo_GPU_NewPos);
					if(gCudaError != cudaSuccess){
						fprintf(gbFile, "ERROR: cudaGLMapBufferObject() Failed for pNewPos : %s\n", cudaGetErrorString(gCudaError));
						uninitialize();
						DestroyWindow(ghwnd);
					}

					//fprintf(gbFile, "Display: After Map Pos \n");


					gCudaError = cudaGLMapBufferObject((void**)&pOldPos, vbo_GPU_OldPos);
					if(gCudaError != cudaSuccess){
						fprintf(gbFile, "ERROR: cudaGLMapBufferObject() Failed for pOldPos : %s\n", cudaGetErrorString(gCudaError));
						uninitialize();
						DestroyWindow(ghwnd);
					}

					//fprintf(gbFile, "Display: After Map OldPos\n");



					AnimateNBody(
						pNewPos, pNewVel, 
						pOldPos, pOldVelo, 
						demos[iDemoCount].fSoftening, demos[iDemoCount].fDumping,
						demos[iDemoCount].fTimeStep,
						particals[giParticalsCount - 1].uiParticals);



					gCudaError = cudaGLUnmapBufferObject(vbo_GPU_NewPos);
					if(gCudaError != cudaSuccess){
						fprintf(gbFile, "ERROR: cudaGLUnmapBufferObject() Failed for pNewPos : %s\n", cudaGetErrorString(gCudaError));
						uninitialize();
						DestroyWindow(ghwnd);
					}

					gCudaError = cudaGLUnmapBufferObject(vbo_GPU_OldPos);
					if(gCudaError != cudaSuccess){
						fprintf(gbFile, "ERROR: cudaGLUnmapBufferObject() Failed for pOldPos : %s\n", cudaGetErrorString(gCudaError));
						uninitialize();
						DestroyWindow(ghwnd);
					}


					glBindVertexArray(vao_Points);
					glBindBuffer(GL_ARRAY_BUFFER, vbo_GPU_NewPos);
					glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
					glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
					glBindBuffer(GL_ARRAY_BUFFER, 0);

					glBindBuffer(GL_ARRAY_BUFFER, vbo_GPU_Color);
					glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * particals[giParticalsCount - 1].uiParticals * 4, gGpuColor, GL_DYNAMIC_DRAW);
					glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 4, GL_FLOAT, GL_FALSE, 0, NULL);
					glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
					glBindBuffer(GL_ARRAY_BUFFER, 0);


					glDrawArrays(GL_POINTS, 0, particals[giParticalsCount - 1].uiParticals);

					glBindVertexArray(0);
					glUseProgram(0);

					iToggle = 2;	
	
				}
				else if(iToggle == 2){

					//fprintf(gbFile, "Display: Toggle2\n");

					gCudaError = cudaGLMapBufferObject((void**)&pNewPos, vbo_GPU_OldPos);
					if(gCudaError != cudaSuccess){
						fprintf(gbFile, "ERROR: cudaGLMapBufferObject() Failed for pNewPos %s\n", cudaGetErrorString(gCudaError));
						uninitialize();
						DestroyWindow(ghwnd);
					}

					//fprintf(gbFile, "Display: After Pos\n");

					gCudaError = cudaGLMapBufferObject((void**)&pOldPos, vbo_GPU_NewPos);
					if(gCudaError != cudaSuccess){
						fprintf(gbFile, "ERROR: cudaGLMapBufferObject() Failed for pOldPos %s\n", cudaGetErrorString(gCudaError));
						uninitialize();
						DestroyWindow(ghwnd);
					}

					//fprintf(gbFile, "Display: After OldPos\n");

					AnimateNBody(
						pNewPos, pOldVelo, 
						pOldPos, pNewVel, 
						demos[iDemoCount].fSoftening, demos[iDemoCount].fDumping,
						demos[iDemoCount].fTimeStep,
						particals[giParticalsCount - 1].uiParticals);



					gCudaError = cudaGLUnmapBufferObject(vbo_GPU_NewPos);
					if(gCudaError != cudaSuccess){
						fprintf(gbFile, "ERROR: cudaGLUnmapBufferObject() Failed for pNewPos : %s\n", cudaGetErrorString(gCudaError));
						uninitialize();
						DestroyWindow(ghwnd);
					}

					gCudaError = cudaGLUnmapBufferObject(vbo_GPU_OldPos);
					if(gCudaError != cudaSuccess){
						fprintf(gbFile, "ERROR: cudaGLUnmapBufferObject() Failed for pOldPos : %s\n", cudaGetErrorString(gCudaError));
						uninitialize();
						DestroyWindow(ghwnd);
					}



					glBindVertexArray(vao_Points);
					
					glBindBuffer(GL_ARRAY_BUFFER, vbo_GPU_OldPos);
					glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
					glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
					glBindBuffer(GL_ARRAY_BUFFER, 0);

					glDrawArrays(GL_POINTS, 0, particals[giParticalsCount - 1].uiParticals);

					glBindVertexArray(0);
					glUseProgram(0);


					iToggle = 1;
					
				}

			 	glDisable(GL_POINT_SPRITE_ARB);
			 	glDisable(GL_BLEND);
			 	glDepthMask(GL_TRUE);	
			}

			break;

		case TECHNOLOGY_USED:

			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 

			TranslateMatrix = mat4::identity();
			ModelViewMatrix = mat4::identity();
			rotateMatrix = mat4::identity();
			ModelViewProjectionMatrix = mat4::identity();

			TranslateMatrix = translate(0.0f, 0.0f, -100.0f);
			ModelViewMatrix = TranslateMatrix * rotateMatrix;

			ModelViewProjectionMatrix = gPerspectiveProjectionMatrix * ModelViewMatrix;
			glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, ModelViewProjectionMatrix);	

			glUniform1i(toggleUniform, 1);

			RenderText("Reference ", -16.0f, 20.0f, 0.15f, vec3(1.0f, 1.0f, 1.0f));
			RenderText("GPU GEMS 3 : Chapter 31. Fast N-Body Simulation with CUDA", -68.0f, 10.0f, 0.10f, vec3(1.0f, 1.0f, 0.0f));
			
			RenderText("Music", -10.00f, -10.0f, 0.150f, vec3(1.0f, 1.0f, 1.0f));
			RenderText("Imagine Dragons : Believer Instrumental", -43.0f, -20.0f, 0.10f, vec3(1.0f, 1.0f, 0.0f));
			

			if(sdkGetTimerValue(&gpSceneTimer) > gfSceneTime){
				giScene = BLACKOUT4;
				gfSceneTime = 1000.0f;
				sdkResetTimer(&gpSceneTimer);
				sdkResetTimer(&gpDemoTimer);
			}

			glDisable(GL_BLEND);
			break;


		case BLACKOUT4:

			if(sdkGetTimerValue(&gpSceneTimer) > gfSceneTime){
				giScene = THANKYOU;
				gfSceneTime = 3000.0f;
				sdkResetTimer(&gpSceneTimer);
			}
			break;



		case THANKYOU:

			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 

			TranslateMatrix = mat4::identity();
			ModelViewMatrix = mat4::identity();
			rotateMatrix = mat4::identity();
			ModelViewProjectionMatrix = mat4::identity();

			TranslateMatrix = translate(0.0f, 0.0f, -100.0f);
			ModelViewMatrix = TranslateMatrix * rotateMatrix;

			ModelViewProjectionMatrix = gPerspectiveProjectionMatrix * ModelViewMatrix;
			glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, ModelViewProjectionMatrix);	

			glUniform1i(toggleUniform, 1);

			RenderText("Thank You ", -20.0f, 15.0f, 0.15f, vec3(1.0f, 1.0f, 1.0f));
			RenderText("Bharat for Brilliant Music Suggestion ", -40.0f, 0.0f, 0.1f, vec3(1.0f, 1.0f, 0.0f));
			RenderText("Prasann da for CUDA Kernel Barrier Help", -46.0f, -10.0f, 0.1f, vec3(1.0f, 1.0f, 0.0f));
			
				
			if(sdkGetTimerValue(&gpSceneTimer) > gfSceneTime){
				giScene = END;
				gfSceneTime = 1000.0f;
				sdkResetTimer(&gpSceneTimer);
				sdkResetTimer(&gpDemoTimer);
			}


			glDisable(GL_BLEND);

			break;

		case END:

			if(sdkGetTimerValue(&gpSceneTimer) > gfSceneTime){
				DestroyWindow(ghwnd);
			}

			break;
	}


	sdkStopTimer(&gpFpsCounter);
	float milisec =  0.0f;
	milisec = sdkGetAverageTimerValue(&gpFpsCounter);

	float fps = 1.0f / (milisec / 1000.0f);

	//fprintf(gbFile, "Platform : %d  Demo : %d  Particles : %d  Milisec : %f  FPS: %f   FPS: %0.1f\n", iPlatform, iDemoCount, giParticalsCount - 1, milisec, fps, fps);

	//memset(gszFps, 0, sizeof(gszFps));
	sprintf(gszFps, "Time in millisecond  : %0.6f\0", milisec);
	sdkResetTimer(&gpFpsCounter);

	SwapBuffers(ghdc);
}




inline float evalHermite(float pA, float pB, float vA, float vB, float u)
{
    float u2= u * u;
    float u3 = u * u * u;
    float B0 = 2 * u3 - 3 * u2 + 1;
    float B1 = -2 * u3 + 3 * u2;
    float B2 = u3 - 2 * u2 + u;
    float B3 = u3 - u;
    return( B0 * pA + B1 * pB + B2 * vA + B3 * vB );
}


unsigned char* createGaussianMap(int N)
{

	void uninitialize(void);

	float *M = NULL;
	M = (float*)malloc(sizeof(float) * 2 * N * N);
	if(M == NULL){
		fprintf(gbFile, "createGaussianMap: Malloc() failed M\n");
		uninitialize();
		DestroyWindow(ghwnd);
	}

	unsigned char *B = NULL;;
	B = (unsigned char*)malloc(sizeof(unsigned char) * 4 * N * N);
	if(B == NULL){
		fprintf(gbFile, "createGaussianMap: Malloc() failed B\n");
		uninitialize();
		DestroyWindow(ghwnd);
	}


	float X, Y, Y2, Dist;
	float Incr = 2.0f / N;
	int i = 0;  
	int j = 0;
	Y = -1.0f;
	

	for (int y = 0; y < N; y++, Y += Incr)
	{
		Y2 = Y * Y;
		X = -1.0f;
		for (int x = 0; x < N; x++, X += Incr, i += 2, j += 4)
		{
		    Dist = (float)sqrtf(X * X + Y2);
		    
		    if (Dist > 1) 
		    	Dist = 1;

		    M[i+1] = M[i] = evalHermite(1.0f, 0, 0, 0, Dist);
		    B[j+3] = B[j+2] = B[j+1] = B[j] = (unsigned char)(M[i] * 255);
		}
	}

	if(M){
		free(M);
		M = NULL;
	}

	return(B);
}    

void CreatePointSpriteTex(int resolution)
{
    unsigned char* data = createGaussianMap(resolution);
    
    glGenTextures(1, &texture_PointSprite);
    glBindTexture(GL_TEXTURE_2D, texture_PointSprite);
   
    glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, resolution, resolution, 0, 
                 GL_RGBA, GL_UNSIGNED_BYTE, data);
   
   glBindTexture(GL_TEXTURE_2D, 0);

    if(data){
    	free(data);
    	data = NULL;
    }
    
}






void RenderText(char *arr, float x, float y, float scale, vec3 color){


	int len = strlen(arr);

	glBindVertexArray(vao_Rect);

	for(int i = 0; i < len; i++){

		struct Characters c = TotalChar[arr[i]];

		float xpos = x + c.bearing[0] * scale;
		float ypos = y - (c.size[1] - c.bearing[1]) * scale;

		float w = c.size[0] * scale;
		float h = c.size[1] * scale;


		float vertices[6][4] = {
			{xpos, ypos + h, 0.0f, 0.0f},
			{xpos, ypos, 0.0f, 1.0f},
			{xpos + w, ypos, 1.0f, 1.0f},

			{xpos, ypos + h, 0.0f, 0.0f},
			{xpos + w, ypos, 1.0f, 1.0f},
			{xpos + w, ypos + h, 1.0f, 0.0f},
		};


		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, c.textureId);
		glUniform1f(samplerUniform, 0);

		glUniform3fv(fontColorUniform, 1, vec3(color[0], color[1], color[2]));


		glBindBuffer(GL_ARRAY_BUFFER, vbo_Rect_Position);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glDrawArrays(GL_TRIANGLE_FAN, 0, 6);

		x = x + (c.advance >> 6) * scale;
	}

	glBindVertexArray(0);

}






void ResetPointsData(void){

	switch(iFlags){

		case RRJ_RANDOM:
			{
				GLfloat fScale = demos[iDemoCount].fClusterScale * max(1.0f, particals[giParticalsCount - 1].uiParticals / 1024.0f);
				GLfloat fVeloScale = fScale * demos[iDemoCount].fVelocityScale;

				int p = 0, v = 0, a = 0, p1 = 0;
				int i = 0;
				int c = 0;
				while(i < particals[giParticalsCount - 1].uiParticals){

					vec3 pos;

					pos[0] = rand() / (float)RAND_MAX * 2 - 1;
					pos[1] = rand() / (float)RAND_MAX * 2 - 1;
					pos[2] = rand() / (float)RAND_MAX * 2 - 1;

					float lenSq = dot(pos, pos);
					//fprintf(gbFile, "LenSq : %f\n", lenSq);

					if(lenSq > 1)
						continue;

					vec4 velo;

					velo[0] = rand() / (float)RAND_MAX * 2 - 1;
					velo[1] = rand() / (float)RAND_MAX * 2 - 1;
					velo[2] = rand() / (float)RAND_MAX * 2 - 1;
					float veloSq = dot(velo, velo);

					//fprintf(gbFile, "veloSq : %f\n", veloSq);

					if(veloSq > 1)
						continue;



					gpGpuPos[p1++] = points_Pos[p++] = pos[0] * fScale;
					gpGpuPos[p1++] = points_Pos[p++] = pos[1] * fScale;
					gpGpuPos[p1++] = points_Pos[p++] = pos[2] * fScale;
					gpGpuPos[p1++] = points_Pos[p++] = 1.0f;

					//fprintf(gbFile, "posX : %f   posY: %f  posZ: %f\n", pos[0] * fScale, pos[1] * fScale, pos[2] * fScale);

					points_Velocity[v++] = velo[0] * fVeloScale;
					points_Velocity[v++] = velo[1] * fVeloScale;
					points_Velocity[v++] = velo[2] * fVeloScale;
					points_Velocity[v++] = 1.0f;

					i++;

				}
			}


			break;

		case RRJ_SHELL:

			{
				GLfloat scale = demos[iDemoCount].fClusterScale;
				GLfloat veloScale = scale * demos[iDemoCount].fVelocityScale;
				GLfloat inner = 2.5f * scale;
				GLfloat outer = 4.0f * scale;

				int p = 0, v = 0, p1 = 0;
				int i = 0;


				while(i < particals[giParticalsCount - 1].uiParticals){

					vec3 pos;

					pos[0] = rand() / (float)RAND_MAX * 2 - 1;
					pos[1] = rand() / (float)RAND_MAX * 2 - 1;
					pos[2] = rand() / (float)RAND_MAX * 2 - 1;

					float len = my_normalize(pos);

					//fprintf(gbFile, "Len : %f\n", len);

					if(len > 1)
						continue;

					gpGpuPos[p1++] = points_Pos[p++] = pos[0] * (inner + (outer - inner) * rand() / (float)RAND_MAX);
					gpGpuPos[p1++] = points_Pos[p++] = pos[1] * (inner + (outer - inner) * rand() / (float)RAND_MAX);
					gpGpuPos[p1++] = points_Pos[p++] = pos[2] * (inner + (outer - inner) * rand() / (float)RAND_MAX);
					gpGpuPos[p1++] = points_Pos[p++] = 1.0f;

					/*fprintf(gbFile, "posX : %f   posY: %f  posZ: %f\n", 
						pos[0] * (inner + (outer - inner) * rand() / (GLfloat)RAND_MAX),
						pos[1] * (inner + (outer - inner) * rand() / (GLfloat)RAND_MAX),
						pos[2] * (inner + (outer - inner) * rand() / (GLfloat)RAND_MAX));*/


					vec3 axis = vec3(0.0f, 0.0f, 1.0f);
					my_normalize(axis);

					if(1 - dot(pos, axis) < 1e-6){
						axis[0] = pos[1];
						axis[1] = pos[0];
						my_normalize(axis);
					}

					vec3 vv = vec3(points_Pos[4*i], points_Pos[4*i+1], points_Pos[4*i+2]);

					vv = cross(vv, axis);

					points_Velocity[v++] = vv[0] * veloScale;
					points_Velocity[v++] = vv[1] * veloScale;
					points_Velocity[v++] = vv[2] * veloScale;
					points_Velocity[v++] = 1.0f;

					i++;

				}

			}

			break;

		case RRJ_EXPAND:

			{
				GLfloat fScale = demos[iDemoCount].fClusterScale * max(1.0f, particals[giParticalsCount - 1].uiParticals / 1024.0f);

				if(fScale < 1.0f)
					fScale = demos[iDemoCount].fClusterScale;

				GLfloat fVeloScale = fScale * demos[iDemoCount].fVelocityScale;

				int p = 0, v = 0, p1 = 0;
				int i = 0;

				while(i < particals[giParticalsCount - 1].uiParticals){

					vec3 pos;

					pos[0] = rand() / (float)RAND_MAX * 2 - 1;
					pos[1] = rand() / (float)RAND_MAX * 2 - 1;
					pos[2] = rand() / (float)RAND_MAX * 2 - 1;

					float lenSq = dot(pos, pos);

					if(lenSq > 1)
						continue;


					gpGpuPos[p1++] = points_Pos[p++] = pos[0] * fScale;
					gpGpuPos[p1++] = points_Pos[p++] = pos[1] * fScale;
					gpGpuPos[p1++] = points_Pos[p++] = pos[2] * fScale;
					gpGpuPos[p1++] = points_Pos[p++] = 1.0f;

					points_Velocity[v++] = pos[0] * fVeloScale;
					points_Velocity[v++] = pos[1] * fVeloScale;
					points_Velocity[v++] = pos[2] * fVeloScale;
					points_Velocity[v++] = 1.0f;

					i++;

				}
			}

			break;

	}



	for(int i = 0; i < particals[giParticalsCount - 1].uiParticals; i++){

		points_Color[i + 0] = rand() / (float)RAND_MAX * 2 - 1;
		points_Color[i + 1] = rand() / (float)RAND_MAX * 2 - 1;
		points_Color[i + 2] = rand() / (float)RAND_MAX * 2 - 1;
		points_Color[i + 3] = 1.0f;

		gGpuColor[i + 0] = rand() / (float)RAND_MAX * 2 - 1;
		gGpuColor[i + 1] = rand() / (float)RAND_MAX * 2 - 1;
		gGpuColor[i + 2] = rand() / (float)RAND_MAX * 2 - 1;
		gGpuColor[i + 4] = 1.0f;



	}	

}


void update(void){

	void FindNewPositionVelocity(void);

	FindNewPositionVelocity();

}



void FindNewPositionVelocity(void){

	void CalculateAcc(void);

	CalculateAcc();

	for(int i = 0; i < particals[giParticalsCount - 1].uiParticals; i++){

		int index = i * 4;
		int indexForce = i * 3;

		// *** F = M * A ***
		points_Force[indexForce + 0] = points_Force[indexForce + 0] * points_Pos[index + 3];
		points_Force[indexForce + 1] = points_Force[indexForce + 1] * points_Pos[index + 3];
		points_Force[indexForce + 2] = points_Force[indexForce + 2] * points_Pos[index + 3];

		// *** New Velocity ***
		points_Velocity[index + 0] = points_Velocity[index + 0] + points_Force[indexForce + 0] * demos[iDemoCount].fTimeStep;
		points_Velocity[index + 1] = points_Velocity[index + 1] + points_Force[indexForce + 1] * demos[iDemoCount].fTimeStep;
		points_Velocity[index + 2] = points_Velocity[index + 2] + points_Force[indexForce + 2] * demos[iDemoCount].fTimeStep;

		points_Velocity[index + 0] *= demos[iDemoCount].fDumping;
		points_Velocity[index + 1] *= demos[iDemoCount].fDumping;
		points_Velocity[index + 2] *= demos[iDemoCount].fDumping;


		// *** New Position ***
		//fprintf(gbFile, "posOld %f %f %f\n", points_Pos[index + 0], points_Pos[index + 1], points_Pos[index + 2]);
		points_Pos[index + 0] = points_Pos[index + 0] + points_Velocity[index + 0] * demos[iDemoCount].fTimeStep;
		points_Pos[index + 1] = points_Pos[index + 1] + points_Velocity[index + 1] * demos[iDemoCount].fTimeStep;
		points_Pos[index + 2] = points_Pos[index + 2] + points_Velocity[index + 2] * demos[iDemoCount].fTimeStep;
		//fprintf(gbFile, "posNew %f %f %f\n", points_Pos[index + 0], points_Pos[index + 1], points_Pos[index + 2]);
	}

}


void CalculateAcc(void){

	vec3 CalculateAccWRT_AllParticals(vec3 acc, GLfloat p0[4], GLfloat p1[4], GLfloat fSoftSq);

	vec3 acc = vec3(0.0f);



	for(int i = 0; i < particals[giParticalsCount - 1].uiParticals; i++){

		int indexForce = 3 * i;
		int j = 0;

		points_Force[indexForce + 0] = 0.0f;
		points_Force[indexForce + 1] = 0.0f;
		points_Force[indexForce + 2] = 0.0f;

		while(j < particals[giParticalsCount - 1].uiParticals){

			acc = CalculateAccWRT_AllParticals(acc, &points_Pos[4*i], &points_Pos[4*j], demos[iDemoCount].fSoftening);
			j++;

			//fprintf(gbFile, "acc1 : %f %f %f\n", acc[0], acc[1], acc[2]);
			
			acc = CalculateAccWRT_AllParticals(acc, &points_Pos[4*i], &points_Pos[4*j], demos[iDemoCount].fSoftening);
			j++;
			
			//fprintf(gbFile, "acc2 : %f %f %f\n", acc[0], acc[1], acc[2]);

			acc = CalculateAccWRT_AllParticals(acc, &points_Pos[4*i], &points_Pos[4*j], demos[iDemoCount].fSoftening);
			j++;
			//fprintf(gbFile, "acc3 : %f %f %f\n", acc[0], acc[1], acc[2]);


			acc =CalculateAccWRT_AllParticals(acc, &points_Pos[4*i], &points_Pos[4*j], demos[iDemoCount].fSoftening);
			j++;

			//fprintf(gbFile, "acc4 : %f %f %f\n", acc[0], acc[1], acc[2]);
			
		}

		//fprintf(gbFile, "acc5 : %f %f %f\n", acc[0], acc[1], acc[2]);

		points_Force[indexForce + 0] = acc[0];
		points_Force[indexForce + 1] = acc[1];
		points_Force[indexForce + 2] = acc[2];

	}

}


vec3 CalculateAccWRT_AllParticals(vec3 acc, GLfloat p0[4], GLfloat p1[4], GLfloat fSoftSq){

	GLfloat fR[3];

	fR[0] = p1[0] - p0[0];
	fR[1] = p1[1] - p0[1];
	fR[2] = p1[2] - p0[2];

	GLfloat fDisSq = (fR[0] * fR[0] + fR[1] * fR[1] + fR[2] * fR[2]);
	fDisSq = fDisSq + (fSoftSq * fSoftSq);


	GLfloat fDistSixth = fDisSq * fDisSq * fDisSq;
	GLfloat invDistCube = 1.0f / sqrt(fDistSixth);

	//GLfloat fInvDis = 1.0f / (GLfloat)sqrt((double)fDisSq);
	//GLfloat fInvCube = fInvDis * fInvDis * fInvDis;

	GLfloat s = p1[3] * invDistCube;

	acc[0] = acc[0] + fR[0] * s;
	acc[1] = acc[1] + fR[1] * s;
	acc[2] = acc[2] + fR[2] * s;

	return(acc);

}




void ResetAllPointsData(int i){

	
	if(points_Pos){
		free(points_Pos);
		points_Pos = NULL;

		points_Pos = (GLfloat*)malloc(sizeof(GLfloat) * 4 * particals[i - 1].uiParticals);
		if(points_Pos == NULL){
			fprintf(gbFile, "ResetAllPointsData: points_Pos malloc \n");
			uninitialize();
			DestroyWindow(ghwnd);
		}
	}

	if(points_Force){
		free(points_Force);
		points_Force = NULL;

		points_Force = (GLfloat*)malloc(sizeof(GLfloat) * 3 * particals[i - 1].uiParticals);
		if(points_Force == NULL){
			fprintf(gbFile, "ResetAllPointsData: points_Force malloc \n");
			uninitialize();
			DestroyWindow(ghwnd);
		}
	}

	if(points_Velocity){
		free(points_Velocity);
		points_Velocity = NULL;

		points_Velocity = (GLfloat*)malloc(sizeof(GLfloat) * 4 * particals[i - 1].uiParticals);
		if(points_Velocity == NULL){
			fprintf(gbFile, "ResetAllPointsData: points_Velocity malloc \n");
			uninitialize();
			DestroyWindow(ghwnd);
		}
	}

	if(points_Color){
		free(points_Color);
		points_Color = NULL;

		points_Color = (GLfloat*)malloc(sizeof(GLfloat) * 4 * particals[i - 1].uiParticals);
		if(points_Color == NULL){
			fprintf(gbFile, "ResetAllPointsData: points_Color malloc \n");
			uninitialize();
			DestroyWindow(ghwnd);
		}
	}


	if(gpGpuPos){
		free(gpGpuPos);
		gpGpuPos = NULL;

		gpGpuPos = (GLfloat*)malloc(sizeof(GLfloat) * 4 * particals[i - 1].uiParticals);
		if(gpGpuPos == NULL){
			fprintf(gbFile, "ResetAllPointsData: gpGpuPos malloc \n");
			uninitialize();
			DestroyWindow(ghwnd);
		}
	}


	if(gGpuColor){
		free(gGpuColor);
		gGpuColor = NULL;

		gGpuColor = (GLfloat*)malloc(sizeof(GLfloat) * 4 * particals[i - 1].uiParticals);
		if(gGpuColor == NULL){
			fprintf(gbFile, "ResetAllPointsData: gGpuColor malloc \n");
			uninitialize();
			DestroyWindow(ghwnd);
		}
	}


	ResetPointsData();


	gCudaError = cudaGLUnregisterBufferObject(vbo_GPU_NewPos);
	if(gCudaError != cudaSuccess){
		fprintf(gbFile, "ResetAllPointsData: cudaMalloc() Failed for pNewPos : %s\n", cudaGetErrorString(gCudaError));
		uninitialize();
		DestroyWindow(ghwnd);
	}

	gCudaError = cudaGLUnregisterBufferObject(vbo_GPU_OldPos);
	if(gCudaError != cudaSuccess){
		fprintf(gbFile, "ResetAllPointsData: cudaMalloc() Failed for pOldPos : %s\n", cudaGetErrorString(gCudaError));
		uninitialize();
		DestroyWindow(ghwnd);
	}


	//Pos
	if(vbo_Points_Position){

		glDeleteBuffers(1, &vbo_Points_Position);
		vbo_Points_Position = 0;

		glGenBuffers(1, &vbo_Points_Position);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Points_Position);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * particals[giParticalsCount - 1].uiParticals * 4, points_Pos, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}


	//Color
	if(vbo_Points_Color){

		glDeleteBuffers(1, &vbo_Points_Color);
		vbo_Points_Color = 0;

		glGenBuffers(1, &vbo_Points_Color);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Points_Color);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * particals[giParticalsCount - 1].uiParticals * 4, points_Color, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}


	// *** GPU New Pos ***
	if(vbo_GPU_NewPos){

		glDeleteBuffers(1, &vbo_GPU_NewPos);
		vbo_GPU_NewPos = 0;

		glGenBuffers(1, &vbo_GPU_NewPos);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_GPU_NewPos);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * particals[giParticalsCount - 1].uiParticals * 4, gpGpuPos, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}


	// *** GPU Old Pos ***
	if(vbo_GPU_OldPos){

		glDeleteBuffers(1, &vbo_GPU_OldPos);
		vbo_GPU_OldPos = 0;

		glGenBuffers(1, &vbo_GPU_OldPos);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_GPU_OldPos);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * particals[giParticalsCount - 1].uiParticals * 4, gpGpuPos, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_OLD_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_OLD_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	// *** GPU Color ***
	if(vbo_GPU_Color){

		glDeleteBuffers(1, &vbo_GPU_Color);
		vbo_GPU_Color = 0;

		glGenBuffers(1, &vbo_GPU_Color);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_GPU_Color);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * particals[giParticalsCount - 1].uiParticals * 4, gGpuColor, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_COLOR, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}


	if(pNewVel){
		cudaFree(pNewVel);
		pNewVel = NULL;

		gCudaError = cudaMalloc((void**)&pNewVel, sizeof(GLfloat) * 4 * particals[giParticalsCount - 1].uiParticals);
		if(gCudaError != cudaSuccess){
			fprintf(gbFile, "ResetAllPointsData: cudaMalloc() Failed for pNewVel : %s\n", cudaGetErrorString(gCudaError));
			uninitialize();
			DestroyWindow(ghwnd);
		}
	}

	if(pOldVelo){ 

		cudaFree(pOldVelo);
		pOldVelo = NULL;

		gCudaError = cudaMalloc((void**)&pOldVelo, sizeof(GLfloat) * 4 * particals[giParticalsCount - 1].uiParticals);
		if(gCudaError != cudaSuccess){
			fprintf(gbFile, "ResetAllPointsData: cudaMalloc() Failed for pOldVelo : %s\n", cudaGetErrorString(gCudaError));
			uninitialize();
			DestroyWindow(ghwnd);
		}

	}


	// *** Register VBO's ***
	gCudaError = cudaGLRegisterBufferObject(vbo_GPU_NewPos);
	if(gCudaError != cudaSuccess){
		fprintf(gbFile, "ResetAllPointsData: cudaGLRegisterBufferObject() Failed for pNewPos : %s\n", cudaGetErrorString(gCudaError));
		uninitialize();
		DestroyWindow(ghwnd);
	}


	gCudaError = cudaGLRegisterBufferObject(vbo_GPU_OldPos);
	if(gCudaError != cudaSuccess){
		fprintf(gbFile, "ResetAllPointsData: cudaGLRegisterBufferObject() Failed for pOldPos : %s\n", cudaGetErrorString(gCudaError));
		uninitialize();
		DestroyWindow(ghwnd);
	}

	fprintf(gbFile, "ResetAllPointsData: Leave\n");


}
#include<Windows.h>
#include<stdio.h>
#include<gl/glew.h>
#include<GL/GL.h>
#include<GL/GLU.h>
#include"vmath.h"
#include"Sphere.h"

#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glu32.lib")
#pragma comment(lib, "Sphere.lib")


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


//For Perspective Matric
mat4 gPerspectiveProjectionMatrix_RRJ;

const float PI = 3.1415926535;


// ***** For Shader Count *****
#define NO_OF_SHADERS 2
#define SHADER_CHOICE 0

GLuint gShaderProgramObject[NO_OF_SHADERS];
GLuint gVertexShaderObject_RRJ[NO_OF_SHADERS];
GLuint gFragmentShaderObject_RRJ[NO_OF_SHADERS];

// ***** Functions *****
void Init_SkyFromAtmosphere(GLuint&, GLuint&, GLuint&);
void Init_GroundFromAtmosphere(GLuint&, GLuint&, GLuint&);

void SetUniformsForShaderProgramObject(GLint);


// ***** Function Pointer and its Array *****
typedef void (*gpfnInitFunctions)(GLuint&, GLuint&, GLuint&);

gpfnInitFunctions gpfnInitFunctionList_RRJ[] = {
	Init_SkyFromAtmosphere
};


GLuint giCameraPositonUniform_RRJ;
GLuint giLightPositionUniform_RRJ;

GLuint giInverseWavelengthUniform_RRJ;

GLuint giCameraHeightUniform_RRJ;
GLuint giCameraHeightSquareUniform_RRJ;

GLuint giOuterRadiusUniform_RRJ;
GLuint giOuterRadiusSquareUniform_RRJ;

GLuint giInnerRadiusUniform_RRJ;
GLuint giInnerRadiusSquareUniform_RRJ;

// ***** R Constant and Mie Constant *****
GLuint giKrESunUniform_RRJ;
GLuint giKmESunUniform_RRJ;

GLuint giKr4PiUniform_RRJ;
GLuint giKm4PiUniform_RRJ;

GLuint giFScaleUniform_RRJ;
GLuint giFScaleDepthUniform_RRJ;
GLuint giFScaleOverScaleDepthUniform_RRJ;

GLuint giGUniform_RRJ;
GLuint giG2Uniform_RRJ;

// ***** For Matrix *****
GLuint giMVPUniform_RRJ;



// ***** For Uniforms Values *****
GLfloat gfCamPos_RRJ[] = {0.0f, 10.00f, 0.0f};
GLfloat gfCamView_RRJ[] = {0.0f, 10.00f, -1.0f};
GLfloat gfCamY_RRJ[] = {0.0f, 1.0f, 0.0f};


GLfloat gfLightPos_RRJ[] = {0.0f, -0.60f, -1.0f};
GLfloat gfInverseWavelength_RRJ[] = {0.650f, 0.570f, 0.475f};
GLfloat gfInverseWavelength4_RRJ[3];

GLfloat gfCameraHeight_RRJ = 0.0f;
GLfloat gfCameraHeight2_RRJ = gfCameraHeight_RRJ * gfCameraHeight_RRJ;

GLfloat gfOuterRadius_RRJ = 10.25f;
GLfloat gfOuterRadius2_RRJ = gfOuterRadius_RRJ * gfOuterRadius_RRJ;

GLfloat gfInnerRadius_RRJ = 10.0f;
GLfloat gfInnerRadius2_RRJ = gfInnerRadius_RRJ * gfInnerRadius_RRJ;

GLfloat gfKrESun_RRJ = 0.0025f * 20.0f;
GLfloat gfKmESun_RRJ = 0.0010f * 20.0f;

GLfloat gfKr4Pi_RRJ = 0.0025f * 4.0f * PI;
GLfloat gfKm4Pi_RRJ = 0.0010f * 4.0f * PI;

GLfloat gfScale_RRJ = 1.0f / (gfOuterRadius_RRJ - gfInnerRadius_RRJ);
GLfloat gfScaleDepth_RRJ = 0.25f;
GLfloat gfScaleOverScaleDepth_RRJ = gfScale_RRJ / gfScaleDepth_RRJ;

GLfloat gfG_RRJ = -0.990f;
GLfloat gfG2_RRJ = gfG_RRJ * gfG_RRJ;


//For Sphere

#define STACK 100
#define SLICES 100

// *** Inner ***
GLuint vao_InnerSphere_RRJ;
GLuint vbo_InnerSphere_Position_RRJ;
GLuint vbo_InnerSphere_Normal_RRJ;
GLuint vbo_InnerSphere_Element_RRJ;

GLfloat innerSphere_Position[STACK * SLICES * 3];
GLfloat innerSphere_Normal[STACK * SLICES * 3];
GLushort innerSphere_Element[STACK * SLICES * 6];

// *** Outer ***
GLuint vao_OuterSphere_RRJ;
GLuint vbo_OuterSphere_Position_RRJ;
GLuint vbo_OuterSphere_Normal_RRJ;
GLuint vbo_OuterSphere_Element_RRJ;

GLfloat OuterSphere_Position[STACK * SLICES * 3];
GLfloat OuterSphere_Normal[STACK * SLICES * 3];
GLushort OuterSphere_Element[STACK * SLICES * 6];




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
	TCHAR szName_RRJ[] = TEXT("RohitRJadhav-PP-Atmospheric Scattering");

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
		TEXT("RohitRJadhav-PP-Atmospheric Scattering"),
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
		fprintf(gbFile_RRJ, "wglCreateContext() Failed!!\n");
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

		case 'W':
		case 'w':
			gfLightPos_RRJ[1] = gfLightPos_RRJ[1] + 0.01f;
			fprintf(gbFile_RRJ, "gfLightPos_RRJ: %f\n", gfLightPos_RRJ[1]);
			break;

		case 'S':
		case 's':
			gfLightPos_RRJ[1] = gfLightPos_RRJ[1] - 0.01f;
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
	void  myMakeSphere(float, int, int, GLfloat[], GLfloat[], GLushort[]);

	PIXELFORMATDESCRIPTOR pfd_RRJ;
	int iPixelFormatIndex_RRJ;
	GLenum Result_RRJ;

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


	gfInverseWavelength4_RRJ[0] = pow(gfInverseWavelength_RRJ[0], 4.0f);
	gfInverseWavelength4_RRJ[1] = pow(gfInverseWavelength_RRJ[1], 4.0f);
	gfInverseWavelength4_RRJ[2] = pow(gfInverseWavelength_RRJ[2], 4.0f);


 	Init_SkyFromAtmosphere(gShaderProgramObject[0], gVertexShaderObject_RRJ[0], gFragmentShaderObject_RRJ[0]);
 	Init_GroundFromAtmosphere(gShaderProgramObject[1], gVertexShaderObject_RRJ[1], gFragmentShaderObject_RRJ[1]);


	/********** Inner Position, Normal and Elements **********/
	myMakeSphere(gfInnerRadius_RRJ, STACK, SLICES, innerSphere_Position, innerSphere_Normal, innerSphere_Element);



	/********** Sphere Vao **********/
	glGenVertexArrays(1, &vao_InnerSphere_RRJ);
	glBindVertexArray(vao_InnerSphere_RRJ);

	/********** Position **********/
	glGenBuffers(1, &vbo_InnerSphere_Position_RRJ);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_InnerSphere_Position_RRJ);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(innerSphere_Position),
		innerSphere_Position,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/********** Normals **********/
	glGenBuffers(1, &vbo_InnerSphere_Normal_RRJ);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_InnerSphere_Normal_RRJ);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(innerSphere_Normal),
		innerSphere_Normal,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	/********** Element Vbo **********/
	glGenBuffers(1, &vbo_InnerSphere_Element_RRJ);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_InnerSphere_Element_RRJ);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(innerSphere_Element), innerSphere_Element, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);


	glBindVertexArray(0);



	/********** Outer Sphere Position, Normal and Elements **********/
	myMakeSphere(gfOuterRadius_RRJ, STACK, SLICES, OuterSphere_Position, OuterSphere_Normal, OuterSphere_Element);



	/********** Sphere Vao **********/
	glGenVertexArrays(1, &vao_OuterSphere_RRJ);
	glBindVertexArray(vao_OuterSphere_RRJ);

	/********** Position **********/
	glGenBuffers(1, &vbo_OuterSphere_Position_RRJ);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_OuterSphere_Position_RRJ);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(OuterSphere_Position),
		OuterSphere_Position,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/********** Normals **********/
	glGenBuffers(1, &vbo_OuterSphere_Normal_RRJ);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_OuterSphere_Normal_RRJ);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(OuterSphere_Normal),
		OuterSphere_Normal,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	/********** Element Vbo **********/
	glGenBuffers(1, &vbo_OuterSphere_Element_RRJ);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_OuterSphere_Element_RRJ);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(OuterSphere_Element), OuterSphere_Element, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);


	glBindVertexArray(0);



	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);

	//glEnable(GL_CULL_FACE);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	gPerspectiveProjectionMatrix_RRJ = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);
	return(0);
}

void uninitialize(void) {

	if (vbo_OuterSphere_Element_RRJ) {
		glDeleteBuffers(1, &vbo_OuterSphere_Element_RRJ);
		vbo_OuterSphere_Element_RRJ = 0;
	}

	if (vbo_OuterSphere_Normal_RRJ) {
		glDeleteBuffers(1, &vbo_OuterSphere_Normal_RRJ);
		vbo_OuterSphere_Normal_RRJ = 0;
	}

	if (vbo_OuterSphere_Position_RRJ) {
		glDeleteBuffers(1, &vbo_OuterSphere_Position_RRJ);
		vbo_OuterSphere_Position_RRJ = 0;
	}

	if (vao_OuterSphere_RRJ) {
		glDeleteVertexArrays(1, &vao_OuterSphere_RRJ);
		vao_OuterSphere_RRJ = 0;
	}


	if (vbo_InnerSphere_Element_RRJ) {
		glDeleteBuffers(1, &vbo_InnerSphere_Element_RRJ);
		vbo_InnerSphere_Element_RRJ = 0;
	}

	if (vbo_InnerSphere_Normal_RRJ) {
		glDeleteBuffers(1, &vbo_InnerSphere_Normal_RRJ);
		vbo_InnerSphere_Normal_RRJ = 0;
	}

	if (vbo_InnerSphere_Position_RRJ) {
		glDeleteBuffers(1, &vbo_InnerSphere_Position_RRJ);
		vbo_InnerSphere_Position_RRJ = 0;
	}

	if (vao_InnerSphere_RRJ) {
		glDeleteVertexArrays(1, &vao_InnerSphere_RRJ);
		vao_InnerSphere_RRJ = 0;
	}

	for(int i = 0; i < NO_OF_SHADERS; i++){
		GLuint iShaderProgramObject = gShaderProgramObject[i];	

		GLsizei ShaderCount;
		GLsizei ShaderNumber;

		if (iShaderProgramObject) {
			glUseProgram(iShaderProgramObject);

			glGetProgramiv(iShaderProgramObject, GL_ATTACHED_SHADERS, &ShaderCount);

			GLuint *pShader = (GLuint*)malloc(sizeof(GLuint) * ShaderCount);
			if (pShader) {
				glGetAttachedShaders(iShaderProgramObject, ShaderCount, &ShaderCount, pShader);
				for (ShaderNumber = 0; ShaderNumber < ShaderCount; ShaderNumber++) {
					glDetachShader(iShaderProgramObject, pShader[ShaderNumber]);
					glDeleteShader(pShader[ShaderNumber]);
					pShader[ShaderNumber] = 0;
					fprintf(gbFile_RRJ, "Shader Deleted\n");
				}
				free(pShader);
				pShader = NULL;
			}
			glDeleteProgram(iShaderProgramObject);
			iShaderProgramObject = 0;
			glUseProgram(0);
		}
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

	gPerspectiveProjectionMatrix_RRJ = mat4::identity();
	gPerspectiveProjectionMatrix_RRJ = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.001f, 100.0f);
}


void display(void) {

	mat4 translateMatrix_RRJ;
	mat4 rotateMatrix_RRJ;
	mat4 modelMatrix_RRJ;
	mat4 viewMatrix_RRJ;
	mat4 modelViewProjectionMatrix_RRJ;

	static GLfloat angle_Sphere_RRJ = 0.0f;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//SetUniformsForShaderProgramObject(gShaderProgramObject[1]);	
	//glUseProgram(gShaderProgramObject[1]);


	///********** Inner Sphere **********/
	//translateMatrix_RRJ = mat4::identity();
	//rotateMatrix_RRJ = mat4::identity();
	//modelMatrix_RRJ = mat4::identity();
	//viewMatrix_RRJ = mat4::identity();
	//modelViewProjectionMatrix_RRJ = mat4::identity();

	////modelMatrix_RRJ = translate(0.0f, 0.0f, 0.0f);
	//viewMatrix_RRJ = lookat(
	// 	vec3(gfCamPos_RRJ[0], gfCamPos_RRJ[1], gfCamPos_RRJ[2]), 
	// 	vec3(gfCamView_RRJ[0], gfCamView_RRJ[1], gfCamView_RRJ[2]), 
	// 	vec3(gfCamY_RRJ[0], gfCamY_RRJ[1], gfCamY_RRJ[2]));
	//
	//modelViewProjectionMatrix_RRJ = gPerspectiveProjectionMatrix_RRJ * viewMatrix_RRJ * modelMatrix_RRJ;

	//glUniformMatrix4fv(giMVPUniform_RRJ, 1, GL_FALSE, modelViewProjectionMatrix_RRJ);

	//glUniform3fv(giCameraPositonUniform_RRJ, 1, gfCamPos_RRJ);
	//glUniform3fv(giLightPositionUniform_RRJ, 1, gfLightPos_RRJ);

	//glUniform3f(giInverseWavelengthUniform_RRJ, 
	//	1.0f / gfInverseWavelength4_RRJ[0], 
	//	1.0f / gfInverseWavelength4_RRJ[1], 
	//	1.0f / gfInverseWavelength4_RRJ[2]);

	//gfCameraHeight_RRJ = sqrt(gfCamPos_RRJ[0] * gfCamPos_RRJ[0] + gfCamPos_RRJ[1] * gfCamPos_RRJ[1] + gfCamPos_RRJ[2] * gfCamPos_RRJ[2]);
	//gfCameraHeight2_RRJ = gfCameraHeight_RRJ * gfCameraHeight_RRJ;

	//glUniform1f(giCameraHeightUniform_RRJ, gfCameraHeight_RRJ);
	//glUniform1f(giCameraHeightSquareUniform_RRJ, gfCameraHeight2_RRJ);

	//glUniform1f(giOuterRadiusUniform_RRJ, gfOuterRadius_RRJ);
	//glUniform1f(giOuterRadiusSquareUniform_RRJ, gfOuterRadius2_RRJ);

	//glUniform1f(giInnerRadiusUniform_RRJ, gfInnerRadius_RRJ);
	//glUniform1f(giInnerRadiusSquareUniform_RRJ, gfInnerRadius2_RRJ);

	//glUniform1f(giKrESunUniform_RRJ, gfKrESun_RRJ);
	//glUniform1f(giKmESunUniform_RRJ, gfKmESun_RRJ);

	//glUniform1f(giKr4PiUniform_RRJ, gfKr4Pi_RRJ);
	//glUniform1f(giKm4PiUniform_RRJ, gfKm4Pi_RRJ);

	//glUniform1f(giFScaleUniform_RRJ, gfScale_RRJ);
	//glUniform1f(giFScaleDepthUniform_RRJ, gfScaleDepth_RRJ);
	//glUniform1f(giFScaleOverScaleDepthUniform_RRJ, gfScaleOverScaleDepth_RRJ);

	//glUniform1f(giGUniform_RRJ, gfG_RRJ);
	//glUniform1f(giG2Uniform_RRJ, gfG2_RRJ);

	////glUniform1i(glGetUniformLocation(gShaderProgramObject[0], "u_choice"), 1);

	//	glBindVertexArray(vao_InnerSphere_RRJ);

	//	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_InnerSphere_Element_RRJ);
	//	glDrawElements(GL_TRIANGLES, STACK * SLICES * 6, GL_UNSIGNED_SHORT, 0);
	//	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	//	glBindVertexArray(0);

	//glUseProgram(0);




	


	/********** Outer Sphere **********/
	SetUniformsForShaderProgramObject(gShaderProgramObject[0]);	
	glUseProgram(gShaderProgramObject[0]);


	//// ***** For Sun *****
	//translateMatrix_RRJ = mat4::identity();
	//rotateMatrix_RRJ = mat4::identity();
	//modelMatrix_RRJ = mat4::identity();
	//viewMatrix_RRJ = mat4::identity();
	//modelViewProjectionMatrix_RRJ = mat4::identity();

	//static GLfloat fSunY = -1.95f;
	////modelMatrix_RRJ = translate(0.0f, gfLightPos_RRJ[1] * 1000.0f, 0.0f);
	//modelMatrix_RRJ = translate(0.0f, fSunY, -10.0f) * scale(0.01f, 0.01f, 0.01f);
	//// viewMatrix_RRJ = lookat(
	////  	vec3(gfCamPos_RRJ[0], gfCamPos_RRJ[1], gfCamPos_RRJ[2]), 
	////  	vec3(gfCamView_RRJ[0], gfCamView_RRJ[1], gfCamView_RRJ[2]), 
	////  	vec3(gfCamY_RRJ[0], gfCamY_RRJ[1], gfCamY_RRJ[2]));

	//fSunY = fSunY + 0.001f;
	//if(fSunY > -0.30f)
	//	fSunY = -0.30f;
	//modelViewProjectionMatrix_RRJ = gPerspectiveProjectionMatrix_RRJ * viewMatrix_RRJ * modelMatrix_RRJ;

	//glUniformMatrix4fv(giMVPUniform_RRJ, 1, GL_FALSE, modelViewProjectionMatrix_RRJ);
	//glUniform1i(glGetUniformLocation(gShaderProgramObject[0], "u_choice"), 1);


	//glBindVertexArray(vao_OuterSphere_RRJ);
	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_OuterSphere_Element_RRJ);
	//glDrawElements(GL_TRIANGLES, STACK * SLICES * 6, GL_UNSIGNED_SHORT, 0);
	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	//glBindVertexArray(0);


	// ***** For Outer Sphere *****
	translateMatrix_RRJ = mat4::identity();
	rotateMatrix_RRJ = mat4::identity();
	modelMatrix_RRJ = mat4::identity();
	viewMatrix_RRJ = mat4::identity();
	modelViewProjectionMatrix_RRJ = mat4::identity();

	modelMatrix_RRJ = translate(0.0f, 0.0f, 0.0f) * rotate(0.0f, angle_Sphere_RRJ, 0.0f);
	viewMatrix_RRJ = lookat(
	 	vec3(gfCamPos_RRJ[0], gfCamPos_RRJ[1], gfCamPos_RRJ[2]), 
	 	vec3(gfCamView_RRJ[0], gfCamView_RRJ[1], gfCamView_RRJ[2]), 
	 	vec3(gfCamY_RRJ[0], gfCamY_RRJ[1], gfCamY_RRJ[2]));
	
	modelViewProjectionMatrix_RRJ = gPerspectiveProjectionMatrix_RRJ * viewMatrix_RRJ * modelMatrix_RRJ;

	glUniformMatrix4fv(giMVPUniform_RRJ, 1, GL_FALSE, modelViewProjectionMatrix_RRJ);

	glUniform3fv(giCameraPositonUniform_RRJ, 1, gfCamPos_RRJ);
	glUniform3fv(giLightPositionUniform_RRJ, 1, gfLightPos_RRJ);

	glUniform3f(giInverseWavelengthUniform_RRJ, 
		1.0f / gfInverseWavelength4_RRJ[0], 
		1.0f / gfInverseWavelength4_RRJ[1], 
		1.0f / gfInverseWavelength4_RRJ[2]);

	gfCameraHeight_RRJ = sqrt(gfCamPos_RRJ[0] * gfCamPos_RRJ[0] + gfCamPos_RRJ[1] * gfCamPos_RRJ[1] + gfCamPos_RRJ[2] * gfCamPos_RRJ[2]);
	gfCameraHeight2_RRJ = gfCameraHeight_RRJ * gfCameraHeight_RRJ;

	glUniform1f(giCameraHeightUniform_RRJ, gfCameraHeight_RRJ);
	glUniform1f(giCameraHeightSquareUniform_RRJ, gfCameraHeight2_RRJ);

	glUniform1f(giOuterRadiusUniform_RRJ, gfOuterRadius_RRJ);
	glUniform1f(giOuterRadiusSquareUniform_RRJ, gfOuterRadius2_RRJ);

	glUniform1f(giInnerRadiusUniform_RRJ, gfInnerRadius_RRJ);
	glUniform1f(giInnerRadiusSquareUniform_RRJ, gfInnerRadius2_RRJ);

	glUniform1f(giKrESunUniform_RRJ, gfKrESun_RRJ);
	glUniform1f(giKmESunUniform_RRJ, gfKmESun_RRJ);

	glUniform1f(giKr4PiUniform_RRJ, gfKr4Pi_RRJ);
	glUniform1f(giKm4PiUniform_RRJ, gfKm4Pi_RRJ);

	glUniform1f(giFScaleUniform_RRJ, gfScale_RRJ);
	glUniform1f(giFScaleDepthUniform_RRJ, gfScaleDepth_RRJ);
	glUniform1f(giFScaleOverScaleDepthUniform_RRJ, gfScaleOverScaleDepth_RRJ);

	glUniform1f(giGUniform_RRJ, gfG_RRJ);
	glUniform1f(giG2Uniform_RRJ, gfG2_RRJ);

	
	// glFrontFace(GL_CW);
	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);
	
	glUniform1i(glGetUniformLocation(gShaderProgramObject[0], "u_choice"), 0);

	glBindVertexArray(vao_OuterSphere_RRJ);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_OuterSphere_Element_RRJ);
	glDrawElements(GL_TRIANGLES, STACK * SLICES * 6, GL_UNSIGNED_SHORT, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	glDisable(GL_BLEND);
	// glFrontFace(GL_CCW);

	glUseProgram(0);


	 angle_Sphere_RRJ = angle_Sphere_RRJ + 1.0f;
	 if(angle_Sphere_RRJ > 360.0f)
	 	angle_Sphere_RRJ = 0.0f;


	// gfLightPos_RRJ[1] = gfLightPos_RRJ[1] + 0.00035f;
	// if(gfLightPos_RRJ[1] > -0.110f)
	// 	gfLightPos_RRJ[1] = -0.110f;

	SwapBuffers(ghdc_RRJ);
}




void myMakeSphere(float fRadius, int  iStack, int iSlices, GLfloat sphere_Position[], GLfloat sphere_Normal[], GLushort sphere_Index[]){

	float longitude;
	float latitude;
	float factorLat = (2.0 * PI) / (iStack);
	float factorLon = PI / (iSlices-1);

	for(int i = 0; i < iStack; i++){
		
		latitude = -PI + i * factorLat;


		for(int j = 0; j < iSlices; j++){

			longitude = (PI) - j * factorLon;

			//console.log(i + "/" + j + ": " + latitude + "/" + longitude);

			float x = fRadius * sin(longitude) * cos((latitude));
			float y = fRadius * sin(longitude) * sin((latitude));
			float z = fRadius * cos((longitude));

			sphere_Position[(i * iSlices * 3)+ (j * 3) + 0] = x;
			sphere_Position[(i * iSlices * 3)+ (j * 3) + 1] = y;
			sphere_Position[(i * iSlices * 3)+ (j * 3) + 2] = z;

			//zconsole.log(i + "/" + j + "   " + x + "/" + y + "/" + z);

			
			sphere_Normal[(i * iSlices * 3)+ (j * 3) + 0] = x;
			sphere_Normal[(i * iSlices * 3)+ (j * 3) + 1] = y;
			sphere_Normal[(i * iSlices * 3)+ (j * 3) + 2] = z;

		}
	}


	int index = 0;
 	for(int i = 0; i < iStack ; i++){
 		for(int j = 0; j < iSlices ; j++){


 			if(i == (iStack - 1)){

 				unsigned short topLeft = (i * iSlices) + j;
	 			unsigned short bottomLeft = ((0) * iSlices) +(j);
	 			unsigned short topRight = topLeft + 1;
	 			unsigned short bottomRight = bottomLeft + 1;


	 			sphere_Index[index] = topLeft;
	 			sphere_Index[index + 1] = bottomLeft;
	 			sphere_Index[index + 2] = topRight;

	 			sphere_Index[index + 3] = topRight;
	 			sphere_Index[index + 4] = bottomLeft;
	 			sphere_Index[index + 5] = bottomRight;

 			}
 			else{

	 			unsigned short topLeft = (i * iSlices) + j;
	 			unsigned short bottomLeft = ((i + 1) * iSlices) +(j);
	 			unsigned short topRight = topLeft + 1;
	 			unsigned short bottomRight = bottomLeft + 1;


	 			sphere_Index[index] = topLeft;
	 			sphere_Index[index + 1] = bottomLeft;
	 			sphere_Index[index + 2] = topRight;

	 			sphere_Index[index + 3] = topRight;
	 			sphere_Index[index + 4] = bottomLeft;
	 			sphere_Index[index + 5] = bottomRight;
 			}

 			

 			index = index + 6;


 		}
 		

 	}
}



void Init_SkyFromAtmosphere(GLuint &iShaderProgramObject, GLuint &iVertexShader, GLuint &iFragmentShader){


	/********** Vertex Shader **********/
	iVertexShader = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *szVertexShaderSourceCode_RRJ =
		"#version 450 core" \
		"\n" \

		"in vec4 vPosition;" \
		"uniform mat4 u_mvp_matrix;" \
		"uniform int u_choice;" \

		"uniform vec3 u_v3CameraPosition;" \
		"uniform vec3 u_v3LightPosition;" \
		"uniform vec3 u_v3InverseWavelength;" \

		"uniform float u_fCameraHeight;" \
		"uniform float u_fCameraHeight2;" \
		"uniform float u_fOuterRadius;" \
		"uniform float u_fOuterRadius2;" \
		"uniform float u_fInnerRadius;" \
		"uniform float u_fInnerRadius2;" \

		"uniform float u_fKrESun;" \
		"uniform float u_fKmESun;" \
		"uniform float u_fKr4PI;" \
		"uniform float u_fKm4PI;" \

		"uniform float u_fScale;" \
		"uniform float u_fScaleDepth;" \
		"uniform float u_fScaleOverScaleDepth;" \

		"const int nSamples = 2;" \
		"const float fSamples = 2.0;" \

		"out vec3 v3Direction_VS;" \
		"out vec3 v3FrontSecColor_VS;" \
		"out vec3 v3FrontColor_VS;" \

		"float scale(float fCos){" \
			
			"float x = 1.0f - fCos;" \
			"return (u_fScaleDepth * exp(-0.00287 + x * (0.459 + x *(3.83 + x * (-6.80 + x * 5.25)))));" \

		"}" \

		"void main(void){" \

			"if(u_choice == 0) { " \

				// Get the ray from the camera to the vertex, and its length (which is the far point of the ray passing through the atmosphere)
				"vec3 v3Pos = vPosition.xyz;" \
				"vec3 v3Ray = v3Pos - u_v3CameraPosition;" \
				"float fFar = length(v3Ray);" \
				"v3Ray = v3Ray / fFar;" \



				// Calculate the ray's starting position, then calculate its scattering offset
				"vec3 v3Start = u_v3CameraPosition;" \
				"float fHeight = length(v3Start);" \
				"float fDepth = exp(u_fScaleOverScaleDepth * (u_fInnerRadius - u_fCameraHeight));" \
				"float fStartAngle = dot(v3Ray, v3Start) / fHeight;" \
				"float fStartOffset = fDepth * scale(fStartAngle);" \


				// Initialize the scattering loop variables
				"float fSampleLength = fFar / fSamples;" \
				"float fScaledLength = fSampleLength * u_fScale;" \
				"vec3 v3SampleRay = v3Ray * fSampleLength;" \
				"vec3 v3SamplePoint = v3Start + v3SampleRay * 0.5;" \


				// Now loop through the sample rays
				"vec3 v3FrontColor = vec3(0.0f, 0.0f, 0.0f);" \

				
				"for(int i=0; i<nSamples; i++){"

					"float fHeight = length(v3SamplePoint);" \

					"float fDepth = exp(u_fScaleOverScaleDepth * (u_fInnerRadius - fHeight));" \
					"float fLightAngle = dot(u_v3LightPosition, v3SamplePoint) / fHeight;" \
					"float fCameraAngle = dot(v3Ray, v3SamplePoint) / fHeight;" \
					

					"float fScatter = (fStartOffset + fDepth * (scale(fLightAngle) - scale(fCameraAngle)));" \

					"vec3 v3Attenuate = exp(-fScatter * (u_v3InverseWavelength * u_fKr4PI + u_fKm4PI));" \

					"v3FrontColor = v3FrontColor + v3Attenuate * (fDepth * fScaledLength);" \

					"v3SamplePoint = v3SamplePoint + v3SampleRay;" \
				"}"

				// Finally, scale the Mie and Rayleigh colors and set up the varying variables for the pixel shader
				"v3FrontSecColor_VS = v3FrontColor * u_fKmESun;" \

				"v3FrontColor_VS = v3FrontColor * (u_v3InverseWavelength * u_fKrESun);" \

				"v3Direction_VS = u_v3CameraPosition - v3Pos;" \

			"}" \

			"gl_Position = u_mvp_matrix * vPosition;" \

			
		
		"}";



			
	glShaderSource(iVertexShader, 1,
		(const GLchar**)&szVertexShaderSourceCode_RRJ, NULL);

	glCompileShader(iVertexShader);

	GLint iShaderCompileStatus_RRJ;
	GLint iInfoLogLength_RRJ;
	GLchar *szInfoLog_RRJ = NULL;
	glGetShaderiv(iVertexShader, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(iVertexShader, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetShaderInfoLog(iVertexShader, iInfoLogLength_RRJ,
					&written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Init_SkyFromAtmosphere: Vertex Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
				exit(0);
			}
		}
	}

	/********** Fragment Shader **********/
	iFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *szFragmentShaderSourceCode =
		"#version 450 core" \
		"\n" \
		
		"in vec3 v3FrontColor_VS;" \
		"in vec3 v3FrontSecColor_VS;" \
		"in vec3 v3Direction_VS;" \

		"uniform vec3 u_v3LightPosition;" \
		"uniform float u_g;" \
		"uniform float u_g2;" \
		"uniform int u_choice;" \

		"out vec4 v4FragColor;" \

		"void main(void){" \

			"if(u_choice == 0){" \

				"float fCos = dot(u_v3LightPosition, v3Direction_VS) / length(v3Direction_VS);" \

				//float fMiePhase = 1.5 * ((1.0 - g2) / (2.0 + g2)) * (1.0 + fCos*fCos) / pow(1.0 + g2 - 2.0*g*fCos, 1.5);
				"float fRayleighPhase = 0.75 * (1 + (fCos * fCos));" \
				"float fMiePhase = 1.5 * ((1.0 - u_g2) / (2.0 + u_g2)) * (1.0f + fCos * fCos) / pow(1.0 + u_g2 - 2.0f * u_g * fCos, 1.5);" \

				//gl_FragColor = gl_Color + fMiePhase * gl_SecondaryColor;
				"vec3 color = fRayleighPhase * v3FrontColor_VS +  v3FrontSecColor_VS;" \

				//gl_FragColor.a = gl_FragColor.b;
				"v4FragColor = vec4(color.x, color.y, color.z, color.z);" \

			"}" \
		
			"else if(u_choice == 1) {" \

				"v4FragColor = vec4(1.0f, 0.50f, 0.0f, 1.0f);" \

			"}" \

		"}";

	glShaderSource(iFragmentShader, 1,
		(const GLchar**)&szFragmentShaderSourceCode, NULL);

	glCompileShader(iFragmentShader);

	iShaderCompileStatus_RRJ = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;

	glGetShaderiv(iFragmentShader, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(iFragmentShader, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetShaderInfoLog(iFragmentShader, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Init_SkyFromAtmosphere: Fragment Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
				exit(0);
			}
		}
	}

	/********** Shader Program Object **********/
	iShaderProgramObject = glCreateProgram();

	glAttachShader(iShaderProgramObject, iVertexShader);
	glAttachShader(iShaderProgramObject, iFragmentShader);

	//glBindAttribLocation(iShaderProgramObject, AMC_ATTRIBUTE_POSITION, "vPosition");
	//glBindAttribLocation(iShaderProgramObject, AMC_ATTRIBUTE_NORMAL, "vNormal");

	glLinkProgram(iShaderProgramObject);

	GLint iProgramLinkingStatus = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;
	glGetProgramiv(iShaderProgramObject, GL_LINK_STATUS, &iProgramLinkingStatus);
	if (iProgramLinkingStatus == GL_FALSE) {
		glGetProgramiv(iShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetProgramInfoLog(iShaderProgramObject, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Shader Program Object Linking Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
				exit(0);
			}
		}
	}

}


void Init_GroundFromAtmosphere(GLuint &iShaderProgramObject, GLuint &iVertexShader, GLuint &iFragmentShader){


	/********** Vertex Shader **********/
	iVertexShader = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *szVertexShaderSourceCode_RRJ =
		"#version 450 core" \
		"\n" \

		"in vec4 vPosition;" \

		"uniform mat4 u_mvp_matrix;" \


		"uniform vec3 u_v3CameraPosition;" \
		"uniform vec3 u_v3LightPosition;" \
		"uniform vec3 u_v3InverseWavelength;" \

		"uniform float u_fCameraHeight;" \
		"uniform float u_fCameraHeight2;" \
		"uniform float u_fOuterRadius;" \
		"uniform float u_fOuterRadius2;" \
		"uniform float u_fInnerRadius;" \
		"uniform float u_fInnerRadius2;" \

		"uniform float u_fKrESun;" \
		"uniform float u_fKmESun;" \
		"uniform float u_fKr4PI;" \
		"uniform float u_fKm4PI;" \

		"uniform float u_fScale;" \
		"uniform float u_fScaleDepth;" \
		"uniform float u_fScaleOverScaleDepth;" \

		"const int nSamples = 2;" \
		"const float fSamples = 2.0;" \

		"out vec3 v3FrontSecColor_VS;" \
		"out vec3 v3FrontColor_VS;" \

		"float scale(float fCos){" \
			
			"float x = 1.0f - fCos;" \
			"return (u_fScaleDepth * exp(-0.00287 + x * (0.459 + x *(3.83 + x * (-6.80 + x * 5.25)))));" \

		"}" \

		"void main(void){" \



			// Get the ray from the camera to the vertex, and its length (which is the far point of the ray passing through the atmosphere)
			"vec3 v3Pos = vPosition.xyz;" \
			"vec3 v3Ray = v3Pos - u_v3CameraPosition;" \
			"float fFar = length(v3Ray);" \
			"v3Ray = v3Ray / fFar;" \

			// Calculate the ray's starting position, then calculate its scattering offset
			"vec3 v3Start = u_v3CameraPosition;" \
			"float fDepth = exp((u_fInnerRadius - u_fCameraHeight) / u_fScaleDepth);" \
			"float fCameraAngle = dot(-v3Ray, v3Pos) / length(v3Pos);" \
			"float fLightAngle = dot(u_v3LightPosition, v3Pos) / length(v3Pos);" \
			"float fCameraScale = scale(fCameraAngle);" \
			"float fLightScale = scale(fLightAngle);" \
			
			"float fCameraOffset = fDepth * fCameraScale;" \
			"float fTemp = (fLightScale + fCameraScale);" \

			// Initialize the scattering loop variables
			"float fSampleLength = fFar / fSamples;" \
			"vec3 v3SampleRay = v3Ray * fSampleLength;" \
			"vec3 v3SamplePoint = v3Start + v3SampleRay * 0.5f;" \

			"float fScaledLength = fSampleLength * u_fScale;" \

			// Now loop through the sample rays
			"vec3 v3FrontColor = vec3(0.0f, 0.0f, 0.0f);" \
			"vec3 v3Attenuate;" \
			"for(int i = 0; i < nSamples; i++){"
				
				"float fHeight = length(v3SamplePoint);" \
				"float fDepth = exp(u_fScaleOverScaleDepth * (u_fInnerRadius - fHeight));" \
				"float fScatter = fDepth * fTemp - fCameraOffset;" \
				"v3Attenuate = exp(-fScatter * (u_v3InverseWavelength * u_fKr4PI + u_fKm4PI));" \
				"v3FrontColor = v3FrontColor + v3Attenuate * (fDepth * fScaledLength);" \
				"v3SamplePoint = v3SamplePoint + v3SampleRay;" \
			"}" \

			"v3FrontColor_VS = v3FrontColor * (u_v3InverseWavelength * u_fKrESun + u_fKmESun);" \

			// Calculate the attenuation factor for the ground
			"v3FrontSecColor_VS = v3Attenuate;" \

			"gl_Position = u_mvp_matrix * vPosition;" \
				
		"}";



			
	glShaderSource(iVertexShader, 1,
		(const GLchar**)&szVertexShaderSourceCode_RRJ, NULL);

	glCompileShader(iVertexShader);

	GLint iShaderCompileStatus_RRJ;
	GLint iInfoLogLength_RRJ;
	GLchar *szInfoLog_RRJ = NULL;
	glGetShaderiv(iVertexShader, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(iVertexShader, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetShaderInfoLog(iVertexShader, iInfoLogLength_RRJ,
					&written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Init_GroundFromAtmosphere: Vertex Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
				exit(0);
			}
		}
	}

	/********** Fragment Shader **********/
	iFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *szFragmentShaderSourceCode =
		"#version 450 core" \
		"\n" \
		
		"in vec3 v3FrontColor_VS;" \
		"in vec3 v3FrontSecColor_VS;" \
		"in vec3 v3Direction_VS;" \

		"uniform vec3 u_v3LightPosition;" \
		"uniform float u_g;" \
		"uniform float u_g2;" \
		"uniform int u_choice;" \

		"out vec4 v4FragColor;" \

		"void main(void){" \


			"vec3 color = v3FrontColor_VS + 0.25 * v3FrontSecColor_VS;" \
			"v4FragColor = vec4(color, color.z);" \

		"}";

	glShaderSource(iFragmentShader, 1,
		(const GLchar**)&szFragmentShaderSourceCode, NULL);

	glCompileShader(iFragmentShader);

	iShaderCompileStatus_RRJ = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;

	glGetShaderiv(iFragmentShader, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(iFragmentShader, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetShaderInfoLog(iFragmentShader, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Init_GroundFromAtmosphere: Fragment Shader Compilation Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
				exit(0);
			}
		}
	}

	/********** Shader Program Object **********/
	iShaderProgramObject = glCreateProgram();

	glAttachShader(iShaderProgramObject, iVertexShader);
	glAttachShader(iShaderProgramObject, iFragmentShader);

	glLinkProgram(iShaderProgramObject);

	GLint iProgramLinkingStatus = 0;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;
	glGetProgramiv(iShaderProgramObject, GL_LINK_STATUS, &iProgramLinkingStatus);
	if (iProgramLinkingStatus == GL_FALSE) {
		glGetProgramiv(iShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (GLchar*)malloc(sizeof(GLchar)* iInfoLogLength_RRJ);
			if (szInfoLog_RRJ != NULL) {
				GLsizei written;
				glGetProgramInfoLog(iShaderProgramObject, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "Shader Program Object Linking Error: %s\n", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
				exit(0);
			}
		}
	}

}




void SetUniformsForShaderProgramObject(GLint iShaderProgramObject){

	glBindAttribLocation(iShaderProgramObject, AMC_ATTRIBUTE_POSITION, "vPosition");

	giMVPUniform_RRJ 					= glGetUniformLocation(iShaderProgramObject, "u_mvp_matrix");
	giCameraPositonUniform_RRJ			= glGetUniformLocation(iShaderProgramObject, "u_v3CameraPosition");
	giLightPositionUniform_RRJ 			= glGetUniformLocation(iShaderProgramObject, "u_v3LightPosition");
	giInverseWavelengthUniform_RRJ		= glGetUniformLocation(iShaderProgramObject, "u_v3InverseWavelength");
	giCameraHeightUniform_RRJ			= glGetUniformLocation(iShaderProgramObject, "u_fCameraHeight");
	giCameraHeightSquareUniform_RRJ 		= glGetUniformLocation(iShaderProgramObject, "u_fCameraHeight2");
	giOuterRadiusUniform_RRJ 				= glGetUniformLocation(iShaderProgramObject, "u_fOuterRadius");
	giOuterRadiusSquareUniform_RRJ		= glGetUniformLocation(iShaderProgramObject, "u_fOuterRadius2");
	giInnerRadiusUniform_RRJ 				= glGetUniformLocation(iShaderProgramObject, "u_fInnerRadius");
	giInnerRadiusSquareUniform_RRJ 		= glGetUniformLocation(iShaderProgramObject, "u_fInnerRadius2");
	giKrESunUniform_RRJ					= glGetUniformLocation(iShaderProgramObject, "u_fKrESun");
	giKmESunUniform_RRJ				= glGetUniformLocation(iShaderProgramObject, "u_fKmESun");
	giKr4PiUniform_RRJ					= glGetUniformLocation(iShaderProgramObject, "u_fKr4PI");
	giKm4PiUniform_RRJ					= glGetUniformLocation(iShaderProgramObject, "u_fKm4PI");
	giFScaleUniform_RRJ					= glGetUniformLocation(iShaderProgramObject, "u_fScale");
	giFScaleDepthUniform_RRJ				= glGetUniformLocation(iShaderProgramObject, "u_fScaleDepth");
	giFScaleOverScaleDepthUniform_RRJ		= glGetUniformLocation(iShaderProgramObject, "u_fScaleOverScaleDepth");
	giGUniform_RRJ						= glGetUniformLocation(iShaderProgramObject, "u_g");
	giG2Uniform_RRJ					= glGetUniformLocation(iShaderProgramObject, "u_g2");

}
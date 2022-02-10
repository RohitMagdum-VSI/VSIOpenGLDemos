#include<windows.h>
#include<stdio.h>
#include<stdlib.h>
#include<GL/glew.h>
#include<GL/gl.h>
#include"vmath.h"
#include"Sphere.h"

#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "Sphere.lib")

using namespace vmath;

enum {
	AMC_ATTRIBUTE_POSITION = 0,
	AMC_ATTRIBUTE_COLOR,
	AMC_ATTRIBUTE_NORMAL,
	AMC_ATTRIBUTE_TEXCOORD0,
};

#define WIN_WIDTH 800
#define WIN_HEIGHT 600


//For Stack
struct STACK {
	mat4 ModelViewMatrix;
	struct STACK *next;
	struct STACK *prev;
};

typedef struct STACK ModelViewStack;
ModelViewStack *TopNode_RRJ = NULL;
int MaxTop_RRJ = 32;
int iTop_RRJ = -1;


//For Planet
int year_RRJ;
int day_RRJ;
int moon_RRJ;

//For FullScreen
bool bIsFullScreen_RRJ = false;
HWND ghwnd_RRJ;
WINDOWPLACEMENT wpPrev_RRJ = { sizeof(WINDOWPLACEMENT) };
DWORD dwStyle_RRJ;

//For OpenGL Context 
HDC ghdc_RRJ = NULL;
HGLRC ghrc_RRJ = NULL;
bool bActiveWindow_RRJ = true;

//For Error
FILE *gbFile_RRJ = NULL;


//For Shader Program Object;
GLint gShaderProgramObject_RRJ;

//For Perspective Matric
mat4 gPerspectiveProjectionMatrix_RRJ;

//For Sphere
GLuint vao_Sphere_RRJ;
GLuint vbo_Sphere_Position_RRJ;
GLuint vbo_Sphere_Normal_RRJ;
GLuint vbo_Sphere_Element_RRJ;
float sphere_vertices_RRJ[1146];
float sphere_normals_RRJ[1146];
float sphere_textures_RRJ[764];
unsigned short sphere_elements_RRJ[2280];
unsigned int gNumVertices_RRJ;
unsigned int gNumElements_RRJ;
GLfloat angle_Sphere_RRJ = 0.0f;

//For Uniform
GLuint mvUniform_RRJ;
GLuint projectionUniform_RRJ;

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow) {

	int initialize(void);
	void display(void);
	void ToggleFullScreen(void);

	fopen_s(&gbFile_RRJ, "Log.txt", "w");
	if (gbFile_RRJ == NULL) {
		MessageBox(NULL, TEXT("ERROR"), TEXT("Log Creation Failed!!\n"), MB_OK);
		exit(-1);
	}
	else
		fprintf(gbFile_RRJ, "Log Created!!\n");



	WNDCLASSEX wndclass_RRJ;
	HWND hwnd_RRJ;
	MSG msg_RRJ;
	TCHAR szName_RRJ[] = TEXT("Rohit_R_Jadhav-SolarSystem");
	int iRet_RRJ;
	bool bDone_RRJ = false;


	wndclass_RRJ.lpszClassName = szName_RRJ;
	wndclass_RRJ.lpszMenuName = NULL;
	wndclass_RRJ.lpfnWndProc = WndProc;

	wndclass_RRJ.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass_RRJ.cbSize = sizeof(WNDCLASSEX);
	wndclass_RRJ.cbClsExtra = 0;
	wndclass_RRJ.cbWndExtra = 0;

	wndclass_RRJ.hInstance = hInstance;
	wndclass_RRJ.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass_RRJ.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass_RRJ.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wndclass_RRJ.hCursor = LoadCursor(NULL, IDC_HAND);

	RegisterClassEx(&wndclass_RRJ);

	hwnd_RRJ = CreateWindowEx(WS_EX_APPWINDOW,
		szName_RRJ,
		TEXT("Rohit_R_Jadhav-SolarSystem"),
		WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
		100, 100,
		WIN_WIDTH, WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL);

	ghwnd_RRJ = hwnd_RRJ;


	iRet_RRJ = initialize();
	if (iRet_RRJ == 1) {
		fprintf(gbFile_RRJ, "ChoosePixelFormat() Failed!!\n");
		DestroyWindow(hwnd_RRJ);
	}
	else if (iRet_RRJ == -2) {
		fprintf(gbFile_RRJ, "setPixelFormat() Failed!!\n");
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
		fprintf(gbFile_RRJ, "initialize() Done!!\n");



	

	ShowWindow(hwnd_RRJ, iCmdShow);
	SetForegroundWindow(hwnd_RRJ);
	SetFocus(hwnd_RRJ);
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
				//For UPdate
			}
			display();
		}
	}

	return((int)msg_RRJ.wParam);
}


LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam) {

	void uninitialize(void);
	void resize(int, int);
	void ToggleFullScreen(void);

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

	case WM_KEYDOWN:
		switch (wParam) {
		case VK_ESCAPE:
			DestroyWindow(hwnd);
			break;
		}
		break;

	case WM_CHAR:
		switch (wParam) {

		case 'F':
		case 'f':
			ToggleFullScreen();
			fprintf(gbFile_RRJ, "In F!\n");
			break;

		case 'Y':
			year_RRJ = (year_RRJ + 3) % 360;
			break;


		case 'y':
			year_RRJ = (year_RRJ - 3) % 360;
			break;

		case 'D':
			day_RRJ = (day_RRJ + 6) % 360;
			break;

		case 'd':
			day_RRJ = (day_RRJ - 6) % 360;
			break;

		case 'M':
			moon_RRJ = (moon_RRJ + 3) % 360;
			break;

		case 'm':
			moon_RRJ = (moon_RRJ - 3) % 360;
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
		if (dwStyle_RRJ & WS_OVERLAPPEDWINDOW) {

			mi_RRJ = { sizeof(MONITORINFO) };

			if (GetWindowPlacement(ghwnd_RRJ, &wpPrev_RRJ) && GetMonitorInfo(MonitorFromWindow(ghwnd_RRJ, MONITORINFOF_PRIMARY), &mi_RRJ)) {
				SetWindowLong(ghwnd_RRJ, GWL_STYLE, dwStyle_RRJ & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghwnd_RRJ, HWND_TOP,
					mi_RRJ.rcMonitor.left,
					mi_RRJ.rcMonitor.top,
					(mi_RRJ.rcMonitor.right - mi_RRJ.rcMonitor.left),
					(mi_RRJ.rcMonitor.bottom - mi_RRJ.rcMonitor.top),
					SWP_NOZORDER | SWP_FRAMECHANGED);

				bIsFullScreen_RRJ = true;
				ShowCursor(FALSE);
			}
		}
	}
	else {
		SetWindowLong(ghwnd_RRJ, GWL_STYLE, dwStyle_RRJ | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd_RRJ, &wpPrev_RRJ);
		SetWindowPos(ghwnd_RRJ, HWND_TOP,
			0, 0, 0, 0,
			SWP_NOSIZE | SWP_NOMOVE | SWP_NOOWNERZORDER | SWP_FRAMECHANGED | SWP_NOZORDER);
		bIsFullScreen_RRJ = false;
		ShowCursor(TRUE);
	}
}

int initialize(void) {

	void uninitialize(void);
	void resize(int, int);

	PIXELFORMATDESCRIPTOR pfd_RRJ;
	int iPixelFormatIndex_RRJ;
	GLenum Result_RRJ;

	//Shader Object;
	GLint iVertexShaderObject_RRJ;
	GLint iFragmentShaderObject_RRJ;

	ghdc_RRJ = GetDC(ghwnd_RRJ);

	pfd_RRJ.nVersion = 1;
	pfd_RRJ.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd_RRJ.iPixelType = PFD_TYPE_RGBA;
	pfd_RRJ.dwFlags = PFD_DOUBLEBUFFER | PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL;

	pfd_RRJ.cColorBits = 32;
	pfd_RRJ.cRedBits = 8;
	pfd_RRJ.cGreenBits = 8;
	pfd_RRJ.cBlueBits = 8;
	pfd_RRJ.cAlphaBits = 8;
	pfd_RRJ.cDepthBits = 24;


	iPixelFormatIndex_RRJ = ChoosePixelFormat(ghdc_RRJ, &pfd_RRJ);
	if (iPixelFormatIndex_RRJ == -1)
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
	}


	/********** Vertex Shader **********/
	iVertexShaderObject_RRJ = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *szVertexShaderSourceCode_RRJ =
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec3 vColor;" \
		"out vec3 outColor;" \
		"uniform mat4 u_mv_matrix;" \
		"uniform mat4 u_projection_matrix;" \
		"void main(void) {" \
		"outColor = vColor;" \
		"gl_Position = u_projection_matrix * u_mv_matrix * vPosition;" \
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
		"in vec3 outColor;" \
		"out vec4 FragColor;" \
		"void main(void) {" \
		"FragColor = vec4(outColor, 1.0);" \
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
	glBindAttribLocation(gShaderProgramObject_RRJ, AMC_ATTRIBUTE_COLOR, "vColor");

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

	mvUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_mv_matrix");
	projectionUniform_RRJ = glGetUniformLocation(gShaderProgramObject_RRJ, "u_projection_matrix");



	/********** Position, Normal and Elements **********/
	getSphereVertexData(sphere_vertices_RRJ, sphere_normals_RRJ, sphere_textures_RRJ, sphere_elements_RRJ);
	gNumVertices_RRJ = getNumberOfSphereVertices();
	gNumElements_RRJ = getNumberOfSphereElements();



	/********** Sphere Vao **********/
	glGenVertexArrays(1, &vao_Sphere_RRJ);
	glBindVertexArray(vao_Sphere_RRJ);

	/********** Position **********/
	glGenBuffers(1, &vbo_Sphere_Position_RRJ);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Sphere_Position_RRJ);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(sphere_vertices_RRJ),
		sphere_vertices_RRJ,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	/********** Color **********/
	glVertexAttrib3f(AMC_ATTRIBUTE_COLOR, 1.0f, 1.0f, 1.0f);


	/********** Element Vbo **********/
	glGenBuffers(1, &vbo_Sphere_Element_RRJ);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_Sphere_Element_RRJ);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(sphere_elements_RRJ), sphere_elements_RRJ, GL_STATIC_DRAW);
	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);


	glBindVertexArray(0);


	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	gPerspectiveProjectionMatrix_RRJ = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);
	return(0);
}



void uninitialize(void) {

	if (vbo_Sphere_Element_RRJ) {
		glDeleteBuffers(1, &vbo_Sphere_Element_RRJ);
		vbo_Sphere_Element_RRJ = 0;
	}

	if (vbo_Sphere_Normal_RRJ) {
		glDeleteBuffers(1, &vbo_Sphere_Normal_RRJ);
		vbo_Sphere_Normal_RRJ = 0;
	}

	if (vbo_Sphere_Position_RRJ) {
		glDeleteBuffers(1, &vbo_Sphere_Position_RRJ);
		vbo_Sphere_Position_RRJ = 0;
	}

	if (vao_Sphere_RRJ) {
		glDeleteVertexArrays(1, &vao_Sphere_RRJ);
		vao_Sphere_RRJ = 0;
	}


	GLsizei ShaderCount;
	GLsizei ShaderNumber;

	if (gShaderProgramObject_RRJ) {
		glUseProgram(gShaderProgramObject_RRJ);

		glGetProgramiv(gShaderProgramObject_RRJ, GL_ATTACHED_SHADERS, &ShaderCount);

		GLuint *pShader = (GLuint*)malloc(sizeof(GLuint) * ShaderCount);
		if (pShader) {
			glGetAttachedShaders(gShaderProgramObject_RRJ, ShaderCount, &ShaderCount, pShader);
			for (ShaderNumber = 0; ShaderNumber < ShaderCount; ShaderNumber++) {
				glDetachShader(gShaderProgramObject_RRJ, pShader[ShaderNumber]);
				glDeleteShader(pShader[ShaderNumber]);
				pShader[ShaderNumber] = 0;
			}
			free(pShader);
			pShader = NULL;
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


	if (ghrc_RRJ == wglGetCurrentContext()) {
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
	gPerspectiveProjectionMatrix_RRJ = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}


void display(void) {

	void my_glPushMatrix(mat4);
	mat4 my_glPopMatrix();


	mat4 modelMatrix_RRJ;
	mat4 viewMatrix_RRJ;
	mat4 modelViewMatrix_RRJ;


	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(gShaderProgramObject_RRJ);



	/********** Sphere **********/
	modelMatrix_RRJ = mat4::identity();
	viewMatrix_RRJ = mat4::identity();
	modelViewMatrix_RRJ = mat4::identity();


	viewMatrix_RRJ = lookat(vec3(0.0f, 0.0f, 3.0f),
		vec3(0.0f, 0.0f, 0.0f),
		vec3(0.0f, 1.0f, 0.0));

	modelViewMatrix_RRJ = viewMatrix_RRJ * modelMatrix_RRJ;


	//Sun
	my_glPushMatrix(modelViewMatrix_RRJ);


	glUniformMatrix4fv(mvUniform_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
	glUniformMatrix4fv(projectionUniform_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);

	glBindVertexArray(vao_Sphere_RRJ);
	glVertexAttrib3f(AMC_ATTRIBUTE_COLOR, 1.0f, 1.0f, 0.0f);
	glDrawElements(GL_TRIANGLES, gNumElements_RRJ, GL_UNSIGNED_SHORT, 0);
	glBindVertexArray(0);


	//Earth
	modelViewMatrix_RRJ = my_glPopMatrix();

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate((GLfloat)year_RRJ, 0.0f, 1.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(1.50f, 0.0f, 0.0f);
	
	my_glPushMatrix(modelViewMatrix_RRJ);

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * scale(0.4f, 0.4f, 0.4f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate((GLfloat)day_RRJ, 0.0f, 1.0f, 0.0f);

	

	glUniformMatrix4fv(mvUniform_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
	glUniformMatrix4fv(projectionUniform_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);

	glBindVertexArray(vao_Sphere_RRJ);
	glVertexAttrib3f(AMC_ATTRIBUTE_COLOR, 0.0f, 0.0f, 0.5f);
	glDrawElements(GL_LINES, gNumElements_RRJ, GL_UNSIGNED_SHORT, 0);
	glBindVertexArray(0);

	


	//Moon
	modelViewMatrix_RRJ = my_glPopMatrix();

	modelViewMatrix_RRJ = modelViewMatrix_RRJ * rotate((GLfloat)moon_RRJ, 0.0f, 1.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * translate(0.50f, 0.0f, 0.0f);
	modelViewMatrix_RRJ = modelViewMatrix_RRJ * scale(0.1f, 0.1f, 0.1f);

	my_glPushMatrix(modelViewMatrix_RRJ);

	glUniformMatrix4fv(mvUniform_RRJ, 1, GL_FALSE, modelViewMatrix_RRJ);
	glUniformMatrix4fv(projectionUniform_RRJ, 1, GL_FALSE, gPerspectiveProjectionMatrix_RRJ);

	glBindVertexArray(vao_Sphere_RRJ);
	glVertexAttrib3f(AMC_ATTRIBUTE_COLOR, 0.50f, 0.50f, 0.5f);
	glDrawElements(GL_LINES, gNumElements_RRJ, GL_UNSIGNED_SHORT, 0);
	glBindVertexArray(0);

	my_glPopMatrix();


	glUseProgram(0);

	SwapBuffers(ghdc_RRJ);
}


void my_glPushMatrix(mat4 matrix) {

	void uninitialize(void);

	ModelViewStack *temp_RRJ = (ModelViewStack*)malloc(sizeof(ModelViewStack));
	if (temp_RRJ == NULL) {
		fprintf(gbFile_RRJ, "ERROR: Malloc Failed!!\n");
		uninitialize();
		DestroyWindow(ghwnd_RRJ);
	}
	else {

		temp_RRJ->ModelViewMatrix = matrix;
		temp_RRJ->next = NULL;

		if (TopNode_RRJ == NULL) {
			TopNode_RRJ = temp_RRJ;
			TopNode_RRJ->prev = NULL;
			fprintf(gbFile_RRJ, "Node Added!!\n");
		}
		else {
			TopNode_RRJ->next = temp_RRJ;
			temp_RRJ->prev = TopNode_RRJ;
			TopNode_RRJ = temp_RRJ;
			fprintf(gbFile_RRJ, "Node Added!!\n");
		}
	}

	if (iTop_RRJ > MaxTop_RRJ) {
		fprintf(gbFile_RRJ, "ERROR: Stack Overflow!!\n");
		uninitialize();
		DestroyWindow(ghwnd_RRJ);
	}



}

mat4 my_glPopMatrix(void) {

	void uninitialize(void);

	ModelViewStack *temp_RRJ = TopNode_RRJ;
	mat4 matrix_RRJ;
	if (temp_RRJ->prev != NULL) {
		TopNode_RRJ = temp_RRJ->prev;
		temp_RRJ->next = NULL;
		temp_RRJ->prev = NULL;
		matrix_RRJ = temp_RRJ->ModelViewMatrix;
		fprintf(gbFile_RRJ, "Node Delete!!\n");
		free(temp_RRJ);
	}
	else {
		temp_RRJ->next = NULL;
		temp_RRJ->prev = NULL;
		matrix_RRJ = temp_RRJ->ModelViewMatrix;
		fprintf(gbFile_RRJ, "Node Delete!!\n");
		free(temp_RRJ);
		TopNode_RRJ = NULL;
	}
	return(matrix_RRJ);

}





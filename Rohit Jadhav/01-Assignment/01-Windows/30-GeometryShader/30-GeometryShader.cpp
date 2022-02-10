#include<windows.h>
#include<stdio.h>
#include<GL/glew.h>
#include<GL/gl.h>
#include"vmath.h"

#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

using namespace vmath;

enum {
	AMC_ATTRIBUTE_POSITION = 1,
	AMC_ATTRIBUTE_COLOR,
	AMC_ATTRIBUTE_NORMAL,
	AMC_ATTRIBUTE_TEXCOORD0,
};

//For Fullscreen
bool bIsFullScreen_RRJ = false;
WINDOWPLACEMENT wpPrev_RRJ = { sizeof(WINDOWPLACEMENT) };
DWORD dwStyle_RRJ;
HWND ghwnd_RRJ;

//For OpenGL
bool bActiveWindow_RRJ = false;
HDC ghdc_RRJ = NULL;
HGLRC ghrc_RRJ = NULL;

//For Error
FILE *gbFile_RRJ = NULL;

//For Shader
GLuint vertexShaderObject_RRJ;
GLuint geometryShaderObject_RRJ;
GLuint fragmentShaderObject_RRJ;
GLuint shaderProgramObject_RRJ;

GLuint vao_Tri_RRJ;
GLuint vbo_Tri_Position_RRJ;

//For Uniform
GLuint mvpUniform_RRJ;


mat4 perspectiveProjectionMatrix_RRJ;

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow) {


	int initialize(void);
	void ToggleFullScreen(void);
	void display(void);

	WNDCLASSEX wndclass_RRJ;
	HWND hwnd_RRJ;
	MSG msg_RRJ;
	TCHAR szName_RRJ[] = TEXT("Rohit_R_Jadhav-GeometryShader");

	int iRet_RRJ;
	bool bDone_RRJ = false;

	fopen_s(&gbFile_RRJ, "Log.txt", "w");
	if (gbFile_RRJ == NULL) {
		MessageBox(NULL, TEXT("Log Creation Failed!!"), TEXT("ERROR"), MB_OK);
		exit(0);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: Log Created!!\n");


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
	wndclass_RRJ.hCursor = LoadCursor(NULL, IDC_ARROW);

	RegisterClassEx(&wndclass_RRJ);

	hwnd_RRJ = CreateWindowEx(WS_EX_APPWINDOW,
		szName_RRJ,
		TEXT("Rohit_R_Jadhav-GeometryShader"),
		WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
		100, 100,
		WIN_WIDTH, WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL);

	ghwnd_RRJ = hwnd_RRJ;

	iRet_RRJ = initialize();
	if (iRet_RRJ == -1) {
		fprintf(gbFile_RRJ, "ERROR: ChoosePixelFormat() Failed!!\n");
		DestroyWindow(hwnd_RRJ);
	}
	else if (iRet_RRJ == -2) {
		fprintf(gbFile_RRJ, "ERROR: SetPixelFormat() Failed!!\n");
		DestroyWindow(hwnd_RRJ);
	}
	else if (iRet_RRJ == -3) {
		fprintf(gbFile_RRJ, "ERROR: wglCreateContext() Failed!!\n");
		DestroyWindow(hwnd_RRJ);
	}
	else if (iRet_RRJ == -4) {
		fprintf(gbFile_RRJ, "ERROR: wglMakeCurrent() Failed!!\n");
		DestroyWindow(hwnd_RRJ);
	}
	else
		fprintf(gbFile_RRJ, "SUCCESS: Initialize() Done!!\n");


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
				//Update
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
	case WM_ACTIVATE:
		if (HIWORD(wParam) == 0)
			bActiveWindow_RRJ = true;
		else
			bActiveWindow_RRJ = false;
		break;

	case WM_ERASEBKGND:
		return(0);

	case WM_SIZE:
		resize(LOWORD(lParam), HIWORD(lParam));
		break;

	case WM_KEYDOWN:
		switch (wParam) {
		case 0X46: //F
			ToggleFullScreen();
			break;

		case VK_ESCAPE:
			DestroyWindow(hwnd);
			break;
		}
		break;

	case WM_CLOSE:
		uninitialize();
		break;

	case WM_DESTROY:
		uninitialize();
		PostQuitMessage(0);
		break;

	}

	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void ToggleFullScreen(void) {

	MONITORINFO mi_RRJ = { sizeof(MONITORINFO) };

	if (bIsFullScreen_RRJ == false) {
		dwStyle_RRJ = GetWindowLong(ghwnd_RRJ, GWL_STYLE);
		if (dwStyle_RRJ & WS_OVERLAPPEDWINDOW) {
			if (GetWindowPlacement(ghwnd_RRJ, &wpPrev_RRJ) && GetMonitorInfo(MonitorFromWindow(ghwnd_RRJ, MONITORINFOF_PRIMARY), &mi_RRJ)) {
				SetWindowLong(ghwnd_RRJ, GWL_STYLE, dwStyle_RRJ & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghwnd_RRJ, HWND_TOP,
					mi_RRJ.rcMonitor.left,
					mi_RRJ.rcMonitor.top,
					(mi_RRJ.rcMonitor.right - mi_RRJ.rcMonitor.left),
					(mi_RRJ.rcMonitor.bottom - mi_RRJ.rcMonitor.top),
					SWP_NOZORDER | SWP_FRAMECHANGED);
			}
			ShowCursor(FALSE);
			bIsFullScreen_RRJ = true;
		}
	}
	else {
		SetWindowLong(ghwnd_RRJ, GWL_STYLE, dwStyle_RRJ | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd_RRJ, &wpPrev_RRJ);
		SetWindowPos(ghwnd_RRJ, HWND_TOP,
			0, 0, 0, 0,
			SWP_NOZORDER | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
		bIsFullScreen_RRJ = false;
	}
}

int initialize(void) {

	void uninitialize(void);
	void resize(int, int);

	PIXELFORMATDESCRIPTOR pfd_RRJ;
	int iPixelFormatIndex_RRJ;
	GLenum result_RRJ;

	ghdc_RRJ = GetDC(ghwnd_RRJ);

	memset(&pfd_RRJ, 0, sizeof(PIXELFORMATDESCRIPTOR));

	pfd_RRJ.nVersion = 1;
	pfd_RRJ.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd_RRJ.iPixelType = PFD_TYPE_RGBA;
	pfd_RRJ.dwFlags = PFD_DOUBLEBUFFER | PFD_SUPPORT_OPENGL | PFD_DRAW_TO_WINDOW;

	pfd_RRJ.cColorBits = 32;
	pfd_RRJ.cRedBits = 8;
	pfd_RRJ.cGreenBits = 8;
	pfd_RRJ.cBlueBits = 8;
	pfd_RRJ.cAlphaBits = 8;
	pfd_RRJ.cDepthBits = 24;

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


	result_RRJ = glewInit();
	if (result_RRJ != GLEW_OK) {
		fprintf(gbFile_RRJ, "ERROR: glewInit() Failed!!\n");
		uninitialize();
		DestroyWindow(ghwnd_RRJ);
	}


	/********** VERTEX SHADER **********/
	vertexShaderObject_RRJ = glCreateShader(GL_VERTEX_SHADER);
	const char *vertexShaderSourceCode_RRJ =
		"#version 450" \
		"\n" \
		"in vec4 vPosition;" \
		"uniform mat4 u_mvp_matrix;" \
		"void main(void) {" \
		"gl_Position = u_mvp_matrix * vPosition;" \
		"}";

	glShaderSource(vertexShaderObject_RRJ, 1, (const char**)&vertexShaderSourceCode_RRJ, NULL);
	glCompileShader(vertexShaderObject_RRJ);

	int iInfoLogLength_RRJ;
	int iShaderCompileStatus_RRJ;
	char *szInfoLog_RRJ = NULL;

	glGetShaderiv(vertexShaderObject_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(vertexShaderObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (char*)malloc(sizeof(char) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ) {
				GLsizei written;

				glGetShaderInfoLog(vertexShaderObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "VERTEX SHADER ERROR: \n %s", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
			}
		}
	}


	/********** GEOMETRY SHADER **********/
	geometryShaderObject_RRJ = glCreateShader(GL_GEOMETRY_SHADER);
	const char *geometryShaderSourceCode_RRJ =
		"#version 450" \
		"\n" \
		"layout(triangles)in;" \
		"layout(triangle_strip, max_vertices = 9)out;" \
		"uniform mat4 u_mvp_matrix;" \
		"void main(void) {" \
			"for(int vertex = 0; vertex < 3; vertex++){" \
		
				"gl_Position = u_mvp_matrix * (gl_in[vertex].gl_Position + vec4(0.0, 1.0, 0.0, 0.0));" \
				"EmitVertex();" \
					
				"gl_Position = u_mvp_matrix * (gl_in[vertex].gl_Position + vec4(-1.0, -1.0, 0.0, 0.0));" \
				"EmitVertex();" \
			
				"gl_Position = u_mvp_matrix * (gl_in[vertex].gl_Position + vec4(1.0, -1.0, 0.0, 0.0));" \
				"EmitVertex();" \

				"EndPrimitive();" \
			"}" \
		"}";


	glShaderSource(geometryShaderObject_RRJ, 1,
		(const char**)&geometryShaderSourceCode_RRJ, NULL);

	glCompileShader(geometryShaderObject_RRJ);

	iInfoLogLength_RRJ = 0;
	iShaderCompileStatus_RRJ = 0;
	szInfoLog_RRJ = NULL;

	glGetShaderiv(geometryShaderObject_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(geometryShaderObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (char*)malloc(sizeof(char) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ) {
				GLsizei written;
				glGetShaderInfoLog(geometryShaderObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "GEOMETRY SHADER ERROR: \n %s", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
			}
		}
	}
		
	



	/********** FRAGMENT SHADER **********/
	fragmentShaderObject_RRJ = glCreateShader(GL_FRAGMENT_SHADER);
	const char *fragmentShaderSourceCode_RRJ =
		"#version 450" \
		"\n" \
		"out vec4 FragColor;" \
		"void main(void) {" \
		"FragColor = vec4(1.0, 1.0, 1.0, 1.0);" \
		"}";


	glShaderSource(fragmentShaderObject_RRJ, 1,
		(const char**)&fragmentShaderSourceCode_RRJ, NULL);

	glCompileShader(fragmentShaderObject_RRJ);

	iInfoLogLength_RRJ = 0;
	iShaderCompileStatus_RRJ = 0;
	szInfoLog_RRJ = NULL;

	glGetShaderiv(fragmentShaderObject_RRJ, GL_COMPILE_STATUS, &iShaderCompileStatus_RRJ);
	if (iShaderCompileStatus_RRJ == GL_FALSE) {
		glGetShaderiv(fragmentShaderObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (char*)malloc(sizeof(char) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ) {
				GLsizei written;
				glGetShaderInfoLog(fragmentShaderObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "FRAGMENT SHADER ERROR: \n %s", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
			}
		}
	}


	/********** SHADER PROGRAM **********/
	shaderProgramObject_RRJ = glCreateProgram();

	glAttachShader(shaderProgramObject_RRJ, vertexShaderObject_RRJ);
	glAttachShader(shaderProgramObject_RRJ, geometryShaderObject_RRJ);
	glAttachShader(shaderProgramObject_RRJ, fragmentShaderObject_RRJ);

	glBindAttribLocation(shaderProgramObject_RRJ, AMC_ATTRIBUTE_POSITION, "vPosition");

	glLinkProgram(shaderProgramObject_RRJ);

	int iProgramLinkStatus_RRJ;
	iInfoLogLength_RRJ = 0;
	szInfoLog_RRJ = NULL;
	glGetProgramiv(shaderProgramObject_RRJ, GL_LINK_STATUS, &iProgramLinkStatus_RRJ);
	if (iProgramLinkStatus_RRJ == GL_FALSE) {
		glGetProgramiv(shaderProgramObject_RRJ, GL_INFO_LOG_LENGTH, &iInfoLogLength_RRJ);
		if (iInfoLogLength_RRJ > 0) {
			szInfoLog_RRJ = (char*)malloc(sizeof(char) * iInfoLogLength_RRJ);
			if (szInfoLog_RRJ) {
				GLsizei written;
				glGetProgramInfoLog(shaderProgramObject_RRJ, iInfoLogLength_RRJ, &written, szInfoLog_RRJ);
				fprintf(gbFile_RRJ, "SHADER PROGRAM ERROR: %s", szInfoLog_RRJ);
				free(szInfoLog_RRJ);
				szInfoLog_RRJ = NULL;
				uninitialize();
				DestroyWindow(ghwnd_RRJ);
			}
		}
	}


	mvpUniform_RRJ = glGetUniformLocation(shaderProgramObject_RRJ, "u_mvp_matrix");
	



	/********** LINE COORDINATES **********/
	float tri_Vertices_RRJ[] = {
		0.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
	};

	glGenVertexArrays(1, &vao_Tri_RRJ);
	glBindVertexArray(vao_Tri_RRJ);

	/********** Position **********/
	glGenBuffers(1, &vbo_Tri_Position_RRJ);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Tri_Position_RRJ);
	glBufferData(GL_ARRAY_BUFFER, sizeof(tri_Vertices_RRJ), tri_Vertices_RRJ, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);


	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	resize(WIN_WIDTH, WIN_HEIGHT);
	return(0);
}


void uninitialize(void) {


	if (bIsFullScreen_RRJ == true) {
		SetWindowLong(ghwnd_RRJ, GWL_STYLE, dwStyle_RRJ | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd_RRJ, &wpPrev_RRJ);
		SetWindowPos(ghwnd_RRJ, HWND_TOP,
			0, 0, 0, 0,
			SWP_NOZORDER | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
		bIsFullScreen_RRJ = false;

	}

	if (vbo_Tri_Position_RRJ) {
		glDeleteBuffers(1, &vbo_Tri_Position_RRJ);
		vbo_Tri_Position_RRJ = 0;
	}

	if (vao_Tri_RRJ) {
		glDeleteVertexArrays(1, &vao_Tri_RRJ);
		vao_Tri_RRJ = 0;
	}

	if (shaderProgramObject_RRJ) {
		glUseProgram(shaderProgramObject_RRJ);

		/*GLint shaderCount_RRJ;
		GLint shaderNo_RRJ;

		glGetShaderiv(shaderProgramObject_RRJ, GL_ATTACHED_SHADERS, &shaderCount_RRJ);
		fprintf(gbFile_RRJ, "INFO: ShaderCount: %d\n", shaderCount_RRJ);
		GLuint *pShaders = (GLuint*)malloc(sizeof(GLuint*) * shaderCount_RRJ);
		if (pShaders) {
			glGetAttachedShaders(shaderProgramObject_RRJ, shaderCount_RRJ, &shaderCount_RRJ, pShaders);
			for (shaderNo_RRJ = 0; shaderNo_RRJ < shaderCount_RRJ; shaderNo_RRJ++) {
				glDetachShader(shaderProgramObject_RRJ, pShaders[shaderNo_RRJ]);
				glDeleteShader(pShaders[shaderNo_RRJ]);
				pShaders[shaderNo_RRJ] = 0;
			}
			free(pShaders);
			pShaders = NULL;
		}*/


		if (fragmentShaderObject_RRJ) {
			glDetachShader(shaderProgramObject_RRJ, fragmentShaderObject_RRJ);
			glDeleteShader(fragmentShaderObject_RRJ);
			fragmentShaderObject_RRJ = 0;
		}

		if (geometryShaderObject_RRJ) {
			glDetachShader(shaderProgramObject_RRJ, geometryShaderObject_RRJ);
			glDeleteShader(geometryShaderObject_RRJ);
			geometryShaderObject_RRJ = 0;
		}

		if (vertexShaderObject_RRJ) {
			glDetachShader(shaderProgramObject_RRJ, vertexShaderObject_RRJ);
			glDeleteShader(vertexShaderObject_RRJ);
			vertexShaderObject_RRJ = 0;
		}


		glUseProgram(0);
		glDeleteProgram(shaderProgramObject_RRJ);
		shaderProgramObject_RRJ = 0;
	}


	if (wglGetCurrentContext() == ghrc_RRJ)
		wglMakeCurrent(NULL, NULL);

	if (ghrc_RRJ) {
		wglDeleteContext(ghrc_RRJ);
		ghrc_RRJ = NULL;
	}

	if (ghdc_RRJ) {
		ReleaseDC(ghwnd_RRJ, ghdc_RRJ);
		ghdc_RRJ = NULL;
	}

	if (gbFile_RRJ) {
		fprintf(gbFile_RRJ, "SUCCESS: Log Close!!\n");
		fclose(gbFile_RRJ);
		gbFile_RRJ = NULL;
	}
}


void resize(int width, int height) {

	if (height == 0)
		height = 1;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	perspectiveProjectionMatrix_RRJ = mat4::identity();
	perspectiveProjectionMatrix_RRJ = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}

void display(void) {

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	mat4 modelViewMatrix_RRJ;
	mat4 modelViewProjectionMatrix_RRJ;


	glUseProgram(shaderProgramObject_RRJ);

	modelViewMatrix_RRJ = mat4::identity();
	modelViewProjectionMatrix_RRJ = mat4::identity();

	modelViewMatrix_RRJ = translate(0.0f, 0.00f, -4.0f);
	modelViewProjectionMatrix_RRJ = perspectiveProjectionMatrix_RRJ * modelViewMatrix_RRJ;

	glUniformMatrix4fv(mvpUniform_RRJ, 1, GL_FALSE, modelViewProjectionMatrix_RRJ);
	


	glBindVertexArray(vao_Tri_RRJ);
	glDrawArrays(GL_TRIANGLES, 0, 3);
	glBindVertexArray(0);


	glUseProgram(0);

	SwapBuffers(ghdc_RRJ);
}


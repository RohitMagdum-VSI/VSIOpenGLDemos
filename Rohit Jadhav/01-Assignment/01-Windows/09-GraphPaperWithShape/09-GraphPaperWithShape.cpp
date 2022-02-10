#include<Windows.h>
#include<stdio.h>
#include<gl/glew.h>
#include<GL/GL.h>
#include"vmath.h"

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

//For Blue_Grid
GLuint vao_Grid;
GLuint vbo_Grid_Position;
//GLuint vbo_Grid_Color;

//For Red_And_Blue_Axis
GLuint vao_Axis;
GLuint vbo_Axis_Position;
GLuint vbo_Axis_Color;


//For Yellow Triangle
GLuint vao_Triangle;
GLuint vbo_Triangle_Position;
GLuint vbo_Triangle_Color;

//For Yellow Rect
GLuint vao_Rect;
GLuint vbo_Rect_Position;
GLuint vbo_Rect_Color;

//For Yellow InCircle
GLuint vao_Circle;
GLuint vbo_Circle_Position;
GLuint vbo_Circle_Color;

GLfloat Incircle_Center[3];
GLfloat Incircle_Radius;

//For Uniform
GLuint mvpUniform;

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
	TCHAR szName[] = TEXT("RohitRJadhav-PP-GraphPaperWithShape");

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
		TEXT("RohitRJadhav-PP-GraphPaperWithShape"),
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
		fprintf(gbFile, "wglCreateContext() Failed!!\n");
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
	void FillGridPosition(GLfloat[]);
	void Calculation(GLfloat[]);



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
		"in vec4 vColor;" \
		"out vec4 outColor;" \
		"uniform mat4 u_mvp_matrix;" \
		"void main(void)" \
		"{" \
		"gl_Position = u_mvp_matrix * vPosition;" \
		"outColor = vColor;" \
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
		"in vec4 outColor;" \
		"out vec4 FragColor;" \
		"void main(void)" \
		"{" \
		"FragColor = outColor;" \
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

	/********** Position  **********/
	GLfloat Grid_Position[40 * 6];
	FillGridPosition(Grid_Position);

	GLfloat Triangle_Position[] = {
		0.0f, 0.70f, 0.0f,
		-0.70f, -0.70f, 0.0f,
		0.70f, -0.70f, 0.0f
	};

	GLfloat Triangle_Color[] = {
		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f
	};

	GLfloat Rect_Position[] = {
		0.70f, 0.70f, 0.0f,
		-0.70f, 0.70f, 0.0f,
		-0.70f, -0.70f, 0.0f,
		0.70f, -0.70f, 0.0f
	};

	GLfloat Rect_Color[] = {
		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f
	};


	/********** To Calculate InCircle Radius and Center **********/
	Calculation(Triangle_Position);


	/********** Grid **********/
	glGenVertexArrays(1, &vao_Grid);
	glBindVertexArray(vao_Grid);

	/********** Position **********/
	glGenBuffers(1, &vbo_Grid_Position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Grid_Position);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(Grid_Position),
		Grid_Position,
		GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/********** Color **********/
	glVertexAttrib3f(AMC_ATTRIBUTE_COLOR, 0.0f, 0.0f, 1.0f);

	glBindVertexArray(0);




	/********** Axis **********/
	glGenVertexArrays(1, &vao_Axis);
	glBindVertexArray(vao_Axis);

	/********** Position **********/
	glGenBuffers(1, &vbo_Axis_Position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Axis_Position);
	glBufferData(GL_ARRAY_BUFFER,
		6 * sizeof(GLfloat),
		NULL,
		GL_DYNAMIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/********** Color **********/
	glGenBuffers(1, &vbo_Axis_Color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Axis_Color);
	glBufferData(GL_ARRAY_BUFFER,
		6 * sizeof(GLfloat),
		NULL,
		GL_DYNAMIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);


	/********** Triangle **********/
	glGenVertexArrays(1, &vao_Triangle);
	glBindVertexArray(vao_Triangle);

	/********** Position **********/
	glGenBuffers(1, &vbo_Triangle_Position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Triangle_Position);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(Triangle_Position),
		Triangle_Position,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/********** Color **********/
	glGenBuffers(1, &vbo_Triangle_Color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Triangle_Color);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(Triangle_Color),
		Triangle_Color,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);



	/********** Rectangle **********/
	glGenVertexArrays(1, &vao_Rect);
	glBindVertexArray(vao_Rect);

	/********** Position **********/
	glGenBuffers(1, &vbo_Rect_Position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Rect_Position);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(Rect_Position),
		Rect_Position,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/********** Color **********/
	glGenBuffers(1, &vbo_Rect_Color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Rect_Color);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(Rect_Color),
		Rect_Color,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);


	/********** Circle **********/
	glGenVertexArrays(1, &vao_Circle);
	glBindVertexArray(vao_Circle);

	/********** Position **********/
	glGenBuffers(1, &vbo_Circle_Position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Circle_Position);
	glBufferData(GL_ARRAY_BUFFER,
		1000 * 3 * sizeof(GLfloat),
		NULL,
		GL_DYNAMIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/********** Color **********/
	glGenBuffers(1, &vbo_Circle_Color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Circle_Color);
	glBufferData(GL_ARRAY_BUFFER,
		3 * 3000 * sizeof(GLfloat),
		NULL,
		GL_DYNAMIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);


	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	gPerspectiveProjectionMatrix = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);
	return(0);
}

void uninitialize(void) {



	if (vbo_Circle_Color) {
		glDeleteBuffers(1, &vbo_Circle_Color);
		vbo_Circle_Color = 0;
	}

	if (vbo_Circle_Position) {
		glDeleteBuffers(1, &vbo_Circle_Position);
		vbo_Circle_Position = 0;
	}

	if (vao_Circle) {
		glDeleteVertexArrays(1, &vao_Circle);
		vao_Circle = 0;
	}

	if (vbo_Rect_Color) {
		glDeleteBuffers(1, &vbo_Rect_Color);
		vbo_Rect_Color = 0;
	}

	if (vbo_Rect_Position) {
		glDeleteBuffers(1, &vbo_Rect_Position);
		vbo_Rect_Position = 0;
	}

	if (vao_Rect) {
		glDeleteVertexArrays(1, &vao_Rect);
		vao_Rect = 0;
	}

	if (vbo_Triangle_Color) {
		glDeleteBuffers(1, &vbo_Triangle_Color);
		vbo_Triangle_Color = 0;
	}

	if (vbo_Triangle_Position) {
		glDeleteBuffers(1, &vbo_Triangle_Position);
		vbo_Triangle_Position = 0;
	}

	if (vao_Triangle) {
		glDeleteVertexArrays(1, &vao_Triangle);
		vao_Triangle = 0;
	}

	if (vbo_Axis_Color) {
		glDeleteBuffers(1, &vbo_Axis_Color);
		vbo_Axis_Color = 0;
	}

	if (vbo_Axis_Position) {
		glDeleteBuffers(1, &vbo_Axis_Position);
		vbo_Axis_Position = 0;
	}

	if (vao_Axis) {
		glDeleteVertexArrays(1, &vao_Axis);
		vao_Axis = 0;
	}


	if (vbo_Grid_Position) {
		glDeleteBuffers(1, &vbo_Grid_Position);
		vbo_Grid_Position = 0;
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

	gPerspectiveProjectionMatrix = mat4::identity();
	gPerspectiveProjectionMatrix = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}

void display(void) {

	void FillCircle_Position(GLfloat[], GLfloat[], int);

	mat4 translateMatrix;
	mat4 modelViewMatrix;
	mat4 modelViewProjectionMatrix;

	GLfloat Axis_Position[6];
	GLfloat Axis_Color[6];

	GLfloat Circle_Position[3000 * 3];
	GLfloat Circle_Color[3000 * 3];

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	glUseProgram(gShaderProgramObject);

	/********** Grid **********/
	translateMatrix = mat4::identity();
	modelViewMatrix = mat4::identity();
	modelViewProjectionMatrix = mat4::identity();

	translateMatrix = translate(0.0f, 0.0f, -3.0f);
	modelViewMatrix = modelViewMatrix * translateMatrix;
	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(mvpUniform,
		1,
		GL_FALSE,
		modelViewProjectionMatrix);

	glBindVertexArray(vao_Grid);

	glDrawArrays(GL_LINES, 0, 4 * 40);

	glBindVertexArray(0);


	/********** Axis **********/

	for (int i = 1; i <= 2; i++) {

		translateMatrix = mat4::identity();
		modelViewMatrix = mat4::identity();
		modelViewProjectionMatrix = mat4::identity();

		translateMatrix = translate(0.0f, 0.0f, -3.0f);
		modelViewMatrix = modelViewMatrix * translateMatrix;
		modelViewProjectionMatrix = gPerspectiveProjectionMatrix * modelViewMatrix;

		glUniformMatrix4fv(mvpUniform,
			1,
			GL_FALSE,
			modelViewProjectionMatrix);


		if (i == 1) {
			Axis_Position[0] = 0.0f;
			Axis_Position[1] = 1.0f;
			Axis_Position[2] = 0.0f;

			Axis_Position[3] = 0.0f;
			Axis_Position[4] = -1.0f;
			Axis_Position[5] = 0.0f;

			Axis_Color[0] = 1.0f;
			Axis_Color[1] = 0.0f;
			Axis_Color[2] = 0.0f;

			Axis_Color[3] = 1.0f;
			Axis_Color[4] = 0.0f;
			Axis_Color[5] = 0.0f;
		}
		else if (i == 2) {
			Axis_Position[0] = -1.0f;
			Axis_Position[1] = 0.0f;
			Axis_Position[2] = 0.0f;

			Axis_Position[3] = 1.0f;
			Axis_Position[4] = 0.0f;
			Axis_Position[5] = 0.0f;


			Axis_Color[0] = 0.0f;
			Axis_Color[1] = 1.0f;
			Axis_Color[2] = 0.0f;

			Axis_Color[3] = 0.0f;
			Axis_Color[4] = 1.0f;
			Axis_Color[5] = 0.0f;

		}

		glBindVertexArray(vao_Axis);

		/***** Position *****/
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Axis_Position);

		glBufferData(GL_ARRAY_BUFFER,
			sizeof(Axis_Position),
			Axis_Position,
			GL_DYNAMIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, 0);

		/***** Color *****/
		glBindBuffer(GL_ARRAY_BUFFER, vbo_Axis_Color);

		glBufferData(GL_ARRAY_BUFFER,
			sizeof(Axis_Color),
			Axis_Color,
			GL_DYNAMIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, 0);


		glDrawArrays(GL_LINES, 0, 4);

		glBindVertexArray(0);

	}



	/********** Triangle **********/
	translateMatrix = mat4::identity();
	modelViewMatrix = mat4::identity();
	modelViewProjectionMatrix = mat4::identity();

	translateMatrix = translate(0.0f, 0.0f, -3.0f);
	modelViewMatrix = modelViewMatrix * translateMatrix;
	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(mvpUniform,
		1,
		GL_FALSE,
		modelViewProjectionMatrix);

	glBindVertexArray(vao_Triangle);

	glDrawArrays(GL_LINE_LOOP, 0, 3);

	glBindVertexArray(0);




	/********** Rectangle **********/
	translateMatrix = mat4::identity();
	modelViewMatrix = mat4::identity();
	modelViewProjectionMatrix = mat4::identity();

	translateMatrix = translate(0.0f, 0.0f, -3.0f);
	modelViewMatrix = modelViewMatrix * translateMatrix;
	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(mvpUniform,
		1,
		GL_FALSE,
		modelViewProjectionMatrix);

	glBindVertexArray(vao_Rect);

	glDrawArrays(GL_LINE_LOOP, 0, 4);

	glBindVertexArray(0);



	/********** Circle **********/
	for (int i = 1; i <= 2; i++) {
		translateMatrix = mat4::identity();
		modelViewMatrix = mat4::identity();
		modelViewProjectionMatrix = mat4::identity();

		translateMatrix = translate(0.0f, 0.0f, -3.0f);
		modelViewMatrix = modelViewMatrix * translateMatrix;
		modelViewProjectionMatrix = gPerspectiveProjectionMatrix * modelViewMatrix;

		glUniformMatrix4fv(mvpUniform,
			1,
			GL_FALSE,
			modelViewProjectionMatrix);

		if (i == 1) {
			FillCircle_Position(Circle_Position, Circle_Color, 1);
		}
		else if (i == 2) {
			FillCircle_Position(Circle_Position, Circle_Color, 2);
		}


		glPointSize(1.500f);
		glBindVertexArray(vao_Circle);

		glBindBuffer(GL_ARRAY_BUFFER, vbo_Circle_Position);

		glBufferData(GL_ARRAY_BUFFER,
			sizeof(Circle_Position),
			Circle_Position,
			GL_DYNAMIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, 0);


		glBindBuffer(GL_ARRAY_BUFFER, vbo_Circle_Color);

		glBufferData(GL_ARRAY_BUFFER,
			sizeof(Circle_Color),
			Circle_Color,
			GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glDrawArrays(GL_POINTS, 0, 3000);

		glBindVertexArray(0);

	}
	glUseProgram(0);



	SwapBuffers(ghdc);
}


void FillGridPosition(GLfloat arr[]) {

	GLint j = 0;


	//Vertical Lines 20

	for (GLfloat i = 0.1f; i < 1.1f; i = i + 0.1f) {

		//1
		arr[j] = 0.0f + i;
		arr[j + 1] = 1.0f;
		arr[j + 2] = 0.0f;

		arr[j + 3] = 0.0f + i;
		arr[j + 4] = -1.0f;
		arr[j + 5] = 0.0f;

		//2
		arr[j + 6] = 0.0f - i;
		arr[j + 7] = 1.0f;
		arr[j + 8] = 0.0f;

		arr[j + 9] = 0.0f - i;
		arr[j + 10] = -1.0f;
		arr[j + 11] = 0.0f;

		j = j + 12;
		fprintf(gbFile, "j: %d\n", j);
	}


	//Horizontal Lines 20 
	fprintf(gbFile, "               j: %d\n", j);
	for (GLfloat i = 0.1f; i < 1.1f; i = i + 0.1f) {
		//1
		arr[j] = -1.0f;
		arr[j + 1] = 0.0f + i;
		arr[j + 2] = 0.0f;

		arr[j + 3] = 1.0f;
		arr[j + 4] = 0.0f + i;
		arr[j + 5] = 0.0f;

		//2
		arr[j + 6] = -1.0f;
		arr[j + 7] = 0.0f - i;
		arr[j + 8] = 0.0f;

		arr[j + 9] = 1.0f;
		arr[j + 10] = 0.0f - i;
		arr[j + 11] = 0.0f;

		j = j + 12;
		fprintf(gbFile, "j: %d\n", j);
	}
}

void Calculation(GLfloat arr[]) {
	GLfloat a, b, c;
	GLfloat s;

	//Distance Formula
	a = (GLfloat)sqrt(pow((arr[6] - arr[3]), 2) + pow((arr[7] - arr[4]), 2));
	b = (GLfloat)sqrt(pow((arr[6] - arr[0]), 2) + pow((arr[7] - arr[1]), 2));
	c = (GLfloat)sqrt(pow((arr[3] - arr[0]), 2) + pow((arr[4] - arr[1]), 2));

	s = (a + b + c) / 2;

	Incircle_Radius = (GLfloat)(sqrt(s * (s - a) * (s - b) * (s - c)) / s);

	Incircle_Center[0] = (a * arr[0] + b * arr[3] + c * arr[6]) / (a + b + c);
	Incircle_Center[1] = (a * arr[1] + b * arr[4] + c * arr[7]) / (a + b + c);
	Incircle_Center[2] = 0.0f;


	fprintf(gbFile, "Incircle_Radius: %f\n", Incircle_Radius);
	fprintf(gbFile, "InCenter x: %f      y: %f      z: %f     \n", Incircle_Center[0], Incircle_Center[1], Incircle_Center[2]);

}


void FillCircle_Position(GLfloat arr[], GLfloat arrColor[], int iFlag) {

	memset(arr, 0, sizeof(GLfloat) * 3000 * 3);

	if (iFlag == 1) {
		//InCircle
		int i = 0;
		for (int i = 0; i < 3000; i = i + 3) {
			GLfloat x = (GLfloat)(2.0f * M_PI * i / 3000);
			arr[i] = (GLfloat)(Incircle_Radius * cos(x)) + Incircle_Center[0];
			arr[i + 1] = (GLfloat)(Incircle_Radius * sin(x)) + Incircle_Center[1];
			arr[i + 2] = 0.0f;


			arrColor[i] = 1.0f;		//R
			arrColor[i + 1] = 1.0f;		//G
			arrColor[i + 2] = 0.0f;		//B
		}
		//fprintf(gbFile, "i: %d\n", i);

	}
	else {
		//Outer Circle
		for (int i = 0; i < 3000; i = i + 3) {
			GLfloat x = (GLfloat)(2.0f * M_PI * i / 3000);
			arr[i] = (GLfloat)(1.0f * cos(x));
			arr[i + 1] = (GLfloat)(1.0f * sin(x));
			arr[i + 2] = 0.0f;

			arrColor[i] = 1.0f;		//R
			arrColor[i + 1] = 1.0f;		//G
			arrColor[i + 2] = 0.0f; 	//B

			//fprintf(gbFile, "x: %f    y: %f    z: %f \n", arr[i], arr[i + 1], arr[i + 2]);
		}
	}

}



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


//For I
GLuint vao_I;
GLuint vbo_I_Position;
GLuint vbo_I_Color;

//For N
GLuint vao_N;
GLuint vbo_N_Position;
GLuint vbo_N_Color;

//For D
GLuint vao_D;
GLuint vbo_D_Position;
GLuint vbo_D_Color;
GLfloat D_Color[] = {
	1.0f, 0.6f, 0.2f, 0.0f,
	0.0705f, 0.533f, 0.0274f, 0.0f,

	1.0f, 0.6f, 0.2f, 0.0f,
	1.0f, 0.6f, 0.2f, 0.0f,

	0.0705f, 0.533f, 0.0274f, 0.0f,
	0.0705f, 0.533f, 0.0274f, 0.0f,

	1.0f, 0.6f, 0.2f, 0.0f,
	0.0705f, 0.533f, 0.0274f, 0.0f
};

GLfloat fD_Fading = 0.0f;



//For A
GLuint vao_A;
GLuint vbo_A_Position;
GLuint vbo_A_Color;

//For V A used in INDIA is Without - therfore V verticaly inverted
GLuint vao_V;
GLuint vbo_V_Position;
GLuint vbo_V_Color;

//For F
GLuint vao_F;
GLuint vbo_F_Position;
GLuint vbo_F_Color;

//For Flag
GLuint vao_Flag;
GLuint vbo_Flag_Position;
GLuint vbo_Flag_Color;

//For Plane's Triangle Part
GLuint vao_Plane_Triangle;
GLuint vbo_Plane_Triangle_Position;
GLuint vbo_Plane_Triangle_Color;

//For Plane's Rectangle Part
GLuint vao_Plane_Rect;
GLuint vbo_Plane_Rect_Position;
GLuint vbo_Plane_Rect_Color;

//For Plane's Polygon Part
GLuint vao_Plane_Polygon;
GLuint vbo_Plane_Polygon_Position;
GLuint vbo_Plane_Polygon_Color;

//For Fading Flag
GLuint vao_Fading_Flag;
GLuint vbo_Fading_Flag_Position;
GLuint vbo_Fading_Flag_Color;


//For Plane Movement and Translation
#define NOT_REACH 0
#define HALF_WAY 1
#define REACH 2
#define END 3

GLfloat Plane1_Count = 1000.0f;
GLfloat Plane2_Count = 1000.0f;
GLfloat Plane3_Count = 1000.0f;

int bPlane1Reached = NOT_REACH;
int bPlane2Reached = NOT_REACH;
int bPlane3Reached = NOT_REACH;
int iFadingFlag1 = 0;
int iFadingFlag2 = 0;
int iFadingFlag3 = 0;


//For Sequence
GLuint iSequence = 1;



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
	TCHAR szName[] = TEXT("RohitRJadhav-PP-DynamicIndia");

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
		TEXT("RohitRJadhav-PP-DynamicIndia"),
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


	/********** Position And Color **********/
	GLfloat I_Position[] = {
		-0.3f, 1.0f, 0.0f,
		0.3f, 1.0f, 0.0f,

		0.0f, 1.0f, 0.0f,
		0.0f, -1.0f, 0.0f,

		-0.3f, -1.0f, 0.0f,
		0.3f, -1.0f, 0.0f
	};


	GLfloat I_Color[] = {
		1.0f, 0.6f, 0.2f,
		1.0f, 0.6f, 0.2f,

		1.0f, 0.6f, 0.2f,
		0.0705f, 0.533f, 0.0274f,

		0.0705f, 0.533f, 0.0274f,
		0.0705f, 0.533f, 0.0274f
	};

	GLfloat N_Position[] = {
		0.0f, 1.06f, 0.0f,
		0.0f, -1.06f, 0.0f,

		0.75f, 1.06f, 0.0f,
		0.75f, -1.06f, 0.0f,

		0.0f, 1.06f, 0.0f,
		0.75f, -1.06f, 0.0f
	};


	GLfloat N_Color[] = {
		1.0f, 0.6f, 0.2f,
		0.0705f, 0.533f, 0.0274f,

		1.0f, 0.6f, 0.2f,
		0.0705f, 0.533f, 0.0274f,

		1.0f, 0.6f, 0.2f,
		0.0705f, 0.533f, 0.0274f
	};

	GLfloat D_Position[] = {
		0.0f, 1.0f, 0.0f,
		0.0f, -1.0f, 0.0f,

		-0.1f, 1.0f, 0.0f,
		0.6f, 1.0f, 0.0f,

		-0.1f, -1.0f, 0.0f,
		0.6f, -1.0f, 0.0f,

		0.6f, 1.0f, 0.0f,
		0.6f, -1.0f, 0.0f
	};

	GLfloat A_Position[] = {
		0.0f, 1.06f, 0.0f,
		-0.5f, -1.06f, 0.0f,

		0.0f, 1.06f, 0.0f,
		0.5f, -1.06f, 0.0f,

		-0.250f, 0.0f, 0.0f,
		0.25f, 0.0f, 0.0f
	};


	GLfloat A_Color[] = {
		1.0f, 0.6f, 0.2f,
		0.0705f, 0.533f, 0.0274f,

		1.0f, 0.6f, 0.2f,
		0.0705f, 0.533f, 0.0274f,

		1.0f, 0.6f, 0.2f,
		0.0705f, 0.533f, 0.0274f
	};


	//For Inverted A
	GLfloat V_Position[] = {
		0.0f, 1.06f, 0.0f,
		-0.5f, -1.06f, 0.0f,

		0.0f, 1.06f, 0.0f,
		0.5f, -1.06f, 0.0f
	};


	GLfloat V_Color[] = {
		1.0f, 0.6f, 0.2f,
		0.0705f, 0.533f, 0.0274f,

		1.0f, 0.6f, 0.2f,
		0.0705f, 0.533f, 0.0274f
	};


	GLfloat F_Position[] = {
		0.10f, 1.0f, 0.0f,
		0.10f, -1.0f, 0.0f,

		0.00f, 1.0f, 0.0f,
		0.90f, 1.0f, 0.0f,

		0.10f, 0.1f, 0.0f,
		0.80f, 0.1f, 0.0f
	};

	GLfloat F_Color[] = {
		1.0f, 0.6f, 0.2f,
		0.0705f, 0.533f, 0.0274f,

		1.0f, 0.6f, 0.2f,
		1.0f, 0.6f, 0.2f,

		0.0705f, 0.533f, 0.0274f,
		0.0705f, 0.533f, 0.0274f
	};


	GLfloat Flag_Position[] = {
		-0.207f, 0.1f, 0.0f,
		0.207f, 0.1f, 0.0f,

		-0.218f, 0.0f, 0.0f,
		0.219f, 0.0f, 0.0f,

		-0.239f, -0.1f, 0.0f,
		0.239f, -0.1f, 0.0f
	};


	GLfloat Flag_Color[] = {
		0.0f, 0.0f, 0.0f,
		1.0f, 0.6f, 0.2f,

		0.0f, 0.0f, 0.0f,
		1.0f, 1.0f, 1.0f,

		0.0f, 0.0f, 0.0f,
		0.0705f, 0.533f, 0.0274f
	};


	GLfloat Plane_Triangle_Position[] = {
		//Front
		5.0f, 0.0f, 0.0f,
		2.50f, 0.65f, 0.0f,
		2.50f, -0.65f, 0.0f
	};

	GLfloat Plane_Triangle_Color[] = {
		//Front
		0.7294f, 0.8862f, 0.9333f,	//Power Blue
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f
	};


	GLfloat Plane_Rect_Position[] = {
		//Middle
		2.50f, 0.65f, 0.0f,
		-2.50f, 0.65f, 0.0f,
		-2.50f, -0.65f, 0.0f,
		2.50f, -0.65f, 0.0f,

		//Upper_Fin
		0.75f, 0.65f, 0.0f,
		-1.20f, 2.5f, 0.0f,
		-2.50f, 2.5f, 0.0f,
		-2.0f, 0.65f, 0.0f,

		//Lower_Fin
		0.75f, -0.65f, 0.0f,
		-1.20f, -2.50f, 0.0f,
		-2.50f, -2.50f, 0.0f,
		-2.0f, -0.65f, 0.0f,

		//Back
		-2.50f, 0.65f, 0.0f,
		-3.0f, 0.75f, 0.0f,
		-3.0f, -0.75f, 0.0f,
		-2.5f, -0.65f, 0.0f
	};


	GLfloat Plane_Rect_Color[] = {
		//Middle
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,

		//Upper_Fin
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,

		//Lower_Fin
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,

		//Back
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f
	};

	GLfloat Plane_Polygon_Position[] = {
		//Upper Tail
		-3.0f, 0.75f, 0.0f,
		-3.90f, 1.5f, 0.0f,
		-4.5f, 1.5f, 0.0f,
		-4.0f, 0.0f, 0.0f,
		-3.0f, 0.0f, 0.0f,

		//Lower Tail
		-3.0f, -0.75f, 0.0f,
		-3.90f, -1.5f, 0.0f,
		-4.5f, -1.5f, 0.0f,
		-4.0f, 0.0f, 0.0f,
		-3.0f, 0.0f, 0.0f
	};

	GLfloat Plane_Polygon_Color[] = {
		//Upper Tail
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,

		//Lower Tail
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f,
		0.7294f, 0.8862f, 0.9333f
	};


	GLfloat Fading_Flag_Color[] = {
		0.0f, 0.0f, 0.0f,
		1.0f, 0.6f, 0.2f,

		0.0f, 0.0f, 0.0f,
		1.0f, 1.0f, 1.0f,

		0.0f, 0.0f, 0.0f,
		0.0705f, 0.533f, 0.0274f
	};

	/********** I **********/
	glGenVertexArrays(1, &vao_I);
	glBindVertexArray(vao_I);

	/********** Position **********/
	glGenBuffers(1, &vbo_I_Position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_I_Position);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(I_Position),
		I_Position,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	/********** Color **********/
	glGenBuffers(1, &vbo_I_Color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_I_Color);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(I_Color),
		I_Color,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);



	/********** N **********/
	glGenVertexArrays(1, &vao_N);
	glBindVertexArray(vao_N);

	/********** Position **********/
	glGenBuffers(1, &vbo_N_Position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_N_Position);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(N_Position),
		N_Position,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	/********** Color **********/
	glGenBuffers(1, &vbo_N_Color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_N_Color);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(N_Color),
		N_Color,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);



	/********** D **********/
	glGenVertexArrays(1, &vao_D);
	glBindVertexArray(vao_D);

	/********** Position **********/
	glGenBuffers(1, &vbo_D_Position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_D_Position);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(D_Position),
		D_Position,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	/********** Color **********/
	glGenBuffers(1, &vbo_D_Color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_D_Color);
	glBufferData(GL_ARRAY_BUFFER,
		8 * 4 * sizeof(GLfloat),
		NULL,
		GL_DYNAMIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR,
		4,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);



	/********** A **********/
	glGenVertexArrays(1, &vao_A);
	glBindVertexArray(vao_A);

	/********** Position **********/
	glGenBuffers(1, &vbo_A_Position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_A_Position);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(A_Position),
		A_Position,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	/********** Color **********/
	glGenBuffers(1, &vbo_A_Color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_A_Color);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(A_Color),
		A_Color,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);


	/********** V **********/
	glGenVertexArrays(1, &vao_V);
	glBindVertexArray(vao_V);

	/********** Position **********/
	glGenBuffers(1, &vbo_V_Position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_V_Position);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(V_Position),
		V_Position,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	/********** Color **********/
	glGenBuffers(1, &vbo_V_Color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_V_Color);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(V_Color),
		V_Color,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);


	/********** F **********/
	glGenVertexArrays(1, &vao_F);
	glBindVertexArray(vao_F);

	/********** Position **********/
	glGenBuffers(1, &vbo_F_Position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_F_Position);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(F_Position),
		F_Position,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	/********** Color **********/
	glGenBuffers(1, &vbo_F_Color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_F_Color);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(F_Color),
		F_Color,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);


	/********** Flag **********/
	glGenVertexArrays(1, &vao_Flag);
	glBindVertexArray(vao_Flag);

	/********** Position **********/
	glGenBuffers(1, &vbo_Flag_Position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Flag_Position);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(Flag_Position),
		Flag_Position,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	/********** Color **********/
	glGenBuffers(1, &vbo_Flag_Color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Flag_Color);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(Flag_Color),
		Flag_Color,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);



	/********** Plane's Triangle Part **********/
	glGenVertexArrays(1, &vao_Plane_Triangle);
	glBindVertexArray(vao_Plane_Triangle);

	/********** Position **********/
	glGenBuffers(1, &vbo_Plane_Triangle_Position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Plane_Triangle_Position);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(Plane_Triangle_Position),
		Plane_Triangle_Position,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/********** Color **********/
	glGenBuffers(1, &vbo_Plane_Triangle_Color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Plane_Triangle_Color);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(Plane_Triangle_Color),
		Plane_Triangle_Color,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);



	/********** Plane's Rectangle Part **********/
	glGenVertexArrays(1, &vao_Plane_Rect);
	glBindVertexArray(vao_Plane_Rect);

	/********** Position **********/
	glGenBuffers(1, &vbo_Plane_Rect_Position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Plane_Rect_Position);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(Plane_Rect_Position),
		Plane_Rect_Position,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/********** Color **********/
	glGenBuffers(1, &vbo_Plane_Rect_Color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Plane_Rect_Color);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(Plane_Rect_Color),
		Plane_Rect_Color,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);




	/********** Plane's Polygon Part **********/
	glGenVertexArrays(1, &vao_Plane_Polygon);
	glBindVertexArray(vao_Plane_Polygon);

	/********** Position **********/
	glGenBuffers(1, &vbo_Plane_Polygon_Position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Plane_Polygon_Position);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(Plane_Polygon_Position),
		Plane_Polygon_Position,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/********** Color **********/
	glGenBuffers(1, &vbo_Plane_Polygon_Color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Plane_Polygon_Color);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(Plane_Polygon_Color),
		Plane_Polygon_Color,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);



	/********** Fading Flag **********/
	glGenVertexArrays(1, &vao_Fading_Flag);
	glBindVertexArray(vao_Fading_Flag);

	/********** Position **********/
	glGenBuffers(1, &vbo_Fading_Flag_Position);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Fading_Flag_Position);
	glBufferData(GL_ARRAY_BUFFER,
		6 * 3 * sizeof(GLfloat),
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
	glGenBuffers(1, &vbo_Fading_Flag_Color);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_Fading_Flag_Color);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(Fading_Flag_Color),
		Fading_Flag_Color,
		GL_STATIC_DRAW);

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

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	gPerspectiveProjectionMatrix = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);
	return(0);
}

void uninitialize(void) {



	//Fading Flag
	if (vbo_Fading_Flag_Color) {
		glDeleteBuffers(1, &vbo_Fading_Flag_Color);
		vbo_Fading_Flag_Color = 0;
	}

	if (vbo_Fading_Flag_Position) {
		glDeleteBuffers(1, &vbo_Fading_Flag_Position);
		vbo_Fading_Flag_Position = 0;
	}

	if (vao_Fading_Flag) {
		glDeleteVertexArrays(1, &vao_Fading_Flag);
		vao_Fading_Flag = 0;
	}



	//Plane Polygon Part
	if (vbo_Plane_Polygon_Color) {
		glDeleteBuffers(1, &vbo_Plane_Polygon_Color);
		vbo_Plane_Polygon_Color = 0;
	}

	if (vbo_Plane_Polygon_Position) {
		glDeleteBuffers(1, &vbo_Plane_Polygon_Position);
		vbo_Plane_Polygon_Position = 0;
	}

	if (vao_Plane_Polygon) {
		glDeleteVertexArrays(1, &vao_Plane_Polygon);
		vao_Plane_Polygon = 0;
	}

	//Plane Rectangle Part
	if (vbo_Plane_Rect_Color) {
		glDeleteBuffers(1, &vbo_Plane_Rect_Color);
		vbo_Plane_Rect_Color = 0;
	}

	if (vbo_Plane_Rect_Position) {
		glDeleteBuffers(1, &vbo_Plane_Rect_Position);
		vbo_Plane_Rect_Position = 0;
	}

	if (vao_Plane_Rect) {
		glDeleteVertexArrays(1, &vao_Plane_Rect);
		vao_Plane_Rect = 0;
	}

	//Plane Triangle Part
	if (vbo_Plane_Triangle_Color) {
		glDeleteBuffers(1, &vbo_Plane_Triangle_Color);
		vbo_Plane_Triangle_Color = 0;
	}

	if (vbo_Plane_Triangle_Position) {
		glDeleteBuffers(1, &vbo_Plane_Triangle_Position);
		vbo_Plane_Triangle_Position = 0;
	}

	if (vao_Plane_Triangle) {
		glDeleteVertexArrays(1, &vao_Plane_Triangle);
		vao_Plane_Triangle = 0;
	}



	//Flag
	if (vbo_Flag_Color) {
		glDeleteBuffers(1, &vbo_Flag_Color);
		vbo_Flag_Color = 0;
	}

	if (vbo_Flag_Position) {
		glDeleteBuffers(1, &vbo_Flag_Position);
		vbo_Flag_Position = 0;
	}

	if (vao_Flag) {
		glDeleteVertexArrays(1, &vao_Flag);
		vao_Flag = 0;
	}

	//F
	if (vbo_F_Color) {
		glDeleteBuffers(1, &vbo_F_Color);
		vbo_F_Color = 0;
	}

	if (vbo_F_Position) {
		glDeleteBuffers(1, &vbo_F_Position);
		vbo_F_Position = 0;
	}

	if (vao_F) {
		glDeleteVertexArrays(1, &vao_F);
		vao_F = 0;
	}


	//V
	if (vbo_V_Color) {
		glDeleteBuffers(1, &vbo_V_Color);
		vbo_V_Color = 0;
	}

	if (vbo_V_Position) {
		glDeleteBuffers(1, &vbo_V_Position);
		vbo_V_Position = 0;
	}

	if (vao_V) {
		glDeleteVertexArrays(1, &vao_V);
		vao_V = 0;
	}

	//A
	if (vbo_A_Color) {
		glDeleteBuffers(1, &vbo_A_Color);
		vbo_A_Color = 0;
	}

	if (vbo_A_Position) {
		glDeleteBuffers(1, &vbo_A_Position);
		vbo_A_Position = 0;
	}

	if (vao_A) {
		glDeleteVertexArrays(1, &vao_A);
		vao_A = 0;
	}

	//D
	if (vbo_D_Color) {
		glDeleteBuffers(1, &vbo_D_Color);
		vbo_D_Color = 0;
	}

	if (vbo_D_Position) {
		glDeleteBuffers(1, &vbo_D_Position);
		vbo_D_Position = 0;
	}

	if (vao_D) {
		glDeleteVertexArrays(1, &vao_D);
		vao_D = 0;
	}

	//N
	if (vbo_N_Color) {
		glDeleteBuffers(1, &vbo_N_Color);
		vbo_N_Color = 0;
	}

	if (vbo_N_Position) {
		glDeleteBuffers(1, &vbo_N_Position);
		vbo_N_Position = 0;
	}

	if (vao_N) {
		glDeleteVertexArrays(1, &vao_N);
		vao_N = 0;
	}

	//I
	if (vbo_I_Color) {
		glDeleteBuffers(1, &vbo_I_Color);
		vbo_I_Color = 0;
	}

	if (vbo_I_Position) {
		glDeleteBuffers(1, &vbo_I_Position);
		vbo_I_Position = 0;
	}

	if (vao_I) {
		glDeleteVertexArrays(1, &vao_I);
		vao_I = 0;
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



mat4 translateMatrix;
mat4 scaleMatrix;
mat4 rotateMatrix;
mat4 modelViewMatrix;
mat4 modelViewProjectionMatrix;


void display(void) {


	void My_Flag(GLfloat, GLfloat, GLfloat, GLfloat);
	void My_Letters(char, GLfloat, GLfloat, GLfloat, GLfloat);
	void My_Plane(GLfloat, GLfloat, GLfloat, GLfloat, GLfloat, GLfloat, GLfloat);
	void My_Fading_Flag(GLfloat, GLfloat, GLfloat, GLfloat);
	void My_D(GLfloat, GLfloat, GLfloat, GLfloat);

	//For India
	static GLfloat fXTranslation = 0.0f;
	static GLfloat fYTranslation = 0.0f;

	//For Plane
	static GLfloat angle_Plane1 = (GLfloat)(M_PI);
	static GLfloat angle_Plane3 = (GLfloat)(M_PI);

	static GLfloat XTrans_Plane1 = 0.0f;
	static GLfloat YTrans_Plane1 = 0.0f;

	static GLfloat XTrans_Plane2 = 0.0f;

	static GLfloat XTrans_Plane3 = 0.0f;
	static GLfloat YTrans_Plane3 = 0.0f;

	static GLfloat ZRot_Plane1 = -60.0f;
	static GLfloat ZRot_Plane3 = 60.0f;



	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	glUseProgram(gShaderProgramObject);

	switch (iSequence) {
	case 1:

		My_Letters('I', -7.50f + fXTranslation, 0.0f, -8.0f, 20.0f);
		fXTranslation = fXTranslation + 0.015f;;
		if ((-7.5f + fXTranslation) >= -2.0f) {
			fXTranslation = 0.0f;
			iSequence = 2;
		}
		break;

	case 2:
		My_Letters('I', -2.0f, 0.0f, -8.0f, 20.0f);
		My_Letters('V', 8.50f - fXTranslation, 0.0f, -8.0f, 20.0f);
		fXTranslation = fXTranslation + 0.015f;;
		if ((8.5f - fXTranslation) <= 2.0f) {
			fXTranslation = 0.0f;
			iSequence = 3;
		}
		break;

	case 3:
		My_Letters('I', -2.0f, 0.0f, -8.0f, 20.0f);
		My_Letters('V', 2.0f, 0.0f, -8.0f, 20.0f);
		My_Letters('N', -1.35f, (6.0f - fYTranslation), -8.0f, 20.0f);
		fYTranslation = fYTranslation + 0.015f;
		if ((6.0f - fYTranslation) < 0.0f) {
			fYTranslation = 0.0f;
			iSequence = 4;
		}
		break;

	case 4:
		My_Letters('I', -2.0f, 0.0f, -8.0f, 20.0f);
		My_Letters('V', 2.0f, 0.0f, -8.0f, 20.0f);
		My_Letters('N', -1.35f, 0.0f, -8.0f, 20.0f);
		My_Letters('I', 1.02f, (-5.0f + fYTranslation), -8.0f, 20.0f);
		fYTranslation = fYTranslation + 0.015f;
		if ((-5.0f + fYTranslation) > 0.0f) {
			fYTranslation = 0.0f;
			iSequence = 5;
		}
		break;

	case 5:

		D_Color[3] = fD_Fading;
		D_Color[7] = fD_Fading;
		D_Color[11] = fD_Fading;
		D_Color[15] = fD_Fading;
		D_Color[19] = fD_Fading;
		D_Color[23] = fD_Fading;
		D_Color[27] = fD_Fading;
		D_Color[31] = fD_Fading;

		My_Letters('I', -2.0f, 0.0f, -8.0f, 20.0f);
		My_Letters('N', -1.35f, 0.0f, -8.0f, 20.0f);
		My_D(-0.15f, 0.0f, -8.0f, 20.0f);
		My_Letters('I', 1.02f, 0.0f, -8.0f, 20.0f);
		My_Letters('V', 2.0f, 0.0f, -8.0f, 20.0f);

		if (fD_Fading > 1.0f) {
			iSequence = 6;
		}
		else
			fD_Fading = fD_Fading + 0.001f;
		break;


	case 6:
		My_Letters('I', -2.0f, 0.0f, -8.0f, 20.0f);
		My_Letters('N', -1.35f, 0.0f, -8.0f, 20.0f);
		//My_Letters('D', -0.15f, 0.0f, -8.0f, 20.0f);
		My_D(-0.15f, 0.0f, -8.0f, 20.0f);
		My_Letters('I', 1.02f, 0.0f, -8.0f, 20.0f);
		My_Letters('V', 2.0f, 0.0f, -8.0f, 20.0f);








		/********** Plane 1 **********/
		if (bPlane1Reached == NOT_REACH) {
			XTrans_Plane1 = (GLfloat)((3.2 * cos(angle_Plane1)) + (-2.5f));
			YTrans_Plane1 = (GLfloat)((4.0f * sin(angle_Plane1)) + (4.0f));
			angle_Plane1 = angle_Plane1 + 0.005f;
			ZRot_Plane1 = ZRot_Plane1 + 0.2f;


			if (angle_Plane1 >= (3.0f * M_PI) / 2.0f) {
				bPlane1Reached = HALF_WAY;
				YTrans_Plane1 = 0.00f;

			}
			else if (ZRot_Plane1 >= 0.0)
				ZRot_Plane1 = 0.0f;

		}
		else if (bPlane1Reached == HALF_WAY) {
			XTrans_Plane1 = XTrans_Plane1 + 0.010f;
			YTrans_Plane1 = 0.00f;

			if (XTrans_Plane1 >= 3.00f) {	//2.6
				bPlane1Reached = REACH;
				angle_Plane1 = (GLfloat)(3.0f * M_PI) / 2.0f;
				ZRot_Plane1 = 0.0f;
			}
		}
		else if (bPlane1Reached == REACH) {

			if (Plane1_Count <= 0.0f) {
				iFadingFlag1 = 2;
				XTrans_Plane1 = (GLfloat)((3.0f * cos(angle_Plane1)) + (3.0f));		//2.6
				YTrans_Plane1 = (GLfloat)((4.0f * sin(angle_Plane1)) + (4.0f));

				if (XTrans_Plane1 >= 6.00f || YTrans_Plane1 >= 4.0f)
					bPlane1Reached = END;

				angle_Plane1 = angle_Plane1 + 0.005f;
				ZRot_Plane1 = ZRot_Plane1 + 0.2f;
			}
			else
				iFadingFlag1 = 1;

			Plane1_Count = Plane1_Count - 1.0f;
		}
		else if (bPlane1Reached == END) {
			angle_Plane1 = 0.0f;
			ZRot_Plane1 = 0.0f;
		}

		/*********** Fading Flag ***********/
		if (bPlane1Reached == NOT_REACH)
			My_Fading_Flag(XTrans_Plane1, YTrans_Plane1, -8.0f, ZRot_Plane1);

		My_Plane(XTrans_Plane1, YTrans_Plane1, -8.0f, 0.18f, 0.18f, 0.0f, ZRot_Plane1);










		/********** Plane 2 **********/
		if (bPlane2Reached == NOT_REACH) {
			if ((-6.0f + XTrans_Plane2) > -2.50f) {
				bPlane2Reached = HALF_WAY;
			}
			else
				XTrans_Plane2 = XTrans_Plane2 + 0.011f;

		}
		else if (bPlane2Reached == HALF_WAY) {
			XTrans_Plane2 = XTrans_Plane2 + 0.010f;
			if ((-6.0f + XTrans_Plane2) >= 3.0f) {	//2.6
				bPlane2Reached = REACH;
			}
		}
		else if (bPlane2Reached == REACH) {
			if (Plane2_Count <= 0.00f) {
				iFadingFlag2 = 2;
				XTrans_Plane2 = XTrans_Plane2 + 0.010f;
			}
			else
				iFadingFlag2 = 1;


			if ((-6.0f + XTrans_Plane2) >= 8.0f)
				bPlane2Reached = END;


			Plane2_Count = Plane2_Count - 1.0f;
		}
		else if (bPlane2Reached == END) {
			XTrans_Plane2 = 14.0f;
		}

		/*********** Fading_Flag **********/
		if (iFadingFlag2 < 2)
			My_Fading_Flag((-6.0f + XTrans_Plane2), 0.0f, -8.0f, 0.0f);

		My_Plane((-6.0f + XTrans_Plane2), 0.0f, -8.0f, 0.18f, 0.18f, 0.0f, 0.0f);







		/********** Plane 3 **********/
		if (bPlane3Reached == NOT_REACH) {
			XTrans_Plane3 = (GLfloat)((3.2 * cos(angle_Plane3)) + (-2.5f));
			YTrans_Plane3 = (GLfloat)((4.0f * sin(angle_Plane3)) + (-4.0f));
			angle_Plane3 = angle_Plane3 - 0.005f;
			ZRot_Plane3 = ZRot_Plane3 - 0.2f;


			if (angle_Plane3 < (M_PI) / 2.0f) {
				bPlane3Reached = HALF_WAY;
				YTrans_Plane3 = 0.00f;

			}
			else if (ZRot_Plane3 < 0.0)
				ZRot_Plane3 = 0.0f;

		}
		else if (bPlane3Reached == HALF_WAY) {
			XTrans_Plane3 = XTrans_Plane3 + 0.010f;
			YTrans_Plane3 = 0.00f;

			if (XTrans_Plane3 >= 3.00f) {	//2.6
				bPlane3Reached = REACH;
				angle_Plane3 = (GLfloat)(M_PI) / 2.0f;
				ZRot_Plane3 = 0.0f;
			}
		}
		else if (bPlane3Reached == REACH) {

			if (Plane3_Count <= 0.0f) {
				iFadingFlag3 = 2;
				XTrans_Plane3 = (GLfloat)((3.0f * cos(angle_Plane3)) + (3.0f));		//2.6
				YTrans_Plane3 = (GLfloat)((4.0f * sin(angle_Plane3)) + (-4.0f));

				if (XTrans_Plane3 >= 6.00f || YTrans_Plane3 < -4.0f)
					bPlane3Reached = END;

				angle_Plane3 = angle_Plane3 - 0.005f;
				ZRot_Plane3 = ZRot_Plane3 - 0.2f;
			}
			else
				iFadingFlag3 = 1;

			Plane3_Count = Plane3_Count - 1.0f;
		}
		else if (bPlane3Reached == END) {
			angle_Plane3 = 0.0f;
			ZRot_Plane3 = 0.0f;
		}



		/*********** Fading Flag ***********/
		if (bPlane2Reached == NOT_REACH)
			My_Fading_Flag(XTrans_Plane3, YTrans_Plane3, -8.0f, ZRot_Plane3);


		My_Plane(XTrans_Plane3, YTrans_Plane3, -8.0f, 0.18f, 0.18f, 0.0f, ZRot_Plane3);


		if (iFadingFlag1 == 2 || iFadingFlag2 == 2 || iFadingFlag3 == 2)
			My_Flag(2.0f, 0.0f, -8.0f, 30.0f);


		break;

	}


	glUseProgram(0);

	SwapBuffers(ghdc);
}



void My_Letters(char c, GLfloat x, GLfloat y, GLfloat z, GLfloat fWidth) {

	translateMatrix = mat4::identity();
	modelViewMatrix = mat4::identity();
	modelViewProjectionMatrix = mat4::identity();

	translateMatrix = translate(x, y, z);
	modelViewMatrix = modelViewMatrix * translateMatrix;
	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(mvpUniform,
		1,
		GL_FALSE,
		modelViewProjectionMatrix);

	glLineWidth(fWidth);

	switch (c) {
	case 'I':
		glBindVertexArray(vao_I);
		glDrawArrays(GL_LINES, 0, 6 * 3);
		glBindVertexArray(0);
		break;

	case 'N':
		glBindVertexArray(vao_N);
		glDrawArrays(GL_LINES, 0, 6 * 3);
		glBindVertexArray(0);
		break;


	case 'A':
		glBindVertexArray(vao_A);
		glDrawArrays(GL_LINES, 0, 4 * 3);
		glBindVertexArray(0);
		break;

	case 'V':
		glBindVertexArray(vao_V);
		glDrawArrays(GL_LINES, 0, 6 * 3);
		glBindVertexArray(0);
		break;


	}


}

void My_D(GLfloat x, GLfloat y, GLfloat z, GLfloat fWidth) {

	translateMatrix = mat4::identity();
	modelViewMatrix = mat4::identity();
	modelViewProjectionMatrix = mat4::identity();

	translateMatrix = translate(x, y, z);
	modelViewMatrix = modelViewMatrix * translateMatrix;
	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(mvpUniform,
		1,
		GL_FALSE,
		modelViewProjectionMatrix);

	glLineWidth(fWidth);

	glBindVertexArray(vao_D);

	glBindBuffer(GL_ARRAY_BUFFER, vbo_D_Color);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(D_Color),
		D_Color,
		GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDrawArrays(GL_LINES, 0, 8 * 3);
	glBindVertexArray(0);

}

void My_Fading_Flag(GLfloat x, GLfloat y, GLfloat z, GLfloat fAngle) {


	static GLfloat Fading_Flag_Position[] = {
		-1.0f, 0.1f, 0.0f,
		-0.50f, 0.1f, 0.0f,

		-1.0f, 0.0f, 0.0f,
		-0.50f, 0.0f, 0.0f,

		-1.0f, -0.1f, 0.0f,
		-0.50f, -0.1f, 0.0f
	};

	if (bPlane2Reached != REACH) {
		Fading_Flag_Position[0] -= 0.005f;
		Fading_Flag_Position[6] -= 0.005f;
		Fading_Flag_Position[12] -= 0.005f;

	}
	else if (bPlane2Reached == REACH) {

		Fading_Flag_Position[0] += 0.007f;
		Fading_Flag_Position[6] += 0.007f;
		Fading_Flag_Position[12] += 0.007f;
	}

	translateMatrix = mat4::identity();
	rotateMatrix = mat4::identity();
	modelViewMatrix = mat4::identity();
	modelViewProjectionMatrix = mat4::identity();

	translateMatrix = translate(x, y, z);
	rotateMatrix = rotate(0.0f, 0.0f, fAngle);
	modelViewMatrix = modelViewMatrix * translateMatrix * rotateMatrix;
	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(mvpUniform,
		1,
		GL_FALSE,
		modelViewProjectionMatrix);

	glLineWidth(30.0f);

	glBindVertexArray(vao_Fading_Flag);

	glBindBuffer(GL_ARRAY_BUFFER, vbo_Fading_Flag_Position);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(Fading_Flag_Position),
		Fading_Flag_Position,
		GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDrawArrays(GL_LINES, 0, 6);

	glBindVertexArray(0);

}



void My_Flag(GLfloat x, GLfloat y, GLfloat z, GLfloat fWidth) {

	translateMatrix = mat4::identity();
	modelViewMatrix = mat4::identity();
	modelViewProjectionMatrix = mat4::identity();

	translateMatrix = translate(x, y, z);
	modelViewMatrix = modelViewMatrix * translateMatrix;
	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(mvpUniform,
		1,
		GL_FALSE,
		modelViewProjectionMatrix);

	glLineWidth(fWidth);
	glBindVertexArray(vao_Flag);
	glDrawArrays(GL_LINES, 0, 6 * 3);
	glBindVertexArray(0);
}


void My_Plane(GLfloat x, GLfloat y, GLfloat z, GLfloat scaleX, GLfloat scaleY, GLfloat scaleZ, GLfloat ZRot_Angle) {

	translateMatrix = mat4::identity();
	scaleMatrix = mat4::identity();
	rotateMatrix = mat4::identity();
	modelViewMatrix = mat4::identity();
	modelViewProjectionMatrix = mat4::identity();

	translateMatrix = translate(x, y, z);
	scaleMatrix = scale(scaleX, scaleY, scaleZ);
	rotateMatrix = rotate(0.0f, 0.0f, ZRot_Angle);
	modelViewMatrix = modelViewMatrix * translateMatrix * scaleMatrix * rotateMatrix;
	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(mvpUniform,
		1,
		GL_FALSE,
		modelViewProjectionMatrix);


	//Triangle
	glBindVertexArray(vao_Plane_Triangle);
	glDrawArrays(GL_TRIANGLES, 0, 3);
	glBindVertexArray(0);

	//Rectangle
	glBindVertexArray(vao_Plane_Rect);

	//For Middle
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

	//For Upper and Lower Fin
	glDrawArrays(GL_TRIANGLE_FAN, 4, 8);

	//For Back
	glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
	glBindVertexArray(0);


	//Polygon
	glBindVertexArray(vao_Plane_Polygon);
	glDrawArrays(GL_TRIANGLE_FAN, 0, 10);
	glBindVertexArray(0);



	//I

	translateMatrix = translate(-1.5f, 0.0f, 0.0f);
	scaleMatrix = scale(0.70f, 0.70f, 0.0f);
	modelViewMatrix = modelViewMatrix * translateMatrix * scaleMatrix;
	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(mvpUniform,
		1,
		GL_FALSE,
		modelViewProjectionMatrix);

	glBindVertexArray(vao_I);
	glDrawArrays(GL_LINES, 0, 6);
	glBindVertexArray(0);






	//A

	translateMatrix = translate(1.0f, 0.0f, 0.0f);
	//scaleMatrix = scale(scaleX, 0.10f, scaleZ);

	modelViewMatrix = modelViewMatrix * translateMatrix;
	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(mvpUniform,
		1,
		GL_FALSE,
		modelViewProjectionMatrix);

	glBindVertexArray(vao_A);
	glDrawArrays(GL_LINES, 0, 6);
	glBindVertexArray(0);




	//F

	translateMatrix = translate(0.7f, 0.0f, 0.0f);

	modelViewMatrix = modelViewMatrix * translateMatrix;
	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(mvpUniform,
		1,
		GL_FALSE,
		modelViewProjectionMatrix);
	glBindVertexArray(vao_F);
	glDrawArrays(GL_LINES, 0, 6);
	glBindVertexArray(0);

}





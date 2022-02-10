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

//For A
GLuint vao_A;
GLuint vbo_A_Position;
GLuint vbo_A_Color;

//For Flag
GLuint vao_Flag;
GLuint vbo_Flag_Position;
GLuint vbo_Flag_Color;
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
	TCHAR szName[] = TEXT("RohitRJadhav-PP-StaticIndia");

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
		TEXT("RohitRJadhav-PP-StaticIndia"),
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

	GLfloat D_Color[] = {
		1.0f, 0.6f, 0.2f,
		0.0705f, 0.533f, 0.0274f,

		1.0f, 0.6f, 0.2f,
		1.0f, 0.6f, 0.2f,

		0.0705f, 0.533f, 0.0274f,
		0.0705f, 0.533f, 0.0274f,

		1.0f, 0.6f, 0.2f,
		0.0705f, 0.533f, 0.0274f
	};


	GLfloat A_Position[] = {
		0.0f, 1.06f, 0.0f,
		-0.5f, -1.06f, 0.0f,

		0.0f, 1.06f, 0.0f,
		0.5f, -1.06f, 0.0f
	};


	GLfloat A_Color[] = {
		1.0f, 0.6f, 0.2f,
		0.0705f, 0.533f, 0.0274f,

		1.0f, 0.6f, 0.2f,
		0.0705f, 0.533f, 0.0274f
	};


	GLfloat Flag_Position[] = {
		-0.207f, 0.06f, 0.0f,
		0.207f, 0.06f, 0.0f,

		-0.218f, 0.0f, 0.0f,
		0.219f, 0.0f, 0.0f,

		-0.235f, -0.06f, 0.0f,
		0.235f, -0.06f, 0.0f
	};


	GLfloat Flag_Color[] = {
		1.0f, 0.6f, 0.2f,
		1.0f, 0.6f, 0.2f,

		1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f,

		0.0705f, 0.533f, 0.0274f,
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
		sizeof(D_Color),
		D_Color,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_COLOR,
		3,
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


	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	gPerspectiveProjectionMatrix = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);
	return(0);
}

void uninitialize(void) {



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
mat4 modelViewMatrix;
mat4 modelViewProjectionMatrix;


void display(void) {

	void My_I(GLfloat, GLfloat, GLfloat, GLfloat);
	void My_N(GLfloat, GLfloat, GLfloat, GLfloat);
	void My_D(GLfloat, GLfloat, GLfloat, GLfloat);
	void My_A(GLfloat, GLfloat, GLfloat, GLfloat);
	void My_Flag(GLfloat, GLfloat, GLfloat, GLfloat);


	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	glUseProgram(gShaderProgramObject);

	//I
	My_I(-2.0f, 0.0f, -8.0f, 20.0f);

	//N
	My_N(-1.35f, 0.0f, -8.0f, 20.0f);

	//D
	My_D(-0.15f, 0.0f, -8.0f, 20.0f);

	//I
	My_I(1.02f, 0.0f, -8.0f, 20.0f);

	//A
	My_A(2.0f, 0.0f, -8.0f, 20.0f);

	//Flag
	My_Flag(2.0f, 0.0f, -8.0f, 20.0f);

	glUseProgram(0);

	SwapBuffers(ghdc);
}

void My_I(GLfloat x, GLfloat y, GLfloat z, GLfloat fWidth) {

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
	glBindVertexArray(vao_I);
	glDrawArrays(GL_LINES, 0, 6 * 3);
	glBindVertexArray(0);
}

void My_N(GLfloat x, GLfloat y, GLfloat z, GLfloat fWidth) {

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
	glBindVertexArray(vao_N);
	glDrawArrays(GL_LINES, 0, 6 * 3);
	glBindVertexArray(0);
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
	glDrawArrays(GL_LINES, 0, 8 * 3);
	glBindVertexArray(0);
}

void My_A(GLfloat x, GLfloat y, GLfloat z, GLfloat fWidth) {

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
	glBindVertexArray(vao_A);
	glDrawArrays(GL_LINES, 0, 4 * 3);
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

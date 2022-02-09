#include <Windows.h>
#include<stdio.h>
#include<gl\glew.h>
#include <gl\GL.h>
#include "..\..\include\vmath.h"
#include "01_3DTexture.h"

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glew32.lib")

#define WIN_WIDTH	800
#define WIN_HEIGHT	600

using namespace vmath;

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

VOID resize(int, int);

//
//	Global variable.
//
HWND g_hWnd;
HDC g_hdc;
HGLRC g_hRC;

DWORD g_dwStyle;
WINDOWPLACEMENT g_WindowPlacementPrev = { sizeof(WINDOWPLACEMENT) };

bool g_boFullScreen = false;
bool g_boActiveWindow = false;
bool g_boEscapeKeyPressed = false;

#define	CLASS_NAME		TEXT("PP : 3D Texture")

#define LOG_FILE_NAME	("log.txt")
//	Handle to log file
FILE *g_fpLogFile = NULL;

GLfloat g_glfAnglePyramid = 0.0f;
GLfloat g_glfAngleCube = 0.0f;

enum
{
	RTR_ATTRIBUTE_POSITION = 0,
	RTR_ATTRIBUTE_COLOR,
	RTR_ATTRIBUTE_NORMAL,
	RTR_ATTRIBUTE_TEXTURE0
};

GLuint g_gluiShaderObjectVertex;
GLuint g_gluiShaderObjectFragment;
GLuint g_gluiShaderObjectProgram;

GLuint g_gluiVAOPyramid;
GLuint g_gluiVAOCube;
GLuint g_gluiVBOPosition;
GLuint g_gluiVBOTexture;
GLint g_gliMVPUniform;

mat4 g_matPerspectiveProjection;

GLuint g_gluiTextureSamplerUniform;
GLuint g_gluiTextureKundali;
GLuint g_gluiTextureStone;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	VOID initialize();
	VOID display();
	VOID uninitialize();
	void update();

	MSG Msg;
	int x, y;
	HWND hWnd;
	int iMaxWidth;
	int iMaxHeight;
	WNDCLASSEX WndClass;
	bool boDone = false;
	TCHAR szClassName[] = CLASS_NAME;

	//
	//	Initialize members of window class.
	//
	WndClass.cbSize = sizeof(WNDCLASSEX);
	WndClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;		//	Change:Added CS_OWNDC.
	WndClass.cbClsExtra = 0;
	WndClass.cbWndExtra = 0;
	WndClass.hInstance = hInstance;
	WndClass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	WndClass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	WndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
	WndClass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	WndClass.lpfnWndProc = WndProc;
	WndClass.lpszClassName = szClassName;
	WndClass.lpszMenuName = NULL;

	//
	//	Register class.
	//
	RegisterClassEx(&WndClass);

	iMaxWidth = GetSystemMetrics(SM_CXFULLSCREEN);
	iMaxHeight = GetSystemMetrics(SM_CYFULLSCREEN);

	x = (iMaxWidth - WIN_WIDTH) / 2;
	y = (iMaxHeight - WIN_HEIGHT) / 2;

	//
	//	Create Window.
	//
	hWnd = CreateWindowEx(
		WS_EX_APPWINDOW,	//	Change: New member get added for CreateWindowEx API.
		szClassName,
		CLASS_NAME,
		WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,		//	Change: Added styles -WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE
		x,
		y,
		WIN_WIDTH,
		WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL
		);
	if (NULL == hWnd)
	{
		return 0;
	}

	g_hWnd = hWnd;

	initialize();

	ShowWindow(hWnd, SW_SHOW);
	SetForegroundWindow(hWnd);
	SetFocus(hWnd);

	//
	//	Message loop.
	//
	while (false == boDone)
	{
		if (PeekMessage(&Msg, NULL, 0, 0, PM_REMOVE))
		{
			if (WM_QUIT == Msg.message)
			{
				boDone = true;
			}
			else
			{
				TranslateMessage(&Msg);
				DispatchMessage(&Msg);
			}
		}
		else
		{
			if (true == g_boActiveWindow)
			{
				if (true == g_boEscapeKeyPressed)
				{
					boDone = true;
				}
				update();
				display();
			}
		}
	}

	uninitialize();

	return((int)Msg.wParam);
}


LRESULT CALLBACK WndProc(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	VOID ToggleFullScreen();
	void InitLight();

	switch (iMsg)
	{
	case WM_ACTIVATE:
		if (0 == HIWORD(wParam))
		{
			g_boActiveWindow = true;
		}
		else
		{
			g_boActiveWindow = false;
		}
		break;


		//case WM_ERASEBKGND:
		//return(0);

	case WM_SIZE:
		resize(LOWORD(lParam), HIWORD(lParam));
		break;


	case WM_CHAR:
		switch (wParam)
		{
		case VK_ESCAPE:
			g_boEscapeKeyPressed = true;
			break;

		case 'f':
		case 'F':
			if (false == g_boFullScreen)
			{
				ToggleFullScreen();
				g_boFullScreen = true;
			}
			else
			{
				ToggleFullScreen();
				g_boFullScreen = false;
			}
			break;

		default:
			break;
		}
		break;

	case WM_DESTROY:
		PostQuitMessage(0);
		break;

	default:
		break;
	}

	return (DefWindowProc(hWnd, iMsg, wParam, lParam));
}


VOID ToggleFullScreen()
{
	MONITORINFO MonitorInfo;

	if (false == g_boFullScreen)
	{
		g_dwStyle = GetWindowLong(g_hWnd, GWL_STYLE);

		if (g_dwStyle & WS_OVERLAPPEDWINDOW)
		{
			MonitorInfo = { sizeof(MonitorInfo) };

			if (GetWindowPlacement(g_hWnd, &g_WindowPlacementPrev) && GetMonitorInfo(MonitorFromWindow(g_hWnd, MONITORINFOF_PRIMARY), &MonitorInfo))
			{
				SetWindowLong(g_hWnd, GWL_STYLE, g_dwStyle & (~WS_OVERLAPPEDWINDOW));
				SetWindowPos(
					g_hWnd,
					HWND_TOP,
					MonitorInfo.rcMonitor.left,
					MonitorInfo.rcMonitor.top,
					MonitorInfo.rcMonitor.right - MonitorInfo.rcMonitor.left,
					MonitorInfo.rcMonitor.bottom - MonitorInfo.rcMonitor.top,
					SWP_NOZORDER | SWP_FRAMECHANGED
					);
			}
		}
		ShowCursor(FALSE);
	}
	else
	{
		SetWindowLong(g_hWnd, GWL_STYLE, g_dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(g_hWnd, &g_WindowPlacementPrev);
		SetWindowPos(g_hWnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);

		ShowCursor(TRUE);
	}
}

VOID initialize()
{
	BOOLEAN LoadGLTextures(GLuint *pgluiTexture, TCHAR arrImageResourceId[]);
	void uninitialize();
	void resize(int, int);

	HDC hDC;
	int iPixelFormatIndex;
	PIXELFORMATDESCRIPTOR pfd;

	fopen_s(&g_fpLogFile, LOG_FILE_NAME, "w");
	if (NULL == g_fpLogFile)
	{
		uninitialize();
		return;
	}

	ZeroMemory(&pfd, sizeof(pfd));

	//
	//	Init Pixel format descriptor structure.
	//
	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 32;
	pfd.cRedBits = 8;
	pfd.cGreenBits = 8;
	pfd.cBlueBits = 8;
	pfd.cAlphaBits = 8;
	pfd.cDepthBits = 32;	//	Change 1: for 3d

	g_hdc = GetDC(g_hWnd);

	hDC = GetDC(g_hWnd);

	ReleaseDC(g_hWnd, hDC);

	iPixelFormatIndex = ChoosePixelFormat(g_hdc, &pfd);
	if (0 == iPixelFormatIndex)
	{
		ReleaseDC(g_hWnd, g_hdc);
		g_hdc = NULL;
	}

	if (FALSE == SetPixelFormat(g_hdc, iPixelFormatIndex, &pfd))
	{
		ReleaseDC(g_hWnd, g_hdc);
		g_hdc = NULL;
	}

	g_hRC = wglCreateContext(g_hdc);
	if (NULL == g_hRC)
	{
		ReleaseDC(g_hWnd, g_hdc);
		g_hdc = NULL;
	}

	if (FALSE == wglMakeCurrent(g_hdc, g_hRC))
	{
		wglDeleteContext(g_hRC);
		g_hRC = NULL;
		ReleaseDC(g_hWnd, g_hdc);
		g_hdc = NULL;
	}

	GLenum glewError = glewInit();
	if (GLEW_OK != glewError)
	{
		uninitialize();
		fprintf(g_fpLogFile, "glewInit() failed, Error :%d", glewError);
		return;
	}

	fprintf(g_fpLogFile, "\n Version : %s", glGetString(GL_VERSION));
	fprintf(g_fpLogFile, "\n Shader Version : %s", glGetString(GL_SHADING_LANGUAGE_VERSION));

	////////////////////////////////////////////////////////////////////
	//+	Shader code

	////////////////////////////////////////////////////////////////////
	//+	Vertex shader.

	//	Create shader.
	g_gluiShaderObjectVertex = glCreateShader(GL_VERTEX_SHADER);

	//	Provide source code.
	const GLchar *szVertexShaderSourceCode =
		"#version 430 core"							\
		"\n"										\
		"in vec4 vPosition;"						\
		"in vec2 vTexture0Coord;"					\
		"out vec2 out_Texture0Coord;"				\
		"uniform mat4 u_mvp_matrix;"				\
		"void main(void)"							\
		"{"											\
		"gl_Position = u_mvp_matrix * vPosition;"	\
		"out_Texture0Coord = vTexture0Coord;"		\
		"}";

	glShaderSource(g_gluiShaderObjectVertex, 1, &szVertexShaderSourceCode, NULL);

	//	Compile shader.
	glCompileShader(g_gluiShaderObjectVertex);

	GLint gliCompileStatus;
	GLint gliInfoLogLength;
	char *pszInfoLog = NULL;
	GLsizei glsiWritten;
	glGetShaderiv(g_gluiShaderObjectVertex, GL_COMPILE_STATUS, &gliCompileStatus);
	if (GL_FALSE == gliCompileStatus)
	{
		glGetShaderiv(g_gluiShaderObjectVertex, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength <= 0)
		{
			fprintf(g_fpLogFile, "GL_INFO_LOG_LENGTH is less than 0.");
			uninitialize();
			exit(0);
		}

		pszInfoLog = (char*)malloc(gliInfoLogLength);
		if (NULL == pszInfoLog)
		{
			fprintf(g_fpLogFile, "malloc failed.");
			uninitialize();
			exit(0);
		}

		glGetShaderInfoLog(g_gluiShaderObjectVertex, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		uninitialize();
		exit(0);
	}
	//-	Vertex shader.
	////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	//+	Fragment shader.

	//	Create shader.
	g_gluiShaderObjectFragment = glCreateShader(GL_FRAGMENT_SHADER);

	//	Provide source code.
	const GLchar *szFragmentShaderSourceCode =
		"#version 430 core"														\
		"\n"																	\
		"in vec2 out_Texture0Coord;"											\
		"out vec4 vFragColor;"													\
		"uniform sampler2D u_texture0_sampler;"									\
		"void main(void)"														\
		"{"																		\
		"vFragColor = texture(u_texture0_sampler,out_Texture0Coord);"			\
		"}";

	glShaderSource(g_gluiShaderObjectFragment, 1, &szFragmentShaderSourceCode, NULL);

	//	Compile shader.
	glCompileShader(g_gluiShaderObjectFragment);

	gliCompileStatus = 0;
	gliInfoLogLength = 0;
	pszInfoLog = NULL;
	glsiWritten = 0;
	glGetShaderiv(g_gluiShaderObjectFragment, GL_COMPILE_STATUS, &gliCompileStatus);
	if (GL_FALSE == gliCompileStatus)
	{
		glGetShaderiv(g_gluiShaderObjectFragment, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength <= 0)
		{
			fprintf(g_fpLogFile, "Fragment : GL_INFO_LOG_LENGTH is less than 0.");
			uninitialize();
			exit(0);
		}

		pszInfoLog = (char*)malloc(gliInfoLogLength);
		if (NULL == pszInfoLog)
		{
			fprintf(g_fpLogFile, "Fragment : malloc failed.");
			uninitialize();
			exit(0);
		}

		glGetShaderInfoLog(g_gluiShaderObjectFragment, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		uninitialize();
		exit(0);
	}
	//-	Fragment shader.
	////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	//+	Shader program.

	//	Create.
	g_gluiShaderObjectProgram = glCreateProgram();

	//	Attach vertex shader to shader program.
	glAttachShader(g_gluiShaderObjectProgram, g_gluiShaderObjectVertex);

	//	Attach Fragment shader to shader program.
	glAttachShader(g_gluiShaderObjectProgram, g_gluiShaderObjectFragment);

	//	IMP:Pre-link binding of shader program object with vertex shader position attribute.
	glBindAttribLocation(g_gluiShaderObjectProgram, RTR_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(g_gluiShaderObjectProgram, RTR_ATTRIBUTE_TEXTURE0, "vTexture0Coord");

	//	Link shader.
	glLinkProgram(g_gluiShaderObjectProgram);

	GLint gliLinkStatus;
	gliInfoLogLength = 0;
	pszInfoLog = NULL;
	glsiWritten = 0;
	glGetShaderiv(g_gluiShaderObjectProgram, GL_LINK_STATUS, &gliLinkStatus);
	if (GL_FALSE == gliLinkStatus)
	{
		glGetProgramiv(g_gluiShaderObjectProgram, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength <= 0)
		{
			fprintf(g_fpLogFile, "Link : GL_INFO_LOG_LENGTH is less than 0.");
			uninitialize();
			exit(0);
		}

		pszInfoLog = (char*)malloc(gliInfoLogLength);
		if (NULL == pszInfoLog)
		{
			fprintf(g_fpLogFile, "Link : malloc failed.");
			uninitialize();
			exit(0);
		}

		glGetProgramInfoLog(g_gluiShaderObjectProgram, gliInfoLogLength, &glsiWritten, pszInfoLog);

		fprintf(g_fpLogFile, pszInfoLog);
		free(pszInfoLog);
		uninitialize();
		exit(0);
	}
	//-	Shader program.
	////////////////////////////////////////////////////////////////////

	//-	Shader code
	////////////////////////////////////////////////////////////////////

	//
	//	The actual locations assigned to uniform variables are not known until the program object is linked successfully.
	//	After a program object has been linked successfully, the index values for uniform variables remain fixed until the next link command occurs.
	//
	g_gliMVPUniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_mvp_matrix");
	if (-1 == g_gliMVPUniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_mvp_matrix) failed.");
		uninitialize();
		exit(0);
	}

	g_gluiTextureSamplerUniform = glGetUniformLocation(g_gluiShaderObjectProgram, "u_texture0_sampler");
	if (-1 == g_gluiTextureSamplerUniform)
	{
		fprintf(g_fpLogFile, "glGetUniformLocation(u_texture0_sampler) failed.");
		uninitialize();
		exit(0);
	}

	////////////////////////////////////////////////////////////////////
	//+	Vertices,color, shader attribute, vbo,vao initialization.

	const GLfloat glfarrPyramidVertices[] =
	{
		//	Front face
		0.0f, 1.0f, 0.0f,	//	apex
		-1.0f, -1.0f, 1.0f,	//	left_bottom
		1.0f, -1.0f, 1.0f,	//	right_bottom
		//	Right face
		0.0f, 1.0f, 0.0f,	//	apex
		1.0f, -1.0f, 1.0f,	//	left_bottom
		1.0f, -1.0f, -1.0f,	//	right_bottom
		//	Back face
		0.0f, 1.0f, 0.0f,	//	apex
		1.0f, -1.0f, -1.0f,	//	left_bottom
		-1.0f, -1.0f, -1.0f,	//	right_bottom
		//	Left face
		0.0f, 1.0f, 0.0f,	//	apex
		-1.0f, -1.0f, -1.0f,	//	left_bottom
		-1.0f, -1.0f, 1.0f,	//	right_bottom

	};

	const GLfloat glfarrPyramidTextCoord[] =
	{
		0.5f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,

		0.5f, 1.0f,
		1.0f, 0.0f,
		0.0f, 0.0f,

		0.5f, 1.0f,
		1.0f, 0.0f,
		0.0f, 0.0f,

		0.5f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
	};

	const GLfloat glfarrCubeVertices[] =
	{
		//	Front face
		1.0f, 1.0f, 1.0f,	//	left top
		1.0f, -1.0f, 1.0f,	//	left bottom
		-1.0f, -1.0f, 1.0f,	//	Right bottom
		-1.0f, 1.0f, 1.0f,	//	Right top

		//	Right face
		1.0f, 1.0f, -1.0f,	//	left top
		1.0f, 1.0f, 1.0f,	//	left bottom
		1.0f, -1.0f, 1.0f,	//	Right bottom
		1.0f, -1.0f, -1.0f,	//	Right top

		//	Top face
		1.0f, 1.0f, -1.0f,	//	left top
		-1.0f, 1.0f, -1.0f,	//	left bottom
		-1.0f, 1.0f, 1.0f,	//	Right bottom
		1.0f, 1.0f, 1.0f,	//	Right top

		//	Front face
		1.0f, 1.0f, -1.0f,	//	left top
		1.0f, -1.0f, -1.0f,	//	left bottom
		-1.0f, -1.0f, -1.0f,	//	Right bottom
		-1.0f, 1.0f, -1.0f,	//	Right top

		//	Right face
		-1.0f, 1.0f, -1.0f,	//	left top
		-1.0f, 1.0f, 1.0f,	//	left bottom
		-1.0f, -1.0f, 1.0f,	//	Right bottom
		-1.0f, -1.0f, -1.0f,	//	Right top

		//	Top face
		1.0f, -1.0f, -1.0f,	//	left top
		-1.0f, -1.0f, -1.0f,	//	left bottom
		-1.0f, -1.0f, 1.0f,	//	Right bottom
		1.0f, -1.0f, 1.0f,	//	Right top
	};

	const GLfloat glfarrCubeTextCoord[] =
	{
		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 1.0f,

		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 1.0f,

		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 1.0f,

		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 1.0f,

		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 1.0f,

		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 1.0f,

	};

	////////////////////////////////////////////////////////////////////
	//+	Pyramid VAO

	glGenVertexArrays(1, &g_gluiVAOPyramid);
	glBindVertexArray(g_gluiVAOPyramid);

	////////////////////////////////////////////////////////////////////
	//+ Vertex position
	glGenBuffers(1, &g_gluiVBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, g_gluiVBOPosition);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glfarrPyramidVertices), glfarrPyramidVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(RTR_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(RTR_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//- Vertex position
	////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	//+ Vertex Texture coord
	glGenBuffers(1, &g_gluiVBOTexture);
	glBindBuffer(GL_ARRAY_BUFFER, g_gluiVBOTexture);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glfarrPyramidTextCoord), glfarrPyramidTextCoord, GL_STATIC_DRAW);
	glVertexAttribPointer(RTR_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(RTR_ATTRIBUTE_TEXTURE0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//- Vertex Color
	////////////////////////////////////////////////////////////////////

	glBindVertexArray(0);

	//-	Pyramid VAO
	////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	//+	Cube VAO
	glGenVertexArrays(1, &g_gluiVAOCube);
	glBindVertexArray(g_gluiVAOCube);

	////////////////////////////////////////////////////////////////////
	//+ Vertex position
	glGenBuffers(1, &g_gluiVBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, g_gluiVBOPosition);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glfarrCubeVertices), glfarrCubeVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(RTR_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(RTR_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//- Vertex position
	////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////
	//+ Vertex Color
	glGenBuffers(1, &g_gluiVBOTexture);
	glBindBuffer(GL_ARRAY_BUFFER, g_gluiVBOTexture);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glfarrCubeTextCoord), glfarrCubeTextCoord, GL_STATIC_DRAW);
	glVertexAttribPointer(RTR_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(RTR_ATTRIBUTE_TEXTURE0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//- Vertex Color
	////////////////////////////////////////////////////////////////////

	glBindVertexArray(0);
	//-	Cube VAO
	////////////////////////////////////////////////////////////////////

	//-	Vertices,color, shader attribute, vbo,vao initialization.
	////////////////////////////////////////////////////////////////////

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	//+	Change 2 For 3D
	glClearDepth(1.0f);

	glEnable(GL_DEPTH_TEST);

	glDepthFunc(GL_LEQUAL);

	//
	//	Optional.
	//
	glShadeModel(GL_SMOOTH);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	//
	//	We will always cull back faces for better performance.
	//	We will this in case of 3-D rotation/graphics.
	//
	//glEnable(GL_CULL_FACE);

	//-	Change 2 For 3D
	//	Change for Texture.
	glEnable(GL_TEXTURE_2D);	//	Enable texture mapping.

	LoadGLTextures(&g_gluiTextureStone, MAKEINTRESOURCE(IDBITMAP_STONE));
	LoadGLTextures(&g_gluiTextureKundali, MAKEINTRESOURCE(IDBITMAP_KUNDALI));

	//	See orthographic projection matrix to identity.
	g_matPerspectiveProjection = mat4::identity();

	//
	//	Resize.
	//
	resize(WIN_WIDTH, WIN_HEIGHT);
}


BOOLEAN LoadGLTextures(GLuint *pgluiTexture, TCHAR arrImageResourceId[])
{
	HBITMAP hBitmap;
	BITMAP bmp;
	BOOLEAN bStatus = FALSE;

	glGenTextures(1, pgluiTexture);	//	1 image

	hBitmap = (HBITMAP)LoadImage(GetModuleHandle(NULL), arrImageResourceId, IMAGE_BITMAP, 0, 0, LR_CREATEDIBSECTION);
	if (!hBitmap)
	{
		return FALSE;
	}

	GetObject(hBitmap, sizeof(bmp), &bmp);

	//glPixelStorei(GL_UNPACK_ALIGNMENT, 4);	//Pixel alignment mode (word alignment / 4 bytes)
	//	Change pixel alignment from 4 to 1 for performace improvement.
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);	//Pixel alignment mode (word alignment / 1 bytes)

	glBindTexture(GL_TEXTURE_2D, *pgluiTexture);	//	Bind texture.

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);	//Generate mipmapped texture(3 byets, width, height & data from bmp)

	glTexImage2D(
		GL_TEXTURE_2D,
		0,	//	Bitmap level of depth(0 for all)
		GL_RGB,//3,	// type of image format which is use by open GL.
		bmp.bmWidth,
		bmp.bmHeight,
		0,	//	Border width (let the implementation decide)
		GL_BGR_EXT,
		GL_UNSIGNED_BYTE,	// Type of last parameter.
		bmp.bmBits
		);

	glGenerateMipmap(GL_TEXTURE_2D);

	DeleteObject(hBitmap);	//	Delete unwanted object.

	return TRUE;
}

void update()
{
	g_glfAngleCube = g_glfAngleCube + 0.1f;

	if (g_glfAngleCube >= 360)
	{
		g_glfAngleCube = 0.0f;
	}

	g_glfAnglePyramid = g_glfAnglePyramid + 0.1f;

	if (g_glfAnglePyramid >= 360)
	{
		g_glfAnglePyramid = 0.0f;
	}
}


VOID display()
{
	mat4 matScale;
	mat4 matRotation;
	mat4 matModelView;
	mat4 matModelViewProjection;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);	//	Change 3 for 3D

	//	Start using opengl program.
	glUseProgram(g_gluiShaderObjectProgram);

	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Triangle

	//	Set modelview and modelviewprojection matrix to identity.
	matModelView = mat4::identity();
	matModelViewProjection = mat4::identity();	//	Good practice to initialize to identity matrix though it will change in next call.
	matRotation = mat4::identity();

	matModelView = translate(-2.0f, 0.0f, -6.0f);

	matModelView = matModelView * rotate(g_glfAnglePyramid, 0.0f, 1.0f, 0.0f);

	//	Multiply the modelview and perspective projection matrix to get modelviewprojection matrix.
	//	Order is very important.
	matModelViewProjection = g_matPerspectiveProjection * matModelView;

	//
	//	Pass above modelviewprojection matrix to the vertex shader in 'u_mvp_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gliMVPUniform, 1, GL_FALSE, matModelViewProjection);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, g_gluiTextureStone);
	glUniform1i(g_gluiTextureSamplerUniform, 0);//	0th sampler enable as we have only 1 taxture sampler in fragment shader.

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOPyramid);

	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glDrawArrays(GL_TRIANGLES, 0, 12); //	3 - each with its x,y,z vertices in triangle vertices array.

	//	Unbind 'VAO'
	glBindVertexArray(0);
	//-	Draw Triangle
	/////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////
	//+	Draw Square

	//	Set modelview and modelviewprojection matrix to identity.
	matModelView = mat4::identity();
	matModelViewProjection = mat4::identity();	//	Good practice to initialize to identity matrix though it will change in next call.

	matModelView = translate(2.0f, 0.0f, -6.0f);

	matModelView = matModelView * scale(0.8f, 0.8f, 0.8f);

	matModelView = matModelView * rotate(g_glfAngleCube, g_glfAngleCube, g_glfAngleCube);

	//	Multiply the modelview and orthographic projection matrix to get modelviewprojection matrix.
	//	Order is very important.
	matModelViewProjection = g_matPerspectiveProjection * matModelView;

	//
	//	Pass above modelviewprojection matrix to the vertex shader in 'u_mvp_matrix' shader variable,
	//	whose position value we already calculated in initialize() using glGetUniformLocation().
	//	Give matrix data to shader.
	//
	glUniformMatrix4fv(g_gliMVPUniform, 1, GL_FALSE, matModelViewProjection);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, g_gluiTextureKundali);
	glUniform1i(g_gluiTextureSamplerUniform, 0);//	0th sampler enable as we have only 1 taxture sampler in fragment shader.

	//	Bind 'VAO'
	glBindVertexArray(g_gluiVAOCube);

	//	Draw either glDrawTriangle, glDrawArrays() or glDrawElements().
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4); //	3 - each with its x,y,z vertices in triangle vertices array.
	glDrawArrays(GL_TRIANGLE_FAN, 4, 4); //	3 - each with its x,y,z vertices in triangle vertices array.
	glDrawArrays(GL_TRIANGLE_FAN, 8, 4); //	3 - each with its x,y,z vertices in triangle vertices array.
	glDrawArrays(GL_TRIANGLE_FAN, 12, 4); //	3 - each with its x,y,z vertices in triangle vertices array.
	glDrawArrays(GL_TRIANGLE_FAN, 16, 4); //	3 - each with its x,y,z vertices in triangle vertices array.
	glDrawArrays(GL_TRIANGLE_FAN, 20, 4); //	3 - each with its x,y,z vertices in triangle vertices array.

	//	Unbind 'VAO'
	glBindVertexArray(0);
	//-	Draw Square
	/////////////////////////////////////////////////////////////////////////////////////////

	//	Stop using opengl program.
	glUseProgram(0);

	SwapBuffers(g_hdc);
}


VOID resize(int iWidth, int iHeight)
{
	if (0 == iHeight)
	{
		iHeight = 1;
	}

	if (0 == iWidth)
	{
		iWidth = 1;
	}

	//	perspective(float fovy, float aspect, float n, float f)
	if (iWidth <= iHeight)
	{
		g_matPerspectiveProjection = perspective(45, (GLfloat)iHeight / (GLfloat)iWidth, 0.1f, 100.0f);
	}
	else
	{
		g_matPerspectiveProjection = perspective(45, (GLfloat)iWidth / (GLfloat)iHeight, 0.1f, 100.0f);
	}

	glViewport(0, 0, iWidth, iHeight);
}

VOID uninitialize()
{
	if (true == g_boFullScreen)
	{
		SetWindowLong(g_hWnd, GWL_STYLE, g_dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(g_hWnd, &g_WindowPlacementPrev);
		SetWindowPos(g_hWnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);

		ShowCursor(TRUE);
	}

	if (g_gluiVBOPosition)
	{
		glDeleteBuffers(1, &g_gluiVBOPosition);
		g_gluiVBOPosition = 0;
	}

	if (g_gluiVBOTexture)
	{
		glDeleteBuffers(1, &g_gluiVBOTexture);
		g_gluiVBOTexture = 0;
	}

	if (g_gluiVAOPyramid)
	{
		glDeleteVertexArrays(1, &g_gluiVAOPyramid);
		g_gluiVAOPyramid = 0;
	}

	if (g_gluiVAOCube)
	{
		glDeleteVertexArrays(1, &g_gluiVAOCube);
		g_gluiVAOCube = 0;
	}

	if (g_gluiTextureStone)
	{
		glDeleteTextures(1, &g_gluiTextureStone);
		g_gluiTextureStone = 0;
	}

	if (g_gluiTextureKundali)
	{
		glDeleteTextures(1, &g_gluiTextureKundali);
		g_gluiTextureKundali = 0;
	}

	if (g_gluiShaderObjectVertex)
	{
		glDetachShader(g_gluiShaderObjectProgram, g_gluiShaderObjectVertex);
		glDeleteShader(g_gluiShaderObjectVertex);
		g_gluiShaderObjectVertex = 0;
	}

	if (g_gluiShaderObjectFragment)
	{
		glDetachShader(g_gluiShaderObjectProgram, g_gluiShaderObjectFragment);
		glDeleteShader(g_gluiShaderObjectFragment);
		g_gluiShaderObjectFragment = 0;
	}

	//
	//	Unlink shader program
	//	This will be useful when detach multiple shaders in loop.
	//	1.glUseProgram(Shader_Program_Object)
	//	2.Get Attach shader list
	//	3.Detach i loop.
	//	4.glUseProgram(0)
	//
	glUseProgram(0);

	if (g_gluiShaderObjectProgram)
	{
		glDeleteProgram(g_gluiShaderObjectProgram);
		g_gluiShaderObjectProgram = 0;
	}

	wglMakeCurrent(NULL, NULL);
	wglDeleteContext(g_hRC);
	g_hRC = NULL;

	ReleaseDC(g_hWnd, g_hdc);
	g_hdc = NULL;

	DestroyWindow(g_hWnd);
	g_hWnd = NULL;

	if (g_fpLogFile)
	{
		fprintf(g_fpLogFile, "\n Log file succesfuly closed.");
		fclose(g_fpLogFile);
		g_fpLogFile = NULL;
	}
}

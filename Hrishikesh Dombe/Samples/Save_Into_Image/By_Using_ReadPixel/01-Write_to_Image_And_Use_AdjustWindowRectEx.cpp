#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_DDS
#include<windows.h>
#include<C:\glew\include\GL\glew.h>
#include<gl/GL.h>
#include<stdio.h>
#include<iostream>
#include"vmath.h"
//#include"stb_image.h"
#include"stb_image_aug.h"
#include"stb_image_aug.c"

#pragma comment(lib,"User32.lib")
#pragma comment(lib,"GDI32.lib")
#pragma comment(lib,"C:\\glew\\lib\\Release\\x64\\glew32.lib")
#pragma comment(lib,"opengl32.lib")

#define WIN_WIDTH 800
#define WIN_HEIGHT 600
#define DEST_FILE ".\\Test_AdjustWindowRectEx.bmp"

using namespace vmath;
using namespace std;

enum
{
	HAD_ATTRIBUTE_POSITION = 0,
	HAD_ATTRIBUTE_COLOR,
	HAD_ATTRIBUTE_NORMAL,
	HAD_ATTRIBUTE_TEXTURE0,
};

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

FILE *gpFile;
HWND ghwnd;
HDC ghdc;
HGLRC ghrc;
DWORD dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };
bool gbActiveWindow = false;
bool gbFullscreen = false;
bool gbIsEscapeKeyPressed = false;
bool gbIsAKeyPressed = false;

GLuint gVertexShaderObject;
GLuint gFragmentShaderObject;
GLuint gShaderProgramObject;
GLuint gVertexShaderObject_Cube;
GLuint gFragmentShaderObject_Cube;
GLuint gShaderProgramObject_Cube;

GLuint gVao_Pyramid, gVao_Cube;
GLuint gVbo_Pos, gVbo_Color, gVbo_Texture, gFbo, gRbo;
GLuint gMVPUniform;

GLuint gTexture;
GLuint gTexture_Image;
GLuint gTexture_sampler_uniform, gTexture_sampler_uniform1;

GLfloat gAngle_Pyramid, gAngle_Cube;

mat4 gPerspectiveProjectionMatrix;

GLenum err;

//For Image
int iStatus = FALSE;
int width, height, nrComponents;
GLenum format;
unsigned char *image = NULL;
unsigned char *image_data = NULL;
GLint viewport[4];
int win_width, win_height;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	void initialize(void);
	void display(void);
	void update(void);
	void uninitialize(int);
	void create_image_file(void);
	void ToggleFullscreen(void);
	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szClassName[] = TEXT("My App");
	bool bDone = false;
	RECT rect;

	if (fopen_s(&gpFile, "Log.txt", "w") != NULL)
	{
		MessageBox(NULL, TEXT("Cannot Create Log File !!!"), TEXT("Error"), MB_OK);
		exit(EXIT_FAILURE);
	}
	else
		fprintf(gpFile, "Log File Created Successfully...\n");

	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.hInstance = hInstance;
	wndclass.lpszClassName = szClassName;
	wndclass.lpszMenuName = NULL;
	wndclass.lpfnWndProc = WndProc;
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);

	RegisterClassEx(&wndclass);

	rect.top = 0;
	rect.bottom = 600;
	rect.left = 0;
	rect.right = 800;
	AdjustWindowRectEx(&rect, WS_OVERLAPPEDWINDOW, FALSE, WS_EX_APPWINDOW);

	hwnd = CreateWindowEx(WS_EX_APPWINDOW, szClassName, TEXT("OpenGLPP : 3D Rotation"), WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE, 100, 100, rect.right - rect.left, rect.bottom - rect.top, NULL, NULL, hInstance, NULL);
	if (hwnd == NULL)
	{
		fprintf(gpFile, "Cannot Create Window...\n");
		uninitialize(1);
	}

	ghwnd = hwnd;

	ShowWindow(hwnd, iCmdShow);
	SetFocus(hwnd);
	SetForegroundWindow(hwnd);

	initialize();

	//ToggleFullscreen();

	/*while (bDone == false)
	{
	if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
	{
	if (msg.message == WM_QUIT)
	bDone = true;
	else
	{
	TranslateMessage(&msg);
	DispatchMessage(&msg);
	}
	}
	else
	{
	if (gbActiveWindow == true)
	{
	if (gbIsEscapeKeyPressed == true)
	bDone = true;
	update();
	display();
	}
	}
	}*/
	display();
	/*image_data = glMapBuffer(GL_TEXTURE_2D, GL_READ_ONLY);
	if (image_data == NULL)
	{
	MessageBox(NULL, TEXT("Cannot Map Buffer"), TEXT("ERROR"), MB_OK);
	}
	glGetTexImage(GL_TEXTURE_2D, 0, nrComponents, GL_UNSIGNED_BYTE, image_data);
	create_image_file();
	glUnmapBuffer(GL_TEXTURE_2D);*/
	uninitialize(0);
	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	void resize(int, int);
	void ToggleFullscreen(void);
	switch (iMsg)
	{
	case WM_ACTIVATE:
		if (HIWORD(wParam) == 0)
			gbActiveWindow = true;
		else
			gbActiveWindow = false;
		break;
	case WM_CREATE:
		break;
	case WM_SIZE:
		//win_width = LOWORD(lParam);
		//win_height = HIWORD(lParam);
		resize(LOWORD(lParam), HIWORD(lParam));
		break;
	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			gbIsEscapeKeyPressed = true;
			break;
		case 0x46:
			if (gbFullscreen == false)
			{
				ToggleFullscreen();
				gbFullscreen = true;
			}
			else
			{
				ToggleFullscreen();
				gbFullscreen = false;
			}
			break;

		case 0x41:
			if (gbIsAKeyPressed == false)
				gbIsAKeyPressed = true;
			else
				gbIsAKeyPressed = false;
			break;
		}
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	}
	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void initialize(void)
{
	void resize(int, int);
	void uninitialize(int);
	void createFboTexture(void);
	int LoadGLTextures(GLuint *, char *);
	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;

	ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));

	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 24;
	pfd.cRedBits = 8;
	pfd.cGreenBits = 8;
	pfd.cBlueBits = 8;
	pfd.cAlphaBits = 8;
	pfd.cDepthBits = 32;

	ghdc = GetDC(ghwnd);
	if (ghdc == NULL)
	{
		fprintf(gpFile, "GetDC() Failed.\n");
		uninitialize(1);
	}

	iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);
	if (iPixelFormatIndex == 0)
	{
		fprintf(gpFile, "ChoosePixelFormat() Failed.\n");
		uninitialize(1);
	}

	if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
	{
		fprintf(gpFile, "SetPixelFormat() Failed.\n");
		uninitialize(1);
	}

	ghrc = wglCreateContext(ghdc);
	if (ghrc == NULL)
	{
		fprintf(gpFile, "wglCreateContext() Failed.\n");
		uninitialize(1);
	}

	if (wglMakeCurrent(ghdc, ghrc) == FALSE)
	{
		fprintf(gpFile, "wglMakeCurrent() Failed");
		uninitialize(1);
	}

	GLenum glew_error = glewInit();
	if (glew_error != GLEW_OK)
	{
		wglDeleteContext(ghrc);
		ghrc = NULL;
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	//Vertex Shader
	gVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *vertexShaderSourceCode =
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec2 vTexture0_Coord;" \
		"uniform mat4 u_mvp_matrix;" \
		"out vec2 out_texture0_coord;" \
		"void main(void)" \
		"{" \
		" vec4 vertices[4] = vec4[4](vec4(1.0, 1.0, 0.5, 1.0),\
		vec4(-1.0, 1.0, 0.5, 1.0),\
		vec4(-1.0, -1.0, 0.5, 1.0),\
		vec4(1.0, -1.0, 0.5, 1.0));\
		vec4 pos = vertices[gl_VertexID]; " \
		//"gl_Position = u_mvp_matrix * vPosition;" 
		"gl_Position = pos;" \
		"out_texture0_coord = vTexture0_Coord;" \
		"}";

	glShaderSource(gVertexShaderObject, 1, (const GLchar **)&vertexShaderSourceCode, NULL);

	glCompileShader(gVertexShaderObject);
	GLint iInfoLogLength = 0;
	GLint iShaderCompiledStatus = 0;
	char *szInfoLog = NULL;

	glGetShaderiv(gVertexShaderObject, GL_COMPILE_STATUS, &iShaderCompiledStatus);
	if (iShaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(gVertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(gVertexShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Vertex Shader Compilation Log : %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize(1);
				exit(0);
			}
		}
	}

	//Fragment Shader
	gFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *fragmentShaderSourceCode =
		"#version 450 core"\
		"\n"\
		"in vec2 out_texture0_coord;" \
		"out vec4 FragColor;"\
		"uniform sampler2D u_texture0_sampler;"\
		"void main(void)"\
		"{"\
		"vec2 text_coord = out_texture0_coord;" \
		"text_coord.y = -text_coord.y;" \
		"vec4 Texture = texture(u_texture0_sampler,out_texture0_coord);"\
		"float average = (Texture.r + Texture.g + Texture.b) / 3.0;" \
		"FragColor = vec4(average,average,average,1.0);" \
		"}";

	glShaderSource(gFragmentShaderObject, 1, (const GLchar **)&fragmentShaderSourceCode, NULL);

	glCompileShader(gFragmentShaderObject);

	glGetShaderiv(gFragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompiledStatus);
	if (iShaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(gFragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(gFragmentShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Fragment Shader Compilation Log : %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize(1);
				exit(0);
			}
		}
	}

	//Shader Program
	gShaderProgramObject = glCreateProgram();

	glAttachShader(gShaderProgramObject, gVertexShaderObject);

	glAttachShader(gShaderProgramObject, gFragmentShaderObject);

	glBindAttribLocation(gShaderProgramObject, HAD_ATTRIBUTE_POSITION, "vPosition");

	glBindAttribLocation(gShaderProgramObject, HAD_ATTRIBUTE_TEXTURE0, "vTexture0_Coord");

	glLinkProgram(gShaderProgramObject);

	GLint iShaderProgramLinkStatus = 0;

	glGetProgramiv(gShaderProgramObject, GL_LINK_STATUS, &iShaderProgramLinkStatus);
	if (iShaderProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(gShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetProgramInfoLog(gShaderProgramObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Shader Program Link Log : %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize(1);
				exit(0);
			}
		}
	}

	gMVPUniform = glGetUniformLocation(gShaderProgramObject, "u_mvp_matrix");

	gTexture_sampler_uniform1 = glGetUniformLocation(gShaderProgramObject_Cube, "u_texture0_sampler");

	/****************For Cube while using FBO****************/
	//Vertex Shader
	gVertexShaderObject_Cube = glCreateShader(GL_VERTEX_SHADER);

	const GLchar *vertexShaderSourceCode_Cube =
		"#version 450 core" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec2 vTexture0_Coord;" \
		"uniform mat4 u_mvp_matrix;" \
		"out vec2 out_texture0_coord;" \
		"void main(void)" \
		"{" \
		"gl_Position = u_mvp_matrix * vPosition;" \
		"out_texture0_coord = vTexture0_Coord;" \
		"}";

	glShaderSource(gVertexShaderObject_Cube, 1, (const GLchar **)&vertexShaderSourceCode_Cube, NULL);

	glCompileShader(gVertexShaderObject_Cube);
	iInfoLogLength = 0;
	iShaderCompiledStatus = 0;
	szInfoLog = NULL;

	glGetShaderiv(gVertexShaderObject_Cube, GL_COMPILE_STATUS, &iShaderCompiledStatus);
	if (iShaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(gVertexShaderObject_Cube, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(gVertexShaderObject_Cube, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Vertex Shader Compilation Log : %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize(1);
				exit(0);
			}
		}
	}

	//Fragment Shader
	gFragmentShaderObject_Cube = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *fragmentShaderSourceCode_Cube =
		"#version 450 core"\
		"\n"\
		"in vec2 out_texture0_coord;" \
		"out vec4 FragColor;"\
		"uniform sampler2D u_texture0_sampler;"\
		"void main(void)"\
		"{"\
		"vec4 Texture = texture(u_texture0_sampler,out_texture0_coord);"\
		"FragColor = Texture;" \
		"}";

	glShaderSource(gFragmentShaderObject_Cube, 1, (const GLchar **)&fragmentShaderSourceCode_Cube, NULL);

	glCompileShader(gFragmentShaderObject_Cube);

	glGetShaderiv(gFragmentShaderObject_Cube, GL_COMPILE_STATUS, &iShaderCompiledStatus);
	if (iShaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(gFragmentShaderObject_Cube, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(gFragmentShaderObject_Cube, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Fragment Shader Compilation Log : %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize(1);
				exit(0);
			}
		}
	}

	//Shader Program
	gShaderProgramObject_Cube = glCreateProgram();

	glAttachShader(gShaderProgramObject_Cube, gVertexShaderObject_Cube);

	glAttachShader(gShaderProgramObject_Cube, gFragmentShaderObject_Cube);

	glBindAttribLocation(gShaderProgramObject_Cube, HAD_ATTRIBUTE_POSITION, "vPosition");

	glBindAttribLocation(gShaderProgramObject_Cube, HAD_ATTRIBUTE_TEXTURE0, "vTexture0_Coord");

	glLinkProgram(gShaderProgramObject_Cube);

	iShaderProgramLinkStatus = 0;

	glGetProgramiv(gShaderProgramObject_Cube, GL_LINK_STATUS, &iShaderProgramLinkStatus);
	if (iShaderProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(gShaderProgramObject_Cube, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetProgramInfoLog(gShaderProgramObject_Cube, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Shader Program Link Log : %s\n", szInfoLog);
				free(szInfoLog);
				uninitialize(1);
				exit(0);
			}
		}
	}

	gMVPUniform = glGetUniformLocation(gShaderProgramObject_Cube, "u_mvp_matrix");

	gTexture_sampler_uniform = glGetUniformLocation(gShaderProgramObject_Cube, "u_texture0_sampler");

	const GLfloat pyramidVertices[] =
	{
		0.0f,1.0f,0.0f,
		-1.0f,-1.0f,1.0f,
		1.0f,-1.0f,1.0f,

		0.0f,1.0f,0.0f,
		1.0f,-1.0f,1.0f,
		1.0f,-1.0f,-1.0f,

		0.0f,1.0f,0.0f,
		1.0f,-1.0f,-1.0f,
		-1.0f,-1.0f,-1.0f,

		0.0f,1.0f,0.0f,
		-1.0f,-1.0f,-1.0f,
		-1.0f,-1.0f,1.0f
	};

	const GLfloat pyramidColor[] =
	{
		1.0f,0.0f,0.0f,
		0.0f,1.0f,0.0f,
		0.0f,0.0f,1.0f,

		1.0f,0.0f,0.0f,
		0.0f,0.0f,1.0f,
		0.0f,1.0f,0.0f,

		1.0f,0.0f,0.0f,
		0.0f,1.0f,0.0f,
		0.0f,0.0f,1.0f,

		1.0f,0.0f,0.0f,
		0.0f,0.0f,1.0f,
		0.0f,1.0f,0.0f
	};

	const GLfloat cubeVertices[] =
	{
		1.0f,1.0f,1.0f,
		-1.0f,1.0f,1.0f,
		-1.0f,-1.0f,1.0f,
		1.0f,-1.0f,1.0f,

		/*1.0f,1.0f,-1.0f,
		1.0f,1.0f,1.0f,
		1.0f,-1.0f,1.0f,
		1.0f,-1.0f,-1.0f,

		-1.0f,1.0f,-1.0f,
		1.0f,1.0f,-1.0f,
		1.0f,-1.0f,-1.0f,
		-1.0f,-1.0f,-1.0f,

		-1.0f,1.0f,1.0f,
		-1.0f,1.0f,-1.0f,
		-1.0f,-1.0f,-1.0f,
		-1.0f,-1.0f,1.0f,

		1.0f,1.0f,-1.0f,
		-1.0f,1.0f,-1.0f,
		-1.0f,1.0f,1.0f,
		1.0f,1.0f,1.0f,

		1.0f,-1.0f,1.0f,
		-1.0f,-1.0f,1.0f,
		-1.0f,-1.0f,-1.0f,
		1.0f,-1.0f,-1.0f*/
	};

	const GLfloat cubeTexcoords[] =
	{
		/*0.0f,0.0f,
		1.0f,0.0f,
		1.0f,1.0f,
		0.0f,1.0f,
		*/
		1.0f,1.0f,
		0.0f,1.0f,
		0.0f,0.0f,
		1.0f,0.0f,

		/*0.0f,0.0f,
		1.0f,0.0f,
		1.0f,1.0f,
		0.0f,1.0f,

		0.0f,0.0f,
		1.0f,0.0f,
		1.0f,1.0f,
		0.0f,1.0f,

		0.0f,0.0f,
		1.0f,0.0f,
		1.0f,1.0f,
		0.0f,1.0f,

		0.0f,0.0f,
		1.0f,0.0f,
		1.0f,1.0f,
		0.0f,1.0f,

		0.0f,0.0f,
		1.0f,0.0f,
		1.0f,1.0f,
		0.0f,1.0f*/
	};

	const GLfloat cubeColor[] =
	{
		1.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,

		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,

		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,

		1.0f, 0.0f, 1.0f,
		1.0f, 0.0f, 1.0f,
		1.0f, 0.0f, 1.0f,
		1.0f, 0.0f, 1.0f,

		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,

		0.0f, 1.0f, 1.0f,
		0.0f, 1.0f, 1.0f,
		0.0f, 1.0f, 1.0f,
		0.0f, 1.0f, 1.0f
	};


	/*****************VAO For Pyramid*****************/
	glGenVertexArrays(1, &gVao_Pyramid);
	/*****************Pyramid Position****************/
	glBindVertexArray(gVao_Pyramid);

	glGenBuffers(1, &gVbo_Pos);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_Pos);
	glBufferData(GL_ARRAY_BUFFER, sizeof(pyramidVertices), pyramidVertices, GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_POSITION);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/*****************Pyramid Color****************/
	glGenBuffers(1, &gVbo_Color);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_Color);
	glBufferData(GL_ARRAY_BUFFER, sizeof(pyramidColor), pyramidColor, GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_COLOR);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	/*****************VAO For Cube*****************/
	glGenVertexArrays(1, &gVao_Cube);
	glBindVertexArray(gVao_Cube);

	/*****************Cube Position****************/
	glGenBuffers(1, &gVbo_Pos);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_Pos);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_POSITION);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/*****************Cube Texture****************/
	glGenBuffers(1, &gVbo_Texture);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_Texture);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cubeTexcoords), cubeTexcoords, GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_TEXTURE0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	/*****************FBO*******************/
	//Create Off-Screen Framebuffer
	glGenFramebuffers(1, &gFbo);
	glBindFramebuffer(GL_FRAMEBUFFER, gFbo);
	//Creating Texture and Color Attachment 
	createFboTexture();


	/*****************RBO*****************/
	//Creating a Render Buffer
	glGenRenderbuffers(1, &gRbo);
	glBindRenderbuffer(GL_RENDERBUFFER, gRbo);
	//Use Renderbuffer to Render the Depth and Stencil(Now Stencil is not used)
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, 800, 600);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);

	//Attach our Renderbuffer to Framebuffer at Depth and Stencil Attachment
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, gRbo);

	//Check if Framebuffer is created successfully
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		MessageBox(ghwnd, TEXT("ERROR :: FRAMEBUFFER :: Framebuffer is not complete !"), TEXT("ERROR"), MB_OK);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glShadeModel(GL_SMOOTH);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	//glEnable(GL_CULL_FACE);
	glEnable(GL_TEXTURE_2D);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	gPerspectiveProjectionMatrix = mat4::identity();
	char filename[] = "a.JPG";
	LoadGLTextures(&gTexture_Image, filename);

	resize(WIN_WIDTH, WIN_HEIGHT);
}

void createFboTexture(void)
{
	//Creating Empty Texture
	glGenTextures(1, &gTexture);
	glBindTexture(GL_TEXTURE_2D, gTexture);

	//Last Parameter is NULL because we Don't have the texture data at initialize time we will produce it at Runtime
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 800, 600, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);

	//Create a Color Attachment and give the empty texture(as Color) to Framebuffer
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gTexture, 0);
}

void display(void)
{
	char str[256];
	void create_image_file(void);
	void uninitialize(int);
	//Bind the Off-Screen Framebuffer so that it can be used to do off-screen rendering
	//glBindFramebuffer(GL_FRAMEBUFFER, gFbo);
	//Clear Color for the background of objects of render to texture 
	glClearColor(0.25f, 0.25f, 0.25f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_DEPTH_TEST);

	//Use Shader Program Object
	glUseProgram(gShaderProgramObject);

	mat4 modelViewMatrix = mat4::identity();
	mat4 modelViewProjectionMatrix = mat4::identity();
	mat4 scaleMatrix = mat4::identity();
	mat4 rotationMatrix = mat4::identity();

	modelViewMatrix = translate(0.0f, 0.0f, -6.0f);

	rotationMatrix = vmath::rotate(180.0f, 1.0f, 0.0f, 0.0f);
	modelViewMatrix = modelViewMatrix*rotationMatrix;

	rotationMatrix = vmath::rotate(gAngle_Pyramid, 0.0f, 1.0f, 0.0f);
	modelViewMatrix = modelViewMatrix*rotationMatrix;

	modelViewProjectionMatrix = gPerspectiveProjectionMatrix*modelViewMatrix;

	glUniformMatrix4fv(gMVPUniform, 1, GL_FALSE, modelViewProjectionMatrix);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, gTexture_Image);
	glUniform1i(gTexture_sampler_uniform1, 0);

	glBindVertexArray(gVao_Cube);

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 16, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 20, 4);
	create_image_file();
	glBindVertexArray(0);

	glUseProgram(0);

	//By binding to 0 means make default framebuffer active
	//glBindFramebuffer(GL_FRAMEBUFFER, 0);
	//Clear Color for Main Screen
	/*modelViewMatrix = mat4::identity();
	modelViewProjectionMatrix = mat4::identity();
	scaleMatrix = mat4::identity();
	rotationMatrix = mat4::identity();
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//glDisable(GL_DEPTH_TEST);

	glUseProgram(gShaderProgramObject_Cube);

	modelViewMatrix = mat4::identity();
	modelViewProjectionMatrix = mat4::identity();

	rotationMatrix = mat4::identity();

	modelViewMatrix = translate(0.0f, 0.0f, -6.0f);

	scaleMatrix = scale(0.75f, 0.75f, 0.75f);
	modelViewMatrix = modelViewMatrix*scaleMatrix;

	rotationMatrix = vmath::rotate(gAngle_Cube, gAngle_Cube, gAngle_Cube);
	modelViewMatrix = modelViewMatrix*rotationMatrix;

	modelViewProjectionMatrix = gPerspectiveProjectionMatrix*modelViewMatrix;

	glUniformMatrix4fv(gMVPUniform, 1, GL_FALSE, modelViewProjectionMatrix);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, gTexture);
	glUniform1i(gTexture_sampler_uniform, 0);

	glBindVertexArray(gVao_Cube);

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 16, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 20, 4);



	glBindVertexArray(0);

	glUseProgram(0);

	glBindTexture(GL_TEXTURE_2D, gTexture_Image);*/
	//GL_INVALID_FRAMEBUFFER_OPERATION;
	/*image_data = glMapBuffer(GL_TEXTURE_2D, GL_READ_ONLY);
	if (image_data == NULL)
	{
	MessageBox(NULL, TEXT("Cannot Map Buffer"), TEXT("ERROR"), MB_OK);
	err = glGetError();
	sprintf(str, "Error : %d", err);
	MessageBox(NULL, str, TEXT("ERROR"), MB_OK);
	uninitialize(1);
	}*/
	/*glGetTexImage(GL_TEXTURE_2D, 0, nrComponents, GL_UNSIGNED_BYTE, image_data);
	if (image_data == NULL)
	{
	MessageBox(NULL, TEXT("Cannot Map Buffer"), TEXT("ERROR"), MB_OK);
	err = glGetError();
	sprintf(str, "Error : %d", err);
	MessageBox(NULL, str, TEXT("ERROR"), MB_OK);
	uninitialize(1);
	}*/
	/*glReadBuffer(GL_FRONT);
	glReadPixels(0, 0, width, height, format, GL_UNSIGNED_BYTE, image_data);
	if (image_data == NULL)
	{
	MessageBox(NULL, TEXT("Cannot Map Buffer"), TEXT("ERROR"), MB_OK);
	err = glGetError();
	sprintf(str, "Error : %d", err);
	MessageBox(NULL, str, TEXT("ERROR"), MB_OK);
	uninitialize(1);
	}
	create_image_file();

	glBindTexture(GL_TEXTURE_2D, 0);
	*/
	SwapBuffers(ghdc);
}

int LoadGLTextures(GLuint *texture, char *filename)
{
	//	HBITMAP hBitmap;
	//BITMAP bmp;


	glGenTextures(1, texture);
	//	hBitmap = (HBITMAP)LoadImage(GetModuleHandle(NULL), imageResourceId, IMAGE_BITMAP, 0, 0, LR_CREATEDIBSECTION);
	image = stbi_load(filename, &width, &height, &nrComponents, 0);
	if (image)
	{
		if (nrComponents == 1)
			format = GL_RED;
		else if (nrComponents == 3)
			format = GL_RGB;
		else if (nrComponents == 4)
			format = GL_RGBA;

		iStatus = TRUE;
		//GetObject(hBitmap, sizeof(bmp), &bmp);v
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glBindTexture(GL_TEXTURE_2D, *texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, format, GL_UNSIGNED_BYTE, image);

		glGenerateMipmap(GL_TEXTURE_2D);

		//DeleteObject(hBitmap);
		stbi_image_free(image);
	}
	return(iStatus);
}

void create_image_file(void)
{
	glGetIntegerv(GL_VIEWPORT, viewport);
	image_data = (unsigned char *)malloc(viewport[2] * viewport[3] * 3);
	glReadPixels(0, 0, viewport[2], viewport[3], GL_RGB, GL_UNSIGNED_BYTE, image_data);
	stbi_write_bmp(DEST_FILE, viewport[2], viewport[3], nrComponents, image_data);

	/*unsigned char TGAheader[12] = { 0,0,2,0,0,0,0,0,0,0,0,0 };

	unsigned char header[6] = { ((int)(viewport[2] % 256)),((int)(viewport[2] / 256)),((int)(viewport[3] % 256)),((int)(viewport[3] / 256)),24,0 };

	char filename[] = "A.tga";
	FILE *image_file = NULL;
	if ((image_file = fopen(filename, "w")) == NULL)
	MessageBox(NULL, TEXT("Cannot Open File to Write"), TEXT("Error"), MB_OK);


	// TGA header schreiben
	fwrite(TGAheader, sizeof(unsigned char), 12, image_file);
	// Header schreiben
	fwrite(header, sizeof(unsigned char), 6, image_file);

	fwrite(image_data, sizeof(unsigned char),
	viewport[2] * viewport[3] * 3, image_file);

	fclose(image_file);*/

	free(image_data);
}

void update(void)
{
	if (gbIsAKeyPressed == true)
	{
		gAngle_Pyramid = gAngle_Pyramid + 1.0f;
		if (gAngle_Pyramid >= 360.0f)
			gAngle_Pyramid = 0.0f;
	}

	if (gbIsAKeyPressed == true)
	{
		gAngle_Cube = gAngle_Cube + 1.0f;
		if (gAngle_Cube >= 360.0f)
			gAngle_Cube = 0.0f;
	}
}

void resize(int width, int height)
{
	if (height == 0)
		height = 1;
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	gPerspectiveProjectionMatrix = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);

	if (gFbo)
	{
		glBindTexture(GL_TEXTURE_2D, gTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glBindTexture(GL_TEXTURE_2D, 0);
	}
	if (gRbo)
	{
		glBindRenderbuffer(GL_RENDERBUFFER, gRbo);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
		glBindRenderbuffer(GL_RENDERBUFFER, 0);
	}
}

void ToggleFullscreen(void)
{
	MONITORINFO mi = { sizeof(MONITORINFO) };
	if (gbFullscreen == false)
	{
		dwStyle = GetWindowLong(ghwnd, GWL_STYLE);
		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{
			if (GetWindowPlacement(ghwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &mi))
			{
				SetWindowLong(ghwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}
		ShowCursor(FALSE);
	}

	else
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
	}
}

void uninitialize(int i_Exit_Flag)
{
	if (gbFullscreen == false)
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
	}

	if (gVao_Pyramid)
	{
		glDeleteVertexArrays(1, &gVao_Pyramid);
		gVao_Pyramid = 0;
	}

	if (gVao_Cube)
	{
		glDeleteVertexArrays(1, &gVao_Cube);
		gVao_Cube = 0;
	}

	if (gVbo_Pos)
	{
		glDeleteBuffers(1, &gVbo_Pos);
		gVbo_Pos = 0;
	}

	if (gVbo_Color)
	{
		glDeleteBuffers(1, &gVbo_Color);
		gVbo_Color = 0;
	}

	if (gFbo)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glDeleteFramebuffers(1, &gFbo);
		gFbo = 0;
	}

	//Detach Shader 
	glDetachShader(gShaderProgramObject, gVertexShaderObject);
	glDetachShader(gShaderProgramObject, gFragmentShaderObject);

	//Delete Shader
	glDeleteShader(gVertexShaderObject);
	gVertexShaderObject = 0;

	glDeleteShader(gFragmentShaderObject);
	gFragmentShaderObject = 0;

	//Delete Program
	glDeleteProgram(gShaderProgramObject);
	gShaderProgramObject = 0;

	//Stray call to glUseProgram(0)
	glUseProgram(0);

	wglMakeCurrent(NULL, NULL);

	if (ghrc != NULL)
	{
		wglDeleteContext(ghrc);
		ghrc = NULL;
	}

	if (ghdc != NULL)
	{
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	if (i_Exit_Flag == 0)
	{
		fprintf(gpFile, "Log File Closed Successfully");
	}
	else if (i_Exit_Flag == 1)
	{
		fprintf(gpFile, "Log File Closed Erroniously");
	}

	fclose(gpFile);
	gpFile = NULL;

	DestroyWindow(ghwnd);
}
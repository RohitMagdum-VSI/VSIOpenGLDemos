#include<windows.h>
#include<C:\glew\include\GL\glew.h>
#include<gl/GL.h>
#include<stdio.h>
#include"vmath.h"
#include"01-Texture.h"

#pragma comment(lib,"User32.lib")
#pragma comment(lib,"GDI32.lib")
#pragma comment(lib,"C:\\glew\\lib\\Release\\x64\\glew32.lib")
#pragma comment(lib,"opengl32.lib")

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

using namespace vmath;
		
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

GLuint gVertexShaderObject;
GLuint gFragmentShaderObject;
GLuint gShaderProgramObject;

GLuint gVao_Cube;
GLuint gVbo_Cube;

GLuint gModelViewMatrixUniform, gProjectionMatrixUniform;
GLuint gLdUniform, gKdUniform, gLightPositionUniform;
GLuint gLKeyPressedUniform;

GLuint gTexture_sampler_uniform;
GLuint gTexture;

GLfloat gAngle_Cube;

mat4 gPerspectiveProjectionMatrix;

bool gbAnimate = false;
bool gbLight = false;
bool gbIsAKeyPressed = false;
bool gbIsLKeyPressed = false;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	void initialize(void);
	void display(void);
	void update(void);
	void uninitialize(int);
	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szClassName[] = TEXT("My App");
	bool bDone = false;

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

	hwnd = CreateWindowEx(WS_EX_APPWINDOW, szClassName, TEXT("OpenGLPP : 3D Texture"), WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE, 100, 100, WIN_WIDTH, WIN_HEIGHT, NULL, NULL, hInstance, NULL);
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

	while (bDone == false)
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
				if(gbAnimate == true)
					update();
				display();
			}
		}
	}

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
			{
				gbAnimate = true;
				gbIsAKeyPressed = true;
			}
			else
			{
				gbAnimate = false;
				gbIsAKeyPressed = false;
			}
			break;

		case 0x4C:
			if (gbIsLKeyPressed == false)
			{
				gbLight = true;
				gbIsLKeyPressed = true;
			}
			else
			{
				gbLight = false;
				gbIsLKeyPressed = false;
			}
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
	int LoadGLTextures(GLuint *, TCHAR[]);
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
		"in vec3 vNormal;" \
		"in vec3 vColor;" \
		"in vec2 vTexture0_Coord;" \
		"uniform mat4 u_model_view_matrix;" \
		"uniform mat4 u_projection_matrix;" \
		"uniform int u_LKeyPressed;" \
		"uniform vec3 u_Ld;" \
		"uniform vec3 u_Kd;" \
		"uniform vec4 u_light_position;" \
		"out vec3 diffuse_light;" \
		"out vec2 out_texture0_coord;" \
		"out vec3 out_color;" \
		"void main(void)" \
		"{" \
		"if (u_LKeyPressed == 1)" \
		"{" \
		"vec4 eyeCoordinates = u_model_view_matrix * vPosition;" \
		"vec3 tnorm = normalize(mat3(u_model_view_matrix) * vNormal);" \
		"vec3 s = normalize(vec3(u_light_position - eyeCoordinates));" \
		"diffuse_light = u_Ld * u_Kd * max(dot(s,tnorm),0.0);" \
		"}" \
		"gl_Position = u_projection_matrix * u_model_view_matrix * vPosition;" \
		"out_texture0_coord = vTexture0_Coord;" \
		"out_color = vColor;" \
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
				fprintf(gpFile, "Vertex Shader Compilation Log : %s\n",szInfoLog);
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
		"in vec3 out_color;" \
		"in vec3 diffuse_light;" \
		"out vec4 FragColor;" \
		"uniform int u_LKeyPressed;" \
		"uniform sampler2D u_texture0_sampler;"\
		"vec4 texture_final;" \
		"void main(void)"\
		"{"\
		"vec4 calculated_color;" \
		"texture_final = texture(u_texture0_sampler,out_texture0_coord);" \
		"if(u_LKeyPressed == 1)" \
		"{" \
		"calculated_color = vec4(diffuse_light,1.0);" \
		"FragColor = texture_final * calculated_color * vec4(out_color,0.0f);"\
		"}" \
		"else" \
		"{" \
		"FragColor = texture_final * vec4(out_color,0.0f);"\
		"}" \
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

	glBindAttribLocation(gShaderProgramObject, HAD_ATTRIBUTE_NORMAL, "vNormal");

	glBindAttribLocation(gShaderProgramObject, HAD_ATTRIBUTE_COLOR, "vColor");

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

	gModelViewMatrixUniform = glGetUniformLocation(gShaderProgramObject, "u_model_view_matrix");
	gProjectionMatrixUniform = glGetUniformLocation(gShaderProgramObject, "u_projection_matrix");
	gLKeyPressedUniform = glGetUniformLocation(gShaderProgramObject, "u_LKeyPressed");
	gLdUniform = glGetUniformLocation(gShaderProgramObject, "u_Ld");
	gKdUniform = glGetUniformLocation(gShaderProgramObject, "u_Kd");
	gLightPositionUniform = glGetUniformLocation(gShaderProgramObject, "u_light_position");
	gTexture_sampler_uniform = glGetUniformLocation(gShaderProgramObject, "u_texture0_sampler");

	const GLfloat cubeVCNT[] =
	{
		1.0f,1.0f,1.0f,1.0f, 0.0f, 0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,
		-1.0f,1.0f,1.0f,1.0f, 0.0f, 0.0f,0.0f,0.0f,1.0f,1.0f,0.0f,
		-1.0f,-1.0f,1.0f,1.0f, 0.0f, 0.0f,0.0f,0.0f,1.0f,1.0f,1.0f,
		1.0f,-1.0f,1.0f,1.0f, 0.0f, 0.0f,0.0f,0.0f,1.0f,0.0f,1.0f,

		1.0f,1.0f,-1.0f,0.0f, 1.0f, 0.0f,1.0f,0.0f,0.0f,0.0f,0.0f,
		1.0f,1.0f,1.0f,0.0f, 1.0f, 0.0f,1.0f,0.0f,0.0f,1.0f,0.0f,
		1.0f,-1.0f,1.0f,0.0f, 1.0f, 0.0f,1.0f,0.0f,0.0f,1.0f,1.0f,
		1.0f,-1.0f,-1.0f,0.0f, 1.0f, 0.0f,1.0f,0.0f,0.0f,0.0f,1.0f,

		-1.0f,1.0f,-1.0f,0.0f, 0.0f, 1.0f,0.0f,0.0f,-1.0f,0.0f,0.0f,
		1.0f,1.0f,-1.0f,0.0f, 0.0f, 1.0f,0.0f,0.0f,-1.0f,1.0f,0.0f,
		1.0f,-1.0f,-1.0f,0.0f, 0.0f, 1.0f,0.0f,0.0f,-1.0f,1.0f,1.0f,
		-1.0f,-1.0f,-1.0f,0.0f, 0.0f, 1.0f,0.0f,0.0f,-1.0f,0.0f,1.0f,

		-1.0f,1.0f,1.0f,1.0f, 0.0f, 1.0f,-1.0f,0.0f,0.0f,0.0f,0.0f,
		-1.0f,1.0f,-1.0f,1.0f, 0.0f, 1.0f,-1.0f,0.0f,0.0f,1.0f,0.0f,
		-1.0f,-1.0f,-1.0f,1.0f, 0.0f, 1.0f,-1.0f,0.0f,0.0f,1.0f,1.0f,
		-1.0f,-1.0f,1.0f,1.0f, 0.0f, 1.0f,-1.0f,0.0f,0.0f,0.0f,1.0f,

		1.0f,1.0f,-1.0f,1.0f, 1.0f, 0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,
		-1.0f,1.0f,-1.0f,1.0f, 1.0f, 0.0f,0.0f,1.0f,0.0f,1.0f,0.0f,
		-1.0f,1.0f,1.0f,1.0f, 1.0f, 0.0f,0.0f,1.0f,0.0f,1.0f,1.0f,
		1.0f,1.0f,1.0f,1.0f, 1.0f, 0.0f,0.0f,1.0f,0.0f,0.0f,1.0f,

		1.0f,-1.0f,1.0f,0.0f, 1.0f, 1.0f,0.0f,-1.0f,0.0f,0.0f,0.0f,
		-1.0f,-1.0f,1.0f,0.0f, 1.0f, 1.0f,0.0f,-1.0f,0.0f,1.0f,0.0f,
		-1.0f,-1.0f,-1.0f,0.0f, 1.0f, 1.0f,0.0f,-1.0f,0.0f,1.0f,1.0f,
		1.0f,-1.0f,-1.0f,0.0f, 1.0f, 1.0f,0.0f,-1.0f,0.0f,0.0f,1.0f,
	};

	/*****************VAO For Cube*****************/
	glGenVertexArrays(1, &gVao_Cube);
	glBindVertexArray(gVao_Cube);

	/*****************Cube Position****************/
	glGenBuffers(1, &gVbo_Cube);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_Cube);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVCNT), cubeVCNT, GL_STATIC_DRAW);

	glVertexAttribPointer(HAD_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(GLfloat), (void *)0);

	glEnableVertexAttribArray(HAD_ATTRIBUTE_POSITION);

	glVertexAttribPointer(HAD_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(GLfloat), (void *)  (3 * sizeof(GLfloat)));

	glEnableVertexAttribArray(HAD_ATTRIBUTE_COLOR);

	glVertexAttribPointer(HAD_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(GLfloat), (void *)  (6 * sizeof(GLfloat)));

	glEnableVertexAttribArray(HAD_ATTRIBUTE_NORMAL);

	glVertexAttribPointer(HAD_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 11 * sizeof(GLfloat), (void *)  (9 * sizeof(GLfloat)));

	glEnableVertexAttribArray(HAD_ATTRIBUTE_TEXTURE0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glShadeModel(GL_SMOOTH);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT , GL_NICEST);
	glEnable(GL_CULL_FACE);

	LoadGLTextures(&gTexture, MAKEINTRESOURCE(IDBITMAP_TEXTURE));

	glEnable(GL_TEXTURE_2D);
	
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	gPerspectiveProjectionMatrix = mat4::identity();

	resize(WIN_WIDTH, WIN_HEIGHT);
}

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
	
	//Use Shader Program Object
	glUseProgram(gShaderProgramObject);

	if (gbLight == true)
	{
		glUniform1i(gLKeyPressedUniform, 1);

		glUniform3f(gLdUniform, 1.0f, 1.0f, 1.0f);
		glUniform3f(gKdUniform, 0.5f, 0.5f, 0.5f);

		float lightPosition[] = { 0.0f,0.0f,2.0f,1.0f };
		glUniform4fv(gLightPositionUniform, 1, (GLfloat *)lightPosition);
	}
	else
	{
		glUniform1i(gLKeyPressedUniform, 0);
	}

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, gTexture);
	glUniform1i(gTexture_sampler_uniform, 0);

	mat4 modelMatrix = mat4::identity();
	mat4 modelViewMatrix = mat4::identity();
	mat4 scaleMatrix = mat4::identity();
	mat4 rotationMatrix = mat4::identity();

	modelMatrix = translate(0.0f, 0.0f, -6.0f);

	scaleMatrix = vmath::scale(0.75f, 0.75f, 0.75f);
	modelViewMatrix = modelMatrix * scaleMatrix;

	rotationMatrix = vmath::rotate(gAngle_Cube, gAngle_Cube, gAngle_Cube);
	modelViewMatrix = modelMatrix*rotationMatrix;

	glUniformMatrix4fv(gModelViewMatrixUniform, 1, GL_FALSE, modelViewMatrix);

	glUniformMatrix4fv(gProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

	glBindVertexArray(gVao_Cube);

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 16, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 20, 4);

	glBindVertexArray(0);

	glUseProgram(0);

	SwapBuffers(ghdc);
}

int LoadGLTextures(GLuint *texture, TCHAR imageResourceId[])
{
	HBITMAP hBitmap;
	BITMAP bmp;
	int iStatus = FALSE;

	glGenTextures(1, texture);
	hBitmap = (HBITMAP)LoadImage(GetModuleHandle(NULL), imageResourceId, IMAGE_BITMAP, 0, 0, LR_CREATEDIBSECTION);
	if (hBitmap)
	{
		iStatus = TRUE;
		GetObject(hBitmap, sizeof(bmp), &bmp);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glBindTexture(GL_TEXTURE_2D, *texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, bmp.bmWidth, bmp.bmHeight, 0, GL_BGR, GL_UNSIGNED_BYTE, bmp.bmBits);

		glGenerateMipmap(GL_TEXTURE_2D);

		DeleteObject(hBitmap);
	}
	return(iStatus);
}

void update(void)
{
	gAngle_Cube = gAngle_Cube + 1.0f;
	if (gAngle_Cube >= 360.0f)
		gAngle_Cube = 0.0f;
}

void resize(int width, int height)
{
	if (height == 0)
		height = 1;
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	gPerspectiveProjectionMatrix = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
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

	if (gVao_Cube)
	{
		glDeleteVertexArrays(1, &gVao_Cube);
		gVao_Cube = 0;
	}

	if (gVbo_Cube)
	{
		glDeleteBuffers(1, &gVbo_Cube);
		gVbo_Cube = 0;
	}

	if (gTexture)
	{
		glDeleteTextures(1, &gTexture);
		gTexture = 0;
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
#include <Windows.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include "../common/vmath.h"
#include "11_TweakSmileyTexture.h"

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glew32.lib")

HDC ghDC = NULL;
HWND ghWnd = NULL;
HGLRC ghRC = NULL;

FILE* gpFile = NULL;

bool gbIsActivate = false;
bool gbIsFullScreen = false;
bool gbIsEscKeyPressed = false;

DWORD dwStyle = 0;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };

GLint giFlag = 0;

GLuint guiVAO;

GLuint guiVBOPosition;
GLuint guiVBOTexture;
GLuint guiMVPUniform;
GLuint guiTextureSamplerUniform;

GLuint guiTextureSmiley;
GLuint guiTextureWhite;
GLubyte TextureWhite[64][64][4];

GLuint guiVertexShaderObject;
GLuint guiFragmentShaderObject;
GLuint guiShaderProgramObject;

vmath::mat4 gPerspectiveProjectionMatrix;

// WinMain
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	MSG msg = { 0 };
	HWND hWnd = NULL;
	bool bDone = false;
	WNDCLASSEX WndClass = { 0 };
	WCHAR wszClassName[] = L"2D Texture";

	fopen_s(&gpFile, "OGLLog.txt", "w");
	if (NULL == gpFile)
	{
		MessageBox(NULL, L"Error while creating log file", L"Error", MB_OK);
		return EXIT_FAILURE;
	}

	// WndClass
	WndClass.cbSize = sizeof(WNDCLASSEX);
	WndClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	WndClass.lpfnWndProc = WndProc;
	WndClass.lpszClassName = wszClassName;
	WndClass.lpszMenuName = NULL;
	WndClass.hInstance = hInstance;
	WndClass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	WndClass.hIcon = (HICON)LoadIcon(NULL, IDI_APPLICATION);
	WndClass.hIconSm = (HICON)LoadIcon(NULL, IDI_APPLICATION);
	WndClass.hCursor = (HCURSOR)LoadCursor(NULL, IDC_ARROW);
	WndClass.cbWndExtra = 0;
	WndClass.cbClsExtra = 0;


	// Register WndClass
	if (!RegisterClassEx(&WndClass))
	{
		fprintf(gpFile, "Error while registering WndClass.\n");
		fclose(gpFile);
		gpFile = NULL;
		return EXIT_FAILURE;
	}

	int iScreenWidth = GetSystemMetrics(SM_CXSCREEN);
	int iScreenHeight = GetSystemMetrics(SM_CYSCREEN);

	// Create Window
	hWnd = CreateWindowEx(
		WS_EX_APPWINDOW,
		wszClassName,
		wszClassName,
		WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
		(iScreenWidth / 2) - (OGL_WINDOW_WIDTH / 2),
		(iScreenHeight / 2) - (OGL_WINDOW_HEIGHT / 2),
		OGL_WINDOW_WIDTH,
		OGL_WINDOW_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL
	);
	if (NULL == hWnd)
	{
		fprintf(gpFile, "Error while creating window.\n");
		fclose(gpFile);
		gpFile = NULL;
		return EXIT_FAILURE;
	}

	ghWnd = hWnd;

	ShowWindow(hWnd, iCmdShow);
	SetForegroundWindow(hWnd);
	SetFocus(hWnd);

	bool bRet = Initialize();
	if (false == bRet)
	{
		fprintf(gpFile, "Error while Initialize().\n");
		fclose(gpFile);
		gpFile = NULL;
		DestroyWindow(hWnd);
		hWnd = NULL;
		return EXIT_FAILURE;
	}

	// Game loop
	while (bDone == false)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (WM_QUIT == msg.message)
			{
				bDone = true;
			}
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		else
		{
			Display();

			if (true == gbIsActivate)
			{
				if (true == gbIsEscKeyPressed)
				{
					bDone = true;
				}
			}
		}
	}

	// Uninitialize
	UnInitialize();

	return (int)msg.wParam;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	switch (iMsg)
	{
	case WM_ACTIVATE:
		if (HIWORD(lParam) == 0)
		{
			gbIsActivate = true;
		}
		else
		{
			gbIsActivate = false;
		}
		break;

	case WM_ERASEBKGND:
		return 0;

	case WM_SIZE:
		Resize(LOWORD(lParam), HIWORD(lParam));
		break;

	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			if (false == gbIsEscKeyPressed)
			{
				gbIsEscKeyPressed = true;
			}
			break;

		case 0x46:
			if (false == gbIsFullScreen)
			{
				ToggleFullScreen();
				gbIsFullScreen = true;
			}
			else
			{
				ToggleFullScreen();
				gbIsFullScreen = false;
			}
			break;

		case 0X31:
		case 0x61:
			giFlag = 1;
			break;

		case 0X32:
		case 0x62:
			giFlag = 2;
			break;

		case 0X33:
		case 0x63:
			giFlag = 3;
			break;

		case 0X34:
		case 0x64:
			giFlag = 4;
			break;

		default:
			giFlag = 0;
			break;
		}
		break;

	case WM_LBUTTONDOWN:
		break;

	case WM_CLOSE:
		UnInitialize();
		break;

	case WM_DESTROY:
		PostQuitMessage(0);
		break;

	default:
		break;
	}

	return DefWindowProc(hWnd, iMsg, wParam, lParam);
}

bool Initialize()
{
	int iPixelFormatIndex = 0;
	PIXELFORMATDESCRIPTOR pfd = { 0 };

	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 32;
	pfd.cRedBits = 8;
	pfd.cGreenBits = 8;
	pfd.cBlueBits = 8;
	pfd.cAlphaBits = 8;
	pfd.cDepthBits = 32;

	ghDC = GetDC(ghWnd);
	if (NULL == ghDC)
	{
		fprintf(gpFile, "Error while GetDC().\n");
		return false;
	}

	iPixelFormatIndex = ChoosePixelFormat(ghDC, &pfd);
	if (0 == iPixelFormatIndex)
	{
		fprintf(gpFile, "Error while ChoosePixelFormat().\n");
		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
		return false;
	}

	if (false == SetPixelFormat(ghDC, iPixelFormatIndex, &pfd))
	{
		fprintf(gpFile, "Error while SetPixelFormat().\n");
		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
		return false;
	}

	ghRC = wglCreateContext(ghDC);
	if (NULL == ghRC)
	{
		fprintf(gpFile, "Error while wglCreateContext().\n");
		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
		return false;
	}

	if (false == wglMakeCurrent(ghDC, ghRC))
	{
		fprintf(gpFile, "Error while wglMakeCurrent().\n");
		wglDeleteContext(ghRC);
		ghRC = NULL;
		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
		return false;
	}

	GLenum glError = glewInit();
	if (GLEW_OK != glError)
	{
		fprintf(gpFile, "Error while glewInit().\n");
		wglDeleteContext(ghRC);
		ghRC = NULL;
		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
		return false;
	}

	// Create vertex shader
	guiVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar* glchVertexShaderSource =
		"#version 430 core" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec2 vTexture0_Coord;" \
		"out vec2 out_texture0_coord;" \
		"uniform mat4 u_mvp_matrix;" \
		"void main(void)" \
		"{" \
		"gl_Position = u_mvp_matrix * vPosition;" \
		"out_texture0_coord = vTexture0_Coord;" \
		"}";

	glShaderSource(guiVertexShaderObject, 1, (const GLchar**)&glchVertexShaderSource, NULL);

	glCompileShader(guiVertexShaderObject);
	GLint gliInfoLogLength = 0;
	GLint gliShaderComileStatus = 0;
	char* pszInfoLog = NULL;

	glGetShaderiv(guiVertexShaderObject, GL_COMPILE_STATUS, &gliShaderComileStatus);
	if (GL_FALSE == gliShaderComileStatus)
	{
		glGetShaderiv(guiVertexShaderObject, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength > 0)
		{
			pszInfoLog = (char*)malloc((size_t)gliInfoLogLength);
			if (NULL != pszInfoLog)
			{
				GLsizei bytesWritten = 0;
				glGetShaderInfoLog(guiVertexShaderObject, gliInfoLogLength, &bytesWritten, pszInfoLog);
				fprintf(gpFile, "Vertex shader compilation Error : %s\n", pszInfoLog);
				free(pszInfoLog);
				pszInfoLog = NULL;
				UnInitialize();
				return false;
			}
		}
	}

	// Create fragment shader
	guiFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar* glchFragmentShaderSource =
		"#version 430 core" \
		"\n" \
		"in vec2 out_texture0_coord;" \
		"out vec4 FragColor;" \
		"uniform sampler2D u_texture0_sampler;" \
		"void main(void)" \
		"{" \
		"FragColor = texture(u_texture0_sampler, out_texture0_coord);" \
		"}";

	glShaderSource(guiFragmentShaderObject, 1, (const GLchar**)&glchFragmentShaderSource, NULL);

	glCompileShader(guiFragmentShaderObject);
	gliInfoLogLength = 0;
	gliShaderComileStatus = 0;
	pszInfoLog = NULL;

	glGetShaderiv(guiFragmentShaderObject, GL_COMPILE_STATUS, &gliShaderComileStatus);
	if (GL_FALSE == gliShaderComileStatus)
	{
		glGetShaderiv(guiFragmentShaderObject, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength > 0)
		{
			GLsizei bytesWritten = 0;
			pszInfoLog = (char*)malloc((size_t)gliInfoLogLength);
			if (NULL != pszInfoLog)
			{
				glGetShaderInfoLog(guiFragmentShaderObject, gliInfoLogLength, &bytesWritten, pszInfoLog);
				fprintf(gpFile, "Fragment shader compilation error : %s\n", pszInfoLog);
				free(pszInfoLog);
				pszInfoLog = NULL;
				UnInitialize();
				return false;
			}
		}
	}

	// Create shader program
	guiShaderProgramObject = glCreateProgram();

	glAttachShader(guiShaderProgramObject, guiVertexShaderObject);
	glAttachShader(guiShaderProgramObject, guiFragmentShaderObject);

	glBindAttribLocation(guiShaderProgramObject, OGL_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(guiShaderProgramObject, OGL_ATTRIBUTE_TEXTURE, "vTexture0_Coord");

	glLinkProgram(guiShaderProgramObject);

	GLint gliProgramLinkStatus = 0;
	gliInfoLogLength = 0;
	pszInfoLog = NULL;

	glGetProgramiv(guiShaderProgramObject, GL_LINK_STATUS, &gliProgramLinkStatus);
	if (GL_FALSE == gliProgramLinkStatus)
	{
		glGetProgramiv(guiShaderProgramObject, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength > 0)
		{
			GLsizei bytesWritten = 0;
			pszInfoLog = (char*)malloc((size_t)gliInfoLogLength);
			if (NULL != pszInfoLog)
			{
				glGetProgramInfoLog(guiShaderProgramObject, gliInfoLogLength, &bytesWritten, pszInfoLog);
				fprintf(gpFile, "Shader program link error : %s\n", pszInfoLog);
				free(pszInfoLog);
				pszInfoLog = NULL;
				UnInitialize();
				return false;
			}
		}
	}

	// get mvp uniform location
	guiMVPUniform = glGetUniformLocation(guiShaderProgramObject, "u_mvp_matrix");
	guiTextureSamplerUniform = glGetUniformLocation(guiShaderProgramObject, "u_texture0_sampler");

	const GLfloat squareVertices[] =
	{
		1.0f, 1.0f, 0.0f,
		-1.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f
	};

	glGenVertexArrays(1, &guiVAO);
	glBindVertexArray(guiVAO);

	glGenBuffers(1, &guiVBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, guiVBOPosition);
	glBufferData(GL_ARRAY_BUFFER, sizeof(squareVertices), squareVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &guiVBOTexture);
	glBindBuffer(GL_ARRAY_BUFFER, guiVBOTexture);
	glBufferData(GL_ARRAY_BUFFER, 2 * 4 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_TEXTURE, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_TEXTURE);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	glShadeModel(GL_SMOOTH);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	glEnable(GL_CULL_FACE);
	glEnable(GL_TEXTURE_2D);

	LoadGLTexture(&guiTextureSmiley, MAKEINTRESOURCE(IDBITMAP_SMILEY));
	LoadGLWhiteTexture();

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	gPerspectiveProjectionMatrix = vmath::mat4::identity();

	Resize(OGL_WINDOW_WIDTH, OGL_WINDOW_HEIGHT);

	return true;
}

void ToggleFullScreen()
{
	MONITORINFO mi = { 0 };

	if (false == gbIsFullScreen)
	{
		dwStyle = GetWindowLong(ghWnd, GWL_STYLE);
		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{
			mi.cbSize = sizeof(MONITORINFO);
			if (
				GetWindowPlacement(ghWnd, &wpPrev) &&
				GetMonitorInfo(MonitorFromWindow(ghWnd, MONITORINFOF_PRIMARY), &mi)
				)
			{
				SetWindowLong(ghWnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(
					ghWnd,
					HWND_TOP,
					mi.rcMonitor.left,
					mi.rcMonitor.top,
					mi.rcMonitor.right - mi.rcMonitor.left,
					mi.rcMonitor.bottom - mi.rcMonitor.top,
					SWP_NOZORDER | SWP_FRAMECHANGED
				);
			}
		}
		ShowCursor(FALSE);
	}
	else
	{
		SetWindowLong(ghWnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghWnd, &wpPrev);
		SetWindowPos(
			ghWnd,
			HWND_TOP,
			0, 0, 0, 0,
			SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED
		);

		ShowCursor(TRUE);
	}
}

void Display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(guiShaderProgramObject);

	vmath::mat4 modelViewMatrix = vmath::mat4::identity();
	vmath::mat4 modelViewProjectionMatrix = vmath::mat4::identity();

	modelViewMatrix = vmath::translate(0.0f, 0.0f, -5.0f);
	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(guiMVPUniform, 1, GL_FALSE, modelViewProjectionMatrix);

	//
	// Texture
	//
	glActiveTexture(GL_TEXTURE0);
	//glBindTexture(GL_TEXTURE_2D, guiTextureSmiley);
	glUniform1i(guiTextureSamplerUniform, 0);

	glBindVertexArray(guiVAO);

	GLfloat squareTexcoord[8] = { 0 };

	if (giFlag == 1)
	{
		glBindTexture(GL_TEXTURE_2D, guiTextureSmiley);

		squareTexcoord[0] = 0.5f;
		squareTexcoord[1] = 0.5f;
		squareTexcoord[2] = 0.0f;
		squareTexcoord[3] = 0.5f;
		squareTexcoord[4] = 0.0f;
		squareTexcoord[5] = 0.0f;
		squareTexcoord[6] = 0.5f;
		squareTexcoord[7] = 0.0f;
	}
	else if (giFlag == 2)
	{
		glBindTexture(GL_TEXTURE_2D, guiTextureSmiley);

		squareTexcoord[0] = 1.0f;
		squareTexcoord[1] = 1.0f;
		squareTexcoord[2] = 0.0f;
		squareTexcoord[3] = 1.0f;
		squareTexcoord[4] = 0.0f;
		squareTexcoord[5] = 0.0f;
		squareTexcoord[6] = 1.0f;
		squareTexcoord[7] = 0.0f;
	}
	else if (giFlag == 3)
	{
		glBindTexture(GL_TEXTURE_2D, guiTextureSmiley);

		squareTexcoord[0] = 2.0f;
		squareTexcoord[1] = 2.0f;
		squareTexcoord[2] = 0.0f;
		squareTexcoord[3] = 2.0f;
		squareTexcoord[4] = 0.0f;
		squareTexcoord[5] = 0.0f;
		squareTexcoord[6] = 2.0f;
		squareTexcoord[7] = 0.0f;
	}
	else if (giFlag == 4)
	{
		glBindTexture(GL_TEXTURE_2D, guiTextureSmiley);

		squareTexcoord[0] = 0.5f;
		squareTexcoord[1] = 0.5f;
		squareTexcoord[2] = 0.5f;
		squareTexcoord[3] = 0.5f;
		squareTexcoord[4] = 0.5f;
		squareTexcoord[5] = 0.5f;
		squareTexcoord[6] = 0.5f;
		squareTexcoord[7] = 0.5f;
	}
	else if (giFlag == 0)
	{
		glBindTexture(GL_TEXTURE_2D, guiTextureWhite);

		squareTexcoord[0] = 1.0f;
		squareTexcoord[1] = 1.0f;
		squareTexcoord[2] = 0.0f;
		squareTexcoord[3] = 1.0f;
		squareTexcoord[4] = 0.0f;
		squareTexcoord[5] = 0.0f;
		squareTexcoord[6] = 1.0f;
		squareTexcoord[7] = 0.0f;
	}

	glBindBuffer(GL_ARRAY_BUFFER, guiVBOTexture);
	glBufferData(GL_ARRAY_BUFFER, sizeof(squareTexcoord), squareTexcoord, GL_DYNAMIC_DRAW);
	glBindTexture(GL_ARRAY_BUFFER, 0);

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

	glBindVertexArray(0);

	glUseProgram(0);

	SwapBuffers(ghDC);
}

void Resize(int iWidth, int iHeight)
{
	if (iHeight == 0)
	{
		iHeight = 1;
	}

	glViewport(0, 0, (GLsizei)iWidth, (GLsizei)iHeight);

	gPerspectiveProjectionMatrix = vmath::perspective(45.0f, (GLfloat)iWidth / (GLfloat)iHeight, 0.1f, 100.0f);
}

void MakeCheckImage(void)
{
	//code
	int i, j;
	for (i = 0; i < 64; i++)
	{
		for (j = 0; j < 64; j++)
		{
			//c = (((i & 0x8) == 0) ^ ((j & 0x8) == 0)) * 255;
			TextureWhite[i][j][0] = (GLubyte)255;
			TextureWhite[i][j][1] = (GLubyte)255;
			TextureWhite[i][j][2] = (GLubyte)255;
			TextureWhite[i][j][3] = (GLubyte)255;
		}
	}
}

int LoadGLTexture(GLuint* texture, TCHAR imageResourceId[])
{
	//local variables
	HBITMAP hBitmap;
	BITMAP bmp;
	int iStatus = FALSE;

	//code
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

void LoadGLWhiteTexture(void)
{
	//function declaration
	void MakeCheckImage(void);

	//code
	MakeCheckImage();

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glGenTextures(1, &guiTextureWhite);
	glBindTexture(GL_TEXTURE_2D, guiTextureWhite);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 64, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, TextureWhite);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
}

void UnInitialize()
{
	if (true == gbIsFullScreen)
	{
		dwStyle = GetWindowLong(ghWnd, GWL_STYLE);
		SetWindowLong(ghWnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghWnd, &wpPrev);
		SetWindowPos(ghWnd, HWND_TOP, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED | SWP_NOMOVE);

		ShowCursor(TRUE);
	}

	if (0 != guiVBOPosition)
	{
		glDeleteBuffers(1, &guiVBOPosition);
		guiVBOPosition = 0;
	}

	if (0 != guiVBOTexture)
	{
		glDeleteBuffers(1, &guiVBOTexture);
		guiVBOTexture = 0;
	}

	if (0 != guiTextureSmiley)
	{
		glDeleteTextures(1, &guiTextureSmiley);
		guiTextureSmiley = 0;
	}

	if (0 != guiTextureWhite)
	{
		glDeleteTextures(1, &guiTextureWhite);
		guiTextureWhite = 0;
	}

	if (0 != guiVAO)
	{
		glDeleteVertexArrays(1, &guiVAO);
		guiVAO = 0;
	}

	glDetachShader(guiShaderProgramObject, guiFragmentShaderObject);
	glDetachShader(guiShaderProgramObject, guiVertexShaderObject);

	glDeleteShader(guiFragmentShaderObject);
	guiFragmentShaderObject = 0;

	glDeleteShader(guiVertexShaderObject);
	guiVertexShaderObject = 0;

	glDeleteProgram(guiShaderProgramObject);
	guiShaderProgramObject = 0;

	glUseProgram(0);

	wglMakeCurrent(NULL, NULL);

	wglDeleteContext(ghRC);
	ghRC = NULL;

	ReleaseDC(ghWnd, ghDC);
	ghDC = NULL;

	if (NULL != gpFile)
	{
		fprintf(gpFile, "Exitting successfully.\n");
		fclose(gpFile);
		gpFile = NULL;
	}

	return;
}
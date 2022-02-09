#include <Windows.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include "../common/vmath.h"
#include "02_Main.h"

#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")

#define ORTHO			200.0f

HDC ghDC = NULL;
HWND ghWnd = NULL;
HGLRC ghRC = NULL;

FILE* gpFile = NULL;

bool gbIsActivate = false;
bool gbIsFullscreen = false;
bool gbIsEscKeyPressed = false;

DWORD gdwStyle = 0;
WINDOWPLACEMENT wpPrev = { 0 };

GLuint guiVAO;
GLuint guiVBO;
GLuint gMVPUniform;

vmath::mat4 gOrthographicProjectionMatrix;

GLuint guiVertexShaderObject = 0;
GLuint guiFragmentShaderObject = 0;
GLuint guiShaderProgramObject = 0;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	MSG msg = { 0 };
	HWND hWnd = NULL;
	bool bDone = false;
	WNDCLASSEX WndClass = { 0 };
	WCHAR wchClassName[] = L"Orth 2D Triangle";

	if (fopen_s(&gpFile, "OGLLog.txt", "w"))
	{
		MessageBox(NULL, L"Error while creating log file", L"Error", MB_OK);
		return EXIT_FAILURE;
	}

	// Fill WndClass
	WndClass.cbSize = sizeof(WNDCLASSEX);
	WndClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	WndClass.cbWndExtra = 0;
	WndClass.cbClsExtra = 0;
	WndClass.lpfnWndProc = WndProc;
	WndClass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	WndClass.hIcon = (HICON)LoadIcon(NULL, IDI_APPLICATION);
	WndClass.hIconSm = (HICON)LoadIcon(NULL, IDI_APPLICATION);
	WndClass.hCursor = (HCURSOR)LoadCursor(NULL, IDC_ARROW);
	WndClass.hInstance = hInstance;
	WndClass.lpszClassName = wchClassName;
	WndClass.lpszMenuName = NULL;

	// Register WndClass
	if (!RegisterClassEx(&WndClass))
	{
		fprintf(gpFile, "Error while registering the wndclass.\n");
		fclose(gpFile);
		return EXIT_FAILURE;
	}

	// Create Window
	hWnd = CreateWindowEx(
		WS_EX_APPWINDOW,
		wchClassName,
		L"Ortho 2D triangle",
		WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
		120,
		100,
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
		return EXIT_FAILURE;
	}

	ghWnd = hWnd;

	ShowWindow(hWnd, iCmdShow);
	SetForegroundWindow(hWnd);
	SetFocus(hWnd);

	// Initialize
	bool bRet = Initialize();
	if (false == bRet)
	{
		fprintf(gpFile, "Error while Initialize().\n");
		fclose(gpFile);
		DestroyWindow(hWnd);
		hWnd = NULL;
		return EXIT_FAILURE;
	}

	// Game loop
	while (false == bDone)
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

	Uninitialize();

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
			if (false == gbIsFullscreen)
			{
				ToggleFullscreen();
				gbIsFullscreen = true;
			}
			else
			{
				ToggleFullscreen();
				gbIsFullscreen = false;
			}
			break;

		default:
			break;
		}
		break;

	case WM_LBUTTONDOWN:
		break;

	case WM_CLOSE:
		Uninitialize();
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
		"uniform mat4 u_mvp_matrix;" \
		"void main(void)" \
		"{" \
		"gl_Position = u_mvp_matrix * vPosition;" \
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
				Uninitialize();
				return false;
			}
		}
	}

	// Create fragment shader
	guiFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar* glchFragmentShaderSource =
		"#version 430 core" \
		"\n" \
		"out vec4 FragColor;" \
		"void main(void)" \
		"{" \
		"FragColor = vec4(0.0, 1.0, 0.0, 1.0);" \
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
				Uninitialize();
				return false;
			}
		}
	}

	// Create shader program
	guiShaderProgramObject = glCreateProgram();

	glAttachShader(guiShaderProgramObject, guiVertexShaderObject);
	glAttachShader(guiShaderProgramObject, guiFragmentShaderObject);

	glBindAttribLocation(guiShaderProgramObject, OGL_ATTRIBUTE_POSITION, "vPosition");

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
				Uninitialize();
				return false;
			}
		}
	}

	// get mvp uniform location
	gMVPUniform = glGetUniformLocation(guiShaderProgramObject, "u_mvp_matrix");

	const GLfloat triangleVertices[] =
	{
		0.0f, 100.0f, 0.0f,
		-100.0f, -100.0f, 0.0f,
		100.0f, -100.0f, 0.0f
	};

	glGenVertexArrays(1, &guiVAO);
	glBindVertexArray(guiVAO);

	glGenBuffers(1, &guiVBO);
	glBindBuffer(GL_ARRAY_BUFFER, guiVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(triangleVertices), triangleVertices, GL_STATIC_DRAW);

	glVertexAttribPointer(OGL_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_POSITION);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	glShadeModel(GL_SMOOTH);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	gOrthographicProjectionMatrix = vmath::mat4::identity();

	Resize(OGL_WINDOW_WIDTH, OGL_WINDOW_HEIGHT);

	return true;
}

void Uninitialize()
{
	if (true == gbIsFullscreen)
	{
		gdwStyle = GetWindowLong(ghWnd, GWL_STYLE);
		SetWindowLong(ghWnd, GWL_STYLE, gdwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghWnd, &wpPrev);
		SetWindowPos(ghWnd, HWND_TOP, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED | SWP_NOMOVE);

		ShowCursor(TRUE);
	}

	if (0 != guiVBO)
	{
		glDeleteBuffers(1, &guiVBO);
		guiVBO = 0;
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

void ToggleFullscreen()
{
	MONITORINFO mi = { 0 };

	if (false == gbIsFullscreen)
	{
		gdwStyle = GetWindowLong(ghWnd, GWL_STYLE);
		if (gdwStyle & WS_OVERLAPPEDWINDOW)
		{
			mi.cbSize = sizeof(MONITORINFO);
			if (
				GetWindowPlacement(ghWnd, &wpPrev) &&
				GetMonitorInfo(MonitorFromWindow(ghWnd, MONITORINFOF_PRIMARY), &mi)
				)
			{
				SetWindowLong(ghWnd, GWL_STYLE, gdwStyle & ~WS_OVERLAPPEDWINDOW);
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
		SetWindowLong(ghWnd, GWL_STYLE, gdwStyle | WS_OVERLAPPEDWINDOW);
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

void Display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(guiShaderProgramObject);

	vmath::mat4 modelViewMatrix = vmath::mat4::identity();
	vmath::mat4 modelViewProjectionMatrix = vmath::mat4::identity();

	modelViewProjectionMatrix = gOrthographicProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(gMVPUniform, 1, GL_FALSE, modelViewProjectionMatrix);

	glBindVertexArray(guiVAO);

	glDrawArrays(GL_TRIANGLES, 0, 3);

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

	if (iWidth < iHeight)
	{
		gOrthographicProjectionMatrix = vmath::ortho(-100.0f, 100.0f, (-100.0f * ((GLfloat)iHeight / (GLfloat)iWidth)), (100.0f * ((GLfloat)iHeight / (GLfloat)iWidth)), -100.0f, 100.0f);
	}
	else
	{
		gOrthographicProjectionMatrix = vmath::ortho((-ORTHO * ((GLfloat)iWidth / (GLfloat)iHeight)), (ORTHO * ((GLfloat)iWidth / (GLfloat)iHeight)), -ORTHO, ORTHO, -1.0f, ORTHO);
	}
}
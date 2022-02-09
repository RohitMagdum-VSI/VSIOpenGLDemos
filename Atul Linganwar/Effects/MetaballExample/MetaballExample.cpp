#include <Windows.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include "../common/vmath.h"
#include "../common/Common.h"
#include "../common/Shapes.h"

#include "MetaballExample.h"

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "GeometricShapes.lib")

HDC ghDC = NULL;
HWND ghWnd = NULL;
HGLRC ghRC = NULL;

FILE* gpFile = NULL;

bool gbIsActivate = false;
bool gbIsFullScreen = false;
bool gbIsEscKeyPressed = false;

GLfloat gMouseX = 0.0f;
GLfloat gMouseY = 0.0f;
bool bIsMouseButtonPressed = false;

DWORD dwStyle = 0;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };

GLfloat gfAngle;
GLfloat gWindowWidth;
GLfloat gWindowHeight;

vmath::mat4 gPerspectiveProjectionMatrix;
vmath::mat4 gOrthographicProjectionMatrix;

vmath::mat4 gmat4ViewMatrix = vmath::mat4::identity();
vmath::mat4 gmat4ModelMatrix = vmath::mat4::identity();
vmath::mat4 gmat4TranslationMatrix = vmath::mat4::identity();
vmath::mat4 gmat4RotationMatrix = vmath::mat4::identity();
vmath::mat4 gmat4ScaleMatrix = vmath::mat4::identity();

// WinMain
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	MSG msg = { 0 };
	HWND hWnd = NULL;
	bool bDone = false;
	WNDCLASSEX WndClass = { 0 };
	WCHAR wszClassName[] = L"First Scene Rotating Planets";

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
			Update();

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
		gWindowWidth = LOWORD(lParam);
		gWindowHeight = HIWORD(lParam);
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

		default:
			break;
		}
		break;

	case WM_LBUTTONDOWN:
		break;

	case WM_RBUTTONDOWN:
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



	gPerspectiveProjectionMatrix = vmath::mat4::identity();
	gOrthographicProjectionMatrix = vmath::mat4::identity();

	Resize((int)gWindowWidth, (int)gWindowHeight);

	return true;
}

void Display(void)
{
	

	SwapBuffers(ghDC);
}

void Update(void)
{
	gfAngle = gfAngle + 0.2f;
	if (gfAngle > 360.0f)
	{
		gfAngle = 0.0f;
	}

	return;
}

int LoadGLTextures(GLuint* texture, TCHAR imageResourceId[])
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

		glTexImage2D(
			GL_TEXTURE_2D,
			0,
			GL_RGB,
			bmp.bmWidth,
			bmp.bmHeight,
			0,
			GL_BGR,
			GL_UNSIGNED_BYTE,
			bmp.bmBits
		);

		glGenerateMipmap(GL_TEXTURE_2D);

		DeleteObject(hBitmap);
	}

	return iStatus;
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
		gOrthographicProjectionMatrix = vmath::ortho(-ORTHO, ORTHO, (-ORTHO * ((GLfloat)iHeight / (GLfloat)iWidth)), (ORTHO * ((GLfloat)iHeight / (GLfloat)iWidth)), -ORTHO, ORTHO);
	}
	else
	{
		gOrthographicProjectionMatrix = vmath::ortho((-ORTHO * ((GLfloat)iWidth / (GLfloat)iHeight)), (ORTHO * ((GLfloat)iWidth / (GLfloat)iHeight)), -ORTHO, ORTHO, -ORTHO, ORTHO);
	}

	gPerspectiveProjectionMatrix = vmath::perspective(45.0f, (GLfloat)iWidth / (GLfloat)iHeight, 0.1f, 100000.0f);
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
		// ShowCursor(FALSE);
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

void ClearMatrices()
{
	gmat4ModelMatrix = vmath::mat4::identity();
	gmat4TranslationMatrix = vmath::mat4::identity();
	gmat4RotationMatrix = vmath::mat4::identity();
	gmat4ScaleMatrix = vmath::mat4::identity();
}
#include <Windows.h>
#include <gl\GL.h>
#include <gl\GLU.h>

#include "03_3DTexture.h"

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glu32.lib")

#define WIN_WIDTH	800
#define WIN_HEIGHT	600

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

GLuint	gluiTextureStone;
GLuint	gluiTextureSmiley;

GLfloat g_fAngleCube;
GLfloat g_fAnglePyramid;

#define	CLASS_NAME		TEXT("03: 3D Texture")

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	VOID initialize();
	VOID display();
	VOID uninitialize();
	VOID UpdateAngle();

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

				UpdateAngle();
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

	case WM_KEYDOWN:
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

	HDC hDC;
	int iPixelFormatIndex;
	PIXELFORMATDESCRIPTOR pfd;

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
	//-	Change 2 For 3D

	//	Change for Texture.
	glEnable(GL_TEXTURE_2D);	//	Enable texture mapping.

	LoadGLTextures(&gluiTextureStone, MAKEINTRESOURCE(IDBITMAP_STONE));
	LoadGLTextures(&gluiTextureSmiley, MAKEINTRESOURCE(IDBITMAP_SMILEY));

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

	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);	//Pixel alignment mode (word alignment / 4 bytes)

	glBindTexture(GL_TEXTURE_2D, *pgluiTexture);	//	Bind texture.

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);	//Generate mipmapped texture(3 byets, width, height & data from bmp)

	gluBuild2DMipmaps(
					GL_TEXTURE_2D,
					3,	// type of image format.
					bmp.bmWidth,
					bmp.bmHeight,
					GL_BGR_EXT,
					GL_UNSIGNED_BYTE,	// Type of last parameter.
					bmp.bmBits
					);

	DeleteObject(hBitmap);	//	Delete unwanted object.

	return TRUE;
}


VOID display()
{
	VOID DrawMultiColorCube();
	VOID DrawMultiColorPyramid();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	//	Change 3 for 3D

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(2.0f, 0.0f, -7.0f);
	glRotatef(g_fAngleCube, 1.0f, 0.0f, 0.0f);
	DrawMultiColorCube();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(-2.0f, 0.0f, -7.0f);
	glRotatef(g_fAnglePyramid, 0.0f, 1.0f, 0.0f);
	DrawMultiColorPyramid();

	SwapBuffers(g_hdc);
}


VOID DrawMultiColorPyramid()
{
	glBindTexture(GL_TEXTURE_2D, gluiTextureStone);

	glBegin(GL_TRIANGLES);
	//Front face.
	glTexCoord2f(0.5f, 1.0f);
	glVertex3f(0.0f, 1.0f, 0.0f);

	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(-1.0f, -1.0f, 1.0f);

	glTexCoord2f(1.0f, 0.0f);
	glVertex3f(1.0f, -1.0f, 1.0f);

	//	Right face
	glTexCoord2f(0.5f, 1.0f);
	glVertex3f(0.0f, 1.0f, 0.0f);

	glTexCoord2f(1.0f, 0.0f);
	glVertex3f(1.0f, -1.0f, 1.0f);

	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(1.0f, -1.0f, -1.0f);

	//	Back face
	glTexCoord2f(0.5f, 1.0f);
	glVertex3f(0.0f, 1.0f, 0.0f);

	glTexCoord2f(1.0f, 0.0f);
	glVertex3f(-1.0f, -1.0f, -1.0f);

	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(1.0f, -1.0f, -1.0f);

	//	Left face
	glTexCoord2f(0.5f, 1.0f);
	glVertex3f(0.0f, 1.0f, 0.0f);

	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(-1.0f, -1.0f, -1.0f);

	glTexCoord2f(1.0f, 0.0f);
	glVertex3f(-1.0f, -1.0f, 1.0f);

	glEnd();
}


VOID DrawMultiColorCube()
{
	glBindTexture(GL_TEXTURE_2D, gluiTextureSmiley);

	glBegin(GL_QUADS);
	//Front face.
	glTexCoord2f(1.0f, 1.0f);
	glVertex3f(1.0f, 1.0f, 1.0f);

	glTexCoord2f(1.0f, 0.0f);
	glVertex3f(1.0f, -1.0f, 1.0f);

	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(-1.0f, -1.0f, 1.0f);

	glTexCoord2f(0.0f, 1.0f);
	glVertex3f(-1.0f, 1.0f, 1.0f);

	//	Right face
	glTexCoord2f(1.0f, 1.0f);
	glVertex3f(1.0f, 1.0f, -1.0f);

	glTexCoord2f(0.0f, 1.0f);
	glVertex3f(1.0f, 1.0f, 1.0f);

	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(1.0f, -1.0f, 1.0f);

	glTexCoord2f(1.0f, 0.0f);
	glVertex3f(1.0f, -1.0f, -1.0f);


	//	Top face
	glTexCoord2f(1.0f, 1.0f);
	glVertex3f(1.0f, 1.0f, -1.0f);

	glTexCoord2f(0.0f, 1.0f);
	glVertex3f(-1.0f, 1.0f, -1.0f);

	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(-1.0f, 1.0f, 1.0f);

	glTexCoord2f(1.0f, 0.0f);
	glVertex3f(1.0f, 1.0f, 1.0f);

	//	Back face.
	glTexCoord2f(0.0f, 1.0f);
	glVertex3f(1.0f, 1.0f, -1.0f);

	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(1.0f, -1.0f, -1.0f);

	glTexCoord2f(1.0f, 0.0f);
	glVertex3f(-1.0f, -1.0f, -1.0f);

	glTexCoord2f(1.0f, 1.0f);
	glVertex3f(-1.0f, 1.0f, -1.0f);

	//	Left face
	glTexCoord2f(0.0f, 1.0f);
	glVertex3f(-1.0f, 1.0f, -1.0f);

	glTexCoord2f(1.0f, 1.0f);
	glVertex3f(-1.0f, 1.0f, 1.0f);

	glTexCoord2f(1.0f, 0.0f);
	glVertex3f(-1.0f, -1.0f, 1.0f);

	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(-1.0f, -1.0f, -1.0f);

	//	Bottom face
	glTexCoord2f(1.0f, 1.0f);
	glVertex3f(1.0f, -1.0f, -1.0f);

	glTexCoord2f(0.0f, 1.0f);
	glVertex3f(-1.0f, -1.0f, -1.0f);

	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(-1.0f, -1.0f, 1.0f);

	glTexCoord2f(1.0f, 0.0f);
	glVertex3f(1.0f, -1.0f, 1.0f);

	glEnd();
}


VOID UpdateAngle()
{
	g_fAngleCube = g_fAngleCube + 0.1f;

	if (g_fAngleCube >= 360)
	{
		g_fAngleCube = 0.0f;
	}

	g_fAnglePyramid = g_fAnglePyramid + 0.1f;

	if (g_fAnglePyramid >= 360)
	{
		g_fAnglePyramid = 0.0f;
	}
}


VOID resize(int iWidth, int iHeight)
{
	if (0 == iHeight)
	{
		iHeight = 1;
	}

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	//
	//	znear and zfar must positive.
	//
	if (iWidth <= iHeight)
	{
		gluPerspective(45, (GLfloat)iHeight / (GLfloat)iWidth, 0.1f, 100.0f);
	}
	else
	{
		gluPerspective(45, (GLfloat)iWidth / (GLfloat)iHeight, 0.1f, 100.0f);
	}

	glViewport(0, 0, iWidth, iHeight);
}

VOID uninitialize()
{
	if (gluiTextureStone)
	{
		glDeleteTextures(1, &gluiTextureStone);
		gluiTextureStone = 0;
	}

	if (gluiTextureSmiley)
	{
		glDeleteTextures(1, &gluiTextureSmiley);
		gluiTextureSmiley = 0;
	}

	if (true == g_boFullScreen)
	{
		SetWindowLong(g_hWnd, GWL_STYLE, g_dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(g_hWnd, &g_WindowPlacementPrev);
		SetWindowPos(g_hWnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);

		ShowCursor(TRUE);
	}

	wglMakeCurrent(NULL, NULL);
	wglDeleteContext(g_hRC);
	g_hRC = NULL;

	ReleaseDC(g_hWnd, g_hdc);
	g_hdc = NULL;

	DestroyWindow(g_hWnd);
	g_hWnd = NULL;
}

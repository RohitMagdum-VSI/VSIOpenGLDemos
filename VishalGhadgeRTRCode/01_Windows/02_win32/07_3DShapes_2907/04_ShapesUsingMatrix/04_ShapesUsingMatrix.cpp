#include <Windows.h>
#include <gl\GL.h>
#include <gl\GLU.h>
#define _USE_MATH_DEFINES		1
#include <math.h>

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

float g_fAnglePyramid = 0.0f;

//
//	Define require matrix.
//
GLfloat g_glfIdentityMatrix[16];	//	For glLoadIdentity()
GLfloat g_glfTranslationMatrix[16];	//	glTranslate()
GLfloat g_glfScaleMatrix[16];		//	glScale()
GLfloat g_glfXRotationMatrix[16];	//	glRotate(angle, 1.0f, 0.0f, 0.0f)
GLfloat g_glfYRotationMatrix[16];	//	glRotate(angle, 0.0f, 1.0f, 0.0f)
GLfloat g_glfZRotationMatrix[16];	//	glRotate(angle, 0.0f, 0.0f, 1.0f)

#define	CLASS_NAME		TEXT("3D : Rotate Cube Using Matrix ")

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

	//+	Initialize Identity matrix.

	//	X
	g_glfIdentityMatrix[0] = 1.0f;
	g_glfIdentityMatrix[1] = 0.0f;
	g_glfIdentityMatrix[2] = 0.0f;
	g_glfIdentityMatrix[3] = 0.0f;

	//	Y
	g_glfIdentityMatrix[4] = 0.0f;
	g_glfIdentityMatrix[5] = 1.0f;
	g_glfIdentityMatrix[6] = 0.0f;
	g_glfIdentityMatrix[7] = 0.0f;

	//	Z
	g_glfIdentityMatrix[8] = 0.0f;
	g_glfIdentityMatrix[9] = 0.0f;
	g_glfIdentityMatrix[10] = 1.0f;
	g_glfIdentityMatrix[11] = 0.0f;

	//	??
	g_glfIdentityMatrix[12] = 0.0f;
	g_glfIdentityMatrix[13] = 0.0f;
	g_glfIdentityMatrix[14] = 0.0f;
	g_glfIdentityMatrix[15] = 1.0f;

	//-	Initialize Identity matrix.

	//+	Initialize Translate matrix.

	//	X
	g_glfTranslationMatrix[0] = 1.0f;
	g_glfTranslationMatrix[1] = 0.0f;
	g_glfTranslationMatrix[2] = 0.0f;
	g_glfTranslationMatrix[3] = 0.0f;

	//	Y
	g_glfTranslationMatrix[4] = 0.0f;
	g_glfTranslationMatrix[5] = 1.0f;
	g_glfTranslationMatrix[6] = 0.0f;
	g_glfTranslationMatrix[7] = 0.0f;

	//	Z
	g_glfTranslationMatrix[8] = 0.0f;
	g_glfTranslationMatrix[9] = 0.0f;
	g_glfTranslationMatrix[10] = 1.0f;
	g_glfTranslationMatrix[11] = 0.0f;

	//	??
	g_glfTranslationMatrix[12] = 0.0f;
	g_glfTranslationMatrix[13] = 0.0f;
	g_glfTranslationMatrix[14] = -6;
	g_glfTranslationMatrix[15] = 1.0f;

	//-	Initialize Translate matrix.

	//+	Initialize Scale matrix.

	//	X
	g_glfScaleMatrix[0] = 0.75f;
	g_glfScaleMatrix[1] = 0.0f;
	g_glfScaleMatrix[2] = 0.0f;
	g_glfScaleMatrix[3] = 0.0f;

	//	Y
	g_glfScaleMatrix[4] = 0.0f;
	g_glfScaleMatrix[5] = 0.75f;
	g_glfScaleMatrix[6] = 0.0f;
	g_glfScaleMatrix[7] = 0.0f;

	//	Z
	g_glfScaleMatrix[8] = 0.0f;
	g_glfScaleMatrix[9] = 0.0f;
	g_glfScaleMatrix[10] = 0.75f;
	g_glfScaleMatrix[11] = 0.0f;

	//	??
	g_glfScaleMatrix[12] = 0.0f;
	g_glfScaleMatrix[13] = 0.0f;
	g_glfScaleMatrix[14] = 0.0f;
	g_glfScaleMatrix[15] = 1.0f;

	//-	Initialize Scale matrix.

	//
	//	Resize.
	//
	resize(WIN_WIDTH, WIN_HEIGHT);
}


VOID display()
{
	VOID DrawMultiColorCube();

	float fAngleRadian;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	//	Change 3 for 3D

	glMatrixMode(GL_MODELVIEW);
	//	glLoadIdentity();
	glLoadMatrixf(g_glfIdentityMatrix);

	//glTranslatef(0.0f, 0.0f, -6.0f);
	glMultMatrixf(g_glfTranslationMatrix);

	//	glScale()
	glMultMatrixf(g_glfScaleMatrix);

	//glRotatef(g_fAnglePyramid, 0.0f, 1.0f, 0.0f);
	fAngleRadian = g_fAnglePyramid * (float)(M_PI / 180);

	//+	Define x-rotation matrix
	//	x
	g_glfXRotationMatrix[0] = 1.0f;
	g_glfXRotationMatrix[1] = 0.0f;
	g_glfXRotationMatrix[2] = 0.0f;
	g_glfXRotationMatrix[3] = 0.0f;

	//	y
	g_glfXRotationMatrix[4] = 0.0f;
	g_glfXRotationMatrix[5] = cos(fAngleRadian);
	g_glfXRotationMatrix[6] = sin(fAngleRadian);
	g_glfXRotationMatrix[7] = 0.0f;

	// z
	g_glfXRotationMatrix[8] = 0.0f;
	g_glfXRotationMatrix[9] = -sin(fAngleRadian);
	g_glfXRotationMatrix[10] = cos(fAngleRadian);
	g_glfXRotationMatrix[11] = 0.0f;

	// ??
	g_glfXRotationMatrix[12] = 0.0f;
	g_glfXRotationMatrix[13] = 0.0f;
	g_glfXRotationMatrix[14] = 0.0f;
	g_glfXRotationMatrix[15] = 1.0f;

	//-	Define x-rotation matrix
	glMultMatrixf(g_glfXRotationMatrix);

	//+	Define y-rotation matrix
	//	x
	g_glfYRotationMatrix[0] = cos(fAngleRadian);
	g_glfYRotationMatrix[1] = 0.0f;
	g_glfYRotationMatrix[2] = - sin(fAngleRadian);
	g_glfYRotationMatrix[3] = 0.0f;

	//	y
	g_glfYRotationMatrix[4] = 0.0f;
	g_glfYRotationMatrix[5] = 1.0f;
	g_glfYRotationMatrix[6] = 0.0f;
	g_glfYRotationMatrix[7] = 0.0f;

	// z
	g_glfYRotationMatrix[8] = sin(fAngleRadian);
	g_glfYRotationMatrix[9] = 0.0f;
	g_glfYRotationMatrix[10] = cos(fAngleRadian);
	g_glfYRotationMatrix[11] = 0.0f;

	// ??
	g_glfYRotationMatrix[12] = 0.0f;
	g_glfYRotationMatrix[13] = 0.0f;
	g_glfYRotationMatrix[14] = 0.0f;
	g_glfYRotationMatrix[15] = 1.0f;

	//-	Define y-rotation matrix
	glMultMatrixf(g_glfYRotationMatrix);

	//+	Define z-rotation matrix
	//	x
	g_glfZRotationMatrix[0] = cos(fAngleRadian);
	g_glfZRotationMatrix[1] = sin(fAngleRadian); 
	g_glfZRotationMatrix[2] = 0.0f;
	g_glfZRotationMatrix[3] = 0.0f;

	//	y
	g_glfZRotationMatrix[4] = -sin(fAngleRadian);
	g_glfZRotationMatrix[5] = cos(fAngleRadian);
	g_glfZRotationMatrix[6] = 0.0f;
	g_glfZRotationMatrix[7] = 0.0f;

	// z
	g_glfZRotationMatrix[8] = 0.0f;
	g_glfZRotationMatrix[9] = 0.0f;
	g_glfZRotationMatrix[10] = 1;
	g_glfZRotationMatrix[11] = 0.0f;

	// ??
	g_glfZRotationMatrix[12] = 0.0f;
	g_glfZRotationMatrix[13] = 0.0f;
	g_glfZRotationMatrix[14] = 0.0f;
	g_glfZRotationMatrix[15] = 1.0f;

	//-	Define y-rotation matrix
	glMultMatrixf(g_glfZRotationMatrix);

	DrawMultiColorCube();

	SwapBuffers(g_hdc);
}


VOID DrawMultiColorCube()
{
	glBegin(GL_QUADS);
	//Front face.
	glColor3f(1.0f, 0.0f, 0.0f);	//	Red
	glVertex3f(1.0f, 1.0f, 1.0f);
	glVertex3f(1.0f, -1.0f, 1.0f);
	glVertex3f(-1.0f, -1.0f, 1.0f);
	glVertex3f(-1.0f, 1.0f, 1.0f);

	//	Right face
	glColor3f(0.0f, 0.0f, 1.0f);	//	Blue
	glVertex3f(1.0f, 1.0f, -1.0f);
	glVertex3f(1.0f, 1.0f, 1.0f);
	glVertex3f(1.0f, -1.0f, 1.0f);
	glVertex3f(1.0f, -1.0f, -1.0f);


	//	Top face
	glColor3f(0.0f, 1.0f, 0.0f);	//	Green
	glVertex3f(1.0f, 1.0f, -1.0f);
	glVertex3f(-1.0f, 1.0f, -1.0f);
	glVertex3f(-1.0f, 1.0f, 1.0f);
	glVertex3f(1.0f, 1.0f, 1.0f);

	//	Back face.
	glColor3f(1.0f, 1.0f, 0.0f);
	glVertex3f(1.0f, 1.0f, -1.0f);
	glVertex3f(1.0f, -1.0f, -1.0f);
	glVertex3f(-1.0f, -1.0f, -1.0f);
	glVertex3f(-1.0f, 1.0f, -1.0f);

	//	Left face
	glColor3f(0.0f, 1.0f, 1.0f);
	glVertex3f(-1.0f, 1.0f, -1.0f);
	glVertex3f(-1.0f, 1.0f, 1.0f);
	glVertex3f(-1.0f, -1.0f, 1.0f);
	glVertex3f(-1.0f, -1.0f, -1.0f);

	//	Bottom face
	glColor3f(1.0f, 0.0f, 1.0f);
	glVertex3f(1.0f, -1.0f, -1.0f);
	glVertex3f(-1.0f, -1.0f, -1.0f);
	glVertex3f(-1.0f, -1.0f, 1.0f);
	glVertex3f(1.0f, -1.0f, 1.0f);

	glEnd();
}



VOID UpdateAngle()
{
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

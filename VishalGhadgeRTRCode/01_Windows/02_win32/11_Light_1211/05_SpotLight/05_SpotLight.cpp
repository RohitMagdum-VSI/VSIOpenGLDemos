#include <Windows.h>
#include <gl\GL.h>
#include <gl\GLU.h>

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

GLfloat g_arrLight0Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrLight0Diffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_arrLight0Specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_arrLight0Position[] = { 0.0f, 0.0f, 1.0f, 0.0f };

GLfloat g_arrLight1Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrLight1Diffuse[] = { 1.0f, 0.0f, 0.0f, 1.0f };	//	Try different colors.
GLfloat g_arrLight1Position[] = { 0.0f, 0.0f, 1.0f, 1.0f };
GLfloat g_arrLight1SpotDirection[] = { 0.0f, 0.0f, -1.0f};


GLfloat g_arrMaterialSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_arrMaterialShininess[] = { 50.0f };

BOOLEAN g_bEnableLight = FALSE;
BOOLEAN g_EnablebLight1 = FALSE;

#define	CLASS_NAME		TEXT("05 : Spot Light")

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	VOID initialize();
	VOID display();
	VOID uninitialize();

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

		case 'l':
		case 'L':
			if (FALSE == g_bEnableLight)
			{
				glEnable(GL_LIGHTING);
				g_bEnableLight = TRUE;
			}
			else
			{
				glDisable(GL_LIGHTING);
				g_bEnableLight = FALSE;
			}
			break;

		case 't':
		case 'T':
			if (FALSE == g_EnablebLight1)
			{
				glDisable(GL_LIGHT0);
				glEnable(GL_LIGHT1);
				g_EnablebLight1 = TRUE;
			}
			else
			{
				glDisable(GL_LIGHT1);
				glEnable(GL_LIGHT0);
				g_EnablebLight1 = FALSE;
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
	void InitLight();

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

	//+	Change For Light
	InitLight();

	glEnable(GL_LIGHT0);

	//
	//	Resize.
	//
	resize(WIN_WIDTH, WIN_HEIGHT);
}

void InitLight()
{
	glLightfv(GL_LIGHT0, GL_AMBIENT, g_arrLight0Ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, g_arrLight0Diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, g_arrLight0Specular);
	glLightfv(GL_LIGHT0, GL_POSITION, g_arrLight0Position);

	glLightfv(GL_LIGHT1, GL_AMBIENT, g_arrLight1Ambient);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, g_arrLight1Diffuse);
	glLightfv(GL_LIGHT1, GL_POSITION, g_arrLight1Position);
	glLightfv(GL_LIGHT1, GL_SPOT_DIRECTION, g_arrLight1SpotDirection);
	glLightf(GL_LIGHT1, GL_SPOT_CUTOFF, 45.0f);
	glLightf(GL_LIGHT1, GL_SPOT_EXPONENT, 0.2f);

	glLightf(GL_LIGHT1, GL_CONSTANT_ATTENUATION, 0.5f);
	glLightf(GL_LIGHT1, GL_LINEAR_ATTENUATION, 0.2f);
	glLightf(GL_LIGHT1, GL_QUADRATIC_ATTENUATION, 0.1f);

	glMaterialfv(GL_FRONT, GL_SPECULAR, g_arrMaterialSpecular);
	glMaterialfv(GL_FRONT, GL_SHININESS, g_arrMaterialShininess);
}

VOID display()
{
	GLUquadric *gluQuadric = NULL;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	//	Change 3 for 3D

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	//
	//	View transaformation.
	//
	//gluLookAt(0.0f, 0.0f, -4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	//glTranslatef(0.0f, 0.0f,-1.0f);	//	Tryout
	glTranslatef(0.0f, 0.0f, -2.0f);

	//
	//	Draw sun.
	//
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	gluQuadric = gluNewQuadric();
	glColor3f(1.0f, 1.0f, 0.0f);
	gluSphere(gluQuadric, 0.25, 40, 40);

	SwapBuffers(g_hdc);
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
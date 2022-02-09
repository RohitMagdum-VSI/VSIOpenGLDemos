#include <Windows.h>
#include <gl\GL.h>
#include <gl\GLU.h>

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glu32.lib")
#pragma comment(lib, "Winmm.lib")

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

#define START_ANIMATE_I		-2.5f
#define START_ANIMATE_N		 2.0f
#define START_ANIMATE_D		0.0f
#define START_ANIMATE_I2	-2.5f
#define START_ANIMATE_A		2.5f
#define START_ANIMATE_FLAG	-4.0f

float g_fAnimateI = START_ANIMATE_I;
float g_fAnimateN = START_ANIMATE_N;
float g_fAnimateD = START_ANIMATE_D;
float g_fAnimateI2 = START_ANIMATE_I2;
float g_fAnimateA = START_ANIMATE_A;
float g_fAnimateFlag = START_ANIMATE_FLAG;
float g_fAnimateFlagX = START_ANIMATE_FLAG;

#define	CLASS_NAME		TEXT("India")

#define	AUDIO_NAME		TEXT("..\\audio\\Revival-Vande.wav")

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	VOID initialize();
	VOID display();
	VOID uninitialize();
	VOID UpdateLetters();

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

				UpdateLetters();
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

		case 'r':
		case 'R':
			g_fAnimateI = START_ANIMATE_I;
			g_fAnimateN = START_ANIMATE_N;
			g_fAnimateD = START_ANIMATE_D;
			g_fAnimateI2 = START_ANIMATE_I2;
			g_fAnimateA = START_ANIMATE_A;
			g_fAnimateFlag = START_ANIMATE_FLAG;
			g_fAnimateFlagX = START_ANIMATE_FLAG;

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

	//
	//	Resize.
	//
	resize(WIN_WIDTH, WIN_HEIGHT);

	//	temp
	//g_fAnimateI = -2.0f;
	//g_fAnimateN = 0.0f;
	//g_fAnimateD = -3.0f;
	PlaySound(AUDIO_NAME, NULL, SND_FILENAME | SND_ASYNC | SND_LOOP);

	ToggleFullScreen();
}


VOID UpdateLetters()
{
#define SPEED	0.0001f

	if (g_fAnimateI <= -2.0f)
	{
		g_fAnimateI = g_fAnimateI + SPEED;
		return;
	}

	if (g_fAnimateN >= 0.0f)
	{
		g_fAnimateN = g_fAnimateN - SPEED;
		return;
	}

	if (g_fAnimateD < 1.0f)
	{
		g_fAnimateD = g_fAnimateD + SPEED;// (0.001f);
		return;
	}

	if (g_fAnimateI2 <= 0.0f)
	{
		g_fAnimateI2 = g_fAnimateI2 + SPEED;
		return;
	}

	if (g_fAnimateA >= 1.5f)
	{
		g_fAnimateA = g_fAnimateA - SPEED;
		return;
	}

	if (g_fAnimateFlag <= 1.5f)
	{
		g_fAnimateFlag = g_fAnimateFlag + (SPEED * 3);
		return;
	}

	if (g_fAnimateFlagX <= -0.2f)
	{
		g_fAnimateFlagX = g_fAnimateFlagX + (SPEED * 2);
		return;
	}
}


VOID display()
{
	VOID DrawI();
	VOID DrawN();
	VOID DrawD();
	VOID DrawA();
	VOID DrawFlagLines();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	//	Change 3 for 3D

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(g_fAnimateI, 0.0f, -3.0f);
	DrawI();

	if (g_fAnimateN == START_ANIMATE_N)
	{
		SwapBuffers(g_hdc);
		return;
	}

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(-1.5f, g_fAnimateN, -3.0f);
	DrawN();

	if (g_fAnimateD == START_ANIMATE_D)
	{
		SwapBuffers(g_hdc);
		return;
	}

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(-0.5f, 0.0f, -3.0f);
	DrawD();

	if (g_fAnimateI2 == START_ANIMATE_I2)
	{
		SwapBuffers(g_hdc);
		return;
	}

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(0.5f, g_fAnimateI2, -3.0f);
	DrawI();

	if (g_fAnimateA == START_ANIMATE_A)
	{
		SwapBuffers(g_hdc);
		return;
	}

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(g_fAnimateA, 0.0f, -3.0f);
	DrawA();

	if (g_fAnimateFlag == START_ANIMATE_FLAG)
	{
		SwapBuffers(g_hdc);
		return;
	}

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(g_fAnimateFlag, 0.0f, -3.0f);
	DrawFlagLines();

	SwapBuffers(g_hdc);
}


VOID DrawI()
{
	glLineWidth(2.0f);
	glBegin(GL_LINES);
	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, 1.0f, 0.0f);
	glColor3f(0.0f, 1.0f, 0.0f);
	glVertex3f(0.0f, -1.0f, 0.0f);
	glEnd();
}


VOID DrawN()
{
	glLineWidth(2.0f);
	glBegin(GL_LINES);
	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, 1.0f, 0.0f);
	glColor3f(0.0f, 1.0f, 0.0f);
	glVertex3f(0.0f, -1.0f, 0.0f);

	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, 1.0f, 0.0f);
	glColor3f(0.0f, 1.0f, 0.0f);
	glVertex3f(0.5f, -1.0f, 0.0f);

	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex3f(0.5f, 1.0f, 0.0f);
	glColor3f(0.0f, 1.0f, 0.0f);
	glVertex3f(0.5f, -1.0f, 0.0f);

	glEnd();
}


VOID DrawD()
{
	GLfloat glfColor;

	glfColor = g_fAnimateD;

	glLineWidth(2.0f);
	glBegin(GL_LINES);
	glColor3f(glfColor, 0.0f, 0.0f);
	glVertex3f(0.0f, 1.0f, 0.0f);
	glColor3f(0.0f, glfColor, 0.0f);
	glVertex3f(0.0f, -1.0f, 0.0f);

	glColor3f(glfColor, 0.0f, 0.0f);
	glVertex3f(0.5f, 1.0f, 0.0f);
	glVertex3f(-0.02f, 1.0f, 0.0f);

	glColor3f(glfColor, 0.0f, 0.0f);
	glVertex3f(0.5f, 1.0f, 0.0f);
	glColor3f(0.0f, glfColor, 0.0f);
	glVertex3f(0.5f, -1.0f, 0.0f);

	glColor3f(0.0f, glfColor, 0.0f);
	glVertex3f(0.5f, -1.0f, 0.0f);
	glVertex3f(-0.02f, -1.0f, 0.0f);

	glEnd();
}


VOID DrawA()
{

	glLineWidth(2.0f);
	glBegin(GL_LINES);
	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, 1.0f, 0.0f);
	glColor3f(0.0f, 1.0f, 0.0f);
	glVertex3f(-0.4f, -1.0f, 0.0f);

	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, 1.0f, 0.0f);
	glColor3f(0.0f, 1.0f, 0.0f);
	glVertex3f(0.4f, -1.0f, 0.0f);
	glEnd();
}


VOID DrawFlagLines()
{
	glLineWidth(1.0f);
	glBegin(GL_LINES);
	glColor3f(1.0f, 0.0f, 0.0f);
	//glVertex3f(-0.2f, 0.005f, 0.0f);
	glVertex3f(g_fAnimateFlagX, 0.005f, 0.0f);
	glVertex3f(0.2f, 0.005f, 0.0f);

	glColor3f(1.0f, 1.0f, 1.0f);
	glVertex3f(g_fAnimateFlagX, 0.0f, 0.0f);
	glVertex3f(0.2f, 0.0f, 0.0f);

	glColor3f(0.0f, 1.0f, 0.0f);
	glVertex3f(g_fAnimateFlagX, -0.005f, 0.0f);
	glVertex3f(0.2f, -0.005f, 0.0f);
	glEnd();
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
	PlaySound(NULL, 0, 0);

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

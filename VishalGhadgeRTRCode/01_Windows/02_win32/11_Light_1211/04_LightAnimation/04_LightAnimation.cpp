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

GLUquadric *g_gluQuadric = NULL;
int g_iSphereSlices = 100;

GLfloat g_fAngleRed;
GLfloat g_fAngleGreen;
GLfloat g_fAngleBlue;

//
//	Light 0 == Red Light
//
GLfloat g_arrLight0Ambient[] = { 0.0f, 0.0f, 0.0f, 0.0f };	//	Decides general light
GLfloat g_arrLight0Diffuse[] = { 1.0f, 0.0f, 0.0f, 0.0f };	//	Decides color of light
GLfloat g_arrLight0Specular[] = { 1.0f, 0.0f, 0.0f, 0.0f };	//	Decides height of light
GLfloat g_arrLight0Position[] = { 0.0f, 0.0f, 0.0f, 0.0f };	//	Runtime give position 

//
//	Light 1 == Green Light
//
GLfloat g_arrLight1Ambient[] = { 0.0f, 0.0f, 0.0f, 0.0f };	//	Decides general light
GLfloat g_arrLight1Diffuse[] = { 0.0f, 1.0f, 0.0f, 0.0f };	//	Decides color of light
GLfloat g_arrLight1Specular[] = { 0.0f, 1.0f, 0.0f, 0.0f };	//	Decides height of light
GLfloat g_arrLight1Position[] = { 0.0f, 0.0f, 0.0f, 0.0f };	//	Runtime give position 

//
//	Light 2 == Blue Light
//
GLfloat g_arrLight2Ambient[] = { 0.0f, 0.0f, 0.0f, 0.0f };	//	Decides general light
GLfloat g_arrLight2Diffuse[] = { 0.0f, 0.0f, 1.0f, 0.0f };	//	Decides color of light
GLfloat g_arrLight2Specular[] = { 0.0f, 0.0f, 1.0f, 0.0f };	//	Decides height of light
GLfloat g_arrLight2Position[] = { 0.0f, 0.0f, 0.0f, 0.0f };	//	Runtime give position 


//
//	Materail
//
GLfloat g_arrMaterialAmbient[] = { 0.0f, 0.0f, 0.0f, 0.0f };
GLfloat g_arrMaterialDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_arrMaterialSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_arrMaterialShininess[] = { 50.0f };

BOOLEAN g_bEnableLight = FALSE;

#define	CLASS_NAME		TEXT("04 : Animated Lights")

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
			g_iSphereSlices = 30;
			break;

		case 'T':
			g_iSphereSlices = 100;
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

	//	Init quadric.
	g_gluQuadric = gluNewQuadric();

	//
	//	Resize.
	//
	resize(WIN_WIDTH, WIN_HEIGHT);
}

void InitLight()
{
	//
	//	Light 0 == Red Light
	//
	glLightfv(GL_LIGHT0, GL_AMBIENT, g_arrLight0Ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, g_arrLight0Diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, g_arrLight0Specular);
	glLightfv(GL_LIGHT0, GL_POSITION, g_arrLight0Position);

	glEnable(GL_LIGHT0);

	//
	//	Light 1 == Green Light
	//
	glLightfv(GL_LIGHT1, GL_AMBIENT, g_arrLight1Ambient);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, g_arrLight1Diffuse);
	glLightfv(GL_LIGHT1, GL_SPECULAR, g_arrLight1Specular);
	glLightfv(GL_LIGHT1, GL_POSITION, g_arrLight1Position);

	glEnable(GL_LIGHT1);

	//
	//	Light 2 == Blue Light
	//
	glLightfv(GL_LIGHT2, GL_AMBIENT, g_arrLight2Ambient);
	glLightfv(GL_LIGHT2, GL_DIFFUSE, g_arrLight2Diffuse);
	glLightfv(GL_LIGHT2, GL_SPECULAR, g_arrLight2Specular);
	glLightfv(GL_LIGHT2, GL_POSITION, g_arrLight2Position);

	glEnable(GL_LIGHT2);

	//
	//	Materail
	//
	glMaterialfv(GL_FRONT, GL_AMBIENT, g_arrMaterialAmbient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, g_arrMaterialDiffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, g_arrMaterialSpecular);
	glMaterialfv(GL_FRONT, GL_SHININESS, g_arrMaterialShininess);
}

VOID display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	//	Change 3 for 3D

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glPushMatrix();
		gluLookAt(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
		glPushMatrix();
			glRotatef(g_fAngleRed, 1, 0, 0);
			g_arrLight0Position[1] = g_fAngleRed;
			glLightfv(GL_LIGHT0, GL_POSITION, g_arrLight0Position);
		glPopMatrix();
		glPushMatrix();
			glRotatef(g_fAngleGreen, 0, 1, 0);
			g_arrLight1Position[0] = g_fAngleGreen;
			glLightfv(GL_LIGHT1, GL_POSITION, g_arrLight1Position);
		glPopMatrix();
		glPushMatrix();
			glRotatef(g_fAngleBlue, 0, 0, 1);
			g_arrLight2Position[0] = g_fAngleBlue;
			glLightfv(GL_LIGHT2, GL_POSITION, g_arrLight2Position);
		glPopMatrix();
		glPushMatrix();
			glTranslatef(0.0f, 0.0f, -3.0f);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			gluSphere(g_gluQuadric, 0.75, g_iSphereSlices, g_iSphereSlices);	//	Update last 2 values from 30 to 100 and compare looks.
		glPopMatrix();
	glPopMatrix();

	SwapBuffers(g_hdc);
}


VOID UpdateAngle()
{
	g_fAngleRed = g_fAngleRed + 0.1f;
	if (g_fAngleRed >= 360)
	{
		g_fAngleRed = 0.0f;
	}

	g_fAngleGreen = g_fAngleGreen + 0.1f;
	if (g_fAngleGreen >= 360)
	{
		g_fAngleGreen = 0.0f;
	}

	g_fAngleBlue = g_fAngleBlue + 0.1f;
	if (g_fAngleBlue >= 360)
	{
		g_fAngleBlue = 0.0f;
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
	if (g_gluQuadric)
	{
		gluDeleteQuadric(g_gluQuadric);
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

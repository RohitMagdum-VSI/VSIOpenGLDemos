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
GLfloat g_fAngleLight = 0.0f;

GLfloat g_fRotateX = 0.0f;
GLfloat g_fRotateY = 0.0f;
GLfloat g_fRotateZ = 0.0f;

GLfloat g_arrLightAmbient[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_arrLightDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_arrLightSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_arrLightPosition[] = { 0.0f, 0.0f, 0.0f, 0.0f };	//	Give position runtime.

GLfloat g_arrMaterialSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat g_arrMaterialShininess[] = { 50.0f };

//
//	Materail 00
//
GLfloat g_arrMaterial00Ambient[] = { 0.0215f, 0.1745f, 0.0215f, 1.0f };
GLfloat g_arrMaterial00Diffuse[] = { 0.07568f, 0.61424f, 0.07568f, 1.0f };
GLfloat g_arrMaterial00Specular[] = { 0.633f, 0.727811f, 0.633f, 1.0f };
GLfloat g_arrMaterial00Shininess[] = { 0.6f * 128.0f };

//
//	Materail 10
//
GLfloat g_arrMaterial10Ambient[] = { 0.135f, 0.2225f, 0.1575f, 1.0f };
GLfloat g_arrMaterial10Diffuse[] = { 0.54f, 0.89f, 0.63f, 1.0f };
GLfloat g_arrMaterial10Specular[] = { 0.316228f, 0.316228f, 0.316228f, 1.0f };
GLfloat g_arrMaterial10Shininess[] = { 0.1f * 128.0f };

//
//	Materail 20
//
GLfloat g_arrMaterial20Ambient[] = { 0.05375f, 0.05f, 0.06625f, 1.0f };
GLfloat g_arrMaterial20Diffuse[] = { 0.18275f, 0.17f, 0.22525f, 1.0f };
GLfloat g_arrMaterial20Specular[] = { 0.332741f, 0.328634f, 0.346435f, 1.0f };
GLfloat g_arrMaterial20Shininess[] = { 0.3f * 128.0f };

//
//	Materail 30
//
GLfloat g_arrMaterial30Ambient[] = { 0.25f, 0.20725f, 0.20725f, 1.0f };
GLfloat g_arrMaterial30Diffuse[] = { 1.0f, 0.829f, 0.829f, 1.0f };
GLfloat g_arrMaterial30Specular[] = { 0.296648f, 0.296648f, 0.296648f, 1.0f };
GLfloat g_arrMaterial30Shininess[] = { 0.088f * 128.0f };

//
//	Materail 40
//
GLfloat g_arrMaterial40Ambient[] = { 0.1745f, 0.01175f, 0.01175f, 1.0f };
GLfloat g_arrMaterial40Diffuse[] = { 0.61424f, 0.04136f, 0.04136f, 1.0f };
GLfloat g_arrMaterial40Specular[] = { 0.727811f, 0.626959f, 0.626959f, 1.0f };
GLfloat g_arrMaterial40Shininess[] = { 0.6f * 128.0f };

//
//	Materail 50
//
GLfloat g_arrMaterial50Ambient[] = { 0.1f, 0.18725f, 0.1745f, 1.0f };
GLfloat g_arrMaterial50Diffuse[] = { 0.396f, 0.74151f, 0.69102f, 1.0f };
GLfloat g_arrMaterial50Specular[] = { 0.297254f, 0.30829f, 0.306678f, 1.0f };
GLfloat g_arrMaterial50Shininess[] = { 0.1f * 128.0f };

//
//	Materail 01
//
GLfloat g_arrMaterial01Ambient[] = { 0.329412f, 0.223529f, 0.027451f, 1.0f };
GLfloat g_arrMaterial01Diffuse[] = { 0.780392f, 0.568627f, 0.113725f, 1.0f };
GLfloat g_arrMaterial01Specular[] = { 0.992157f, 0.941176f, 0.807843f, 1.0f };
GLfloat g_arrMaterial01Shininess[] = { 0.21794872f * 128.0f };

//
//	Materail 11
//
GLfloat g_arrMaterial11Ambient[] = { 0.2125f, 0.1275f, 0.054f, 1.0f };
GLfloat g_arrMaterial11Diffuse[] = { 0.714f, 0.4284f, 0.18144f, 1.0f };
GLfloat g_arrMaterial11Specular[] = { 0.393548f, 0.271906f, 0.166721f, 1.0f };
GLfloat g_arrMaterial11Shininess[] = { 0.2f * 128.0f };

//
//	Materail 21
//
GLfloat g_arrMaterial21Ambient[] = { 0.25f, 0.25f, 0.25f, 1.0f };
GLfloat g_arrMaterial21Diffuse[] = { 0.4f, 0.4f, 0.4f, 1.0f };
GLfloat g_arrMaterial21Specular[] = { 0.774597f, 0.774597f, 0.774597f, 1.0f };
GLfloat g_arrMaterial21Shininess[] = { 0.6f * 128.0f };

//
//	Materail 31
//
GLfloat g_arrMaterial31Ambient[] = { 0.19125f, 0.0735f, 0.0225f, 1.0f };
GLfloat g_arrMaterial31Diffuse[] = { 0.7038f, 0.27048f, 0.0828f, 1.0f };
GLfloat g_arrMaterial31Specular[] = { 0.256777f, 0.137622f, 0.296648f, 1.0f };
GLfloat g_arrMaterial31Shininess[] = { 0.1f * 128.0f };

//
//	Materail 41
//
GLfloat g_arrMaterial41Ambient[] = { 0.24725f, 0.1995f, 0.0745f, 1.0f };
GLfloat g_arrMaterial41Diffuse[] = { 0.75164f, 0.60648f, 0.22648f, 1.0f };
GLfloat g_arrMaterial41Specular[] = { 0.628281f, 0.555802f, 0.366065f, 1.0f };
GLfloat g_arrMaterial41Shininess[] = { 0.4f * 128.0f };

//
//	Materail 51
//
GLfloat g_arrMaterial51Ambient[] = { 0.19225f, 0.19225f, 0.19225f, 1.0f };
GLfloat g_arrMaterial51Diffuse[] = { 0.50754f, 0.50754f, 0.50754f, 1.0f };
GLfloat g_arrMaterial51Specular[] = { 0.508273f, 0.508273f, 0.508273f, 1.0f };
GLfloat g_arrMaterial51Shininess[] = { 0.4f * 128.0f };

//
//	Materail 02
//
GLfloat g_arrMaterial02Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrMaterial02Diffuse[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrMaterial02Specular[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrMaterial02Shininess[] = { 0.25f * 128.0f };

//
//	Materail 12
//
GLfloat g_arrMaterial12Ambient[] = { 0.0f, 0.1f, 0.06f, 1.0f };
GLfloat g_arrMaterial12Diffuse[] = { 0.0f, 0.50980392f, 0.50980392f, 1.0f };
GLfloat g_arrMaterial12Specular[] = { 0.50980392f, 0.50980392f, 0.50980392f, 1.0f };
GLfloat g_arrMaterial12Shininess[] = { 0.25f * 128.0f };

//
//	Materail 22
//
GLfloat g_arrMaterial22Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrMaterial22Diffuse[] = { 0.1f, 0.35f, 0.1f, 1.0f };
GLfloat g_arrMaterial22Specular[] = { 0.45f, 0.45f, 0.45f, 1.0f };
GLfloat g_arrMaterial22Shininess[] = { 0.25f * 128.0f };

//
//	Materail 32
//
GLfloat g_arrMaterial32Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrMaterial32Diffuse[] = { 0.5f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrMaterial32Specular[] = { 0.7f, 0.6f, 0.6f, 1.0f };
GLfloat g_arrMaterial32Shininess[] = { 0.25f * 128.0f };

//
//	Materail 42
//
GLfloat g_arrMaterial42Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrMaterial42Diffuse[] = { 0.55f, 0.55f, 0.55f, 1.0f };
GLfloat g_arrMaterial42Specular[] = { 0.70f, 0.70f, 0.70f, 1.0f };
GLfloat g_arrMaterial42Shininess[] = { 0.25f * 128.0f };

//
//	Materail 52
//
GLfloat g_arrMaterial52Ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrMaterial52Diffuse[] = { 0.5f, 0.5f, 0.0f, 1.0f };
GLfloat g_arrMaterial52Specular[] = { 0.60f, 0.60f, 0.50f, 1.0f };
GLfloat g_arrMaterial52Shininess[] = { 0.25f * 128.0f };

//
//	Materail 03
//
GLfloat g_arrMaterial03Ambient[] = { 0.02f, 0.02f, 0.02f, 1.0f };
GLfloat g_arrMaterial03Diffuse[] = { 0.01f, 0.01f, 0.01f, 1.0f };
GLfloat g_arrMaterial03Specular[] = { 0.4f, 0.4f, 0.4f, 1.0f };
GLfloat g_arrMaterial03Shininess[] = { 0.078125f * 128.0f };

//
//	Materail 13
//
GLfloat g_arrMaterial13Ambient[] = { 0.0f, 0.05f, 0.05f, 1.0f };
GLfloat g_arrMaterial13Diffuse[] = { 0.4f, 0.5f, 0.5f, 1.0f };
GLfloat g_arrMaterial13Specular[] = { 0.04f, 0.7f, 0.7f, 1.0f };
GLfloat g_arrMaterial13Shininess[] = { 0.078125f * 128.0f };

//
//	Materail 23
//
GLfloat g_arrMaterial23Ambient[] = { 0.0f, 0.05f, 0.0f, 1.0f };
GLfloat g_arrMaterial23Diffuse[] = { 0.4f, 0.5f, 0.4f, 1.0f };
GLfloat g_arrMaterial23Specular[] = { 0.04f, 0.7f, 0.04f, 1.0f };
GLfloat g_arrMaterial23Shininess[] = { 0.078125f * 128.0f };

//
//	Materail 33
//
GLfloat g_arrMaterial33Ambient[] = { 0.05f, 0.0f, 0.0f, 1.0f };
GLfloat g_arrMaterial33Diffuse[] = { 0.5f, 0.4f, 0.4f, 1.0f };
GLfloat g_arrMaterial33Specular[] = { 0.7f, 0.04f, 0.04f, 1.0f };
GLfloat g_arrMaterial33Shininess[] = { 0.078125f * 128.0f };

//
//	Materail 43
//
GLfloat g_arrMaterial43Ambient[] = { 0.05f, 0.05f, 0.05f, 1.0f };
GLfloat g_arrMaterial43Diffuse[] = { 0.5f, 0.5f, 0.5f, 1.0f };
GLfloat g_arrMaterial43Specular[] = { 0.7f, 0.7f, 0.7f, 1.0f };
GLfloat g_arrMaterial43Shininess[] = { 0.78125f * 128.0f };

//
//	Materail 53
//
GLfloat g_arrMaterial53Ambient[] = { 0.05f, 0.05f, 0.0f, 1.0f };
GLfloat g_arrMaterial53Diffuse[] = { 0.5f, 0.5f, 0.4f, 1.0f };
GLfloat g_arrMaterial53Specular[] = { 0.7f, 0.7f, 0.04f, 1.0f };
GLfloat g_arrMaterial53Shininess[] = { 0.078125f * 128.0f };


GLfloat g_arrLightModelAmbient[] = { 0.2f, 0.2f, 0.2f, 0.0f };
GLfloat g_arrLightModelLocalViewer[] = { 0.0f };

BOOLEAN g_bEnableLight = FALSE;

#define	CLASS_NAME		TEXT("04 : Lights With Different Materials")

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

		case 'x':
		case 'X':
			//g_fAngleLight = 0.0f;
			g_fRotateX = 1.0f;
			g_fRotateY = 0.0f;
			g_fRotateZ = 0.0f;
			break;

		case 'y':
		case 'Y':
			//g_fAngleLight = 0.0f;
			g_fRotateX = 0.0f;
			g_fRotateY = 1.0f;
			g_fRotateZ = 0.0f;
			break;

		case 'z':
		case 'Z':
			//g_fAngleLight = 0.0f;
			g_fRotateX = 0.0f;
			g_fRotateY = 0.0f;
			g_fRotateZ = 1.0f;
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

	glClearColor(0.25f, 0.25f, 0.25f, 0.0f);	//	Grey color.

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

	glEnable(GL_AUTO_NORMAL);
	glEnable(GL_NORMALIZE);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, g_arrLightModelAmbient);
	glLightModelfv(GL_LIGHT_MODEL_LOCAL_VIEWER, g_arrLightModelLocalViewer);

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
	glLightfv(GL_LIGHT0, GL_AMBIENT, g_arrLightAmbient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, g_arrLightDiffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, g_arrLightSpecular);
	//glLightfv(GL_LIGHT0, GL_POSITION, g_arrLightPosition);

	glEnable(GL_LIGHT0);
}

VOID display()
{
	void Sphere00();
	void Sphere10();
	void Sphere20();
	void Sphere30();
	void Sphere40();
	void Sphere50();

	void Sphere01();
	void Sphere11();
	void Sphere21();
	void Sphere31();
	void Sphere41();
	void Sphere51();

	void Sphere02();
	void Sphere12();
	void Sphere22();
	void Sphere32();
	void Sphere42();
	void Sphere52();

	void Sphere03();
	void Sphere13();
	void Sphere23();
	void Sphere33();
	void Sphere43();
	void Sphere53();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	//	Change 3 for 3D

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Sphere00();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Sphere10();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Sphere20();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Sphere30();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Sphere40();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Sphere50();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Sphere01();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Sphere11();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Sphere21();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Sphere31();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Sphere41();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Sphere51();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Sphere02();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Sphere12();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Sphere22();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Sphere32();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Sphere42();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Sphere52();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Sphere03();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Sphere13();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Sphere23();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Sphere33();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Sphere43();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Sphere53();

	SwapBuffers(g_hdc);
}

void Sphere00()
{
	glPushMatrix();
	//gluLookAt(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	glTranslatef(0.0f, 0.0f, -2.0f);
	glPushMatrix();
	if (1.0f == g_fRotateX)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[1] = g_fAngleLight;
		g_arrLightPosition[0] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateY)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateZ)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[2] = g_arrLightPosition[1] = 0.0f;
	}
	else
	{
		g_arrLightPosition[0] = g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
		g_arrLightPosition[2] = 1.0f;
	}
	glLightfv(GL_LIGHT0, GL_POSITION, g_arrLightPosition);
	glPopMatrix();
	glPushMatrix();
	glTranslatef(-5.0f, 4.0f, -10.0f);
	glMaterialfv(GL_FRONT, GL_AMBIENT, g_arrMaterial00Ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, g_arrMaterial00Diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, g_arrMaterial00Specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, g_arrMaterial00Shininess);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluSphere(g_gluQuadric, 0.50, g_iSphereSlices, g_iSphereSlices);
	glPopMatrix();
	glPopMatrix();
}

void Sphere10()
{
	glPushMatrix();
		//gluLookAt(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
		glTranslatef(0.0f, 0.0f, -2.0f);
		glPushMatrix();
			if (1.0f == g_fRotateX)
			{
				glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
				g_arrLightPosition[1] = g_fAngleLight;
				g_arrLightPosition[0] = g_arrLightPosition[2] = 0.0f;
			}
			else if (1.0f == g_fRotateY)
			{
				glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
				g_arrLightPosition[0] = g_fAngleLight;
				g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
			}
			else if (1.0f == g_fRotateZ)
			{
				glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
				g_arrLightPosition[0] = g_fAngleLight;
				g_arrLightPosition[2] = g_arrLightPosition[1] = 0.0f;
			}
			else
			{
				g_arrLightPosition[0] = g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
				g_arrLightPosition[2] = 1.0f;
			}
			glLightfv(GL_LIGHT0, GL_POSITION, g_arrLightPosition);
		glPopMatrix();
		glPushMatrix();
			glTranslatef(-5.0f, 2.5f, -10.0f);
			glMaterialfv(GL_FRONT, GL_AMBIENT, g_arrMaterial10Ambient);
			glMaterialfv(GL_FRONT, GL_DIFFUSE, g_arrMaterial10Diffuse);
			glMaterialfv(GL_FRONT, GL_SPECULAR, g_arrMaterial10Specular);
			glMaterialfv(GL_FRONT, GL_SHININESS, g_arrMaterial10Shininess);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			gluSphere(g_gluQuadric, 0.50, g_iSphereSlices, g_iSphereSlices);
		glPopMatrix();
	glPopMatrix();
}

void Sphere20()
{
	glPushMatrix();
	//gluLookAt(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	glTranslatef(0.0f, 0.0f, -2.0f);
	glPushMatrix();
	if (1.0f == g_fRotateX)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[1] = g_fAngleLight;
		g_arrLightPosition[0] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateY)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateZ)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[2] = g_arrLightPosition[1] = 0.0f;
	}
	else
	{
		g_arrLightPosition[0] = g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
		g_arrLightPosition[2] = 1.0f;
	}
	glLightfv(GL_LIGHT0, GL_POSITION, g_arrLightPosition);
	glPopMatrix();
	glPushMatrix();
	glTranslatef(-5.0f, 1.0f, -10.0f);
	glMaterialfv(GL_FRONT, GL_AMBIENT, g_arrMaterial20Ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, g_arrMaterial20Diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, g_arrMaterial20Specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, g_arrMaterial20Shininess);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluSphere(g_gluQuadric, 0.50, g_iSphereSlices, g_iSphereSlices);
	glPopMatrix();
	glPopMatrix();
}

void Sphere30()
{
	glPushMatrix();
	//gluLookAt(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	glTranslatef(0.0f, 0.0f, -2.0f);
	glPushMatrix();
	if (1.0f == g_fRotateX)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[1] = g_fAngleLight;
		g_arrLightPosition[0] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateY)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateZ)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[2] = g_arrLightPosition[1] = 0.0f;
	}
	else
	{
		g_arrLightPosition[0] = g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
		g_arrLightPosition[2] = 1.0f;
	}
	glLightfv(GL_LIGHT0, GL_POSITION, g_arrLightPosition);
	glPopMatrix();
	glPushMatrix();
	glTranslatef(-5.0f, -0.5f, -10.0f);
	glMaterialfv(GL_FRONT, GL_AMBIENT, g_arrMaterial30Ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, g_arrMaterial30Diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, g_arrMaterial30Specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, g_arrMaterial30Shininess);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluSphere(g_gluQuadric, 0.50, g_iSphereSlices, g_iSphereSlices);
	glPopMatrix();
	glPopMatrix();
}

void Sphere40()
{
	glPushMatrix();
	//gluLookAt(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	glTranslatef(0.0f, 0.0f, -2.0f);
	glPushMatrix();
	if (1.0f == g_fRotateX)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[1] = g_fAngleLight;
		g_arrLightPosition[0] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateY)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateZ)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[2] = g_arrLightPosition[1] = 0.0f;
	}
	else
	{
		g_arrLightPosition[0] = g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
		g_arrLightPosition[2] = 1.0f;
	}
	glLightfv(GL_LIGHT0, GL_POSITION, g_arrLightPosition);
	glPopMatrix();
	glPushMatrix();
	glTranslatef(-5.0f, -2.0f, -10.0f);
	glMaterialfv(GL_FRONT, GL_AMBIENT, g_arrMaterial40Ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, g_arrMaterial40Diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, g_arrMaterial40Specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, g_arrMaterial40Shininess);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluSphere(g_gluQuadric, 0.50, g_iSphereSlices, g_iSphereSlices);
	glPopMatrix();
	glPopMatrix();
}

void Sphere50()
{
	glPushMatrix();
	//gluLookAt(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	glTranslatef(0.0f, 0.0f, -2.0f);
	glPushMatrix();
	if (1.0f == g_fRotateX)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[1] = g_fAngleLight;
		g_arrLightPosition[0] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateY)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateZ)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[2] = g_arrLightPosition[1] = 0.0f;
	}
	else
	{
		g_arrLightPosition[0] = g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
		g_arrLightPosition[2] = 1.0f;
	}
	glLightfv(GL_LIGHT0, GL_POSITION, g_arrLightPosition);
	glPopMatrix();
	glPushMatrix();
	glTranslatef(-5.0f, -3.5f, -10.0f);
	glMaterialfv(GL_FRONT, GL_AMBIENT, g_arrMaterial50Ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, g_arrMaterial50Diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, g_arrMaterial50Specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, g_arrMaterial50Shininess);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluSphere(g_gluQuadric, 0.50, g_iSphereSlices, g_iSphereSlices);
	glPopMatrix();
	glPopMatrix();
}

void Sphere01()
{
	glPushMatrix();
	//gluLookAt(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	glTranslatef(0.0f, 0.0f, -2.0f);
	glPushMatrix();
	if (1.0f == g_fRotateX)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[1] = g_fAngleLight;
		g_arrLightPosition[0] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateY)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateZ)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[2] = g_arrLightPosition[1] = 0.0f;
	}
	else
	{
		g_arrLightPosition[0] = g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
		g_arrLightPosition[2] = 1.0f;
	}
	glLightfv(GL_LIGHT0, GL_POSITION, g_arrLightPosition);
	glPopMatrix();
	glPushMatrix();
	glTranslatef(-2.0f, 4.0f, -10.0f);
	glMaterialfv(GL_FRONT, GL_AMBIENT, g_arrMaterial01Ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, g_arrMaterial01Diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, g_arrMaterial01Specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, g_arrMaterial01Shininess);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluSphere(g_gluQuadric, 0.50, g_iSphereSlices, g_iSphereSlices);
	glPopMatrix();
	glPopMatrix();
}

void Sphere11()
{
	glPushMatrix();
	//gluLookAt(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	glTranslatef(0.0f, 0.0f, -2.0f);
	glPushMatrix();
	if (1.0f == g_fRotateX)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[1] = g_fAngleLight;
		g_arrLightPosition[0] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateY)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateZ)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[2] = g_arrLightPosition[1] = 0.0f;
	}
	else
	{
		g_arrLightPosition[0] = g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
		g_arrLightPosition[2] = 1.0f;
	}
	glLightfv(GL_LIGHT0, GL_POSITION, g_arrLightPosition);
	glPopMatrix();
	glPushMatrix();
	glTranslatef(-2.0f, 2.5f, -10.0f);
	glMaterialfv(GL_FRONT, GL_AMBIENT, g_arrMaterial11Ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, g_arrMaterial11Diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, g_arrMaterial11Specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, g_arrMaterial11Shininess);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluSphere(g_gluQuadric, 0.50, g_iSphereSlices, g_iSphereSlices);
	glPopMatrix();
	glPopMatrix();
}

void Sphere21()
{
	glPushMatrix();
	//gluLookAt(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	glTranslatef(0.0f, 0.0f, -2.0f);
	glPushMatrix();
	if (1.0f == g_fRotateX)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[1] = g_fAngleLight;
		g_arrLightPosition[0] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateY)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateZ)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[2] = g_arrLightPosition[1] = 0.0f;
	}
	else
	{
		g_arrLightPosition[0] = g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
		g_arrLightPosition[2] = 1.0f;
	}
	glLightfv(GL_LIGHT0, GL_POSITION, g_arrLightPosition);
	glPopMatrix();
	glPushMatrix();
	glTranslatef(-2.0f, 1.0f, -10.0f);
	glMaterialfv(GL_FRONT, GL_AMBIENT, g_arrMaterial21Ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, g_arrMaterial21Diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, g_arrMaterial21Specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, g_arrMaterial21Shininess);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluSphere(g_gluQuadric, 0.50, g_iSphereSlices, g_iSphereSlices);
	glPopMatrix();
	glPopMatrix();
}

void Sphere31()
{
	glPushMatrix();
	//gluLookAt(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	glTranslatef(0.0f, 0.0f, -2.0f);
	glPushMatrix();
	if (1.0f == g_fRotateX)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[1] = g_fAngleLight;
		g_arrLightPosition[0] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateY)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateZ)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[2] = g_arrLightPosition[1] = 0.0f;
	}
	else
	{
		g_arrLightPosition[0] = g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
		g_arrLightPosition[2] = 1.0f;
	}
	glLightfv(GL_LIGHT0, GL_POSITION, g_arrLightPosition);
	glPopMatrix();
	glPushMatrix();
	glTranslatef(-2.0f, -0.5f, -10.0f);
	glMaterialfv(GL_FRONT, GL_AMBIENT, g_arrMaterial31Ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, g_arrMaterial31Diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, g_arrMaterial31Specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, g_arrMaterial31Shininess);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluSphere(g_gluQuadric, 0.50, g_iSphereSlices, g_iSphereSlices);
	glPopMatrix();
	glPopMatrix();
}

void Sphere41()
{
	glPushMatrix();
	//gluLookAt(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	glTranslatef(0.0f, 0.0f, -2.0f);
	glPushMatrix();
	if (1.0f == g_fRotateX)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[1] = g_fAngleLight;
		g_arrLightPosition[0] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateY)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateZ)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[2] = g_arrLightPosition[1] = 0.0f;
	}
	else
	{
		g_arrLightPosition[0] = g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
		g_arrLightPosition[2] = 1.0f;
	}
	glLightfv(GL_LIGHT0, GL_POSITION, g_arrLightPosition);
	glPopMatrix();
	glPushMatrix();
	glTranslatef(-2.0f, -2.0f, -10.0f);
	glMaterialfv(GL_FRONT, GL_AMBIENT, g_arrMaterial41Ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, g_arrMaterial41Diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, g_arrMaterial41Specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, g_arrMaterial41Shininess);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluSphere(g_gluQuadric, 0.50, g_iSphereSlices, g_iSphereSlices);
	glPopMatrix();
	glPopMatrix();
}

void Sphere51()
{
	glPushMatrix();
	//gluLookAt(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	glTranslatef(0.0f, 0.0f, -2.0f);
	glPushMatrix();
	if (1.0f == g_fRotateX)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[1] = g_fAngleLight;
		g_arrLightPosition[0] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateY)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateZ)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[2] = g_arrLightPosition[1] = 0.0f;
	}
	else
	{
		g_arrLightPosition[0] = g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
		g_arrLightPosition[2] = 1.0f;
	}
	glLightfv(GL_LIGHT0, GL_POSITION, g_arrLightPosition);
	glPopMatrix();
	glPushMatrix();
	glTranslatef(-2.0f, -3.5f, -10.0f);
	glMaterialfv(GL_FRONT, GL_AMBIENT, g_arrMaterial51Ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, g_arrMaterial51Diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, g_arrMaterial51Specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, g_arrMaterial51Shininess);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluSphere(g_gluQuadric, 0.50, g_iSphereSlices, g_iSphereSlices);
	glPopMatrix();
	glPopMatrix();
}

void Sphere02()
{
	glPushMatrix();
	//gluLookAt(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	glTranslatef(0.0f, 0.0f, -2.0f);
	glPushMatrix();
	if (1.0f == g_fRotateX)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[1] = g_fAngleLight;
		g_arrLightPosition[0] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateY)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateZ)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[2] = g_arrLightPosition[1] = 0.0f;
	}
	else
	{
		g_arrLightPosition[0] = g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
		g_arrLightPosition[2] = 1.0f;
	}
	glLightfv(GL_LIGHT0, GL_POSITION, g_arrLightPosition);
	glPopMatrix();
	glPushMatrix();
	glTranslatef(1.0f, 4.0f, -10.0f);
	glMaterialfv(GL_FRONT, GL_AMBIENT, g_arrMaterial02Ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, g_arrMaterial02Diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, g_arrMaterial02Specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, g_arrMaterial02Shininess);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluSphere(g_gluQuadric, 0.50, g_iSphereSlices, g_iSphereSlices);
	glPopMatrix();
	glPopMatrix();
}

void Sphere12()
{
	glPushMatrix();
	//gluLookAt(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	glTranslatef(0.0f, 0.0f, -2.0f);
	glPushMatrix();
	if (1.0f == g_fRotateX)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[1] = g_fAngleLight;
		g_arrLightPosition[0] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateY)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateZ)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[2] = g_arrLightPosition[1] = 0.0f;
	}
	else
	{
		g_arrLightPosition[0] = g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
		g_arrLightPosition[2] = 1.0f;
	}
	glLightfv(GL_LIGHT0, GL_POSITION, g_arrLightPosition);
	glPopMatrix();
	glPushMatrix();
	glTranslatef(1.0f, 2.5f, -10.0f);
	glMaterialfv(GL_FRONT, GL_AMBIENT, g_arrMaterial12Ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, g_arrMaterial12Diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, g_arrMaterial12Specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, g_arrMaterial12Shininess);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluSphere(g_gluQuadric, 0.50, g_iSphereSlices, g_iSphereSlices);
	glPopMatrix();
	glPopMatrix();
}

void Sphere22()
{
	glPushMatrix();
	//gluLookAt(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	glTranslatef(0.0f, 0.0f, -2.0f);
	glPushMatrix();
	if (1.0f == g_fRotateX)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[1] = g_fAngleLight;
		g_arrLightPosition[0] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateY)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateZ)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[2] = g_arrLightPosition[1] = 0.0f;
	}
	else
	{
		g_arrLightPosition[0] = g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
		g_arrLightPosition[2] = 1.0f;
	}
	glLightfv(GL_LIGHT0, GL_POSITION, g_arrLightPosition);
	glPopMatrix();
	glPushMatrix();
	glTranslatef(1.0f, 1.0f, -10.0f);
	glMaterialfv(GL_FRONT, GL_AMBIENT, g_arrMaterial22Ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, g_arrMaterial22Diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, g_arrMaterial22Specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, g_arrMaterial22Shininess);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluSphere(g_gluQuadric, 0.50, g_iSphereSlices, g_iSphereSlices);
	glPopMatrix();
	glPopMatrix();
}

void Sphere32()
{
	glPushMatrix();
	//gluLookAt(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	glTranslatef(0.0f, 0.0f, -2.0f);
	glPushMatrix();
	if (1.0f == g_fRotateX)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[1] = g_fAngleLight;
		g_arrLightPosition[0] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateY)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateZ)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[2] = g_arrLightPosition[1] = 0.0f;
	}
	else
	{
		g_arrLightPosition[0] = g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
		g_arrLightPosition[2] = 1.0f;
	}
	glLightfv(GL_LIGHT0, GL_POSITION, g_arrLightPosition);
	glPopMatrix();
	glPushMatrix();
	glTranslatef(1.0f, -0.5f, -10.0f);
	glMaterialfv(GL_FRONT, GL_AMBIENT, g_arrMaterial32Ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, g_arrMaterial32Diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, g_arrMaterial32Specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, g_arrMaterial32Shininess);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluSphere(g_gluQuadric, 0.50, g_iSphereSlices, g_iSphereSlices);
	glPopMatrix();
	glPopMatrix();
}

void Sphere42()
{
	glPushMatrix();
	//gluLookAt(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	glTranslatef(0.0f, 0.0f, -2.0f);
	glPushMatrix();
	if (1.0f == g_fRotateX)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[1] = g_fAngleLight;
		g_arrLightPosition[0] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateY)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateZ)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[2] = g_arrLightPosition[1] = 0.0f;
	}
	else
	{
		g_arrLightPosition[0] = g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
		g_arrLightPosition[2] = 1.0f;
	}
	glLightfv(GL_LIGHT0, GL_POSITION, g_arrLightPosition);
	glPopMatrix();
	glPushMatrix();
	glTranslatef(1.0f, -2.0f, -10.0f);
	glMaterialfv(GL_FRONT, GL_AMBIENT, g_arrMaterial42Ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, g_arrMaterial42Diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, g_arrMaterial42Specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, g_arrMaterial42Shininess);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluSphere(g_gluQuadric, 0.50, g_iSphereSlices, g_iSphereSlices);
	glPopMatrix();
	glPopMatrix();
}

void Sphere52()
{
	glPushMatrix();
	//gluLookAt(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	glTranslatef(0.0f, 0.0f, -2.0f);
	glPushMatrix();
	if (1.0f == g_fRotateX)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[1] = g_fAngleLight;
		g_arrLightPosition[0] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateY)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateZ)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[2] = g_arrLightPosition[1] = 0.0f;
	}
	else
	{
		g_arrLightPosition[0] = g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
		g_arrLightPosition[2] = 1.0f;
	}
	glLightfv(GL_LIGHT0, GL_POSITION, g_arrLightPosition);
	glPopMatrix();
	glPushMatrix();
	glTranslatef(1.0f, -3.5f, -10.0f);
	glMaterialfv(GL_FRONT, GL_AMBIENT, g_arrMaterial52Ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, g_arrMaterial52Diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, g_arrMaterial52Specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, g_arrMaterial52Shininess);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluSphere(g_gluQuadric, 0.50, g_iSphereSlices, g_iSphereSlices);
	glPopMatrix();
	glPopMatrix();
}

void Sphere03()
{
	glPushMatrix();
	//gluLookAt(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	glTranslatef(0.0f, 0.0f, -2.0f);
	glPushMatrix();
	if (1.0f == g_fRotateX)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[1] = g_fAngleLight;
		g_arrLightPosition[0] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateY)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateZ)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[2] = g_arrLightPosition[1] = 0.0f;
	}
	else
	{
		g_arrLightPosition[0] = g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
		g_arrLightPosition[2] = 1.0f;
	}
	glLightfv(GL_LIGHT0, GL_POSITION, g_arrLightPosition);
	glPopMatrix();
	glPushMatrix();
	glTranslatef(4.0f, 4.0f, -10.0f);
	glMaterialfv(GL_FRONT, GL_AMBIENT, g_arrMaterial03Ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, g_arrMaterial03Diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, g_arrMaterial03Specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, g_arrMaterial03Shininess);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluSphere(g_gluQuadric, 0.50, g_iSphereSlices, g_iSphereSlices);
	glPopMatrix();
	glPopMatrix();
}

void Sphere13()
{
	glPushMatrix();
	//gluLookAt(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	glTranslatef(0.0f, 0.0f, -2.0f);
	glPushMatrix();
	if (1.0f == g_fRotateX)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[1] = g_fAngleLight;
		g_arrLightPosition[0] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateY)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateZ)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[2] = g_arrLightPosition[1] = 0.0f;
	}
	else
	{
		g_arrLightPosition[0] = g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
		g_arrLightPosition[2] = 1.0f;
	}
	glLightfv(GL_LIGHT0, GL_POSITION, g_arrLightPosition);
	glPopMatrix();
	glPushMatrix();
	glTranslatef(4.0f, 2.5f, -10.0f);
	glMaterialfv(GL_FRONT, GL_AMBIENT, g_arrMaterial13Ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, g_arrMaterial13Diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, g_arrMaterial13Specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, g_arrMaterial13Shininess);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluSphere(g_gluQuadric, 0.50, g_iSphereSlices, g_iSphereSlices);
	glPopMatrix();
	glPopMatrix();
}

void Sphere23()
{
	glPushMatrix();
	//gluLookAt(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	glTranslatef(0.0f, 0.0f, -2.0f);
	glPushMatrix();
	if (1.0f == g_fRotateX)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[1] = g_fAngleLight;
		g_arrLightPosition[0] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateY)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateZ)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[2] = g_arrLightPosition[1] = 0.0f;
	}
	else
	{
		g_arrLightPosition[0] = g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
		g_arrLightPosition[2] = 1.0f;
	}
	glLightfv(GL_LIGHT0, GL_POSITION, g_arrLightPosition);
	glPopMatrix();
	glPushMatrix();
	glTranslatef(4.0f, 1.0f, -10.0f);
	glMaterialfv(GL_FRONT, GL_AMBIENT, g_arrMaterial23Ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, g_arrMaterial23Diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, g_arrMaterial23Specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, g_arrMaterial23Shininess);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluSphere(g_gluQuadric, 0.50, g_iSphereSlices, g_iSphereSlices);
	glPopMatrix();
	glPopMatrix();
}

void Sphere33()
{
	glPushMatrix();
	//gluLookAt(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	glTranslatef(0.0f, 0.0f, -2.0f);
	glPushMatrix();
	if (1.0f == g_fRotateX)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[1] = g_fAngleLight;
		g_arrLightPosition[0] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateY)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateZ)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[2] = g_arrLightPosition[1] = 0.0f;
	}
	else
	{
		g_arrLightPosition[0] = g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
		g_arrLightPosition[2] = 1.0f;
	}
	glLightfv(GL_LIGHT0, GL_POSITION, g_arrLightPosition);
	glPopMatrix();
	glPushMatrix();
	glTranslatef(4.0f, -0.5f, -10.0f);
	glMaterialfv(GL_FRONT, GL_AMBIENT, g_arrMaterial33Ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, g_arrMaterial33Diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, g_arrMaterial33Specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, g_arrMaterial33Shininess);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluSphere(g_gluQuadric, 0.50, g_iSphereSlices, g_iSphereSlices);
	glPopMatrix();
	glPopMatrix();
}

void Sphere43()
{
	glPushMatrix();
	//gluLookAt(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	glTranslatef(0.0f, 0.0f, -2.0f);
	glPushMatrix();
	if (1.0f == g_fRotateX)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[1] = g_fAngleLight;
		g_arrLightPosition[0] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateY)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateZ)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[2] = g_arrLightPosition[1] = 0.0f;
	}
	else
	{
		g_arrLightPosition[0] = g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
		g_arrLightPosition[2] = 1.0f;
	}
	glLightfv(GL_LIGHT0, GL_POSITION, g_arrLightPosition);
	glPopMatrix();
	glPushMatrix();
	glTranslatef(4.0f, -2.0f, -10.0f);
	glMaterialfv(GL_FRONT, GL_AMBIENT, g_arrMaterial43Ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, g_arrMaterial43Diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, g_arrMaterial43Specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, g_arrMaterial43Shininess);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluSphere(g_gluQuadric, 0.50, g_iSphereSlices, g_iSphereSlices);
	glPopMatrix();
	glPopMatrix();
}

void Sphere53()
{
	glPushMatrix();
	//gluLookAt(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	glTranslatef(0.0f, 0.0f, -2.0f);
	glPushMatrix();
	if (1.0f == g_fRotateX)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[1] = g_fAngleLight;
		g_arrLightPosition[0] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateY)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
	}
	else if (1.0f == g_fRotateZ)
	{
		glRotatef(g_fAngleLight, g_fRotateX, g_fRotateY, g_fRotateZ);
		g_arrLightPosition[0] = g_fAngleLight;
		g_arrLightPosition[2] = g_arrLightPosition[1] = 0.0f;
	}
	else
	{
		g_arrLightPosition[0] = g_arrLightPosition[1] = g_arrLightPosition[2] = 0.0f;
		g_arrLightPosition[2] = 1.0f;
	}
	glLightfv(GL_LIGHT0, GL_POSITION, g_arrLightPosition);
	glPopMatrix();
	glPushMatrix();
	glTranslatef(4.0f, -3.5f, -10.0f);
	glMaterialfv(GL_FRONT, GL_AMBIENT, g_arrMaterial53Ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, g_arrMaterial53Diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, g_arrMaterial53Specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, g_arrMaterial53Shininess);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluSphere(g_gluQuadric, 0.50, g_iSphereSlices, g_iSphereSlices);
	glPopMatrix();
	glPopMatrix();
}

VOID UpdateAngle()
{
	g_fAngleLight = g_fAngleLight + 1.0f;
	if (g_fAngleLight >= 360.0f)
	{
		g_fAngleLight = 0.0f;
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

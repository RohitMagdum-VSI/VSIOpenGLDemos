#include<windows.h>
#include<gl/GL.h>
#include<gl/GLU.h>
#include<stdio.h>

#pragma comment(lib,"User32.lib")
#pragma comment(lib,"GDI32.lib")
#pragma comment(lib,"opengl32.lib")
#pragma comment(lib,"glu32.lib")

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

HWND ghwnd;
HDC ghdc;
HGLRC ghrc;
FILE *gpFile;
WINDOWPLACEMENT wpPrev;
DWORD dwStyle;

bool gbFullscreen = false;
bool gbActiveWindow = false;
bool gbIsEscapeKeyPressed = false;
bool gbIsLKeyPressed = false;

GLfloat light1_ambient[] = { 0.0f,0.0f,0.0f,0.0f };
GLfloat light1_diffuse[] = { 1.0f,0.0f,0.0f,0.0f };
GLfloat light1_specular[] = { 1.0f,0.0f,0.0f,0.0f };
GLfloat Red_light1_position[] = { 0.0f,0.0f,0.0f,0.0f };

GLfloat light2_ambient[] = { 0.0f,0.0f,0.0f,0.0f };
GLfloat light2_diffuse[] = { 0.0f,1.0f,0.0f,0.0f };
GLfloat light2_specular[] = { 0.0f,1.0f,0.0f,0.0f };
GLfloat Green_light2_position[] = { 0.0f,0.0f,0.0f,0.0f };

GLfloat light3_ambient[] = { 0.0f,0.0f,0.0f,0.0f };
GLfloat light3_diffuse[] = { 0.0f,0.0f,1.0f,0.0f };
GLfloat light3_specular[] = { 0.0f,0.0f,1.0f,0.0f };
GLfloat Blue_light3_position[] = { 0.0f,0.0f,0.0f,0.0f };

GLfloat material_ambient[] = { 0.0f,0.0f,0.0f,0.0f };
GLfloat material_diffuse[] = { 1.0f,1.0f,1.0f,1.0f };
GLfloat material_specular[] = { 1.0f,1.0f,1.0f,1.0f };
GLfloat material_shininess[] = { 50.0f,50.0f,50.0f,50.0f };

GLfloat gfRed_Angle_Light = 0.0f;
GLfloat gfGreen_Angle_Light = 0.0f;
GLfloat gfBlue_Angle_Light = 0.0f;

GLUquadric *quadric = NULL;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	void uninitialize(int);
	void initialize(void);
	void display(void);
	void update(void);

	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szClassName[] = TEXT("My OpenGL");
	bool bDone = false;

	if (fopen_s(&gpFile, "Log.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Cannot Create File!!!"), TEXT("ERROR"), MB_OK);
		exit(EXIT_FAILURE);
	}
	else
		fprintf(gpFile, "Log File Is Created...\n");

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

	hwnd = CreateWindowEx(WS_EX_APPWINDOW, szClassName, TEXT("OpenGL"), WS_OVERLAPPEDWINDOW|WS_CLIPCHILDREN|WS_CLIPSIBLINGS|WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT, WIN_WIDTH, WIN_HEIGHT, NULL, NULL, hInstance, NULL);
	if (hwnd == NULL)
	{
		fprintf(gpFile, "CreateWindow() Failed");
		uninitialize(1);
	}

	ghwnd = hwnd;

	ShowWindow(ghwnd,iCmdShow);
	SetForegroundWindow(ghwnd);
	SetFocus(ghwnd);

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
	void uninitialize(void);

	//code
	switch (iMsg)
	{
	case WM_ACTIVATE:
		if (HIWORD(wParam) == 0)
			gbActiveWindow = true;
		else
			gbActiveWindow = false;
		break;
	case WM_ERASEBKGND:
		return(0);
	case WM_SIZE:
		resize(LOWORD(lParam), HIWORD(lParam));
		break;
	case WM_KEYDOWN:
		switch (LOWORD(wParam))
		{
		case VK_ESCAPE:
			gbIsEscapeKeyPressed = true;
			break;
		/*case VK_LEFT:
			gfTranslate_X_Light = gfTranslate_X_Light - 0.1f;
			break;
		case VK_RIGHT:
			gfTranslate_X_Light = gfTranslate_X_Light + 0.1f;
			break;*/
		}
		break;
	case WM_CHAR:
		switch(wParam)
		{
		case 'L':
		case 'l':
			if (gbIsLKeyPressed == false)
			{
				glEnable(GL_LIGHTING);
				gbIsLKeyPressed = true;
			}
			else
			{
				glDisable(GL_LIGHTING);
				gbIsLKeyPressed = false;
			}
			break;
		case 'F':
		case 'f':
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
		default:
			break;
		}
		break;
	case WM_LBUTTONDOWN:
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	default:
		break;
	}
	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void initialize(void)
{
	void uninitialize(int);
	void resize(int, int);
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
		fprintf(gpFile, "GetDC() Failed");
		//uninitialize(1);
	}

	iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);
	if (iPixelFormatIndex == 0)
	{
		fprintf(gpFile, "ChoosePixelFormat() Failed");
		//uninitialize(1);
	}

	if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
	{
		fprintf(gpFile, "SetPixelFormat() Failed");
		//uninitialize(1);
	}

	ghrc = wglCreateContext(ghdc);
	if (ghrc == NULL)
	{
		fprintf(gpFile, "wglCreateContext() Failed");
		//uninitialize(1);
	}

	if (wglMakeCurrent(ghdc, ghrc) == FALSE)
	{
		fprintf(gpFile, "wglMakeCurrent() Failed");
		//uninitialize(1);
	}

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glShadeModel(GL_SMOOTH);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT , GL_NICEST);

	glLightfv(GL_LIGHT0, GL_AMBIENT, light1_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light1_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light1_specular);
	glEnable(GL_LIGHT0);

	glLightfv(GL_LIGHT1, GL_AMBIENT, light2_ambient);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, light2_diffuse);
	glLightfv(GL_LIGHT1, GL_SPECULAR, light2_specular);
	glEnable(GL_LIGHT1);

	glLightfv(GL_LIGHT2, GL_AMBIENT, light3_ambient);
	glLightfv(GL_LIGHT2, GL_DIFFUSE, light3_diffuse);
	glLightfv(GL_LIGHT2, GL_SPECULAR, light3_specular);
	glEnable(GL_LIGHT2);

	glMaterialfv(GL_FRONT, GL_AMBIENT, material_ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, material_diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, material_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, material_shininess);

	resize(WIN_WIDTH, WIN_HEIGHT);
}

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glPushMatrix();
	gluLookAt(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);

	glPushMatrix();
	glRotatef(gfRed_Angle_Light, 1.0f, 0.0f, 0.0f);
	Red_light1_position[1] = gfRed_Angle_Light;
	glLightfv(GL_LIGHT0, GL_POSITION, Red_light1_position);
	glPopMatrix();

	glPushMatrix();
	glRotatef(gfGreen_Angle_Light, 0.0f, 1.0f, 0.0f);
	Green_light2_position[0] = gfGreen_Angle_Light;
	glLightfv(GL_LIGHT1, GL_POSITION, Green_light2_position);
	glPopMatrix();

	glPushMatrix();
	glRotatef(gfBlue_Angle_Light, 0.0f, 0.0f, 1.0f);
	Blue_light3_position[0] = gfBlue_Angle_Light;
	glLightfv(GL_LIGHT2, GL_POSITION, Blue_light3_position);
	glPopMatrix();

	glPushMatrix();
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	quadric = gluNewQuadric();
	glTranslatef(0.0f, 0.0f, -6.0f);
	gluSphere(quadric, 1.0, 30, 30);
	glPopMatrix();

	glPopMatrix();

	SwapBuffers(ghdc);
}

void update(void)
{
	if (gfRed_Angle_Light >= 360.0f)
		gfRed_Angle_Light = 0.0f;

	if (gfGreen_Angle_Light >= 360.0f)
		gfGreen_Angle_Light = 0.0f;

	if (gfBlue_Angle_Light >= 360.0f)
		gfBlue_Angle_Light = 0.0f;

	gfRed_Angle_Light = gfRed_Angle_Light + 0.3f;
	gfGreen_Angle_Light = gfGreen_Angle_Light + 0.3f;
	gfBlue_Angle_Light = gfBlue_Angle_Light + 0.3f;
}

void resize(int width, int height)
{
	if (height == 0)
		height = 1;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
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
				SetWindowPos(ghwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.left, SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}
		ShowCursor(FALSE);
	}

	else
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
	}
}

void uninitialize(int iExit_Flag)
{
	if(gbFullscreen==true)
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
	}

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

	if (iExit_Flag == 1)
		fprintf(gpFile, "Log file closed due to some error!!!\n");
	else if (iExit_Flag == 0)
		fprintf(gpFile, "Log file closed Successfully...\n");

	DestroyWindow(ghwnd);
}

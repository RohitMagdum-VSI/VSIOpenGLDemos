//Header Files
#include<windows.h>
#include<gl/GL.h>
#include<stdio.h>

//Pre-processing Directives
#pragma comment(lib,"User32.lib")
#pragma comment(lib,"GDI32.lib")
#pragma comment(lib,"OpenGL32.lib")

#define WIN_WIDTH 800
#define WIN_HEIGHT 800

//CallBack Function Declaration
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

//Global Variables
HWND ghwnd;
HDC ghdc;
HGLRC ghrc;
DWORD dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };
FILE *gpFile;
bool gbActiveWindow = false;
bool gbFullscreen = false;
bool gbIsEscapeKeyPressed = false;
float gi_Y_Vertex, gi_X_Vertex;

//WinMain
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	void initialize(void);
	void display(void);
	void uninitialize(int);
	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szClassName[] = TEXT("My App");
	bool bDone = false;

	if (fopen_s(&gpFile, "Log.txt", "w") != NULL)
	{
		MessageBox(NULL, TEXT("Cannot Create Log File"), TEXT("ERROR"), MB_OK);
		exit(EXIT_FAILURE);
	}
	else
		fprintf(gpFile, "Log File is Created Successfully...\n");

	//Initialize WndClass
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

	hwnd = CreateWindowEx(WS_EX_APPWINDOW, szClassName, TEXT("OpenGL Window "), WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT, WIN_WIDTH, WIN_HEIGHT, NULL, NULL, hInstance, NULL);
	if (hwnd == NULL)
	{
		fprintf(gpFile, "Cannot Create Window...\n");
		exit(EXIT_FAILURE);
	}

	ghwnd = hwnd;

	ShowWindow(hwnd, iCmdShow);
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);

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
				display();
			}
		}
	}

	uninitialize(0);
	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	void ToggleFullscreen(void);
	void resize(int, int);
	switch (iMsg)
	{
	case WM_ACTIVATE:
		if (HIWORD(wParam) == 0)
			gbActiveWindow = true;
		else
			gbActiveWindow = false;
		break;
	case WM_CREATE:
		break;
	case WM_SIZE:
		resize(LOWORD(lParam), HIWORD(lParam));
		break;
	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			gbIsEscapeKeyPressed = true;
			break;
		case 0x46:
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
		}
		break;
	case WM_LBUTTONDOWN:
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	}
	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void initialize()
{
	void uninitialize(int);
	int iPixelFormatIndex;
	PIXELFORMATDESCRIPTOR pfd;

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

	ghdc = GetDC(ghwnd);
	if (ghdc == NULL)
	{
		fprintf(gpFile, "GetDC() Failed...\n");
		uninitialize(1);
	}

	iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);
	if (iPixelFormatIndex == 0)
	{
		fprintf(gpFile, "ChoosePixelFormat() Failed...\n");
		uninitialize(1);
	}

	if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
	{
		fprintf(gpFile, "SetPixelFormat() Failed...\n");
		uninitialize(1);
	}

	ghrc = wglCreateContext(ghdc);
	if (ghrc == NULL)
	{
		fprintf(gpFile, "wglCreateContext() Failed...\n");
		uninitialize(1);
	}

	if (wglMakeCurrent(ghdc, ghrc) == FALSE)
	{
		fprintf(gpFile, "wglMakeCurrent() Failed...\n");
		uninitialize(1);
	}
}

void resize(int width, int height)
{
	if (height == 0)
		height = 1;
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);
}

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT);

	glLineWidth(3.0f);
	glBegin(GL_LINES);
	glColor3f(0.0f, 1.0f, 0.0f);
	glVertex3f(1.0f, 0.0f, 0.0f);
	glVertex3f(-1.0f, 0.0f, 0.0f);
	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, 1.0f, 0.0f);
	glVertex3f(0.0f, -1.0f, 0.0f);
	glEnd();

	glLineWidth(1.0f);
	glBegin(GL_LINES);
	glColor3f(0.0f, 0.0f, 1.0f);
	for (gi_Y_Vertex = -1.0f; gi_Y_Vertex <= 0.0f; gi_Y_Vertex = gi_Y_Vertex + 0.05f)
	{
		glVertex3f(1.0f, gi_Y_Vertex, 0.0f);
		glVertex3f(-1.0f, gi_Y_Vertex, 0.0f);
	}
	for (gi_Y_Vertex = 0.05f; gi_Y_Vertex <= 1.0f; gi_Y_Vertex = gi_Y_Vertex + 0.05f)
	{
		glVertex3f(1.0f, gi_Y_Vertex, 0.0f);
		glVertex3f(-1.0f, gi_Y_Vertex, 0.0f);
	}
	for (gi_X_Vertex = -1.0f; gi_X_Vertex <= 0.0f; gi_X_Vertex = gi_X_Vertex + 0.05f)
	{
		glVertex3f(gi_X_Vertex, 1.0f, 0.0f);
		glVertex3f(gi_X_Vertex, -1.0f, 0.0f);
	}
	for (gi_X_Vertex = 0.05f; gi_X_Vertex <= 1.0f; gi_X_Vertex = gi_X_Vertex + 0.05f)
	{
		glVertex3f(gi_X_Vertex, 1.0f, 0.0f);
		glVertex3f(gi_X_Vertex, -1.0f, 0.0f);
	}
	glEnd();

	glBegin(GL_LINE_LOOP);
	glColor3f(1.0f, 1.0f, 0.0f);
	glVertex3f(0.5f, 0.5f, 0.0f);
	glVertex3f(-0.5f, 0.5f, 0.0f);
	glVertex3f(-0.5f, -0.5f, 0.0f);
	glVertex3f(0.5f, -0.5f, 0.0f);
	glEnd();

	SwapBuffers(ghdc);
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
				SetWindowPos(ghwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}
		ShowCursor(FALSE);
	}
	else
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
	}
}

void uninitialize(int iExit_Flag)
{
	if (gbFullscreen == true)
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
	}

	if (ghdc != NULL)
	{
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	if (ghrc != NULL)
	{
		wglDeleteContext(ghrc);
		ghrc = NULL;
	}

	if (iExit_Flag == 0)
	{
		fprintf(gpFile, "Log File Closed Successfully...Calling Destroy Window()\n");
		fclose(gpFile);
		DestroyWindow(ghwnd);
	}
	else if (iExit_Flag == 1)
	{
		fprintf(gpFile, "Log File Closed Successfully...Calling exit(EXIT_FAILURE)\n");
		fclose(gpFile);
		exit(EXIT_FAILURE);
	}
}
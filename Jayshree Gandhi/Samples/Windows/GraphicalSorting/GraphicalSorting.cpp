#include<windows.h>
#include<gl/GL.h>
#include<gl/GLU.h>
#include<stdio.h>
#include<stdlib.h>

#pragma comment(lib,"opengl32.lib")
#pragma comment(lib,"glu32.lib")

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

GLubyte space[] = { 0x00,
					0x00,
					0x00,
					0x00,
					0x00,
					0x00,
					0x00,
					0x00,
					0x00,
					0x00,
					0x00,
					0x00,
					0x00 };

GLubyte letters[][13] = {
	{0x00, 0x00, 0xc3, 0xc3, 0xc3, 0xc3, 0xff, 0xc3, 0xc3, 0xc3, 0x66, 0x3c, 0x18},
	{0x00, 0x00, 0xfe, 0xc7, 0xc3, 0xc3, 0xc7, 0xfe, 0xc7, 0xc3, 0xc3, 0xc7, 0xfe},
	{0x00, 0x00, 0x7e, 0xe7, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xe7, 0x7e},
	{0x00, 0x00, 0xfc, 0xce, 0xc7, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xc7, 0xce, 0xfc},
	{0x00, 0x00, 0xff, 0xc0, 0xc0, 0xc0, 0xc0, 0xfc, 0xc0, 0xc0, 0xc0, 0xc0, 0xff},
	{0x00, 0x00, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xfc, 0xc0, 0xc0, 0xc0, 0xff},
	{0x00, 0x00, 0x7e, 0xe7, 0xc3, 0xc3, 0xcf, 0xc0, 0xc0, 0xc0, 0xc0, 0xe7, 0x7e},
	{0x00, 0x00, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xff, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3},
	{0x00, 0x00, 0x7e, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x7e},
	{0x00, 0x00, 0x7c, 0xee, 0xc6, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06},
	{0x00, 0x00, 0xc3, 0xc6, 0xcc, 0xd8, 0xf0, 0xe0, 0xf0, 0xd8, 0xcc, 0xc6, 0xc3},
	{0x00, 0x00, 0xff, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0},
	{0x00, 0x00, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xdb, 0xff, 0xff, 0xe7, 0xc3},
	{0x00, 0x00, 0xc7, 0xc7, 0xcf, 0xcf, 0xdf, 0xdb, 0xfb, 0xf3, 0xf3, 0xe3, 0xe3},
	{0x00, 0x00, 0x7e, 0xe7, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xe7, 0x7e},
	{0x00, 0x00, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xfe, 0xc7, 0xc3, 0xc3, 0xc7, 0xfe},
	{0x00, 0x00, 0x3f, 0x6e, 0xdf, 0xdb, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0x66, 0x3c},
	{0x00, 0x00, 0xc3, 0xc6, 0xcc, 0xd8, 0xf0, 0xfe, 0xc7, 0xc3, 0xc3, 0xc7, 0xfe},
	{0x00, 0x00, 0x7e, 0xe7, 0x03, 0x03, 0x07, 0x7e, 0xe0, 0xc0, 0xc0, 0xe7, 0x7e},
	{0x00, 0x00, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0xff},
	{0x00, 0x00, 0x7e, 0xe7, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3},
	{0x00, 0x00, 0x18, 0x3c, 0x3c, 0x66, 0x66, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3},
	{0x00, 0x00, 0xc3, 0xe7, 0xff, 0xff, 0xdb, 0xdb, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3},
	{0x00, 0x00, 0xc3, 0x66, 0x66, 0x3c, 0x3c, 0x18, 0x3c, 0x3c, 0x66, 0x66, 0xc3},
	{0x00, 0x00, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3c, 0x3c, 0x66, 0x66, 0xc3},
	{0x00, 0x00, 0xff, 0xc0, 0xc0, 0x60, 0x30, 0x7e, 0x0c, 0x06, 0x03, 0x03, 0xff}
};

GLubyte digits[][13] = {
	{0x00, 0xFF, 0xFF, 0xC3, 0xC3, 0xC3, 0xC3, 0xC3, 0xC3, 0xC3, 0xC3, 0xFF, 0xFF},
	{0x00, 0xFF, 0xFF, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x98, 0xD8, 0x78, 0x38},
	{0x00, 0xFF, 0xFF, 0xC0, 0xC0, 0xC0, 0xFF, 0xFF, 0x03, 0x03, 0x03, 0xFF, 0xFF},
	{0x00, 0xFF, 0xFF, 0x03, 0x03, 0x03, 0xFF, 0xFF, 0x03, 0x03, 0x03, 0xFF, 0xFF},
	{0x00, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0xFF, 0xFF, 0xC3, 0xC3, 0xC3, 0xC3},
	{0x00, 0xFF, 0xFF, 0x03, 0x03, 0x03, 0xFF, 0xFF, 0xC0, 0xC0, 0xC0, 0xFF, 0xFF},
	{0x00, 0xFF, 0xFF, 0xC3, 0xC3, 0xC3, 0xFF, 0xFF, 0xC0, 0xC0, 0xC0, 0xFF, 0xFF},
	{0x00, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0xFF, 0xFF},
	{0x00, 0xFF, 0xFF, 0xC3, 0xC3, 0xC3, 0xFF, 0xFF, 0xC3, 0xC3, 0xC3, 0xFF, 0xFF},
	{0x00, 0xFF, 0xFF, 0x03, 0x03, 0x03, 0xFF, 0xFF, 0xC3, 0xC3, 0xC3, 0xFF, 0xFF}
};

GLuint fontOffset;

bool gbFullScreen = false;
DWORD dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };
HWND ghWnd = NULL;
HDC ghdc = NULL;
HGLRC ghrc = NULL;
bool gbActiveWindow = false;
FILE *gpFile = NULL;


void ToggleFullScreen(void);
int initialize(void);
void resize(int, int);
void display(void);
void update(void);
void uninitialize(void);

void makeRasterFont(void);
void printString(const char *);

char keyPressed;
void menu(void);

//quick sort
int qsLength = 100;
int qsDelay = 5000;
int quickSortArray[100];

int qsPartition(int[], int, int);
void qsQuickSort(int[], int, int);
void qsSort(int[], int);
void qsRenderFunction(void);
void qsRandomizeArray(int[], int);
void qsSwap(int, int);
void qsMySwap(int, int);

//Bubble sort
int bsLength = 100;
int bubbleSortArray[100];
int k = 0;
int s = 0;

void bsBubbleSort(int[], int);
void bsRandomizeArray(int[], int);
void bsSwap(int, int);
void bsMySwap(int, int);
void bsRenderFunction(void);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szAppName[] = TEXT("MY GRAPHICAL SORTING WINDOW");
	bool bDone = false;
	int iRet = 0;

	if (fopen_s(&gpFile, "Log.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Log file can not be created"), TEXT("Error"), MB_OK);
		exit(0);
	}
	else
	{
		fprintf(gpFile, "Log file created successfully...\n");
	}

	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.lpfnWndProc = WndProc;
	wndclass.hInstance = hInstance;
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass.lpszClassName = szAppName;
	wndclass.lpszMenuName = NULL;
	wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);

	RegisterClassEx(&wndclass);

	hwnd = CreateWindowEx(WS_EX_APPWINDOW,
		szAppName,
		TEXT("Graphical Sorting"),
		WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
		100,
		100,
		WIN_WIDTH,
		WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL);

	ghWnd = hwnd;

	iRet = initialize();
	if (iRet == -1)
	{
		fprintf(gpFile, "ChoosePixelFormat() Failed\n");
		DestroyWindow(hwnd);
	}
	else if (iRet == -2)
	{
		fprintf(gpFile, "SetPixelFormat() Failed\n");
		DestroyWindow(hwnd);
	}
	else if (iRet == -3)
	{
		fprintf(gpFile, "wglCreateContext() Failed\n");
		DestroyWindow(hwnd);
	}
	else if (iRet == -4)
	{
		fprintf(gpFile, "wglMakeCurrent() Failed\n");
		DestroyWindow(hwnd);
	}
	else
	{
		fprintf(gpFile, "Initialization succeded\n");
	}

	ToggleFullScreen();
	ShowWindow(hwnd, iCmdShow);
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);

	while (bDone == false)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
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
			//Play game here
			if (gbActiveWindow == true)
			{
				//code
				//here call update
				update();
			}
			display();
		}
	}

	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{

	switch (iMsg)
	{
	case WM_SETFOCUS:
		gbActiveWindow = true;
		break;

	case WM_KILLFOCUS:
		gbActiveWindow = false;
		break;

	case WM_SIZE:
		resize(LOWORD(lParam), HIWORD(lParam));
		break;

	case WM_ERASEBKGND:
		return(0);
		break;

	case WM_CLOSE:
		DestroyWindow(hwnd);
		break;

	case WM_CHAR:
		switch (wParam)
		{
		case VK_ESCAPE:
			DestroyWindow(hwnd);
			break;

		case 'Q':
		case 'q':
			keyPressed = 'Q';
			break;

		case 'B':
		case 'b':
			keyPressed = 'B';
			break;

		case 'S':
		case 's':
			qsSort(quickSortArray, qsLength);
			break;

		case 'R':
		case 'r':
			qsRandomizeArray(quickSortArray, qsLength);
			break;

		default:
			keyPressed = 'M';
			break;
		}
		break;

	case WM_DESTROY:
		uninitialize();
		PostQuitMessage(0);
		break;
	}

	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void ToggleFullScreen(void)
{
	MONITORINFO mi;

	if (gbFullScreen == false)
	{
		dwStyle = GetWindowLong(ghWnd, GWL_STYLE);

		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{
			mi = { sizeof(MONITORINFO) };

			if (GetWindowPlacement(ghWnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghWnd, MONITORINFOF_PRIMARY), &mi))
			{
				SetWindowLong(ghWnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);

				SetWindowPos(ghWnd,
					HWND_TOP,
					mi.rcMonitor.left,
					mi.rcMonitor.top,
					mi.rcMonitor.right - mi.rcMonitor.left,
					mi.rcMonitor.bottom - mi.rcMonitor.top,
					SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}

		ShowCursor(FALSE);
		gbFullScreen = true;
	}
}

int initialize(void)
{
	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;

	ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));

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

	ghdc = GetDC(ghWnd);

	iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);
	if (iPixelFormatIndex == 0)
	{
		return(-1);
	}

	if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
	{
		return(-2);
	}

	ghrc = wglCreateContext(ghdc);
	if (ghrc == NULL)
	{
		return(-3);
	}

	if (wglMakeCurrent(ghdc, ghrc) == FALSE)
	{
		return(-4);
	}

	makeRasterFont();
	keyPressed = 'M';

	//quick sort
	for (int i = 0; i < qsLength; i++)
	{
		quickSortArray[i] = i;
	}
	qsRandomizeArray(quickSortArray, qsLength);

	//bubble sort
	for (int i = 0; i < bsLength; i++)
	{
		bubbleSortArray[i] = i;
	}
	bsRandomizeArray(bubbleSortArray, bsLength);

	glShadeModel(GL_SMOOTH);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	resize(WIN_WIDTH, WIN_HEIGHT);

	return(0);
}

void resize(int width, int height)
{
	if (height == 0)
	{
		height = 1;
	}

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);

	glLoadIdentity();

	if (keyPressed == 'M')
	{
		menu();
	}
	else if (keyPressed == 'Q')
	{
		qsRenderFunction();
	}
	else if (keyPressed == 'B')
	{
		bsRenderFunction();
	}


	SwapBuffers(ghdc);

}

void update(void)
{

}


void uninitialize(void)
{

	if (gbFullScreen == true)
	{
		SetWindowLong(ghWnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);

		SetWindowPlacement(ghWnd, &wpPrev);

		SetWindowPos(ghWnd,
			HWND_TOP,
			0,
			0,
			0,
			0,
			SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER);

		ShowCursor(TRUE);
	}

	if (wglGetCurrentContext() == ghrc)
	{
		wglMakeCurrent(NULL, NULL);
	}

	if (ghrc)
	{
		wglDeleteContext(ghrc);
		ghrc = NULL;
	}

	if (ghdc)
	{
		ReleaseDC(ghWnd, ghdc);
		ghdc = NULL;
	}

	if (gpFile)
	{
		fprintf(gpFile, "Log file closed successfully\n");
		fclose(gpFile);
		gpFile = NULL;
	}


}

void makeRasterFont(void)
{
	GLuint i, j, k, s;
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	fontOffset = glGenLists(128);

	for (i = 0, j = 'A'; i < 26; i++, j++)
	{
		glNewList(fontOffset + j, GL_COMPILE);
		glBitmap(8, 13, 0.0f, 2.0f, 10.0f, 0.0f, letters[i]);
		glEndList();
	}

	for (k = 0, s = '0'; k < 10; k++, s++)
	{
		glNewList(fontOffset + s, GL_COMPILE);
		glBitmap(8, 13, 0.0f, 2.0f, 10.0f, 0.0f, digits[k]);
		glEndList();
	}

	glNewList(fontOffset + ' ', GL_COMPILE);
	glBitmap(8, 13, 0.0f, 2.0f, 10.0f, 0.0f, space);
	glEndList();
}

void printString(const char *s)
{
	glPushAttrib(GL_LIST_BIT);
	glListBase(fontOffset);
	glCallLists(strlen(s), GL_UNSIGNED_BYTE, (GLubyte *)s);
	glPopAttrib();
}

void menu(void)
{
	glTranslatef(0.0f, 0.0f, -3.0f);

	glColor3f(0.0f, 1.0f, 1.0f);
	glRasterPos2f(-0.3f, 1.0f);
	printString("GRAPHICAL SORTING");

	glColor3f(1.0f, 0.0f, 1.0f);
	glRasterPos2f(-0.3f, 0.9f);
	printString("PRESS");

	glColor3f(1.0f, 1.0f, 0.0f);
	glRasterPos2f(-0.3f, 0.8f);
	printString("Q FOR QUICK SORT");

	glRasterPos2f(-0.3f, 0.7f);
	printString("B FOR BUBBLE SORT");

}

//quick sort
int qsPartition(int a[], int low, int high)
{
	int lowIndex = low - 1;
	int pivot = a[high];

	for (int i = low; i < high; i++)
	{
		if (a[i] <= pivot)
		{
			++lowIndex;
			qsSwap(lowIndex, i);
		}
	}

	++lowIndex;

	qsSwap(lowIndex, high);

	return (lowIndex);
}

void qsQuickSort(int a[], int low, int high)
{
	if (low < high)
	{
		int pi = qsPartition(a, low, high);

		qsQuickSort(a, low, pi - 1);
		qsQuickSort(a, pi + 1, high);
	}
}

void qsSort(int a[], int length)
{
	qsQuickSort(a, 0, length - 1);
}

void qsRenderFunction(void)
{

	float l = (float)qsLength;
	float widthAdder = 1.0f / l;

	glLoadIdentity();
	glTranslatef(0.0f, 0.0f, -2.5f);

	glColor3f(0.0f, 0.0f, 1.0f);
	glRasterPos2f(1.1f, 1.0f);
	printString("PRESS");

	glRasterPos2f(1.1f, 0.9f);
	printString("R TO RANDOMIZE");

	glRasterPos2f(1.1f, 0.8f);
	printString("S TO SORT");

	glRasterPos2f(1.1f, 0.7f);
	printString("M FOR MAIN MENU");


	for (int i = 0; i < qsLength; i++)
	{

		glBegin(GL_QUADS);

		float arrayIndexHeightRatio = 2 * (quickSortArray[i] + 1) / l;
		float widthIndexAdder = 2 * i / l;

		float leftX = -1 + widthIndexAdder;
		float rightX = leftX + widthAdder;
		float bottomY = -1;
		float topY = bottomY + arrayIndexHeightRatio;

		//top right
		glColor3f(0.0f, 0.0f, 1.0f);
		glVertex3f(rightX, topY, 0.0f);

		//top left
		glColor3f(0.0f, 0.0f, 1.0f);
		glVertex3f(leftX, topY, 0.0f);

		//bottom left
		glColor3f(0.0f, 0.0f, 0.0f);
		glVertex3f(leftX, bottomY, 0.0f);

		//bottom right
		glColor3f(0.0f, 0.0f, 0.5019f);
		glVertex3f(rightX, bottomY, 0.0f);

		glEnd();
	}
}

void qsRandomizeArray(int a[], int length)
{
	for (int i = length - 1; i > 0; --i)
	{
		qsSwap(quickSortArray[i], quickSortArray[rand() % (i + 1)]);
	}
}


void qsSwap(int index1, int index2)
{
	qsMySwap(index1, index2);
}

void qsMySwap(int index1, int index2)
{
	int temp = quickSortArray[index1];
	quickSortArray[index1] = quickSortArray[index2];
	quickSortArray[index2] = temp;
}

//bubble sort
void bsRenderFunction(void)
{

	float l = (float)bsLength;
	float widthAdder = 1.0f / l;

	glLoadIdentity();
	glTranslatef(0.0f, 0.0f, -2.5f);

	glColor3f(0.0f, 1.0f, 0.0f);
	glRasterPos2f(1.2f, 1.0f);
	printString("PRESS");

	glRasterPos2f(1.2f, 0.9f);
	printString("M FOR MAIN MENU");

	for (int i = 0; i < bsLength; i++)
	{

		glBegin(GL_QUADS);

		float arrayIndexHeightRatio = 2 * (bubbleSortArray[i] + 1) / l;
		float widthIndexAdder = 2 * i / l;

		float leftX = -1 + widthIndexAdder;
		float rightX = leftX + widthAdder;
		float bottomY = -1;
		float topY = bottomY + arrayIndexHeightRatio;

		//top right
		glColor3f(0.0f, 1.0f, 0.0f);
		glVertex3f(rightX, topY, 0.0f);

		//top left
		glColor3f(0.0f, 1.0f, 0.0f);
		glVertex3f(leftX, topY, 0.0f);

		//bottom left
		glColor3f(0.0f, 0.0f, 0.0f);
		glVertex3f(leftX, bottomY, 0.0f);

		//bottom right
		glColor3f(0.0f, 0.5019f, 0.0f);
		glVertex3f(rightX, bottomY, 0.0f);

		glEnd();
	}

	int a = bubbleSortArray[s];
	int b = bubbleSortArray[s + 1];
	if (a > b)
	{
		bsSwap(s, s + 1);
	}

	if (k < bsLength)
	{
		s = s + 1;
		if (s >= bsLength - k - 1)
		{
			s = 0;
			k = k + 1;
		}
	}
}

void bsSwap(int index1, int index2)
{
	bsMySwap(index1, index2);
}

void bsMySwap(int index1, int index2)
{
	int temp = bubbleSortArray[index1];
	bubbleSortArray[index1] = bubbleSortArray[index2];
	bubbleSortArray[index2] = temp;
}

void bsRandomizeArray(int arr[], int length)
{
	for (int i = length - 1; i > 0; --i)
	{
		bsSwap(arr[i], arr[rand() % (i + 1)]);
	}
}

void bsBubbleSort(int arr[], int length)
{
	for (int i = 0; i < length; i++)
	{
		for (int j = 0; j < length - i - 1; j++)
		{
			int a = arr[j];
			int b = arr[j + 1];

			if (a > b)
			{
				bsSwap(j, j + 1);
			}
		}

	}
}

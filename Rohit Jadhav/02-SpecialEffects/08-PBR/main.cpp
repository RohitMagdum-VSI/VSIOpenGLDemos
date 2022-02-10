#include<Windows.h>
#include<stdio.h>

#include<gl/glew.h>
#include<gl/GL.h>


#include"vmath.h"
#include<assert.h>
#include<string.h>


#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")


#define WIN_WIDTH 800
#define WIN_HEIGHT 600

using namespace vmath;


enum {
	AMC_ATTRIBUTE_POSITION = 0,
	AMC_ATTRIBUTE_COLOR,
	AMC_ATTRIBUTE_NORMAL,
	AMC_ATTRIBUTE_TEXCOORD0,
};

#define STB_IMAGE_IMPLEMENTATION

#include"main.h"
#include"01-Utility/stb_texture.h"
#include"01-Utility/01-KtxLoader.h"
#include"01-Utility/02-ModelLoading.h"
#include"01-Utility/03-Camera.h"
#include"01-Utility/04-Utils.h"
#include"01-Utility/05-Shapes.h"

#include"02-PBR/PBR.h"



//For FullScreen 
bool bIsFullScreen = false;
HWND ghwnd;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };
DWORD dwStyle;

//For SuperMan
bool bActiveWindow;
HDC ghdc = NULL;
HGLRC ghrc = NULL;

//For Error
FILE *gpFile;


//For Global Matrix
mat4 global_PerspectiveProjectionMatrix;
mat4 global_ViewMatrix;

//For Camera
CAMERA c;
GLfloat gCameraSpeed = 0.10;
GLfloat fPitchAngle = 20.0f;
GLfloat fYawAngle = -90.0f;


//For Time
DWORD gdwStartTime = 0;



LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR szCmdLine, int iCmdShow) {


	if (fopen_s(&gpFile, "Log.txt", "w") != 0) {
		MessageBox(NULL, TEXT("Log Creation Failed!!"), TEXT("Error"), MB_OK);
		exit(0);
	}
	else
		fprintf(gpFile, "Log Created!!\n");

	int initialize(void);
	void display(void);
	void ToggleFullScreen(void);


	gdwStartTime = GetTickCount();

	initialize_Camera(c);

	int iRet;
	bool bDone = false;

	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szName[] = TEXT("Rohit_R_Jadhav-PBR-v1");

	wndclass.lpszClassName = szName;
	wndclass.lpszMenuName = NULL;
	wndclass.lpfnWndProc = WndProc;

	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;

	wndclass.hInstance = hInstance;
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(MYICON));
	wndclass.hIconSm = LoadIcon(hInstance, MAKEINTRESOURCE(MYICON));
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);

	RegisterClassEx(&wndclass);

	hwnd = CreateWindowEx(WS_EX_APPWINDOW,
		szName,
		TEXT("Rohit_R_Jadhav-PBR-v1"),
		WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_VISIBLE,
		100, 100,
		WIN_WIDTH, WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL);

	ghwnd = hwnd;

	SetFocus(hwnd);
	SetForegroundWindow(hwnd);

	iRet = initialize();
	if (iRet == -1) {
		fprintf(gpFile, "ChoosePixelFormat() Failed!!\n");
		DestroyWindow(hwnd);
	}
	else if (iRet == -2) {
		fprintf(gpFile, "SetPixelFormat() Failed!!\n");
		DestroyWindow(hwnd);
	}
	else if (iRet == -3) {
		fprintf(gpFile, "wglCreateContext() Failed!!\n");
		DestroyWindow(hwnd);
	}
	else if (iRet == -4) {
		fprintf(gpFile, "wglMakeCurrent() Failed!!\n");
		DestroyWindow(hwnd);
	}
	else
		fprintf(gpFile, "initialize() done!!\n\n");

	ShowWindow(hwnd, iCmdShow);
	ToggleFullScreen();


	while (bDone == false) {
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
			if (msg.message == WM_QUIT)
				bDone = true;
			else {
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		else {
			if (bActiveWindow == true){
				// update();
			}
			display();
		}
	}
	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {

	void uninitialize(void);
	void resize(int, int);
	void ToggleFullScreen(void);

	switch (msg) {
	case WM_SETFOCUS:
		bActiveWindow = true;
		break;
	case WM_KILLFOCUS:
		bActiveWindow = false;
		break;
	case WM_SIZE:
		resize(LOWORD(lParam), HIWORD(lParam));
		break;
	case WM_KEYDOWN:
		switch (wParam) {
		case 'F':
			ToggleFullScreen();
			break;

		case 'W':
			moveForwardStright(c, gCameraSpeed);
			break;

		case 'S':
			moveBackwardStright(c, gCameraSpeed);
			break;

		case 'A':
			moveLeft(c, gCameraSpeed);
			break;

		case 'D':
			moveRight(c, gCameraSpeed);
			break;

		case VK_ESCAPE:
			DestroyWindow(hwnd);
			break;
		}
		break;


	case WM_CHAR:
		switch(wParam){
			case 'Q':
				fPitchAngle += 1.0f;
				setCameraFrontUsingAngle(c, fPitchAngle, fYawAngle);
				break;

			case 'q':
				fPitchAngle -= 1.0f;
				setCameraFrontUsingAngle(c, fPitchAngle, fYawAngle);
				break;


			case 'E':
				fYawAngle += 1.0f;
				setCameraFrontUsingAngle(c, fPitchAngle, fYawAngle);
				break;

			case 'e':
				fYawAngle -= 1.0f;
				setCameraFrontUsingAngle(c, fPitchAngle, fYawAngle);
				break;

		}
		break;


	case WM_ERASEBKGND:
		return(0);

	case WM_DESTROY:
		uninitialize();
		PostQuitMessage(0);
		break;
	}
	return(DefWindowProc(hwnd, msg, wParam, lParam));
}

void ToggleFullScreen(void) {
	MONITORINFO mi;

	if (bIsFullScreen == false) {
		dwStyle = GetWindowLong(ghwnd, GWL_STYLE);
		mi = { sizeof(MONITORINFO) };
		if (dwStyle & WS_OVERLAPPEDWINDOW) {
			if (GetWindowPlacement(ghwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &mi)) {
				SetWindowLong(ghwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghwnd,
					HWND_TOP,
					mi.rcMonitor.left,
					mi.rcMonitor.top,
					(mi.rcMonitor.right - mi.rcMonitor.left),
					(mi.rcMonitor.bottom - mi.rcMonitor.top),
					SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}
		ShowCursor(FALSE);
		bIsFullScreen = true;
	}
	else {
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd,
			HWND_TOP,
			0, 0, 0, 0,
			SWP_NOZORDER | SWP_NOSIZE | SWP_NOMOVE | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
		bIsFullScreen = false;
	}
}




int initialize(void) {

	GLenum Result;
	void resize(int, int);
	void uninitialize(void);


	PIXELFORMATDESCRIPTOR pfd;
	int iPixelTypeIndex;

	memset(&pfd, NULL, sizeof(PIXELFORMATDESCRIPTOR));

	ghdc = GetDC(ghwnd);

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

	iPixelTypeIndex = ChoosePixelFormat(ghdc, &pfd);
	if (iPixelTypeIndex == 0)
		return(-1);

	if (SetPixelFormat(ghdc, iPixelTypeIndex, &pfd) == FALSE)
		return(-2);

	ghrc = wglCreateContext(ghdc);
	if (ghrc == NULL)
		return(-3);

	if (wglMakeCurrent(ghdc, ghrc) == FALSE)
		return(-4);

	Result = glewInit();
	if (Result != GLEW_OK) {
		fprintf(gpFile, "glewInit() Failed!!\n");
		uninitialize();
		DestroyWindow(ghwnd);
	}


	fprintf(gpFile, "\nOpenGL Version : %s\n", glGetString(GL_VERSION));
	fprintf(gpFile, "Renderer : %s\n", glGetString(GL_RENDERER));
	fprintf(gpFile, "Vendor : %s\n", glGetString(GL_VENDOR));
	fprintf(gpFile, "GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));


	initialize_Shapes();

	initialize_PBR();


	
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);

	
	glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
	resize(WIN_WIDTH, WIN_HEIGHT);

	return(0);
}





void uninitialize(void) {


	if(bIsFullScreen == true){

		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd,
			HWND_TOP,
			0, 0, 0, 0,
			SWP_NOZORDER | SWP_NOSIZE | SWP_NOMOVE | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
		bIsFullScreen = false;
	}


	uninitialize_Shape();
	uninitialize_PBR();


	if (wglGetCurrentContext() == ghrc)
		wglMakeCurrent(NULL, NULL);

	if (ghrc) {
		wglDeleteContext(ghrc);
		ghrc = NULL;
	}

	if (ghdc) {
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	if (gpFile) {
		fprintf(gpFile, "\n\nLog Close!!\n");
		fclose(gpFile);
		gpFile = NULL;
	}
}



void resize(int width, int height) {

	if (height == 0)
		height = 1;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	global_PerspectiveProjectionMatrix = mat4::identity();
	global_PerspectiveProjectionMatrix = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);

}

void display(void) {

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	display_PBR();

	
	SwapBuffers(ghdc);
}


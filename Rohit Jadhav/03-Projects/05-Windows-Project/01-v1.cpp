#include<Windows.h>
#include<stdio.h>

#include<gl/glew.h>
#include<gl/GL.h>


#include"vmath.h"
#include<assert.h>
#include<string.h>
#include<Mmsystem.h>


#include<ft2build.h>
#include FT_FREETYPE_H

#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "freetype.lib")
#pragma comment(lib, "winmm.lib")

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

using namespace vmath;


enum {
	ATTRIBUTE_POSITION = 0,
	ATTRIBUTE_COLOR,
	ATTRIBUTE_NORMAL,
	ATTRIBUTE_TEXCOORD0,
	ATTRIBUTE_HEIGHTMAP,
	ATTRIBUTE_SLOPE,
};


#include"01-v1.h"
#include"00-Common/01-KtxLoader.h"
#include"00-Common/02-ModelLoading.h"
#include"00-Common/03-Camera.h"
#include"00-Common/04-Utils.h"

//For Models
extern MODEL *godray_gpModelTree;
extern MODEL *gpModelLightHouse;



#include"01-Fire/01-Fire.h"
#include"04-Model/04-Model.h"

#include"02-Terrain/02-Terrain.h"
#include"03-GodRays/03-GodRays.h"

#include"05-Water/05-FFT-Water.h"
#include"06-Font/06-Font.h"
#include"08-Fade/08-Fade.h"




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
mat4 PerspectiveProjectionMatrix;
mat4 globalViewMatrix;

//For Camera
CAMERA c;
GLfloat gCameraSpeed = 0.10;
GLfloat fPitchAngle = 20.0f;
GLfloat fYawAngle = -90.0f;


//For Models
MODEL *godray_gpModelTree;
MODEL *gpModelLightHouse;


//For Scene
#define BLK0 9

#define AMC 0
#define BLK1 10

#define GRP 1
#define PRESENT 2
#define BLK2 11

#define DEMO 3

#define PHOTO 4
#define BLK3 12

#define NAME 5
#define BLK4 13

#define CREDIT 6
#define BLK5 14

#define THANKS 7
#define BLK6 15

#define END 8

GLuint gSceneNo = BLK0;


//For Time
DWORD gdwStartTime = 0;
bool gIsPlay = false;

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


	initialize_Camera(c);

	int iRet;
	bool bDone = false;

	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szName[] = TEXT("Rohit_R_Jadhav-Demo");

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
		TEXT("Rohit_R_Jadhav-Demo"),
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

		case 'P':
			gIsPlay = true;
			gdwStartTime = GetTickCount();
			PlaySound(MAKEINTRESOURCE(ID_SONG), GetModuleHandle(NULL), SND_RESOURCE | SND_ASYNC | SND_LOOP);
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


void initialize_Cuda(void){

	int devCount = 0;
	error = cudaGetDeviceCount(&devCount);
	if(error != cudaSuccess){
		fprintf(gpFile, "cudaGetDeviceCount() Failed!!\n");
		uninitialize();
		exit(0);
	}
	else if(devCount == 0){
		fprintf(gpFile, "devCount == 0\n");
		uninitialize();
		exit(0);
	}
	else{
		fprintf(gpFile, "DevCount: %d\n", devCount);
		cudaSetDevice(0);
	}
}


int initialize(void) {

	GLenum Result;
	void resize(int, int);
	void uninitialize(void);


	initialize_Cuda();


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



	// // *************** Model ***************
	// godray_gpModelTree = initialize_ModelWithFileName("04-Model/tree1.txt");
	// if(godray_gpModelTree == NULL){
	// 	fprintf(gpFile, "ERROR: godray_gpModelTree == NULL");
	// 	uninitialize();
	// 	exit(0);
	// }

	// LoadModel(godray_gpModelTree);

	// *************** Model ***************
	gpModelLightHouse = initialize_ModelWithFileName("04-Model/lightHouse2.txt");
	if(gpModelLightHouse == NULL){
		fprintf(gpFile, "ERROR: gpModelLightHouse == NULL");
		uninitialize();
		exit(0);
	}

	LoadModel(gpModelLightHouse);

	gpModelLightHouse->texture = LoadTexture(MAKEINTRESOURCE(ID_LIGHT_HOUSE_TEX), 0);




	initialize_Font();

	initialize_ModelWithLight();


	initialize_Fire();

	initialize_Terrain();

	initialize_GodRays_Final();

	initialize_Water();

	initialize_Fade();
	

	
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);

	
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	resize(WIN_WIDTH, WIN_HEIGHT);


	//For Calculation of Normals
	//display_Terrain_TF_Pass();

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


	// if(godray_gpModelTree)
	// 	uninitialize_Model(godray_gpModelTree);

	if(gpModelLightHouse)
		uninitialize_Model(gpModelLightHouse);


	uninitialize_Font();

	uninitialize_ModelWithLight();

	uninitialize_Fire();

	uninitialize_Terrain();

	uninitialize_GodRays_Final();

	uninitialize_Water();

	uninitialize_Fade();


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

	PerspectiveProjectionMatrix = mat4::identity();
	PerspectiveProjectionMatrix = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);

}

void display(void) {


	static GLfloat fade = 1.0f;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if(gIsPlay == true){
		
		GLfloat currentTime = (GLfloat)((GetTickCount() - gdwStartTime) / 1000.0f);

		switch(gSceneNo){

			case BLK0:
				if(currentTime > 5.0f)
					gSceneNo = AMC;
				break;

			case AMC:
				display_Font("ASTROMEDICOMP", -30.0f, 0.0f, 0.150f, vec3(1.0f, 1.0f, 1.0f));
				if(currentTime > 10.0f)
					gSceneNo = BLK1;
				break;


			case BLK1:
				if(currentTime > 12.0f)
					gSceneNo = GRP;
				break;

			case GRP:
				display_Font("COMPUTE GROUP", -25.0f, 0.0f, 0.120f, vec3(1.0f, 1.0f, 1.0f));
				display_Font("PRESENTS...", 0.0f, -10.0f, 0.10f, vec3(1.0f, 1.0f, 0.0f));
				if(currentTime > 16.0f){
					gSceneNo = BLK2;
					fade = 1.0f;
				}
				break;

			case BLK2:	

				display_GodRays_Moon();
				display_Fade(fade);

				fade = fade - 0.009f;

				if(currentTime > 18.0f){
					gSceneNo = DEMO;
					fPitchAngle = 20.0f;
					fade = 0.0f;
				}	

				break;


			case DEMO:

				if(currentTime < 30.0f){
					fPitchAngle -= 0.025f;
					setCameraFrontUsingAngle(c, fPitchAngle, fYawAngle);
				}
				else if(currentTime < 45.0f){
					moveBackwardStright(c, gCameraSpeed);
				}
				
				display_GodRays_Moon();

				display_GodRays_LightHouse();

				display_Water();

				display_Terrain();


				if(currentTime > 58.0f && currentTime < 60.0f){
					display_Fade(fade);
					fade += 0.01f;
				}

				display_Fade(fade);

				if(currentTime > 60.0f)
					gSceneNo = PHOTO;

				break;

			case PHOTO:
				display_Font("PHOTO",  -15.0f, 0.0f, 0.20f, vec3(1.0f, 1.0f, 1.0f));
				if(currentTime > 63.0f)
					gSceneNo = BLK3;
				break;


			case BLK3:
				if(currentTime > 64.0f)
					gSceneNo = NAME;
				break;



			case NAME:
				display_Font("Created by",  -10.0f, 10.0f, 0.10f, vec3(1.0f, 1.0f, 0.0f));
				display_Font("Rohit Ramesh Jadhav",  -33.0f, 0.0f, 0.150f, vec3(1.0f, 1.0f, 1.0f));
				if(currentTime > 67.0f)
					gSceneNo = BLK4;
				break;


			case BLK4:
				if(currentTime > 68.0f)
					gSceneNo = CREDIT;
				break;

			case CREDIT:

				display_Font("Effects",  -10.0f, 20.0f, 0.100f, vec3(1.0f, 1.0f, 0.0f));
				display_Font("1. CUDA FFT Ocean : CUDA Samples",  -43.0f, 10.0f, 0.10f, vec3(1.0f, 1.0f, 1.0f));
				display_Font("2. God Rays : GPU GEMS 3",  -30.0f, 0.0f, 0.10f, vec3(1.0f, 1.0f, 1.0f));

				display_Font("Music",  -10.0f, -10.0f, 0.10f, vec3(1.0f, 1.0f, 0.0f));
				display_Font("Kitaro : In the Beginning",  -25.0f, -20.0f, 0.10f, vec3(1.0f, 1.0f, 1.0f));

				if(currentTime > 71.0f)
					gSceneNo = BLK5;
				break;


			case BLK5:
				if(currentTime > 72.0f)
					gSceneNo = END;
				break;


			case END:
				display_Font("The End",  -10.0f, 0.0f, 0.150f, vec3(1.0f, 1.0f, 1.0f));
				if(currentTime > 76.0f)
					DestroyWindow(ghwnd);
				break;

		}
	}


	

		// display_GodRays_Moon();

		// display_GodRays_LightHouse();

		// display_Water();

		// display_Terrain();
		// //display_Fire();	
		
	SwapBuffers(ghdc);
}


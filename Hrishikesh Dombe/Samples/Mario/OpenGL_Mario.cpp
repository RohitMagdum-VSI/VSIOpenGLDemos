#define _USE_MATH_DEFINES

#include<windows.h>
#include<gl/GL.h>
#include<gl/GLU.h>
#include<stdio.h>
#include<math.h>
#include <time.h>
#include <cstdlib>
#include<mmsystem.h>
#include"01-OpenGL_Alphabets.h"
#include"Error_String.h"
#include"PowerString.h"
#include"accessPrivateMemebr.h"
#include"diamondProblem.h"
#include"usePointer.h"
#include"textVirtualInheritance.h"
#include"textConst.h"
#include"textFixedValue.h"

#pragma comment(lib,"User32.lib")
#pragma comment(lib,"GDI32.lib")
#pragma comment(lib,"opengl32.lib")
#pragma comment(lib,"glu32.lib")
#pragma comment(lib,"winmm.lib")

#define WIN_WIDTH 800
#define WIN_HEIGHT 600
#define PI 3.1415

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

FILE *gpFile;
HWND ghwnd;
HDC ghdc;
HGLRC ghrc;
DWORD dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };
bool gbActiveWindow = false;
bool gbFullscreen = false;
bool gbIsEscapeKeyPressed = false;
bool gbLeft_Leg = false;
bool gbRight_Leg = false;
bool gbMario_Jump = false;
bool gbMario_Jump_Complete = false;
bool gbStart_Hurdles = false;
bool gbStart_Power = false;
bool gbOpenGL_Complete_Flag = false;
GLfloat gfVertex_X_Increment_Value = 0.0f;
GLdouble gfVertex_X_Increment_Value_For_Cactus = 45.0f;
GLfloat gfVertex_X_Pipe_Increment_Value = 0.0f;
GLfloat gfTranslate_X_Tile = -4.2f;
GLfloat fMario_Leg_Increment_Value = 0.0f;
GLfloat fMario_Leg_Decrement_Value = 0.0f;
GLfloat gfMario_Y_Translate = -1.3f;
GLfloat gfTranslate_X_Mushroom = 1.0f;
GLfloat gfIncrement_For_Power;
GLfloat gfStage_Assembly_Text_Complete = 0.0f;
GLfloat gfMario_X_Translate = -2.0f;

GLint giHurdles = 1;
GLint giPowers = 0;
GLint giStage = 1;

GLfloat gfTranslate_X_Star = -4.2f;
GLdouble gfTranslate_X_Cactus = -4.2;

//Global Variables for cactus
double gAngleForJaw1 = 8.0f;
bool giJaw1ResetFlag = false;
double gAngleForJaw2 = 350.0f;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	void initialize(void);
	void display(void);
	void update(void);
	void Display_Game(void);

	void ToggleFullscreen(void);
	void uninitialize(int);

	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szClassName[] = TEXT("My App");
	bool bDone = false;

	if (fopen_s(&gpFile, "Log.txt", "w") != NULL)
	{
		MessageBox(NULL, TEXT("Cannot Create Log File !!!"), TEXT("Error"), MB_OK);
		exit(EXIT_FAILURE);
	}
	else
		fprintf(gpFile, "Log File Created Successfully...\n");

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

	hwnd = CreateWindowEx(WS_EX_APPWINDOW, szClassName, TEXT("OpenGL"), WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE, 100, 100, WIN_WIDTH, WIN_HEIGHT, NULL, NULL, hInstance, NULL);
	if (hwnd == NULL)
	{
		fprintf(gpFile, "Cannot Create Window...\n");
		uninitialize(1);
	}

	ghwnd = hwnd;

	ShowWindow(hwnd, iCmdShow);
	SetFocus(hwnd);
	SetForegroundWindow(hwnd);

	initialize();
	ToggleFullscreen();

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
				//Display_Game();
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
		case 0x41:
			fprintf(gpFile, "gfTranslate_X_Tile : %f\n", gfTranslate_X_Tile);
			break;

		case VK_SPACE:
			gbMario_Jump = true;
			break;
		}
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	}
	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void initialize(void)
{
	void resize(int, int);
	void uninitialize(int);
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
		fprintf(gpFile, "GetDC() Failed.\n");
		uninitialize(1);
	}

	iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);
	if (iPixelFormatIndex == 0)
	{
		fprintf(gpFile, "ChoosePixelFormat() Failed.\n");
		uninitialize(1);
	}

	if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
	{
		fprintf(gpFile, "SetPixelFormat() Failed.\n");
		uninitialize(1);
	}

	ghrc = wglCreateContext(ghdc);
	if (ghrc == NULL)
	{
		fprintf(gpFile, "wglCreateContext() Failed.\n");
		uninitialize(1);
	}

	if (wglMakeCurrent(ghdc, ghrc) == FALSE)
	{
		fprintf(gpFile, "wglMakeCurrent() Failed");
		uninitialize(1);
	}

	sndPlaySound("mario.wav", SND_ASYNC);

	glClearColor(0.0f, 0.0f, 1.0f, 0.0f);

	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glShadeModel(GL_SMOOTH);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT , GL_NICEST);

	resize(WIN_WIDTH, WIN_HEIGHT);
}

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();


	void update();



	void Draw_Stage_Assembly(void);
	void Display_Game(void);

	if (gfStage_Assembly_Text_Complete < 1.0f)
		Draw_Stage_Assembly();
	else
	{
		Display_Game();
		update();
	}


	gfStage_Assembly_Text_Complete = gfStage_Assembly_Text_Complete + 0.01f;
}

void Display_Game(void)
{
	void DrawTiles(void);
	void DrawBricks(void);
	void DrawGiftBox(void);
	void DrawPipe(void);
	void Draw_Star(void);
	void Draw_Hurdles(void);
	void Draw_Powers(void);
	void Draw_Cactus(void);
	void Draw_Clouds(void);
	void ShowMushroom(void);
	void Draw_Mario(void);
	void Draw_Stage_C(void);
	void Draw_Stage_C_Plus_Plus(void);
	void Draw_Stage_Win_32(void);
	void Draw_Stage_OpenGL(void);
	void Draw_Stage_Assembly(void);
	void Draw_DLL(void);
	void Draw_No_Code_Security(void);
	void Draw_Transformation_Matrix(void);
	void Draw_Projection_Matrix(void);
	void Draw_Andhar(void);
	void Draw_Fragment(void);
	void uninitialize(int);

	if (gfTranslate_X_Mushroom > -2.4f && gfTranslate_X_Mushroom < -0.3f)
	{
		glClearColor(1.0f, 0.0f, 0.0f, 0.0f);	
		gbStart_Hurdles = true;
	}
		
	else
	{
		glClearColor(0.0f, 0.0f, 1.0f, 0.0f);
		gbStart_Hurdles = false;
	}

	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
	
	if (gbStart_Hurdles == true)
	{
		Draw_Hurdles();
	}

	if (gbStart_Power == true && gfIncrement_For_Power < 70.0f)
	{
		Draw_Powers();
		gfIncrement_For_Power = gfIncrement_For_Power + 1.0f;
	}
	DrawTiles();

	DrawPipe();

	Draw_Star();

	Draw_Cactus();

	Draw_Clouds();

	static bool bFlag = false;
	static bool bFlag1 = false;
	static bool bFlag2 = false;
	static bool bFlag3 = false;
	static bool bFlag4 = false;
	static bool bFlag5 = false;
	static bool bFlag6 = false;
	static bool bFlag7 = false;
	static bool bFlag8 = false;

	if (gfTranslate_X_Tile < -15 && gfTranslate_X_Tile > -16 && bFlag1 == false)
	{
		bFlag = true;
		bFlag1 = true;
		bFlag4 = false;
	}
	if (gfTranslate_X_Tile < -31 && gfTranslate_X_Tile > -32 && bFlag3 == false)
	{
		bFlag = true;
		bFlag3 = true;
		bFlag1 = false;
	}
	if (gfTranslate_X_Tile < -46 && gfTranslate_X_Tile > -47 && bFlag4 == false)
	{
		bFlag = true;
		bFlag4 = true;
		bFlag3 = false;
	}
	if (gfTranslate_X_Tile < -70 && gbOpenGL_Complete_Flag==false)
	{
		gfTranslate_X_Tile = -4.2f;
		gfTranslate_X_Star = -4.2f;
		giStage++;
	}
	if (gfTranslate_X_Tile < -60)
	{
		if (giStage == 1)
			Draw_Stage_C();
		if (giStage == 2)
			Draw_Stage_C_Plus_Plus();
		if (giStage == 3)
			Draw_Stage_Win_32();
		if (giStage == 4)
			Draw_Stage_OpenGL();
		if (giStage == 5)
			gbOpenGL_Complete_Flag = true;
	}

	if (bFlag == true)
	{
		glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


		gfTranslate_X_Tile = gfTranslate_X_Tile + 4.0f;
		gfTranslate_X_Star = gfTranslate_X_Star + 4.0f;
		bFlag = false;
		bFlag2 = true;
	}
	if (bFlag2 == true)
	{
		ShowMushroom();
		if (gfTranslate_X_Mushroom > -2.5f)
			gfTranslate_X_Mushroom = gfTranslate_X_Mushroom - 0.04f;
		
		else
		{
			//gbMario_Jump = true;
			bFlag2 = false;
			gfTranslate_X_Mushroom = 1.5f;
			giHurdles++;
			giPowers++;
			gbStart_Power = true;
			gfIncrement_For_Power = 0.0f;
		}
			
	}

	if (gbOpenGL_Complete_Flag == true)
	{
		Draw_Fragment();
		if (gfMario_Y_Translate < 1.5f)
			gfMario_Y_Translate = gfMario_Y_Translate + 0.05f;
		if(gfMario_Y_Translate >= 1.45f)
			gfMario_X_Translate = gfMario_X_Translate + 0.03f;
	}

	Draw_Mario();

	SwapBuffers(ghdc);
}

void Draw_Fragment(void)
{
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(-3.0f, 0.5f, -10.0f);
	Draw_F();

	glLoadIdentity();

	glTranslatef(-2.0f, 0.5f, -10.0f);
	Draw_R();

	glLoadIdentity();

	glTranslatef(-1.0f, 0.5f, -10.0f);
	Draw_A();

	glLoadIdentity();

	glTranslatef(0.0f, 0.5f, -10.0f);
	Draw_G();

	glLoadIdentity();

	glTranslatef(0.8f, 0.5f, -10.0f);
	Draw_M();

	glLoadIdentity();

	glTranslatef(1.8f, 0.5f, -10.0f);
	Draw_E();

	glLoadIdentity();

	glTranslatef(2.8f, 0.5f, -10.0f);
	Draw_N();

	glLoadIdentity();

	glTranslatef(3.8f, 0.5f, -10.0f);
	Draw_T();

}


void Draw_Andhar(void)
{
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(-3.0f, 0.0f, -10.0f);
	Draw_A();

	glLoadIdentity();

	glTranslatef(-2.0f, 0.0f, -10.0f);
	Draw_N();

	glLoadIdentity();

	glTranslatef(-1.0f, 0.0f, -10.0f);
	Draw_D();

	glLoadIdentity();

	glTranslatef(0.0f, 0.0f, -10.0f);
	Draw_H();

	glLoadIdentity();

	glTranslatef(1.0f, 0.0f, -10.0f);
	Draw_A();

	glLoadIdentity();

	glTranslatef(2.0f, 0.0f, -10.0f);
	Draw_R();
}

void Draw_Transformation_Matrix(void)
{
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(-5.0f, 1.5f, -10.0f);
	Draw_T();

	glLoadIdentity();

	glTranslatef(-4.3f, 1.5f, -10.0f);
	Draw_R();

	glLoadIdentity();

	glTranslatef(-3.5f, 1.5f, -10.0f);
	Draw_A();

	glLoadIdentity();

	glTranslatef(-3.0f, 1.5f, -10.0f);
	Draw_N();

	glLoadIdentity();

	glTranslatef(-2.3f, 1.5f, -10.0f);
	Draw_S();

	glLoadIdentity();

	glTranslatef(-1.6f, 1.5f, -10.0f);
	Draw_F();

	glLoadIdentity();

	glTranslatef(-0.7f, 1.5f, -10.0f);
	Draw_O();

	glLoadIdentity();

	glTranslatef(-0.2f, 1.5f, -10.0f);
	Draw_R();

	glLoadIdentity();

	glTranslatef(0.5f, 1.5f, -10.0f);
	Draw_M();

	glLoadIdentity();

	glTranslatef(1.4f, 1.5f, -10.0f);
	Draw_A();

	glLoadIdentity();

	glTranslatef(1.8f, 1.5f, -10.0f);
	Draw_T();

	glLoadIdentity();

	glTranslatef(2.5f, 1.5f, -10.0f);
	Draw_I();

	glLoadIdentity();

	glTranslatef(3.3f, 1.5f, -10.0f);
	Draw_O();

	glLoadIdentity();

	glTranslatef(3.7f, 1.5f, -10.0f);
	Draw_N();

	glLoadIdentity();

	glTranslatef(-3.0f, 0.0f, -10.0f);
	Draw_M();

	glLoadIdentity();

	glTranslatef(-2.0f, 0.0f, -10.0f);
	Draw_A();

	glLoadIdentity();

	glTranslatef(-1.7f, 0.0f, -10.0f);
	Draw_T();

	glLoadIdentity();

	glTranslatef(-1.0f, 0.0f, -10.0f);
	Draw_R();

	glLoadIdentity();

	glTranslatef(-0.2f, 0.0f, -10.0f);
	Draw_I();

	glLoadIdentity();

	glTranslatef(0.7f, 0.0f, -10.0f);
	Draw_X();
}

void Draw_Projection_Matrix(void)
{
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(-4.0f, 1.5f, -10.0f);
	Draw_P();

	glLoadIdentity();

	glTranslatef(-3.3f, 1.5f, -10.0f);
	Draw_R();

	glLoadIdentity();

	glTranslatef(-2.3f, 1.5f, -10.0f);
	Draw_O();

	glLoadIdentity();

	glTranslatef(-1.8f, 1.5f, -10.0f);
	Draw_J();

	glLoadIdentity();

	glTranslatef(-1.2f, 1.5f, -10.0f);
	Draw_E();

	glLoadIdentity();

	glTranslatef(-0.2f, 1.5f, -10.0f);
	Draw_C();

	glLoadIdentity();

	glTranslatef(0.0f, 1.5f, -10.0f);
	Draw_T();

	glLoadIdentity();

	glTranslatef(0.8f, 1.5f, -10.0f);
	Draw_I();

	glLoadIdentity();

	glTranslatef(1.5f, 1.5f, -10.0f);
	Draw_O();

	glLoadIdentity();

	glTranslatef(2.0f, 1.5f, -10.0f);
	Draw_N();


	glLoadIdentity();

	glTranslatef(-3.0f, 0.0f, -10.0f);
	Draw_M();

	glLoadIdentity();

	glTranslatef(-2.0f, 0.0f, -10.0f);
	Draw_A();

	glLoadIdentity();

	glTranslatef(-1.7f, 0.0f, -10.0f);
	Draw_T();

	glLoadIdentity();

	glTranslatef(-1.0f, 0.0f, -10.0f);
	Draw_R();

	glLoadIdentity();

	glTranslatef(-0.2f, 0.0f, -10.0f);
	Draw_I();

	glLoadIdentity();

	glTranslatef(0.7f, 0.0f, -10.0f);
	Draw_X();

}

void Draw_No_Code_Security(void)
{
	glPointSize(3.0f);
	glLineWidth(3.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(-4.0f, 1.0f, -10.0f);
	Draw_N();

	glLoadIdentity();

	glTranslatef(-3.0f, 1.0f, -10.0f);
	Draw_O();

	glLoadIdentity();

	glTranslatef(-2.0f, 1.0f, -10.0f);
	Draw_C();

	glLoadIdentity();

	glTranslatef(-1.5f, 1.0f, -10.0f);
	Draw_O();

	glLoadIdentity();

	glTranslatef(-1.0f, 1.0f, -10.0f);
	Draw_D();

	glLoadIdentity();

	glTranslatef(-0.3f, 1.0f, -10.0f);
	Draw_E();

	glLoadIdentity();

	glTranslatef(0.7f, 1.0f, -10.0f);
	Draw_S();

	glLoadIdentity();

	glTranslatef(1.4f, 1.0f, -10.0f);
	Draw_E();

	glLoadIdentity();

	glTranslatef(2.5f, 1.0f, -10.0f);
	Draw_C();

	glLoadIdentity();

	glTranslatef(2.8f, 1.0f, -10.0f);
	Draw_U();

	glLoadIdentity();

	glTranslatef(3.5f, 1.0f, -10.0f);
	Draw_R();

	glLoadIdentity();

	glTranslatef(4.0f, 1.0f, -10.0f);
	Draw_I();

	glLoadIdentity();

	glTranslatef(4.7f, 1.0f, -10.0f);
	Draw_T();

	glLoadIdentity();

	glTranslatef(5.4f, 1.0f, -10.0f);
	Draw_Y();

}

void Draw_Stage_Assembly(void)
{
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(-2.0f, 1.0f, -10.0f);
	Draw_S();

	glLoadIdentity();

	glTranslatef(-1.2f, 1.0f, -10.0f);
	Draw_T();

	glLoadIdentity();

	glTranslatef(0.0f, 1.0f, -10.0f);
	Draw_A();

	glLoadIdentity();

	glTranslatef(1.0f, 1.0f, -10.0f);
	Draw_G();

	glLoadIdentity();

	glTranslatef(1.8f, 1.0f, -10.0f);
	Draw_E();

	glLoadIdentity();

	glTranslatef(-2.5f, -0.5f, -10.0f);
	Draw_A();

	glLoadIdentity();

	glTranslatef(-2.0f, -0.5f, -10.0f);
	Draw_S();

	glLoadIdentity();

	glTranslatef(-1.3f, -0.5f, -10.0f);
	Draw_S();

	glLoadIdentity();

	glTranslatef(-0.6f, -0.5f, -10.0f);
	Draw_E();

	glLoadIdentity();

	glTranslatef(0.1f, -0.5f, -10.0f);
	Draw_M();

	glLoadIdentity();

	glTranslatef(0.8f, -0.5f, -10.0f);
	Draw_B();

	glLoadIdentity();

	glTranslatef(1.5f, -0.5f, -10.0f);
	Draw_L();

	glLoadIdentity();

	glTranslatef(2.0f, -0.5f, -10.0f);
	Draw_Y();

	SwapBuffers(ghdc);
}

void Draw_Stage_C(void)
{
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(-2.0f, 1.0f, -10.0f);
	Draw_S();

	glLoadIdentity();

	glTranslatef(-1.2f, 1.0f, -10.0f);
	Draw_T();

	glLoadIdentity();

	glTranslatef(0.0f, 1.0f, -10.0f);
	Draw_A();

	glLoadIdentity();

	glTranslatef(1.0f, 1.0f, -10.0f);
	Draw_G();

	glLoadIdentity();

	glTranslatef(1.8f, 1.0f, -10.0f);
	Draw_E();

	glLoadIdentity();

	glTranslatef(0.0f, -0.5f, -10.0f);
	Draw_C();
}

void Draw_Stage_C_Plus_Plus(void)
{
	void Draw_Plus(void);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(-2.0f, 1.0f, -10.0f);
	Draw_S();

	glLoadIdentity();

	glTranslatef(-1.2f, 1.0f, -10.0f);
	Draw_T();

	glLoadIdentity();

	glTranslatef(0.0f, 1.0f, -10.0f);
	Draw_A();

	glLoadIdentity();

	glTranslatef(1.0f, 1.0f, -10.0f);
	Draw_G();

	glLoadIdentity();

	glTranslatef(1.8f, 1.0f, -10.0f);
	Draw_E();

	glLoadIdentity();

	glTranslatef(-1.0f, -0.5f, -10.0f);
	Draw_C();

	glLoadIdentity();
	
	glTranslatef(0.0f, -0.5f, -10.0f);
	Draw_Plus();

	glLoadIdentity();
	
	glTranslatef(1.2f, -0.5f, -10.0f);
	Draw_Plus();
}

void Draw_Stage_Win_32(void)
{
	void Draw_3(void);
	void Draw_2(void);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(-2.0f, 1.0f, -10.0f);
	Draw_S();

	glLoadIdentity();

	glTranslatef(-1.2f, 1.0f, -10.0f);
	Draw_T();

	glLoadIdentity();

	glTranslatef(0.0f, 1.0f, -10.0f);
	Draw_A();

	glLoadIdentity();

	glTranslatef(1.0f, 1.0f, -10.0f);
	Draw_G();

	glLoadIdentity();

	glTranslatef(1.8f, 1.0f, -10.0f);
	Draw_E();

	glLoadIdentity();

	glTranslatef(-1.5f, -0.2f, -10.0f);
	Draw_W();

	glLoadIdentity();

	glTranslatef(-0.25f, -0.2f, -10.0f);
	Draw_I();
	
	glLoadIdentity();

	glTranslatef(0.5f, -0.2f, -10.0f);
	Draw_N();

	Draw_3();
	Draw_2();
}

void Draw_Stage_OpenGL(void)
{
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(-2.0f, 1.0f, -10.0f);
	Draw_S();

	glLoadIdentity();

	glTranslatef(-1.2f, 1.0f, -10.0f);
	Draw_T();

	glLoadIdentity();

	glTranslatef(0.0f, 1.0f, -10.0f);
	Draw_A();

	glLoadIdentity();

	glTranslatef(1.0f, 1.0f, -10.0f);
	Draw_G();

	glLoadIdentity();

	glTranslatef(1.8f, 1.0f, -10.0f);
	Draw_E();

	glLoadIdentity();

	glTranslatef(-1.5f, -0.2f, -10.0f);
	Draw_O();

	glLoadIdentity();

	glTranslatef(-0.9f, -0.2f, -10.0f);
	Draw_P();

	glLoadIdentity();

	glTranslatef(-0.2f, -0.2f, -10.0f);
	Draw_E();

	glLoadIdentity();

	glTranslatef(0.5f, -0.2f, -10.0f);
	Draw_N();

	glLoadIdentity();

	glTranslatef(1.6f, -0.2f, -10.0f);
	Draw_G();

	glLoadIdentity();

	glTranslatef(2.0f, -0.2f, -10.0f);
	Draw_L();

}

void update(void)
{
	void update_Tiles(void);
	void update_Mario(void);

	update_Tiles();

	update_Mario();

}

void update_Mario(void)
{
	if (fMario_Leg_Decrement_Value > -0.2f && gbRight_Leg == false)
	{
		fMario_Leg_Decrement_Value = fMario_Leg_Decrement_Value - 0.008f;
		if (fMario_Leg_Decrement_Value < -0.19f)
			gbRight_Leg = true;
	}
	else if (fMario_Leg_Decrement_Value < 0.0f && gbRight_Leg == true)
	{
		fMario_Leg_Decrement_Value = fMario_Leg_Decrement_Value + 0.008f;
		if (fMario_Leg_Decrement_Value > -0.01f)
			gbRight_Leg = false;
	}

	if (fMario_Leg_Increment_Value < 0.2f && gbLeft_Leg == false)
	{
		fMario_Leg_Increment_Value = fMario_Leg_Increment_Value + 0.008f;
		if (fMario_Leg_Increment_Value > 0.19f)
			gbLeft_Leg = true;
	}
	else if (fMario_Leg_Increment_Value > 0.0f && gbLeft_Leg == true)
	{
		fMario_Leg_Increment_Value = fMario_Leg_Increment_Value - 0.008f;
		if (fMario_Leg_Increment_Value < 0.01f)
			gbLeft_Leg = false;
	}

	if (gbMario_Jump == true)
	{
		if (gfMario_Y_Translate < 1.3f && gbMario_Jump_Complete == false)
		{
			gfMario_Y_Translate = gfMario_Y_Translate + 0.03f;
			if (gfMario_Y_Translate > 1.29f)
				gbMario_Jump_Complete = true;
		}
		else if (gfMario_Y_Translate > -1.3f && gbMario_Jump_Complete == true)
		{
			gfMario_Y_Translate = gfMario_Y_Translate - 0.03f;
			if (gfMario_Y_Translate < -1.28f)
			{
				gbMario_Jump_Complete = false;
				gbMario_Jump = false;
			}
		}
	}
}

void update_Tiles(void)
{
	gfTranslate_X_Tile = gfTranslate_X_Tile - 0.03f;
}


void DrawTiles(void)
{
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(gfTranslate_X_Tile, -2.25f, -6.0f);

	for (gfVertex_X_Increment_Value = 0.0f; gfVertex_X_Increment_Value < 120.0f; gfVertex_X_Increment_Value = gfVertex_X_Increment_Value + 0.51f)
	{
		glBegin(GL_QUADS);
		glColor3f(1.0f, 0.3f, 0.0f);
		glVertex3f(0.25f + gfVertex_X_Increment_Value, 0.25f, 0.0f);
		glVertex3f(-0.25f + gfVertex_X_Increment_Value, 0.25f, 0.0f);
		glVertex3f(-0.25f + gfVertex_X_Increment_Value, -0.25f, 0.0f);
		glVertex3f(0.25f + gfVertex_X_Increment_Value, -0.25f, 0.0f);
		glEnd();

		glLineWidth(2.0f);
		glBegin(GL_LINES);
		glColor3f(0.0f, 0.0f, 0.0f);
		glVertex3f(0.1f + gfVertex_X_Increment_Value, 0.1f, 0.0f);
		glVertex3f(0.25f + gfVertex_X_Increment_Value, 0.1f, 0.0f);
		glVertex3f(0.1f + gfVertex_X_Increment_Value, 0.25f, 0.0f);
		glVertex3f(0.1f + gfVertex_X_Increment_Value, -0.05f, 0.0f);
		glVertex3f(0.1f + gfVertex_X_Increment_Value, -0.05f, 0.0f);
		glVertex3f(0.0f + gfVertex_X_Increment_Value, -0.2f, 0.0f);
		glVertex3f(0.0f + gfVertex_X_Increment_Value, -0.2f, 0.0f);
		glVertex3f(0.0f + gfVertex_X_Increment_Value, -0.25f, 0.0f);
		glVertex3f(0.0f + gfVertex_X_Increment_Value, -0.2f, 0.0f);
		glVertex3f(-0.25f + gfVertex_X_Increment_Value, -0.1f, 0.0f);
		glEnd();

		glBegin(GL_LINE_LOOP);
		glColor3f(0.0f, 0.0f, 0.0f);
		glVertex3f(0.25f + gfVertex_X_Increment_Value, 0.25f, 0.0f);
		glVertex3f(-0.25f + gfVertex_X_Increment_Value, 0.25f, 0.0f);
		glVertex3f(-0.25f + gfVertex_X_Increment_Value, -0.25f, 0.0f);
		glVertex3f(0.25f + gfVertex_X_Increment_Value, -0.25f, 0.0f);
		glEnd();
	}

	glLoadIdentity();

	glTranslatef(gfTranslate_X_Tile, -1.74f, -6.0f);

	for (gfVertex_X_Increment_Value = 0.0f; gfVertex_X_Increment_Value < 120.0f; gfVertex_X_Increment_Value = gfVertex_X_Increment_Value + 0.51f)
	{
		glBegin(GL_QUADS);
		glColor3f(1.0f, 0.3f, 0.0f);
		glVertex3f(0.25f + gfVertex_X_Increment_Value, 0.25f, 0.0f);
		glVertex3f(-0.25f + gfVertex_X_Increment_Value, 0.25f, 0.0f);
		glVertex3f(-0.25f + gfVertex_X_Increment_Value, -0.25f, 0.0f);
		glVertex3f(0.25f + gfVertex_X_Increment_Value, -0.25f, 0.0f);
		glEnd();

		glLineWidth(2.0f);
		glBegin(GL_LINES);
		glColor3f(0.0f, 0.0f, 0.0f);
		glVertex3f(0.1f + gfVertex_X_Increment_Value, 0.1f, 0.0f);
		glVertex3f(0.25f + gfVertex_X_Increment_Value, 0.1f, 0.0f);
		glVertex3f(0.1f + gfVertex_X_Increment_Value, 0.25f, 0.0f);
		glVertex3f(0.1f + gfVertex_X_Increment_Value, -0.05f, 0.0f);
		glVertex3f(0.1f + gfVertex_X_Increment_Value, -0.05f, 0.0f);
		glVertex3f(0.0f + gfVertex_X_Increment_Value, -0.2f, 0.0f);
		glVertex3f(0.0f + gfVertex_X_Increment_Value, -0.2f, 0.0f);
		glVertex3f(0.0f + gfVertex_X_Increment_Value, -0.25f, 0.0f);
		glVertex3f(0.0f + gfVertex_X_Increment_Value, -0.2f, 0.0f);
		glVertex3f(-0.25f + gfVertex_X_Increment_Value, -0.1f, 0.0f);
		glEnd();

		glBegin(GL_LINE_LOOP);
		glColor3f(0.0f, 0.0f, 0.0f);
		glVertex3f(0.25f + gfVertex_X_Increment_Value, 0.25f, 0.0f);
		glVertex3f(-0.25f + gfVertex_X_Increment_Value, 0.25f, 0.0f);
		glVertex3f(-0.25f + gfVertex_X_Increment_Value, -0.25f, 0.0f);
		glVertex3f(0.25f + gfVertex_X_Increment_Value, -0.25f, 0.0f);
		glEnd();
	}

}

void DrawBricks(void)
{
	
}

void DrawGiftBox(void)
{
	GLfloat iRadius = 0.1f;
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(0.5f, 1.0f, -6.0f);
	glBegin(GL_QUADS);
	glColor3f(1.0f, 0.55f, 0.0f);
	glVertex3f(0.25f, 0.25f, 0.0f);
	glVertex3f(-0.25f, 0.25f, 0.0f);
	glVertex3f(-0.25f, -0.25f, 0.0f);
	glVertex3f(0.25f, -0.25f, 0.0f);
	glEnd();

	glLoadIdentity();

	glLineWidth(2.0f);
	glTranslatef(0.5f, 1.0f, -6.0f);
	glBegin(GL_LINE_LOOP);
	glColor3f(0.0f, 0.0f, 0.0f);
	glVertex3f(0.25f, 0.25f, 0.0f);
	glVertex3f(-0.25f, 0.25f, 0.0f);
	glVertex3f(-0.25f, -0.25f, 0.0f);
	glVertex3f(0.25f, -0.25f, 0.0f);
	glEnd();

	glPointSize(5.0f);
	glColor3f(0.6f, 0.3f, 0.0f);
	glBegin(GL_POINTS);
	glVertex3f(0.2f, 0.2f, 0.0f);
	glVertex3f(-0.2f, 0.2f, 0.0f);
	glVertex3f(-0.2f, -0.2f, 0.0f);
	glVertex3f(0.2f, -0.2f, 0.0f);
	glVertex3f(0.0f, -0.16f, 0.0f);
	glEnd();

	glBegin(GL_POINTS);
	glColor3f(0.6f, 0.3f, 0.0f);
	for (GLfloat angle = 0.0f; angle < 1.0f*PI; angle = angle + 0.0001f)
	{
		glVertex3f(iRadius*cos(angle), iRadius*sin(angle) + 0.05f, 0.0f);
	}
	glEnd();

	glLineWidth(6.0f);
	glBegin(GL_LINES);
	glColor3f(0.6f, 0.3f, 0.0f);
	glVertex3f(0.1f, 0.05f, 0.0f);
	glVertex3f(0.0f, -0.05f, 0.0f);
	glVertex3f(0.0f, -0.04f, 0.0f);
	glVertex3f(0.0f, -0.12f, 0.0f);
	glEnd();
}

void DrawPipe(void)
{
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(gfTranslate_X_Tile , -1.0f, -6.0f);
	gfVertex_X_Pipe_Increment_Value = 15.0f;
		glBegin(GL_QUADS);
		glColor3f(0.0f, 1.0f, 0.0f);
		glVertex3f(0.4f + gfVertex_X_Pipe_Increment_Value, 0.5f, 0.0f);
		glVertex3f(-0.4f + gfVertex_X_Pipe_Increment_Value, 0.5f, 0.0f);
		glVertex3f(-0.4f + gfVertex_X_Pipe_Increment_Value, -0.48f, 0.0f);
		glVertex3f(0.4f + gfVertex_X_Pipe_Increment_Value, -0.48f, 0.0f);
		glEnd();

		glBegin(GL_QUADS);
		glColor3f(0.0f, 1.0f, 0.0f);
		glVertex3f(0.5f + gfVertex_X_Pipe_Increment_Value, 0.8f, 0.0f);
		glVertex3f(-0.5f + gfVertex_X_Pipe_Increment_Value, 0.8f, 0.0f);
		glVertex3f(-0.5f + gfVertex_X_Pipe_Increment_Value, 0.5f, 0.0f);
		glVertex3f(0.5f + gfVertex_X_Pipe_Increment_Value, 0.5f, 0.0f);
		glEnd();

		glLineWidth(2.0f);
		glBegin(GL_LINE_LOOP);
		glColor3f(0.0f, 0.0f, 0.0f);
		glVertex3f(0.4f + gfVertex_X_Pipe_Increment_Value, 0.5f, 0.0f);
		glVertex3f(-0.4f + gfVertex_X_Pipe_Increment_Value, 0.5f, 0.0f);
		glVertex3f(-0.4f + gfVertex_X_Pipe_Increment_Value, -0.48f, 0.0f);
		glVertex3f(0.4f + gfVertex_X_Pipe_Increment_Value, -0.48f, 0.0f);
		glEnd();

		glBegin(GL_LINE_LOOP);
		glColor3f(0.0f, 0.0f, 0.0f);
		glVertex3f(0.5f + gfVertex_X_Pipe_Increment_Value, 0.8f, 0.0f);
		glVertex3f(-0.5f + gfVertex_X_Pipe_Increment_Value, 0.8f, 0.0f);
		glVertex3f(-0.5f + gfVertex_X_Pipe_Increment_Value, 0.5f, 0.0f);
		glVertex3f(0.5f + gfVertex_X_Pipe_Increment_Value, 0.5f, 0.0f);
		glEnd();
}

void Draw_Hurdles(void)
{
	void Draw_Stack_Overflow(void);
	void Draw_ANSI(void);
	void DrawDanglingPointer(void);
	void DrawCompilerTable(void);
	void DrawHeterogeneousDatatypes(void);

	glColor3f(0.0f, 0.0f, 0.0f);

	if (giHurdles == 1)
		DrawAddressingModes();
	else if (giHurdles == 2)
		DrawStackManipulation();
	else if (giHurdles == 3)
		DrawSegmentationFault();
	else if (giHurdles == 4)
		DrawDanglingPointer();
	else if (giHurdles == 5)
		DrawCompilerTable();
	else if (giHurdles == 6)
		DrawHeterogeneousDatatypes();
	else if (giHurdles == 7)
		drawTextAccessPrivateMemebr();
	else if (giHurdles == 8)
		drawTextDiamondProblem();
	else if (giHurdles == 9)
		drawTextFixedValue();
	else if (giHurdles == 10)
		Draw_ANSI();
	else if (giHurdles == 11)
		Draw_Stack_Overflow();
	else if (giHurdles == 12)
		Draw_No_Code_Security();
	else if (giHurdles == 13)
		Draw_Andhar();
	else if (giHurdles == 14)
		Draw_Andhar();
	else if (giHurdles == 15)
		Draw_Andhar();	
}

void Draw_Powers(void)
{
	glLineWidth(3.0f);
	glPointSize(3.0f);
	void Draw_MSDN(void);
	void Draw_WCHAR(void);
	void Draw_DLL(void);
	void DrawInitializePointer(void);
	void DrawGlobalVariables(void);
	void DrawUseStructure(void);
	void Draw_Transformation_Matrix(void);
	void Draw_Projection_Matrix(void);
	void Draw_Viewport(void);


	fprintf(gpFile, "%d\n", giPowers);


	glColor3f(0.0f, 0.0f, 0.0f);

	if (giPowers == 1)
		DrawIndirectIndexing();
	else if (giPowers == 2)
		DrawCallingConvention();
	else if (giPowers == 3)
		DrawMMUSegmentation();
	else if (giPowers == 4)
		DrawInitializePointer();
	else if (giPowers == 5)
		DrawGlobalVariables();
	else if (giPowers == 6)
		DrawUseStructure();
	else if (giPowers == 7)
		drawTextUsePointer();
	else if (giPowers == 8)
		drawTextVirtualFunction();
	else if (giPowers == 9)
		drawTextConst();
	else if (giPowers == 10)
		Draw_WCHAR();
	else if (giPowers == 11)
		Draw_MSDN();
	else if (giPowers == 12)
		Draw_DLL();
	else if (giPowers == 13)
		Draw_Transformation_Matrix();
	else if (giPowers == 14)
		Draw_Projection_Matrix();
	else if (giPowers == 15)
		Draw_Viewport();
}

void Draw_Viewport()
{
	glLineWidth(3.0f);
	glPointSize(3.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glColor3f(0.0f, 0.0f, 0.0f);
	glTranslatef(-3.0f, 1.0f, -10.0f);
	Draw_V();

	glLoadIdentity();

	glTranslatef(-2.0f, 1.0f, -10.0f);
	Draw_I();

	glLoadIdentity();

	glTranslatef(-1.0f, 1.0f, -10.0f);
	Draw_E();

	glLoadIdentity();

	glTranslatef(-0.2f, 1.0f, -10.0f);
	Draw_W();

	glLoadIdentity();

	glTranslatef(1.2f, 1.0f, -10.0f);
	Draw_P();

	glLoadIdentity();

	glTranslatef(2.2f, 1.0f, -10.0f);
	Draw_O();

	glLoadIdentity();

	glTranslatef(3.0f, 1.0f, -10.0f);
	Draw_R();

	glLoadIdentity();

	glTranslatef(4.0f, 1.0f, -10.0f);
	Draw_T();
}

void Draw_DLL(void)
{
	glLineWidth(3.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(-1.0f, 1.0f, -10.0f);
	Draw_D();

	glLoadIdentity();

	glTranslatef(0.0f, 1.0f, -10.0f);
	Draw_L();

	glLoadIdentity();

	glTranslatef(1.0f, 1.0f, -10.0f);
	Draw_L();
}

void Draw_WCHAR(void)
{
	glLineWidth(3.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(-2.0f, 1.0f, -10.0f);
	Draw_W();

	glLoadIdentity();

	glTranslatef(-0.4f, 1.0f, -10.0f);
	Draw_C();

	glLoadIdentity();

	glTranslatef(-0.1f, 1.0f, -10.0f);
	Draw_H();

	glLoadIdentity();

	glTranslatef(1.0f, 1.0f, -10.0f);
	Draw_A();

	glLoadIdentity();

	glTranslatef(1.6f, 1.0f, -10.0f);
	Draw_R();
}

void resize(int width, int height)
{
	if (height == 0)
		height = 1;
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(45.0f,(GLfloat)width/(GLfloat)height,0.1f,100.0f);
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
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
	}
}

void uninitialize(int i_Exit_Flag)
{
	if (gbFullscreen == false)
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
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

	if (i_Exit_Flag == 0)
		fprintf(gpFile, "Log File Closed Successfully...\n");

	else
		fprintf(gpFile, "Log File Closed Erroneously!!!\n");

	DestroyWindow(ghwnd);
}

void Draw_Clouds(void)
{
	void DrawCloud_1(void);
	void DrawCloud_2(void);
	void DrawCloud_3(void);
	void DrawCloud_4(void);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(gfTranslate_X_Tile, 2.3f, -8.0f);
	DrawCloud_1();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(gfTranslate_X_Tile, 1.3f, -8.0f);
	DrawCloud_2();
	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(gfTranslate_X_Tile, 2.2f, -8.0f);
	DrawCloud_3();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	
	glTranslatef(gfTranslate_X_Tile, 3.8f, -8.0f);
	DrawCloud_4();
}

void Draw_Stack_Overflow(void)
{
	glLineWidth(3.0f);
	/***********************S*********************/
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(-3.0f, 1.0f, -10.0f);
	glRotatef(270.0f, 0.0f, 0.0f, 1.0f);

	glPointSize(3.0f);
	glBegin(GL_POINTS);
	GLfloat fRadius_For_S = 0.25f;
	for (GLfloat angle = 1.8f; angle < 2.0f*PI; angle = angle + 0.01f)
	{
		glVertex3f(fRadius_For_S*cos(angle) - 0.24f, fRadius_For_S*sin(angle) + 0.2f, 0.0f);
	}
	glEnd();

	glBegin(GL_POINTS);
	for (GLfloat angle = -0.9f; angle < 1.0f*PI; angle = angle + 0.01f)
	{
		glVertex3f(fRadius_For_S*cos(angle) + 0.25f, fRadius_For_S*sin(angle) + 0.2f, 0.0f);
	}
	glEnd();

	/***********************T*********************/
	glRotatef(90.0f, 0.0f, 0.0f, 1.0f);

	glBegin(GL_LINES);
	glVertex3f(0.9f, 0.5f, 0.0f);
	glVertex3f(0.9f, -0.5f, 0.0f);
	glVertex3f(0.65f, 0.5f, 0.0f);
	glVertex3f(1.15f, 0.5f, 0.0f);
	glEnd();

	/***********************A*********************/
	glLineWidth(3.0f);
	glBegin(GL_LINES);
	glVertex3f(1.2f, -0.5f, 0.0f);
	glVertex3f(1.5f, 0.5f, 0.0f);
	glVertex3f(1.5f, 0.5f, 0.0f);
	glVertex3f(1.8f, -0.5f, 0.0f);
	glVertex3f(1.33f, -0.1f, 0.0f);
	glVertex3f(1.70f, -0.1f, 0.0f);
	glEnd();

	/***********************C*********************/
	glRotatef(90.0f, 0.0f, 0.0f, 1.0f);

	glPointSize(3.0f);
	glBegin(GL_POINTS);
	GLfloat fRadius_For_C = 0.45f;
	for (GLfloat angle = 0.0f; angle < 1.0f*PI; angle = angle + 0.01f)
	{
		glVertex3f(fRadius_For_C*cos(angle), fRadius_For_C*sin(angle) - 2.4f, 0.0f);
	}
	glEnd();

	/***********************K*********************/
	glRotatef(270.0f, 0.0f, 0.0f, 1.0f);

	glBegin(GL_LINES);
	glVertex3f(2.6f, 0.5f, 0.0f);
	glVertex3f(2.6f, -0.5f, 0.0f);
	glVertex3f(2.6f, -0.1f, 0.0f);
	glVertex3f(3.0f, 0.5f, 0.0f);
	glVertex3f(2.7f, 0.03f, 0.0f);
	glVertex3f(3.0f, -0.5f, 0.0f);
	glEnd();

	/***********************O*********************/
	glScalef(0.6f, 1.0f, 1.0f);

	glPointSize(3.0f);
	glBegin(GL_POINTS);
	GLfloat fRadius_For_O = 0.45f;
	for (GLfloat angle = 0.0f; angle < 2.0f*PI; angle = angle + 0.01f)
	{
		glVertex3f(fRadius_For_O*cos(angle) + 6.4f, fRadius_For_O*sin(angle), 0.0f);
	}
	glEnd();

	/***********************V*********************/
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(1.2f, 1.0f, -10.0f);

	glBegin(GL_LINES);
	glVertex3f(0.0f, 0.5f, 0.0f);
	glVertex3f(0.25f, -0.5f, 0.0f);
	glVertex3f(0.5f, 0.5f, 0.0f);
	glVertex3f(0.25f, -0.5f, 0.0f);
	glEnd();

	/***********************E*********************/
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(1.8f, 1.0f, -10.0f);

	glBegin(GL_LINES);
	glVertex3f(0.0f, 0.5f, 0.0f);
	glVertex3f(0.0f, -0.5f, 0.0f);
	glVertex3f(0.0f, 0.5f, 0.0f);
	glVertex3f(0.5f, 0.5f, 0.0f);
	glVertex3f(0.0f, -0.5f, 0.0f);
	glVertex3f(0.5f, -0.5f, 0.0f);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(0.35f, 0.0f, 0.0f);
	glEnd();

	/***********************R*********************/
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(2.4f, 1.0f, -10.0f);

	glBegin(GL_LINES);
	glVertex3f(0.0f, 0.5f, 0.0f);
	glVertex3f(0.0f, -0.5f, 0.0f);
	glVertex3f(0.0f, 0.5f, 0.0f);
	glVertex3f(0.2f, 0.5f, 0.0f);
	glVertex3f(0.0f, -0.01f, 0.0f);
	glVertex3f(0.2f, -0.01f, 0.0f);
	glVertex3f(0.0f, -0.01f, 0.0f);
	glVertex3f(0.4f, -0.5f, 0.0f);
	glEnd();

	glRotatef(270.0f, 0.0f, 0.0f, 1.0f);

	glPointSize(3.0f);
	glBegin(GL_POINTS);
	GLfloat fRadius_For_R = 0.25f;
	for (GLfloat angle = 0.0f; angle < 1.0f*PI; angle = angle + 0.01f)
	{
		glVertex3f(fRadius_For_R*cos(angle) - 0.24f, fRadius_For_R*sin(angle) + 0.2f, 0.0f);
	}
	glEnd();

	/***********************F*********************/
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(3.0f, 1.0f, -10.0f);

	glBegin(GL_LINES);
	glVertex3f(0.0f, 0.5f, 0.0f);
	glVertex3f(0.0f, -0.5f, 0.0f);
	glVertex3f(0.0f, 0.5f, 0.0f);
	glVertex3f(0.5f, 0.5f, 0.0f);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(0.35f, 0.0f, 0.0f);
	glEnd();

	/***********************L*********************/
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(3.6f, 1.0f, -10.0f);

	glBegin(GL_LINES);
	glVertex3f(0.0f, 0.5f, 0.0f);
	glVertex3f(0.0f, -0.5f, 0.0f);
	glVertex3f(0.0f, -0.5f, 0.0f);
	glVertex3f(0.5f, -0.5f, 0.0f);
	glEnd();

	/***********************O*********************/
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(4.4f, 1.0f, -10.0f);
	glScalef(0.6f, 1.0f, 1.0f);

	glPointSize(3.0f);
	glBegin(GL_POINTS);
	for (GLfloat angle = 0.0f; angle < 2.0f*PI; angle = angle + 0.01f)
	{
		glVertex3f(fRadius_For_O*cos(angle), fRadius_For_O*sin(angle), 0.0f);
	}
	glEnd();

	/***********************W*********************/
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(4.8f, 1.0f, -10.0f);

	glPointSize(3.0f);
	glBegin(GL_LINES);
	glVertex3f(0.0f, 0.5f, 0.0f);
	glVertex3f(0.25f, -0.5f, 0.0f);
	glVertex3f(0.5f, 0.5f, 0.0f);
	glVertex3f(0.25f, -0.5f, 0.0f);
	glVertex3f(0.5f, 0.5f, 0.0f);
	glVertex3f(0.75f, -0.5f, 0.0f);
	glVertex3f(0.75f, -0.5f, 0.0f);
	glVertex3f(1.0f, 0.5f, 0.0f);
	glEnd();
}

void Draw_MSDN(void)
{
	void Draw_M(void);
	void Draw_S(void);
	void Draw_D(void);
	void Draw_N(void);

	glLineWidth(3.0f);
	/***********************M*********************/
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(-1.6f, 1.0f, -10.0f);

	Draw_M();

	/***********************S*********************/
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(-0.8f, 1.0f, -10.0f);

	Draw_S();

	/***********************D*********************/
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(0.0f, 1.0f, -10.0f);

	Draw_D();

	/***********************N*********************/
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(0.8f, 1.0f, -10.0f);

	Draw_N();
}

void Draw_ANSI(void)
{
	void Draw_A(void);
	void Draw_N(void);
	void Draw_S(void);
	void Draw_I(void);

	/***********************A*********************/
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(-1.2f, 1.0f, -10.0f);

	Draw_A();

	/***********************N*********************/
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(-0.6f, 1.0f, -10.0f);

	Draw_N();

	/***********************S*********************/
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(0.3f, 1.0f, -10.0f);

	Draw_S();

	/***********************I*********************/
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(1.0f, 1.0f, -10.0f);

	Draw_I();
}

void Draw_Star(void)
{
	void drawStar(void);
	drawStar();
	gfTranslate_X_Star = gfTranslate_X_Star - 0.03f;
}

void Draw_Cactus(void)
{
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	// Base and Leaves...!!
	glTranslatef(gfTranslate_X_Tile, -3.3f, -15.0f);
	glColor3d(0.64, 0.57, 0.25);
	glBegin(GL_POLYGON);
	glVertex3d(-1.0 + gfVertex_X_Increment_Value_For_Cactus, -0.5, 0);
	glVertex3d(-1.0 + gfVertex_X_Increment_Value_For_Cactus, 0.5, 0);
	glVertex3d(1.0 + gfVertex_X_Increment_Value_For_Cactus, -0.5, 0);
	glVertex3d(1.0 + gfVertex_X_Increment_Value_For_Cactus, 0.5, 0);
	glEnd();
	// Stem..!!
	glBegin(GL_QUADS);
	glVertex3d(-0.2 + gfVertex_X_Increment_Value_For_Cactus, -0.5, 0);
	glVertex3d(0.2 + gfVertex_X_Increment_Value_For_Cactus, -0.5, 0);
	glVertex3d(0.2 + gfVertex_X_Increment_Value_For_Cactus, 1.0, 0);
	glVertex3d(-0.2 + gfVertex_X_Increment_Value_For_Cactus, 1.0, 0);
	glEnd();
	//jaw1
	//glRotated( gAngleForJaw1, 0.0, 0.0, 1.0 );
	//glRotated( 10.0, 0.0, 0.0, 1.0 );
	//Tooth
	glColor3f(1.0, 1.0, 1.0);
	glBegin(GL_TRIANGLES);
	glVertex3d(0.0 + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(0.0) + 1.5, 0.0);
	glVertex3d(sin(PI - 0.08) + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(0.5) + 1.5, 0.0);
	glVertex3d(0.0 + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(0.8) + 1.5, 0.0);
	glVertex3d(0.0 + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(0.8) + 1.5, 0.0);
	glVertex3d(sin(PI - 0.07) + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(0.95) + 1.5, 0.0);
	glVertex3d(0.0 + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(1.1) + 1.5, 0.0);
	glVertex3d(0.0 + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(1.1) + 1.5, 0.0);
	glVertex3d(sin(PI - 0.055) + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(1.25) + 1.5, 0.0);
	glVertex3d(0.0 + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(1.4) + 1.5, 0.0);
	glVertex3d(0.0 + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(1.4) + 1.5, 0.0);
	glVertex3d(sin(PI - 0.053) + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(1.55) + 1.5, 0.0);
	glVertex3d(0.0 + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(1.7) + 1.5, 0.0);
	glVertex3d(0.0 + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(1.7) + 1.5, 0.0);
	glVertex3d(sin(PI - 0.05) + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(1.85) + 1.5, 0.0);
	glVertex3d(0.0 + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(2.0) + 1.5, 0.0);
	glVertex3d(0.0 + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(2.0) + 1.5, 0.0);
	glVertex3d(sin(PI - 0.04) + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(2.15) + 1.5, 0.0);
	glVertex3d(0.0 + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(2.30) + 1.5, 0.0);
	glEnd();
	glColor3d(0.00, 0.50, 0.00);
	glBegin(GL_POLYGON);
	for (float angle = 0.0f; (angle < PI); angle = angle + 0.1f)
	{
		glVertex3d(-0.5f*sin(PI/4 + angle) - 0.095 + gfVertex_X_Increment_Value_For_Cactus, (0.7f*cos(PI/4 + angle)) + 1.5, 0.0f);
	}
	glEnd();

	//glLoadIdentity();
	//jaw2
	//glRotated( gAngleForJaw2, 0.0, 0.0, 1.0 );

	glColor3f(1.0, 1.0, 1.0);
	glBegin(GL_TRIANGLES);
	glVertex3d(0.0 + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(0.0) + 1.5, 0.0);
	glVertex3d(-sin(PI - 0.08) + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(0.5) + 1.5, 0.0);
	glVertex3d(0.0 + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(0.8) + 1.5, 0.0);
	glVertex3d(0.0 + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(0.8) + 1.5, 0.0);
	glVertex3d(-sin(PI - 0.07) + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(0.95) + 1.5, 0.0);
	glVertex3d(0.0 + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(1.1) + 1.5, 0.0);
	glVertex3d(0.0 + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(1.1) + 1.5, 0.0);
	glVertex3d(-sin(PI - 0.055) + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(1.25) + 1.5, 0.0);
	glVertex3d(0.0 + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(1.4) + 1.5, 0.0);
	glVertex3d(0.0 + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(1.4) + 1.5, 0.0);
	glVertex3d(-sin(PI - 0.053) + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(1.55) + 1.5, 0.0);
	glVertex3d(0.0 + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(1.7) + 1.5, 0.0);
	glVertex3d(0.0 + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(1.7) + 1.5, 0.0);
	glVertex3d(-sin(PI - 0.05) + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(1.85) + 1.5, 0.0);
	glVertex3d(0.0 + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(2.0) + 1.5, 0.0);
	glVertex3d(0.0 + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(2.0) + 1.5, 0.0);
	glVertex3d(-sin(PI - 0.04) + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(2.15) + 1.5, 0.0);
	glVertex3d(0.0 + gfVertex_X_Increment_Value_For_Cactus, 0.7f*cos(2.30) + 1.5, 0.0);
	glEnd();

	glColor3d(0.00, 0.50, 0.00);
	glBegin(GL_POLYGON);
	for (float angle = 0.0f; (angle < PI); angle = angle + 0.1f)
	{
		glVertex3d(0.5f*sin(PI/4 + angle) + 0.095 + gfVertex_X_Increment_Value_For_Cactus, (0.7f*cos(PI/4 + angle)) + 1.5, 0.0f);
	}
	glEnd();
}

void DrawCloud_1()
{
	GLfloat X = 1.8f;
	GLfloat Y = 0.4f;
	int i;
	int triangleAmount = 1000;
	GLfloat radius = 0.5f;
	GLfloat twicePi = 2.0f * 3.14;
	glEnable(GL_LINE_SMOOTH);
	glLineWidth(5.0f);
	for (GLfloat gfCloud_1_Increment_Value = 0.0f; gfCloud_1_Increment_Value < 250.0f; gfCloud_1_Increment_Value = gfCloud_1_Increment_Value + 18.0f)
	{
		X = X + gfCloud_1_Increment_Value;
		glBegin(GL_POLYGON);
		glColor3f(1.0f, 1.0f, 1.0f);
		for (i = 0; i <= triangleAmount; i++)
		{
			glVertex3f(X + (radius * cos(i * twicePi / triangleAmount)), Y + (radius * sin(i * twicePi / triangleAmount)), 0.0f);
		}
		X += 0.4f;
		Y += 0.1f;
		radius = 0.35f;
		for (i = 0; i <= triangleAmount; i++)
		{
			glVertex3f(X + (radius * cos(i * twicePi / triangleAmount)), Y + (radius * sin(i * twicePi / triangleAmount)), 0.0f);
		}
		X -= 0.9f;
		Y -= 0.3f;
		radius = 0.3f;
		for (i = 0; i <= triangleAmount; i++)
		{
			glVertex3f(X + (radius * cos(i * twicePi / triangleAmount)), Y + (radius * sin(i * twicePi / triangleAmount)), 0.0f);
		}
		X += 0.35f;
		Y -= 0.2f;
		radius = 0.35f;
		for (i = 0; i <= triangleAmount; i++)
		{
			glVertex3f(X + (radius * cos(i * twicePi / triangleAmount)), Y + (radius * sin(i * twicePi / triangleAmount)), 0.0f);
		}
		glEnd();
	}
}

void DrawCloud_2()
{
	GLfloat X = -2.4f;
	GLfloat Y = 1.4f;
	int i;
	int triangleAmount = 1000;
	GLfloat radius = 0.5f;
	GLfloat twicePi = 2.0f * 3.14;
	glEnable(GL_LINE_SMOOTH);

	glLineWidth(5.0f);
	for (GLfloat gfCloud_1_Increment_Value = 6.0f; gfCloud_1_Increment_Value < 250.0f; gfCloud_1_Increment_Value = gfCloud_1_Increment_Value + 18.0f)
	{
		X = X + gfCloud_1_Increment_Value;

		glBegin(GL_POLYGON);
		glColor3f(1.0f, 1.0f, 1.0f);
		for (i = 0; i <= triangleAmount; i++)
		{
			glVertex3f(X + (radius * cos(i * twicePi / triangleAmount)), Y + (radius * sin(i * twicePi / triangleAmount)), 0.0f);
		}
		X += 0.4f;
		Y += 0.1f;
		radius = 0.35f;
		for (i = 0; i <= triangleAmount; i++)
		{
			glVertex3f(X + (radius * cos(i * twicePi / triangleAmount)), Y + (radius * sin(i * twicePi / triangleAmount)), 0.0f);
		}
		X -= 0.9f;
		Y -= 0.3f;
		radius = 0.3f;
		for (i = 0; i <= triangleAmount; i++)
		{
			glVertex3f(X + (radius * cos(i * twicePi / triangleAmount)), Y + (radius * sin(i * twicePi / triangleAmount)), 0.0f);
		}

		glEnd();
	}
}

void DrawCloud_3()
{
	GLfloat X = -1.6f;
	GLfloat Y = 0.35f;
	int i;
	int triangleAmount = 1000;
	GLfloat radius = 0.5f;
	GLfloat twicePi = 2.0f * 3.14;
	glEnable(GL_LINE_SMOOTH);
	glLineWidth(5.0f);
	for (GLfloat gfCloud_1_Increment_Value = 12.0f; gfCloud_1_Increment_Value < 250.0f; gfCloud_1_Increment_Value = gfCloud_1_Increment_Value + 18.0f)
	{
		X = X + gfCloud_1_Increment_Value;

		glBegin(GL_POLYGON);
		glColor3f(1.0f, 1.0f, 1.0f);

		for (i = 0; i <= triangleAmount; i++)
		{
			glVertex3f(X + (radius * cos(i * twicePi / triangleAmount)), Y + (radius * sin(i * twicePi / triangleAmount)), 0.0f);
		}

		X += 0.4f;
		Y += 0.2f;
		radius = 0.4f;
		for (i = 0; i <= triangleAmount; i++)
		{
			glVertex3f(X + (radius * cos(i * twicePi / triangleAmount)), Y + (radius * sin(i * twicePi / triangleAmount)), 0.0f);
		}

		X += 0.4f;
		Y += 0.1f;
		radius = 0.4f;
		for (i = 0; i <= triangleAmount; i++)
		{
			glVertex3f(X + (radius * cos(i * twicePi / triangleAmount)), Y + (radius * sin(i * twicePi / triangleAmount)), 0.0f);
		}
		X -= 1.0f;
		Y -= 0.3f;
		radius = 0.3f;
		for (i = 0; i <= triangleAmount; i++)
		{
			glVertex3f(X + (radius * cos(i * twicePi / triangleAmount)), Y + (radius * sin(i * twicePi / triangleAmount)), 0.0f);
		}
		X += 0.6f;
		Y -= 0.25f;
		radius = 0.35f;
		for (i = 0; i <= triangleAmount; i++)
		{
			glVertex3f(X + (radius * cos(i * twicePi / triangleAmount)), Y + (radius * sin(i * twicePi / triangleAmount)), 0.0f);
		}
		glEnd();
	}
}

void DrawCloud_4()
{
	GLfloat X = -2.6f;
	GLfloat Y = -1.3f;
	int i;
	int triangleAmount = 1000;
	GLfloat radius = 0.5f;
	GLfloat twicePi = 2.0f * 3.14;
	glEnable(GL_LINE_SMOOTH);
	glLineWidth(5.0f);
	for (GLfloat gfCloud_1_Increment_Value = 18.0f; gfCloud_1_Increment_Value < 250.0f; gfCloud_1_Increment_Value = gfCloud_1_Increment_Value + 18.0f)
	{
		X = X + gfCloud_1_Increment_Value;

		glBegin(GL_POLYGON);
		glColor3f(1.0f, 1.0f, 1.0f);


		X += 0.4f;
		Y += 0.2f;
		radius = 0.4f;
		for (i = 0; i <= triangleAmount; i++)
		{
			glVertex3f(X + (radius * cos(i * twicePi / triangleAmount)), Y + (radius * sin(i * twicePi / triangleAmount)), 0.0f);
		}

		X += 0.4f;
		Y += 0.1f;
		radius = 0.4f;
		for (i = 0; i <= triangleAmount; i++)
		{
			glVertex3f(X + (radius * cos(i * twicePi / triangleAmount)), Y + (radius * sin(i * twicePi / triangleAmount)), 0.0f);
		}
		X -= 1.0f;
		Y -= 0.3f;
		radius = 0.3f;
		for (i = 0; i <= triangleAmount; i++)
		{
			glVertex3f(X + (radius * cos(i * twicePi / triangleAmount)), Y + (radius * sin(i * twicePi / triangleAmount)), 0.0f);
		}
		X += 0.6f;
		Y -= 0.25f;
		radius = 0.35f;
		for (i = 0; i <= triangleAmount; i++)
		{
			glVertex3f(X + (radius * cos(i * twicePi / triangleAmount)), Y + (radius * sin(i * twicePi / triangleAmount)), 0.0f);
		}
		glEnd();
	}
}

void ShowMushroom()
{
	float xx, yy, r = 0.2f;
	int n = 40, i;

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(gfTranslate_X_Mushroom, -3.7f, -15.0f);
	glColor3f(0.94, 0.63, 0.25);
	glBegin(GL_POLYGON);
	{
		for (i = 3.0f*n / 4; i > (1 * n / 2 - n / 8); i--)
		{
			xx = r*cos((float)i * 2 * PI / n) + 0.2;
			yy = r*sin((float)i * 2 * PI / n) + 0.5;
			glVertex3f(xx, yy, 0.0f);
		}
		glVertex3f(0.4, 1.0, 0.0f);
		glVertex3f(0.6, 1.0, 0.0f);
		for (i = 5; i > -n / 4; i--)
		{
			xx = r*cos((float)i * 2 * PI / n) + 0.8;
			yy = r*sin((float)i * 2 * PI / n) + 0.5;
			glVertex3f(xx, yy, 0.0f);
		}

	}
	glEnd();



	r = 0.16f;
	glColor3f(1.0, 0.0, 0.0);
	glBegin(GL_POLYGON);
	{
		for (i = 0; i < n; i++)
		{
			xx = r*cos((float)i * 2 * PI / n) + 0.3;
			yy = r*sin((float)i * 2 * PI / n) + 0.56;
			glVertex3f(xx, yy, 0.0f);
		}
	}
	glEnd();

	r = 0.12f;
	glColor3f(1.0, 0.0, 0.0);
	glBegin(GL_POLYGON);
	{
		for (i = 0; i < n; i++)
		{
			xx = r*cos((float)i * 2 * PI / n) + 0.7;
			yy = r*sin((float)i * 2 * PI / n) + 0.75;
			glVertex3f(xx, yy, 0.0f);
		}
	}
	glEnd();


	r = 0.2f;
	glColor3f(1.0, 1.0, 1.0);
	glBegin(GL_POLYGON);
	{
		for (i = 0; i < n; i++)
		{
			if ((i >= 0 && i < n / 8) || (i >= 3 * n / 8 && i < 5 * n / 8) || (i >= 7 * n / 8 && i < n))
			{
				xx = 1.4*r*cos((float)i * 2 * PI / n) + 0.5;
				yy = r*sin((float)i * 2 * PI / n) + 0.2;
			}
			else
			{
				xx = r*cos((float)i * 2 * PI / n) + 0.5;
				yy = r*sin((float)i * 2 * PI / n) + 0.2;
			}
			glVertex3f(xx, yy, 0.0f);
		}
	}
	glEnd();
}

void Draw_Mario(void)
{
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(gfMario_X_Translate, gfMario_Y_Translate, -8.0f);
	glScalef(0.4f, 0.4f, 0.4f);
	//Red Color
	//1
	glBegin(GL_QUADS);
	glColor3f(0.698f, 0.2f, 0.1411f);
	glVertex3f(1.1f, 1.0f, 0.0f);
	glVertex3f(1.1f, 1.2f, 0.0f);
	glVertex3f(-0.2f, 1.2f, 0.0f);
	glVertex3f(-0.2f, 1.0f, 0.0f);
	glEnd();

	//2
	glBegin(GL_QUADS);
	glVertex3f(0.7f, 1.2f, 0.0f);
	glVertex3f(0.7f, 1.4f, 0.0f);
	glVertex3f(0.0f, 1.4f, 0.0f);
	glVertex3f(0.0f, 1.2f, 0.0f);
	glEnd();

	//Green Color
	//3
	glBegin(GL_QUADS);
	glColor3f(0.4156f, 0.4352f, 0.0f);
	glVertex3f(-0.2f, 1.0f, 0.0f);
	glVertex3f(0.2f, 1.0f, 0.0f);
	glVertex3f(0.2f, 0.8f, 0.0f);
	glVertex3f(-0.2f, 0.8f, 0.0f);
	glEnd();

	//4
	glBegin(GL_QUADS);
	glVertex3f(0.1f, 0.8f, 0.0f);
	glVertex3f(-0.05f, 0.8f, 0.0f);
	glVertex3f(-0.05f, 0.4f, 0.0f);
	glVertex3f(0.1f, 0.4f, 0.0f);
	glEnd();

	//5
	glBegin(GL_QUADS);
	glVertex3f(0.1f, 0.4f, 0.0f);
	glVertex3f(0.2f, 0.4f, 0.0f);
	glVertex3f(0.2f, 0.6f, 0.0f);
	glVertex3f(0.1f, 0.6f, 0.0f);
	glEnd();

	//6
	glBegin(GL_QUADS);
	glVertex3f(-0.2f, 0.8f, 0.0f);
	glVertex3f(-0.4f, 0.8f, 0.0f);
	glVertex3f(-0.4f, 0.4f, 0.0f);
	glVertex3f(-0.2f, 0.4f, 0.0f);
	glEnd();

	//7
	glBegin(GL_QUADS);
	glVertex3f(-0.4f, 0.4f, 0.0f);
	glVertex3f(-0.05f, 0.4f, 0.0f);
	glVertex3f(-0.05f, 0.2f, 0.0f);
	glVertex3f(-0.4f, 0.2f, 0.0f);
	glEnd();

	//8
	glBegin(GL_QUADS);
	glVertex3f(0.7f, 1.0f, 0.0f);
	glVertex3f(0.5f, 1.0f, 0.0f);
	glVertex3f(0.5f, 0.6f, 0.0f);
	glVertex3f(0.7f, 0.6f, 0.0f);
	glEnd();

	//9
	glBegin(GL_QUADS);
	glVertex3f(0.7f, 0.6f, 0.0f);
	glVertex3f(0.9f, 0.6f, 0.0f);
	glVertex3f(0.9f, 0.4f, 0.0f);
	glVertex3f(0.7f, 0.4f, 0.0f);
	glEnd();

	//10
	glBegin(GL_QUADS);
	glVertex3f(0.5f, 0.4f, 0.0f);
	glVertex3f(0.5f, 0.2f, 0.0f);
	glVertex3f(1.1f, 0.2f, 0.0f);
	glVertex3f(1.1f, 0.4f, 0.0f);
	glEnd();

	//White Color
	//11
	glBegin(GL_QUADS);
	glColor3f(0.898f, 0.6431f, 0.1882f);
	glVertex3f(-0.05f, 0.8f, 0.0f);
	glVertex3f(-0.05f, 0.4f, 0.0f);
	glVertex3f(-0.2f, 0.4f, 0.0f);
	glVertex3f(-0.2f, 0.8f, 0.0f);
	glEnd();

	//12
	glBegin(GL_QUADS);
	glVertex3f(-0.05f, 0.2f, 0.0f);
	glVertex3f(-0.05f, 0.0f, 0.0f);
	glVertex3f(1.0f, 0.0f, 0.0f);
	glVertex3f(1.0f, 0.2f, 0.0f);
	glEnd();

	//13
	glBegin(GL_QUADS);
	glVertex3f(-0.05f, 0.4f, 0.0f);
	glVertex3f(-0.05f, 0.2f, 0.0f);
	glVertex3f(0.5f, 0.2f, 0.0f);
	glVertex3f(0.5f, 0.4f, 0.0f);
	glEnd();

	//14
	glBegin(GL_QUADS);
	glVertex3f(0.7f, 0.6f, 0.0f);
	glVertex3f(0.2f, 0.6f, 0.0f);
	glVertex3f(0.2f, 0.4f, 0.0f);
	glVertex3f(0.7f, 0.4f, 0.0f);
	glEnd();

	//15
	glBegin(GL_QUADS);
	glVertex3f(0.1f, 0.6f, 0.0f);
	glVertex3f(0.1f, 0.8f, 0.0f);
	glVertex3f(0.5f, 0.8f, 0.0f);
	glVertex3f(0.5f, 0.6f, 0.0f);
	glEnd();

	//16
	glBegin(GL_QUADS);
	glVertex3f(0.2f, 1.0f, 0.0f);
	glVertex3f(0.2f, 0.8f, 0.0f);
	glVertex3f(0.5f, 0.8f, 0.0f);
	glVertex3f(0.5f, 1.0f, 0.0f);
	glEnd();

	//17
	glBegin(GL_QUADS);
	glVertex3f(0.7f, 1.0f, 0.0f);
	glVertex3f(0.7f, 0.6f, 0.0f);
	glVertex3f(0.9f, 0.6f, 0.0f);
	glVertex3f(0.9f, 1.0f, 0.0f);
	glEnd();

	//18
	glBegin(GL_QUADS);
	glVertex3f(0.9f, 0.4f, 0.0f);
	glVertex3f(1.1f, 0.4f, 0.0f);
	glVertex3f(1.1f, 0.8f, 0.0f);
	glVertex3f(0.9f, 0.8f, 0.0f);
	glEnd();

	//19
	glBegin(GL_QUADS);
	glVertex3f(1.1f, 0.4f, 0.0f);
	glVertex3f(1.3f, 0.4f, 0.0f);
	glVertex3f(1.3f, 0.6f, 0.0f);
	glVertex3f(1.1f, 0.6f, 0.0f);
	glEnd();

	//Green
	//20
	glBegin(GL_QUADS);
	glColor3f(0.4156f, 0.4352f, 0.0f);
	glVertex3f(0.1f, 0.0f, 0.0f);
	glVertex3f(-0.2f, 0.0f, 0.0f);
	glVertex3f(-0.2f, -0.6f, 0.0f);
	glVertex3f(0.1f, -0.6f, 0.0f);
	glEnd();

	//21
	glBegin(GL_QUADS);
	glVertex3f(-0.2f, -0.2f, 0.0f);
	glVertex3f(-0.4f, -0.2f, 0.0f);
	glVertex3f(-0.4f, -0.6f, 0.0f);
	glVertex3f(-0.2f, -0.6f, 0.0f);
	glEnd();

	//22
	glBegin(GL_QUADS);
	glVertex3f(-0.4f, -0.4f, 0.0f);
	glVertex3f(-0.4f, -0.6f, 0.0f);
	glVertex3f(-0.6f, -0.6f, 0.0f);
	glVertex3f(-0.6f, -0.4f, 0.0f);
	glEnd();

	//23
	glBegin(GL_QUADS);
	glVertex3f(-0.05f, -0.6f, 0.0f);
	glVertex3f(-0.05f, -0.8f, 0.0f);
	glVertex3f(-0.2f, -0.8f, 0.0f);
	glVertex3f(-0.2f, -0.6f, 0.0f);
	glEnd();

	//Red
	//24
	glBegin(GL_QUADS);
	glColor3f(0.698f, 0.2f, 0.1411f);
	glVertex3f(0.1f, 0.0f, 0.0f);
	glVertex3f(0.3f, 0.0f, 0.0f);
	glVertex3f(0.3f, -0.6f, 0.0f);
	glVertex3f(0.1f, -0.6f, 0.0f);
	glEnd();

	//25
	glBegin(GL_QUADS);
	glVertex3f(0.3f, -0.6f, 0.0f);
	glVertex3f(0.5f, -0.6f, 0.0f);
	glVertex3f(0.5f, -0.4f, 0.0f);
	glVertex3f(0.3f, -0.4f, 0.0f);
	glEnd();

	//26
	glBegin(GL_QUADS);
	glVertex3f(0.5f, -0.6f, 0.0f);
	glVertex3f(0.7f, -0.6f, 0.0f);
	glVertex3f(0.7f, -0.2f, 0.0f);
	glVertex3f(0.5f, -0.2f, 0.0f);
	glEnd();

	//27
	glBegin(GL_QUADS);
	glVertex3f(0.3f, -0.6f, 0.0f);
	glVertex3f(0.3f, -0.8f, 0.0f);
	glVertex3f(0.5f, -0.8f, 0.0f);
	glVertex3f(0.5f, -0.6f, 0.0f);
	glEnd();

	//28
	glBegin(GL_QUADS);
	glVertex3f(-0.05f, -0.6f, 0.0f);
	glVertex3f(-0.05f, -0.8f, 0.0f);
	glVertex3f(0.1f, -0.8f, 0.0f);
	glVertex3f(0.1f, -0.6f, 0.0f);
	glEnd();

	//29
	glBegin(GL_QUADS);
	glVertex3f(0.7f, -0.6f, 0.0f);
	glVertex3f(0.9f, -0.6f, 0.0f);
	glVertex3f(0.9f, -0.8f, 0.0f);
	glVertex3f(0.7f, -0.8f, 0.0f);
	glEnd();

	//30
	glBegin(GL_QUADS);
	glVertex3f(-0.05f, -0.8f, 0.0f);
	glVertex3f(-0.05f, -1.0f, 0.0f);
	glVertex3f(0.9f, -1.0f, 0.0f);
	glVertex3f(0.9f, -0.8f, 0.0f);
	glEnd();

	//31
	glBegin(GL_QUADS);
	glVertex3f(0.3f, -1.0f, 0.0f);
	glVertex3f(0.3f, -1.2f, 0.0f);
	glVertex3f(0.5f, -1.2f, 0.0f);
	glVertex3f(0.5f, -1.0f, 0.0f);
	glEnd();

	//32
	glBegin(GL_QUADS);
	glVertex3f(0.3f, -1.0f, 0.0f);
	glVertex3f(0.3f, -1.4f, 0.0f);
	glVertex3f(-0.2f, -1.4f, 0.0f);
	glVertex3f(-0.2f, -1.0f, 0.0f);
	glEnd();

	//33
	glBegin(GL_QUADS);
	glVertex3f(0.5f, -1.0f, 0.0f);
	glVertex3f(0.5f, -1.4f, 0.0f);
	glVertex3f(1.1f, -1.4f, 0.0f);
	glVertex3f(1.1f, -1.0f, 0.0f);
	glEnd();

	//Green
	//34
	glBegin(GL_QUADS);
	glColor3f(0.4156f, 0.4352f, 0.0f);
	glVertex3f(0.3f, 0.0f, 0.0f);
	glVertex3f(0.5f, 0.0f, 0.0f);
	glVertex3f(0.5f, -0.4f, 0.0f);
	glVertex3f(0.3f, -0.4f, 0.0f);
	glEnd();

	//35
	glBegin(GL_QUADS);
	glVertex3f(0.5f, 0.0f, 0.0f);
	glVertex3f(0.5f, -0.2f, 0.0f);
	glVertex3f(0.7f, -0.2f, 0.0f);
	glVertex3f(0.7f, 0.0f, 0.0f);
	glEnd();

	//36
	glBegin(GL_QUADS);
	glVertex3f(0.7f, -0.2f, 0.0f);
	glVertex3f(0.7f, -0.6f, 0.0f);
	glVertex3f(1.1f, -0.6f, 0.0f);
	glVertex3f(1.1f, -0.2f, 0.0f);
	glEnd();

	//37
	glBegin(GL_QUADS);
	glVertex3f(1.1f, -0.4f, 0.0f);
	glVertex3f(1.4f, -0.4f, 0.0f);
	glVertex3f(1.4f, -0.6f, 0.0f);
	glVertex3f(1.1f, -0.6f, 0.0f);
	glEnd();

	//38
	glBegin(GL_QUADS);
	glVertex3f(1.1f, -0.6f, 0.0f);
	glVertex3f(0.9f, -0.6f, 0.0f);
	glVertex3f(0.9f, -0.8f, 0.0f);
	glVertex3f(1.1f, -0.8f, 0.0f);
	glEnd();

	//39
	glBegin(GL_QUADS);
	glVertex3f(0.1f + fMario_Leg_Increment_Value, -1.4f, 0.0f);
	glVertex3f(-0.3f + fMario_Leg_Increment_Value, -1.4f, 0.0f);
	glVertex3f(-0.3f + fMario_Leg_Increment_Value, -1.8f, 0.0f);
	glVertex3f(0.1f + fMario_Leg_Increment_Value, -1.8f, 0.0f);
	glEnd();

	//40
	glBegin(GL_QUADS);
	glVertex3f(0.1f + fMario_Leg_Increment_Value, -1.8f, 0.0f);
	glVertex3f(0.3f + fMario_Leg_Increment_Value, -1.8f, 0.0f);
	glVertex3f(0.3f + fMario_Leg_Increment_Value, -1.6f, 0.0f);
	glVertex3f(0.1f + fMario_Leg_Increment_Value, -1.6f, 0.0f);
	glEnd();

	//41
	glBegin(GL_QUADS);
	glVertex3f(0.7f + fMario_Leg_Decrement_Value, -1.4f, 0.0f);
	glVertex3f(0.7f + fMario_Leg_Decrement_Value, -1.8f, 0.0f);
	glVertex3f(1.1f + fMario_Leg_Decrement_Value, -1.8f, 0.0f);
	glVertex3f(1.1f + fMario_Leg_Decrement_Value, -1.4f, 0.0f);
	glEnd();

	//42
	glBegin(GL_QUADS);
	glVertex3f(1.1f + fMario_Leg_Decrement_Value, -1.8f, 0.0f);
	glVertex3f(1.3f + fMario_Leg_Decrement_Value, -1.8f, 0.0f);
	glVertex3f(1.3f + fMario_Leg_Decrement_Value, -1.6f, 0.0f);
	glVertex3f(1.1f + fMario_Leg_Decrement_Value, -1.6f, 0.0f);
	glEnd();

	

	//White
	//43
	glBegin(GL_QUADS);
	glColor3f(0.898f, 0.6431f, 0.1882f);
	glVertex3f(0.3f, -0.6f, 0.0f);
	glVertex3f(0.1f, -0.6f, 0.0f);
	glVertex3f(0.1f, -0.8f, 0.0f);
	glVertex3f(0.3f, -0.8f, 0.0f);
	glEnd();

	//44
	glBegin(GL_QUADS);
	glColor3f(0.898f, 0.6431f, 0.1882f);
	glVertex3f(0.5f, -0.6f, 0.0f);
	glVertex3f(0.7f, -0.6f, 0.0f);
	glVertex3f(0.7f, -0.8f, 0.0f);
	glVertex3f(0.5f, -0.8f, 0.0f);
	glEnd();

	//45
	glBegin(GL_QUADS);
	glVertex3f(-0.6f, -0.6f, 0.0f);
	glVertex3f(-0.2f, -0.6f, 0.0f);
	glVertex3f(-0.2f, -1.2f, 0.0f);
	glVertex3f(-0.6f, -1.2f, 0.0f);
	glEnd();

	//46
	glBegin(GL_QUADS);
	glVertex3f(-0.2f, -0.8f, 0.0f);
	glVertex3f(-0.2f, -1.0f, 0.0f);
	glVertex3f(-0.05f, -1.0f, 0.0f);
	glVertex3f(-0.05f, -0.8f, 0.0f);
	glEnd();

	//47
	glBegin(GL_QUADS);
	glVertex3f(1.4f, -0.6f, 0.0f);
	glVertex3f(1.1f, -0.6f, 0.0f);
	glVertex3f(1.1f, -1.2f, 0.0f);
	glVertex3f(1.4f, -1.2f, 0.0f);
	glEnd();

	//48
	glBegin(GL_QUADS);
	glVertex3f(1.1f, -0.8f, 0.0f);
	glVertex3f(0.9f, -0.8f, 0.0f);
	glVertex3f(0.9f, -1.0f, 0.0f);
	glVertex3f(1.1f, -1.0f, 0.0f);
	glEnd();
}


GLfloat gfIncrement_X_Star_Value;

void drawStar(void)
{
	//glTranslatef(gfTranslate_X_Star, -1.0f, -16.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(gfTranslate_X_Star, -2.0f, -14.0f);

	/*for (gfIncrement_X_Star_Value = 30.0f; gfIncrement_X_Star_Value < 250.0f; gfIncrement_X_Star_Value = gfIncrement_X_Star_Value + 30.0f)
	{
	*/
	gfIncrement_X_Star_Value = 30.0f;
		glBegin(GL_TRIANGLES);
		glColor3f(1.0f, 160.0f / 255.0f, 68.0f / 255.0f);
		glVertex3f(0.0f + gfIncrement_X_Star_Value, 0.7f, 0.0f);
		glVertex3f(-0.3f + gfIncrement_X_Star_Value, 0.0f, 0.0f);
		glVertex3f(0.3f + gfIncrement_X_Star_Value, 0.0f, 0.0f);
		glEnd();
		glBegin(GL_TRIANGLES);
		glVertex3f(0.3f + gfIncrement_X_Star_Value, 0.0f, 0.0f);
		glVertex3f(1.0f + gfIncrement_X_Star_Value, 0.0f, 0.0f);
		glVertex3f(0.5f + gfIncrement_X_Star_Value, -0.5f, 0.0f);
		glEnd();
		glBegin(GL_TRIANGLES);
		glVertex3f(-0.3f + gfIncrement_X_Star_Value, 0.0f, 0.0f);
		glVertex3f(-1.0f + gfIncrement_X_Star_Value, 0.0f, 0.0f);
		glVertex3f(-0.5f + gfIncrement_X_Star_Value, -0.5f, 0.0f);
		glEnd();
		glBegin(GL_TRIANGLES);
		glVertex3f(-0.5f + gfIncrement_X_Star_Value, -0.5f, 0.0f);
		glVertex3f(-0.6f + gfIncrement_X_Star_Value, -1.2f, 0.0f);
		glVertex3f(0.0f + gfIncrement_X_Star_Value, -0.9f, 0.0f);
		glEnd();
		glBegin(GL_TRIANGLES);
		glVertex3f(0.5f + gfIncrement_X_Star_Value, -0.5f, 0.0f);
		glVertex3f(0.6f + gfIncrement_X_Star_Value, -1.2f, 0.0f);
		glVertex3f(0.0f + gfIncrement_X_Star_Value, -0.9f, 0.0f);
		glEnd();

		glBegin(GL_POLYGON);
		glVertex3f(-0.3f + gfIncrement_X_Star_Value, 0.0f, 0.0f);
		glVertex3f(0.3f + gfIncrement_X_Star_Value, 0.0f, 0.0f);
		glVertex3f(0.5f + gfIncrement_X_Star_Value, -0.5f, 0.0f);
		glVertex3f(0.0f + gfIncrement_X_Star_Value, -0.9f, 0.0f);
		glVertex3f(-0.5f + gfIncrement_X_Star_Value, -0.5f, 0.0f);
		glEnd();

		glPointSize(8.0f);
		glBegin(GL_POINTS);
		glColor3f(1.0f + gfIncrement_X_Star_Value, 0.0f, 0.0f);
		glVertex3f(-0.2f + gfIncrement_X_Star_Value, -0.2f, 0.0f);
		glVertex3f(0.2f + gfIncrement_X_Star_Value, -0.2f, 0.0f);
		glEnd();
	//}
}

void Draw_Plus(void)
{
	glLineWidth(3.0f);
	glBegin(GL_LINES);
	glVertex3f(-0.5f, 0.0f, 0.0f);
	glVertex3f(0.5f, 0.0f, 0.0f);
	glVertex3f(0.0f, -0.5f, 0.0f);
	glVertex3f(0.0f, 0.5f, 0.0f);
	glEnd();
}

void Draw_3(void)
{
	glLineWidth(3.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(-1.4f, -0.05f, -3.0f);
	glScalef(2.0f, 2.0f, 2.0f);
	glBegin(GL_LINES);
	glVertex3f(0.95f, 0.075f, 0.0f);
	glVertex3f(0.95f, -0.075f, 0.0f);
	glVertex3f(0.95f, 0.075f, 0.0f);
	glVertex3f(0.88f, 0.075f, 0.0f);
	glVertex3f(0.95f, -0.075f, 0.0f);
	glVertex3f(0.88f, -0.075f, 0.0f);
	glVertex3f(0.95f, 0.0f, 0.0f);
	glVertex3f(0.88f, 0.0f, 0.0f);
	glEnd();
}

void Draw_2(void)
{
	glLineWidth(3.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(0.6f, -1.8f, -3.0f);
	glScalef(2.0f, 2.0f, 2.0f);
	glBegin(GL_LINES);
	glVertex3f(0.0f, 0.95f, 0.0f);
	glVertex3f(0.07f, 0.95f, 0.0f);
	glVertex3f(0.07f, 0.95f, 0.0f);
	glVertex3f(0.07f, 0.875f, 0.0f);
	glVertex3f(0.07f, 0.875f, 0.0f);
	glVertex3f(0.0f, 0.875f, 0.0f);
	glVertex3f(0.0f, 0.875f, 0.0f);
	glVertex3f(0.0f, 0.80f, 0.0f);
	glVertex3f(0.0f, 0.80f, 0.0f);
	glVertex3f(0.07f, 0.80f, 0.0f);
	glEnd();
}


void DrawHeterogeneousDatatypes(void)
{
	double gdHeteroStart = -5.0;
	double gdHeteroPosHolder = 2.2;

	//Heterogeneous
	glLoadIdentity();
	glTranslated(gdHeteroStart + 0.7, gdHeteroPosHolder, -10.0);
	Draw_H();

	glLoadIdentity();
	glTranslated(gdHeteroStart + 1.4, gdHeteroPosHolder, -10.0);
	Draw_E();

	glLoadIdentity();
	glTranslated(gdHeteroStart + 2.1, gdHeteroPosHolder, -10.0);
	Draw_T();

	glLoadIdentity();
	glTranslated(gdHeteroStart + 2.8, gdHeteroPosHolder, -10.0);
	Draw_E();

	glLoadIdentity();
	glTranslated(gdHeteroStart + 3.5, gdHeteroPosHolder, -10.0);
	Draw_R();

	glLoadIdentity();
	glTranslated(gdHeteroStart + 4.3, gdHeteroPosHolder, -10.0);
	Draw_O();

	glLoadIdentity();
	glTranslated(gdHeteroStart + 5.1, gdHeteroPosHolder, -10.0);
	Draw_G();

	glLoadIdentity();
	glTranslated(gdHeteroStart + 5.6, gdHeteroPosHolder, -10.0);
	Draw_E();

	glLoadIdentity();
	glTranslated(gdHeteroStart + 6.3, gdHeteroPosHolder, -10.0);
	Draw_N();

	glLoadIdentity();
	glTranslated(gdHeteroStart + 7.0, gdHeteroPosHolder, -10.0);
	Draw_E();

	glLoadIdentity();
	glTranslated(gdHeteroStart + 7.9, gdHeteroPosHolder, -10.0);
	Draw_O();

	glLoadIdentity();
	glTranslated(gdHeteroStart + 8.4, gdHeteroPosHolder, -10.0);
	Draw_U();

	glLoadIdentity();
	glTranslated(gdHeteroStart + 9.1, gdHeteroPosHolder, -10.0);
	Draw_S();

	//Datatypes
	glLoadIdentity();
	glTranslated(gdHeteroStart + 2.0, gdHeteroPosHolder - 1.5, -10.0);
	Draw_D();
	glLoadIdentity();
	glTranslated(gdHeteroStart + 3.0, gdHeteroPosHolder - 1.5, -10.0);
	Draw_A();
	glLoadIdentity();
	glTranslated(gdHeteroStart + 3.5, gdHeteroPosHolder - 1.5, -10.0);
	Draw_T();
	glLoadIdentity();
	glTranslated(gdHeteroStart + 4.5, gdHeteroPosHolder - 1.5, -10.0);
	Draw_A();
	glLoadIdentity();
	glTranslated(gdHeteroStart + 5.0, gdHeteroPosHolder - 1.5, -10.0);
	Draw_T();
	glLoadIdentity();
	glTranslated(gdHeteroStart + 5.7, gdHeteroPosHolder - 1.5, -10.0);
	Draw_Y();
	glLoadIdentity();
	glTranslated(gdHeteroStart + 6.5, gdHeteroPosHolder - 1.5, -10.0);
	Draw_P();
	glLoadIdentity();
	glTranslated(gdHeteroStart + 7.2, gdHeteroPosHolder - 1.5, -10.0);
	Draw_E();
	glLoadIdentity();
	glTranslated(gdHeteroStart + 8.0, gdHeteroPosHolder - 1.5, -10.0);
	Draw_S();

}

void DrawUseStructure(void)
{
	double gdUseStructStart = -4.0;
	double gdUseStructPosHolder = 1;

	glLoadIdentity();
	glTranslated(gdUseStructStart, gdUseStructPosHolder, -10.0);
	Draw_U();

	glLoadIdentity();
	glTranslated(gdUseStructStart + 0.7, gdUseStructPosHolder, -10.0);
	Draw_S();

	glLoadIdentity();
	glTranslated(gdUseStructStart + 1.4, gdUseStructPosHolder, -10.0);
	Draw_E();

	glLoadIdentity();
	glTranslated(gdUseStructStart + 2.4, gdUseStructPosHolder, -10.0);
	Draw_S();

	glLoadIdentity();
	glTranslated(gdUseStructStart + 3.1, gdUseStructPosHolder, -10.0);
	Draw_T();

	glLoadIdentity();
	glTranslated(gdUseStructStart + 3.8, gdUseStructPosHolder, -10.0);
	Draw_R();

	glLoadIdentity();
	glTranslated(gdUseStructStart + 4.4, gdUseStructPosHolder, -10.0);
	Draw_U();

	glLoadIdentity();
	glTranslated(gdUseStructStart + 5.59, gdUseStructPosHolder, -10.0);
	Draw_C();

	glLoadIdentity();
	glTranslated(gdUseStructStart + 5.8, gdUseStructPosHolder, -10.0);
	Draw_T();

	glLoadIdentity();
	glTranslated(gdUseStructStart + 6.5, gdUseStructPosHolder, -10.0);
	Draw_U();

	glLoadIdentity();
	glTranslated(gdUseStructStart + 7.2, gdUseStructPosHolder, -10.0);
	Draw_R();

	glLoadIdentity();
	glTranslated(gdUseStructStart + 7.9, gdUseStructPosHolder, -10.0);
	Draw_E();


}

void DrawGlobalVariables(void)
{
	double gdUseStart = -3.8;
	double gdUsePosHolder = 2.2;

	double gdVariableStart = -3.8;
	double gdVariablePosHolder = 1.0;
	//use
	glLoadIdentity();
	glTranslated(gdUseStart, gdUsePosHolder, -10.0);
	Draw_U();
	glLoadIdentity();
	glTranslated(gdUseStart + 0.7, gdUsePosHolder, -10.0);
	Draw_S();
	glLoadIdentity();
	glTranslated(gdUseStart + 1.4, gdUsePosHolder, -10.0);
	Draw_E();

	//global
	glLoadIdentity();
	glTranslated(gdUseStart + 3.0, gdUsePosHolder, -10.0);
	Draw_G();
	glLoadIdentity();
	glTranslated(gdUseStart + 3.5, gdUsePosHolder, -10.0);
	Draw_L();
	glLoadIdentity();
	glTranslated(gdUseStart + 4.4, gdUsePosHolder, -10.0);
	Draw_O();
	glLoadIdentity();
	glTranslated(gdUseStart + 4.9, gdUsePosHolder, -10.0);
	Draw_B();
	glLoadIdentity();
	glTranslated(gdUseStart + 5.8, gdUsePosHolder, -10.0);
	Draw_A();
	glLoadIdentity();
	glTranslated(gdUseStart + 6.5, gdUsePosHolder, -10.0);
	Draw_L();

	//varaiables
	glLoadIdentity();
	glTranslated(gdVariableStart, gdVariablePosHolder, -10.0);
	Draw_V();

	glLoadIdentity();
	glTranslated(gdVariableStart + 0.9, gdVariablePosHolder, -10.0);
	Draw_A();

	glLoadIdentity();
	glTranslated(gdVariableStart + 1.4, gdVariablePosHolder, -10.0);
	Draw_R();

	glLoadIdentity();
	glTranslated(gdVariableStart + 2.1, gdVariablePosHolder, -10.0);
	Draw_I();

	glLoadIdentity();
	glTranslated(gdVariableStart + 3.0, gdVariablePosHolder, -10.0);
	Draw_A();

	glLoadIdentity();
	glTranslated(gdVariableStart + 3.5, gdVariablePosHolder, -10.0);
	Draw_B();

	glLoadIdentity();
	glTranslated(gdVariableStart + 4.2, gdVariablePosHolder, -10.0);
	Draw_L();

	glLoadIdentity();
	glTranslated(gdVariableStart + 4.9, gdVariablePosHolder, -10.0);
	Draw_E();

	glLoadIdentity();
	glTranslated(gdVariableStart + 5.6, gdVariablePosHolder, -10.0);
	Draw_S();

}

void DrawCompilerTable(void)
{
	double gdCompilerStart = -4.3;
	double gdCompilerPosHolder = 2.3;

	double gdLimitStart = -4.8;
	double gdLimitPosHolder = 1.0;
	//compiler 
	glLoadIdentity();
	glTranslated(gdCompilerStart, gdCompilerPosHolder, -10.0);
	Draw_C();

	glLoadIdentity();
	glTranslated(gdCompilerStart + 0.4, gdCompilerPosHolder, -10.0);
	Draw_O();

	glLoadIdentity();
	glTranslated(gdCompilerStart + 0.8, gdCompilerPosHolder, -10.0);
	Draw_M();

	glLoadIdentity();
	glTranslated(gdCompilerStart + 1.5, gdCompilerPosHolder, -10.0);
	Draw_P();

	glLoadIdentity();
	glTranslated(gdCompilerStart + 2.0, gdCompilerPosHolder, -10.0);
	Draw_I();

	glLoadIdentity();
	glTranslated(gdCompilerStart + 2.7, gdCompilerPosHolder, -10.0);
	Draw_L();

	glLoadIdentity();
	glTranslated(gdCompilerStart + 3.3, gdCompilerPosHolder, -10.0);
	Draw_E();

	glLoadIdentity();
	glTranslated(gdCompilerStart + 3.99, gdCompilerPosHolder, -10.0);
	Draw_R();

	//table 
	glLoadIdentity();
	glTranslated(gdCompilerStart + 5.0, gdCompilerPosHolder, -10.0);
	Draw_T();

	glLoadIdentity();
	glTranslated(gdCompilerStart + 5.9, gdCompilerPosHolder, -10.0);
	Draw_A();

	glLoadIdentity();
	glTranslated(gdCompilerStart + 6.4, gdCompilerPosHolder, -10.0);
	Draw_B();

	glLoadIdentity();
	glTranslated(gdCompilerStart + 7.1, gdCompilerPosHolder, -10.0);
	Draw_L();

	glLoadIdentity();
	glTranslated(gdCompilerStart + 7.8, gdCompilerPosHolder, -10.0);
	Draw_E();

	//limit
	glLoadIdentity();
	glTranslated(gdLimitStart, gdLimitPosHolder, -10.0);
	Draw_L();
	glLoadIdentity();
	glTranslated(gdLimitStart + 0.6, gdLimitPosHolder, -10.0);
	Draw_I();
	glLoadIdentity();
	glTranslated(gdLimitStart + 1.3, gdLimitPosHolder, -10.0);
	Draw_M();
	glLoadIdentity();
	glTranslated(gdLimitStart + 2.0, gdLimitPosHolder, -10.0);
	Draw_I();
	glLoadIdentity();
	glTranslated(gdLimitStart + 2.7, gdLimitPosHolder, -10.0);
	Draw_T();

	//exceeded
	glLoadIdentity();
	glTranslated(gdLimitStart + 3.7, gdLimitPosHolder, -10.0);
	Draw_E();
	glLoadIdentity();
	glTranslated(gdLimitStart + 4.4, gdLimitPosHolder, -10.0);
	Draw_X();
	glLoadIdentity();
	glTranslated(gdLimitStart + 5.5, gdLimitPosHolder, -10.0);
	Draw_C();
	glLoadIdentity();
	glTranslated(gdLimitStart + 5.7, gdLimitPosHolder, -10.0);
	Draw_E();
	glLoadIdentity();
	glTranslated(gdLimitStart + 6.4, gdLimitPosHolder, -10.0);
	Draw_E();
	glLoadIdentity();
	glTranslated(gdLimitStart + 7.09, gdLimitPosHolder, -10.0);
	Draw_D();

	glLoadIdentity();
	glTranslated(gdLimitStart + 7.9, gdLimitPosHolder, -10.0);
	Draw_E();
	glLoadIdentity();
	glTranslated(gdLimitStart + 8.6, gdLimitPosHolder, -10.0);

	Draw_D();

}

void DrawInitializePointer(void)
{
	double gdInitializeStart = -3.7;
	double gdInitializePosHolder = 2.2;

	double gdPointerStart = -2.4;
	double gdPointerPosHolder = 1.0;
	//Initialize
	//gdInitializeStart
	glLoadIdentity();
	glTranslated(gdInitializeStart, gdInitializePosHolder, -10.0);
	Draw_I();

	glLoadIdentity();
	glTranslated(gdInitializeStart + 0.7, gdInitializePosHolder, -10.0);
	Draw_N();

	glLoadIdentity();
	glTranslated(gdInitializeStart + 1.4, gdInitializePosHolder, -10.0);
	Draw_I();

	glLoadIdentity();
	glTranslated(gdInitializeStart + 2.1, gdInitializePosHolder, -10.0);
	Draw_T();

	glLoadIdentity();
	glTranslated(gdInitializeStart + 2.8, gdInitializePosHolder, -10.0);
	Draw_I();

	glLoadIdentity();
	glTranslated(gdInitializeStart + 3.8, gdInitializePosHolder, -10.0);
	Draw_A();

	glLoadIdentity();
	glTranslated(gdInitializeStart + 4.3, gdInitializePosHolder, -10.0);
	Draw_L();

	glLoadIdentity();
	glTranslated(gdInitializeStart + 5.0, gdInitializePosHolder, -10.0);
	Draw_I();

	glLoadIdentity();
	glTranslated(gdInitializeStart + 5.7, gdInitializePosHolder, -10.0);
	Draw_Z();

	glLoadIdentity();
	glTranslated(gdInitializeStart + 6.4, gdInitializePosHolder, -10.0);
	Draw_E();


	//Pointer
	glLoadIdentity();
	glTranslated(gdPointerStart, gdPointerPosHolder, -10.0);
	Draw_P();

	glLoadIdentity();
	glTranslated(gdPointerStart + 0.9, gdPointerPosHolder, -10.0);
	Draw_O();

	glLoadIdentity();
	glTranslated(gdPointerStart + 1.3, gdPointerPosHolder, -10.0);
	Draw_I();

	glLoadIdentity();
	glTranslated(gdPointerStart + 2.0, gdPointerPosHolder, -10.0);
	Draw_N();

	glLoadIdentity();
	glTranslated(gdPointerStart + 2.7, gdPointerPosHolder, -10.0);
	Draw_T();

	glLoadIdentity();
	glTranslated(gdPointerStart + 3.3, gdPointerPosHolder, -10.0);
	Draw_E();

	glLoadIdentity();
	glTranslated(gdPointerStart + 4.0, gdPointerPosHolder, -10.0);
	Draw_R();
}

void DrawDanglingPointer(void)
{
	double gdDanglingStart = -3.0;
	double gdDanglingPosHolder = 2.2;

	double gdPointerStart = -2.4;
	double gdPointerPosHolder = 1.0;

	glTranslated(gdDanglingStart, gdDanglingPosHolder, -10.0);
	Draw_D();

	glLoadIdentity();
	glTranslated(gdDanglingStart + 1.0, gdDanglingPosHolder, -10.0);
	Draw_A();

	glLoadIdentity();
	glTranslated(gdDanglingStart + 1.5, gdDanglingPosHolder, -10.0);
	Draw_N();

	glLoadIdentity();
	glTranslated(gdDanglingStart + 2.6, gdDanglingPosHolder, -10.0);
	Draw_G();

	glLoadIdentity();
	glTranslated(gdDanglingStart + 3.0, gdDanglingPosHolder, -10.0);
	Draw_L();

	glLoadIdentity();
	glTranslated(gdDanglingStart + 3.6, gdDanglingPosHolder, -10.0);
	Draw_I();

	glLoadIdentity();
	glTranslated(gdDanglingStart + 4.3, gdDanglingPosHolder, -10.0);
	Draw_N();

	glLoadIdentity();
	glTranslated(gdDanglingStart + 5.4, gdDanglingPosHolder, -10.0);
	Draw_G();

	//Pointer
	glLoadIdentity();
	glTranslated(gdPointerStart, gdPointerPosHolder, -10.0);
	Draw_P();

	glLoadIdentity();
	glTranslated(gdPointerStart + 0.9, gdPointerPosHolder, -10.0);
	Draw_O();

	glLoadIdentity();
	glTranslated(gdPointerStart + 1.3, gdPointerPosHolder, -10.0);
	Draw_I();

	glLoadIdentity();
	glTranslated(gdPointerStart + 2.0, gdPointerPosHolder, -10.0);
	Draw_N();

	glLoadIdentity();
	glTranslated(gdPointerStart + 2.7, gdPointerPosHolder, -10.0);
	Draw_T();

	glLoadIdentity();
	glTranslated(gdPointerStart + 3.3, gdPointerPosHolder, -10.0);
	Draw_E();

	glLoadIdentity();
	glTranslated(gdPointerStart + 4.0, gdPointerPosHolder, -10.0);
	Draw_R();

}

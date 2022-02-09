#include<windows.h>
#include<gl/GL.h>
#include<gl/GLU.h>
#include<mmsystem.h>

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

#pragma comment(lib,"User32.lib")
#pragma comment(lib,"GDI32.lib")
#pragma comment(lib,"opengl32.lib")
#pragma comment(lib,"glu32.lib")
#pragma comment(lib,"winmm.lib")

LRESULT CALLBACK WndProc(HWND,UINT,WPARAM,LPARAM);

HWND ghwnd=NULL;
HDC ghdc=NULL;
HGLRC ghrc=NULL;

DWORD dwStyle;
WINDOWPLACEMENT wpPrev={ sizeof(WINDOWPLACEMENT) };

GLfloat gIDrawFlag = -2.5f;
GLfloat gNDrawFlag = 2.2f;
GLfloat gDDrawFlag = 0.0f;
GLfloat gI1DrawFlag = -2.0f;
GLfloat gADrawFlag = 2.5f;
GLfloat gTriDrawFlag = -3.0f;
GLfloat gVertexDrawFlag = -4.2f;
GLfloat gGreen_Speed_Flag = 0.00005f;
GLfloat gOrange_Speed_Flag = 0.0002f;

GLfloat gD_L_VL_R_Flag = 0.0f;
GLfloat gD_L_VL_G_Flag = 0.0f;
GLfloat gD_L_VL_B_Flag = 0.0f;
GLfloat gD_L_VL_R_Flag1 = 0.0f;
GLfloat gD_L_VL_G_Flag1 = 0.0f;
GLfloat gD_L_VL_B_Flag1 = 0.0f;

GLfloat gD_R_VL_R_Flag = 0.0f;
GLfloat gD_R_VL_G_Flag = 0.0f;
GLfloat gD_R_VL_B_Flag = 0.0f;
GLfloat gD_R_VL_R_Flag1 = 0.0f;
GLfloat gD_R_VL_G_Flag1 = 0.0f;
GLfloat gD_R_VL_B_Flag1 = 0.0f;

GLfloat gD_T_HL_R_Flag = 0.0f;
GLfloat gD_T_HL_G_Flag = 0.0f;
GLfloat gD_T_HL_B_Flag = 0.0f;

GLfloat gD_B_HL_R_Flag = 0.0f;
GLfloat gD_B_HL_G_Flag = 0.0f;
GLfloat gD_B_HL_B_Flag = 0.0f;

bool gbD_Orange_Complete_Flag = false;
bool gbD_Green_Complete_Flag = false;
//bool gbD_Complete_Flag = false;

bool gbActiveWindow=false;
bool gbEscapeKeyIsPressed=false;
bool gbFullscreen=false;

bool bICheckFlag = false;
bool bNCheckFlag = false;
bool gbDCheckFlag = false;
bool bI1CheckFlag = false;
bool bACheckFlag = false;

int WINAPI WinMain(HINSTANCE hInstance,HINSTANCE hPrevInstance,LPSTR lpszCmdLine,int nCmdShow)
{
	void initialize(void);
	void display(void);
	void uninitialize(void);
	void ToggleFullScreen(void);

	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szClassName[]=TEXT("OpenGL Fixed Function Pipeline by native windowing");
	bool bDone=false;
	
	wndclass.cbSize=sizeof(WNDCLASSEX);
	wndclass.style=CS_HREDRAW|CS_VREDRAW|CS_OWNDC;
	wndclass.cbClsExtra=0;
	wndclass.cbWndExtra=0;
	wndclass.lpfnWndProc=WndProc;
	wndclass.lpszClassName=szClassName;
	wndclass.lpszMenuName=NULL;
	wndclass.hInstance=hInstance;
	wndclass.hIcon=LoadIcon(NULL,IDI_APPLICATION);
	wndclass.hIconSm=LoadIcon(NULL,IDI_APPLICATION);
	wndclass.hCursor=LoadCursor(NULL,IDC_ARROW);
	wndclass.hbrBackground=(HBRUSH)GetStockObject(BLACK_BRUSH);

	RegisterClassEx(&wndclass);

	hwnd=CreateWindowEx(WS_EX_APPWINDOW,szClassName,TEXT("My OpenGL First Native Window"),WS_OVERLAPPEDWINDOW|WS_CLIPCHILDREN|WS_CLIPSIBLINGS|WS_VISIBLE,0,0,WIN_WIDTH,WIN_HEIGHT,NULL,NULL,hInstance,NULL);
	
	ghwnd=hwnd;

	ShowWindow(hwnd,SW_SHOW);
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);

	initialize();
	ToggleFullScreen();
	sndPlaySound("Maa_Tujhe_Salam.wav", SND_ASYNC);

	while(bDone==false)
	{
		if(PeekMessage(&msg,NULL,0,0,PM_REMOVE))
		{
			if(msg.message==WM_QUIT)
				bDone=true;
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		else
		{
			display();
			if(gbActiveWindow==true)
			{
				if(gbEscapeKeyIsPressed==true)
					bDone=true;
			}
		}
	}
	uninitialize();
	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd,UINT iMsg,WPARAM wParam,LPARAM lParam)
{
	void display(void);
	void resize(int,int);
	void ToggleFullScreen(void);
	void uninitialize(void);

	switch(iMsg)
	{
	case WM_ACTIVATE:
		if(HIWORD(wParam)==0)
			gbActiveWindow=true;
		else
			gbActiveWindow=false;
		break;
	case WM_SIZE:
		resize(LOWORD(lParam),HIWORD(lParam));
		break;
	case WM_KEYDOWN:
		switch(wParam)
		{
		case VK_ESCAPE:
			if(gbEscapeKeyIsPressed==false)
				gbEscapeKeyIsPressed=true;
			break;
		case 0x46:
			if(gbFullscreen==false)
			{
				ToggleFullScreen();
				gbFullscreen=true;
			}
			else
			{
				ToggleFullScreen();
				gbFullscreen=false;
			}
			break;
		default:
			break;
		}
		break;
	case WM_LBUTTONDOWN:
		break;
	case WM_CLOSE:
		uninitialize();
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
		case WM_QUIT:
		PostQuitMessage(0);
		break;
	default:
		break;
	}
	return(DefWindowProc(hwnd,iMsg,wParam,lParam));
}

void ToggleFullScreen(void)
{
	MONITORINFO mi={sizeof(MONITORINFO)};
	if(gbFullscreen==false)
	{
		dwStyle=GetWindowLong(ghwnd,GWL_STYLE);
		if(dwStyle & WS_OVERLAPPEDWINDOW)
		{
			//mi= { sizeof(MONITORINFO) };
			if(GetWindowPlacement(ghwnd,&wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd,MONITORINFOF_PRIMARY),&mi))
			{
				SetWindowLong(ghwnd,GWL_STYLE,dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghwnd,HWND_TOP,mi.rcMonitor.left,mi.rcMonitor.top,mi.rcMonitor.right-mi.rcMonitor.left,mi.rcMonitor.bottom-mi.rcMonitor.top,SWP_NOZORDER|SWP_FRAMECHANGED);
			}
		}
		ShowCursor(FALSE);
	} 

	else
	{
		SetWindowLong(ghwnd,GWL_STYLE,dwStyle|WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd,&wpPrev);
		SetWindowPos(ghwnd,HWND_TOP,0,0,0,0,SWP_NOMOVE|SWP_NOSIZE|SWP_NOOWNERZORDER|SWP_NOZORDER|SWP_FRAMECHANGED);
		
		ShowCursor(TRUE);
	}
}


void initialize(void)
{
	void resize(int,int);
	void ToggleFullScreen(void);

	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;

	ZeroMemory(&pfd,sizeof(PIXELFORMATDESCRIPTOR));

	pfd.nSize=sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion=1;
	pfd.dwFlags=PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL|PFD_DOUBLEBUFFER;
	pfd.cColorBits=32;
	pfd.cRedBits=8;
	pfd.cGreenBits=8;
	pfd.cBlueBits=8;
	pfd.cAlphaBits=8;
	pfd.cDepthBits=24;

	ghdc=GetDC(ghwnd);

	iPixelFormatIndex=ChoosePixelFormat(ghdc,&pfd);
	if(iPixelFormatIndex==0)
	{
		ReleaseDC(ghwnd,ghdc);
		ghdc=NULL;
	}

	if(SetPixelFormat(ghdc,iPixelFormatIndex,&pfd)==false)
	{
		ReleaseDC(ghwnd,ghdc);
		ghdc=NULL;
	}

	ghrc=wglCreateContext(ghdc);
	if(ghrc==NULL)
	{
		ReleaseDC(ghwnd,ghdc);
		ghdc=NULL;
	}

	if(wglMakeCurrent(ghdc,ghrc)==false)
	{
		wglDeleteContext(ghrc);
		ghrc=NULL;
		ReleaseDC(ghwnd,ghdc);
		ghdc=NULL;
	}

	resize(WIN_WIDTH,WIN_HEIGHT);
}

/*Saffron: (RGB: 255, 153, 51) (hex code: #FF9933)
White: (RGB: 255, 255, 255) (hex code: #FFFFFF)
Green: (RGB: 19, 136, 8) (hex code: #138808)*/

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT);

		//For I;
	glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		if (gIDrawFlag <= -1.9f)
		{
			gIDrawFlag = gIDrawFlag + 0.0001f;
		}
		else
		{
			bICheckFlag = true;
		}

		glTranslatef(gIDrawFlag, 0.0f, -3.0f);
		glScalef(0.5f, 0.5f, 0.5f);
		glBegin(GL_QUADS);
		glColor3f(1.0f, 0.6f, 0.2f);
		glVertex3f(0.15f, 1.7f, 0.0f);
		glVertex3f(0.1f, 1.7f, 0.0f);
		glColor3f(0.075f, 0.533f, 0.0314f);
		glVertex3f(0.1f, -1.7f, 0.0f);
		glVertex3f(0.15f, -1.7f, 0.0f);
		glEnd();

		if (bICheckFlag == true)
		{
			//For N 
			glLoadIdentity();

			if (gNDrawFlag >= 0.0f)
			{
				gNDrawFlag = gNDrawFlag - 0.00015f;
			}
			else
				bNCheckFlag = true;

			glTranslatef(-1.4f, gNDrawFlag, -3.0f);
			glScalef(0.5f, 0.5f, 0.5f);
			glBegin(GL_QUADS);
			//For Left Quads of N
			glColor3f(1.0f, 0.6f, 0.2f);
			glVertex3f(0.15f, 1.7f, 0.0f);
			glVertex3f(0.1f, 1.7f, 0.0f);
			glColor3f(0.075f, 0.533f, 0.0314f);
			glVertex3f(0.1f, -1.7f, 0.0f);
			glVertex3f(0.15f, -1.7f, 0.0f);
			//For Right Quads Of N	
			glColor3f(1.0f, 0.6f, 0.2f);
			glVertex3f(1.22f, 1.7f, 0.0f);
			glVertex3f(1.17f, 1.7f, 0.0f);
			glColor3f(0.075f, 0.533f, 0.0314f);
			glVertex3f(1.17f, -1.7f, 0.0f);
			glVertex3f(1.22f, -1.7f, 0.0f);
			//For Center Quads Of N
			glColor3f(1.0f, 0.6f, 0.2f);
			glVertex3f(0.15f, 1.7f, 0.0f);
			glVertex3f(0.1f, 1.7f, 0.0f);
			glColor3f(0.075f, 0.533f, 0.0314f);
			glVertex3f(1.17f, -1.7f, 0.0f);
			glVertex3f(1.22f, -1.7f, 0.0f);
			glEnd();
		}

		if (bNCheckFlag == true)
		{
			//For D
			glLoadIdentity();
			glTranslatef(-0.3f, 0.0f, -3.0f);
			glScalef(0.5f, 0.5f, 0.5f);
			glBegin(GL_QUADS);
			//Left Vertical line
			if (gD_L_VL_R_Flag < 1.0f | gD_L_VL_G_Flag < 0.6f | gD_L_VL_B_Flag < 0.2f)
			{
				if (gD_L_VL_R_Flag < 1.0f)
					gD_L_VL_R_Flag = gD_L_VL_R_Flag + gOrange_Speed_Flag;

				if (gD_L_VL_G_Flag < 0.6f)
					gD_L_VL_G_Flag = gD_L_VL_G_Flag + gOrange_Speed_Flag;

				if (gD_L_VL_B_Flag < 0.2f)
					gD_L_VL_B_Flag = gD_L_VL_B_Flag + gOrange_Speed_Flag;
			}
			else
				gbD_Orange_Complete_Flag = true;

			glColor3f(gD_L_VL_R_Flag, gD_L_VL_G_Flag, gD_L_VL_B_Flag);
			glVertex3f(0.15f, 1.7f, 0.0f);
			glVertex3f(0.1f, 1.7f, 0.0f);
			if (gD_L_VL_R_Flag1 < 0.075f | gD_L_VL_G_Flag1 < 0.533f | gD_L_VL_B_Flag1 < 0.0314f)
			{
				if (gD_L_VL_R_Flag1 < 0.075f)
					gD_L_VL_R_Flag1 = gD_L_VL_R_Flag1 + gGreen_Speed_Flag;

				if (gD_L_VL_G_Flag1 < 0.533f)
					gD_L_VL_G_Flag1 = gD_L_VL_G_Flag1 + gGreen_Speed_Flag;

				if (gD_L_VL_B_Flag1 < 0.0314f)
					gD_L_VL_B_Flag1 = gD_L_VL_B_Flag1 + gGreen_Speed_Flag;
			}
			else
				gbD_Green_Complete_Flag = true;

			glColor3f(gD_L_VL_R_Flag1, gD_L_VL_G_Flag1, gD_L_VL_B_Flag1);
			glVertex3f(0.1f, -1.7f, 0.0f);
			glVertex3f(0.15f, -1.7f, 0.0f);
			//Right Vertical line
			if (gD_R_VL_R_Flag < 1.0f | gD_R_VL_G_Flag < 0.6f | gD_R_VL_B_Flag < 0.2f)
			{
				if (gD_R_VL_R_Flag < 1.0f)
					gD_R_VL_R_Flag = gD_R_VL_R_Flag + gOrange_Speed_Flag;

				if (gD_R_VL_G_Flag < 0.6f)
					gD_R_VL_G_Flag = gD_R_VL_G_Flag + gOrange_Speed_Flag;

				if (gD_R_VL_B_Flag < 0.2f)
					gD_R_VL_B_Flag = gD_R_VL_B_Flag + gOrange_Speed_Flag;
			}
			glColor3f(gD_R_VL_R_Flag, gD_R_VL_G_Flag, gD_R_VL_B_Flag);
			glVertex3f(1.37, 1.7f, 0.0f);
			glVertex3f(1.32f, 1.7f, 0.0f);
			if (gD_R_VL_R_Flag1 < 0.075f | gD_R_VL_G_Flag1 < 0.533f | gD_R_VL_B_Flag1 < 0.0314f)
			{
				if (gD_R_VL_R_Flag1 < 0.075f)
					gD_R_VL_R_Flag1 = gD_R_VL_R_Flag1 + gGreen_Speed_Flag;

				if (gD_R_VL_G_Flag1 < 0.533f)
					gD_R_VL_G_Flag1 = gD_R_VL_G_Flag1 + gGreen_Speed_Flag;

				if (gD_R_VL_B_Flag1 < 0.0314f)
					gD_R_VL_B_Flag1 = gD_R_VL_B_Flag1 + gGreen_Speed_Flag;
			}
			glColor3f(gD_R_VL_R_Flag1, gD_R_VL_G_Flag1, gD_R_VL_B_Flag1);
			glVertex3f(1.32f, -1.7f, 0.0f);
			glVertex3f(1.37f, -1.7f, 0.0f);
			//Top Horizontal line
			if (gD_T_HL_R_Flag < 1.0f | gD_T_HL_G_Flag < 0.6f | gD_T_HL_B_Flag < 0.2f)
			{
				if (gD_T_HL_R_Flag < 1.0f)
					gD_T_HL_R_Flag = gD_T_HL_R_Flag + gOrange_Speed_Flag;

				if (gD_T_HL_G_Flag < 0.6f)
					gD_T_HL_G_Flag = gD_T_HL_G_Flag + gOrange_Speed_Flag;

				if (gD_T_HL_B_Flag < 0.2f)
					gD_T_HL_B_Flag = gD_T_HL_B_Flag + gOrange_Speed_Flag;
			}
			glColor3f(gD_T_HL_R_Flag, gD_T_HL_G_Flag, gD_T_HL_B_Flag);
			glVertex3f(1.37f, 1.7f, 0.0f);
			glVertex3f(-0.1f, 1.7f, 0.0f);
			glVertex3f(-0.1f, 1.75f, 0.0f);
			glVertex3f(1.37f, 1.75f, 0.0f);
			//Bottom Horizontal line
			if (gD_B_HL_R_Flag < 0.075f | gD_B_HL_G_Flag < 0.533f | gD_B_HL_B_Flag < 0.0314f)
			{
				if (gD_B_HL_R_Flag < 0.075f)
					gD_B_HL_R_Flag = gD_B_HL_R_Flag + gGreen_Speed_Flag;

				if (gD_B_HL_G_Flag < 0.533f)
					gD_B_HL_G_Flag = gD_B_HL_G_Flag + gGreen_Speed_Flag;

				if (gD_B_HL_B_Flag < 0.0314f)
					gD_B_HL_B_Flag = gD_B_HL_B_Flag + gGreen_Speed_Flag;
			}
			glColor3f(gD_B_HL_R_Flag, gD_B_HL_G_Flag, gD_B_HL_B_Flag);
			glVertex3f(1.37f, -1.7f, 0.0f);
			glVertex3f(-0.1f, -1.7f, 0.0f);
			glVertex3f(-0.1f, -1.75f, 0.0f);
			glVertex3f(1.37f, -1.75f, 0.0f);
			glEnd();

			if (gbD_Orange_Complete_Flag == true && gbD_Green_Complete_Flag == true)
				gbDCheckFlag = true;
		}

		if (gbDCheckFlag == true)
		{
			//For I;
			glLoadIdentity();

			if (gI1DrawFlag <= 0.0f)
			{
				gI1DrawFlag = gI1DrawFlag + 0.00015f;
			}
			else
				bI1CheckFlag = true;

			glTranslatef(0.8f, gI1DrawFlag, -3.0f);
			glScalef(0.5f, 0.5f, 0.5f);
			glBegin(GL_QUADS);
			glColor3f(1.0f, 0.6f, 0.2f);
			glVertex3f(0.15f, 1.7f, 0.0f);
			glVertex3f(0.1f, 1.7f, 0.0f);
			glColor3f(0.075f, 0.533f, 0.0314f);
			glVertex3f(0.1f, -1.7f, 0.0f);
			glVertex3f(0.15f, -1.7f, 0.0f);
			glEnd();
		}

		if (bI1CheckFlag == true)
		{
			//For A;
			glLoadIdentity();

			if (gADrawFlag >= 1.9f)
			{
				gADrawFlag = gADrawFlag - 0.0001f;
			}
			else
				bACheckFlag = true;

			glTranslatef(gADrawFlag, 0.0f, -3.0f);
			glScalef(0.5f, 0.5f, 0.5f);
			glBegin(GL_QUADS);
			glColor3f(1.0f, 0.6f, 0.2f);
			glVertex3f(-0.60f, 1.7f, 0.0f);
			glVertex3f(-0.65f, 1.7f, 0.0f);
			glColor3f(0.075f, 0.533f, 0.0314f);
			glVertex3f(0.1f, -1.7f, 0.0f);
			glVertex3f(0.15f, -1.7f, 0.0f);
			glColor3f(1.0f, 0.6f, 0.2f);
			glVertex3f(-0.60f, 1.7f, 0.0f);
			glVertex3f(-0.65f, 1.7f, 0.0f);
			glColor3f(0.075f, 0.533f, 0.0314f);
			glVertex3f(-1.25f, -1.7f, 0.0f);
			glVertex3f(-1.30f, -1.7f, 0.0f);
			glEnd();
		}

		if (bACheckFlag == true)
		{
			//For TriColor
			glLoadIdentity();

			if (gTriDrawFlag <= 1.0f)
			{
				gTriDrawFlag = gTriDrawFlag + 0.0002f;
			}

			if (gTriDrawFlag >= 1.0f)
			{
				if (gVertexDrawFlag <= 0.56f)
				{
					gVertexDrawFlag = gVertexDrawFlag + 0.0002f;
				}
			}

			glTranslatef(gTriDrawFlag, -0.3f, -3.0f);
			glScalef(0.75f, 0.75f, 0.75f);
			glBegin(GL_QUADS);
			glColor3f(1.0f, 0.6f, 0.2f);
			glVertex3f(1.02f, 0.41f, 0.0f);
			glVertex3f(gVertexDrawFlag, 0.41f, 0.0f);
			glVertex3f(gVertexDrawFlag, 0.4f, 0.0f);
			glVertex3f(1.02f, 0.4f, 0.0f);
			glColor3f(1.0f, 1.0f, 1.0f);
			glVertex3f(1.02f, 0.39f, 0.0f);
			glVertex3f(gVertexDrawFlag, 0.39f, 0.0f);
			glVertex3f(gVertexDrawFlag, 0.38f, 0.0f);
			glVertex3f(1.02f, 0.38f, 0.0f);
			glColor3f(0.075f, 0.533f, 0.0314f);
			glVertex3f(1.02f, 0.37f, 0.0f);
			glVertex3f(gVertexDrawFlag, 0.37f, 0.0f);
			glVertex3f(gVertexDrawFlag, 0.36f, 0.0f);
			glVertex3f(1.03f, 0.36f, 0.0f);
			glEnd();
		}
	
	SwapBuffers(ghdc);
}

void resize(int width,int height)
{
	if(height==0)
		height=1;
	glViewport(0,0,(GLsizei)width,(GLsizei)height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(45.0f,(GLfloat)width/(GLfloat)height,0.1f,100.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void uninitialize(void)
{
	if(gbFullscreen==true)
	{
		dwStyle=GetWindowLong(ghwnd,GWL_STYLE);
		SetWindowLong(ghwnd,GWL_STYLE,dwStyle|WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd,&wpPrev);
		SetWindowPos(ghwnd,HWND_TOP,0,0,0,0,SWP_NOMOVE|SWP_NOSIZE|SWP_NOOWNERZORDER|SWP_NOZORDER|SWP_FRAMECHANGED);

		ShowCursor(TRUE);
	}
	wglMakeCurrent(NULL,NULL);
	wglDeleteContext(ghrc);
	ghrc=NULL;

	ReleaseDC(ghwnd,ghdc);
	ghdc=NULL;

	DestroyWindow(ghwnd);
}
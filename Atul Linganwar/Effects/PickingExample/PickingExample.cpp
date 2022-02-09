#include<Windows.h>
#include<stdio.h>
#include<stdlib.h>

#include<gl/glew.h>
#include<gl/GL.h>

#include "../common/vmath.h"
#include "VertexDataPicking.h"
#include "VAOsPicking.h"
#include "ShaderProgramPicking.h"
#include "PickingExample.h"
#include "DrawPlanets.h"

#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")

//global variables
FILE* gpFile = NULL;

HWND ghwnd = NULL;
HDC ghdc = NULL;
HGLRC ghrc = NULL;

DWORD dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };

bool gbActiveWindow = false;
bool gbEscapeKeyIsPressed = false;
bool gbFullscreen = false;

VAO vaoSun(VAO_COLOR, QUAD);
VAO vaoMercury(VAO_COLOR, QUAD);
VAO vaoVenus(VAO_COLOR, QUAD);

VAO vaoSunTexture(VAO_TEXTURE, QUAD);
VAO vaoMercuryTexture(VAO_TEXTURE, QUAD);
VAO vaoVenusTexture(VAO_TEXTURE, QUAD);

ShaderProgram PickingShaderProgram(PICKING_COLOR_SHADER);
ShaderProgram TextureShaderProgram(TEXTURE_SHADER);
ShaderProgram TextureProgramQuad(SHADER_TYPE::QUAD_TEXTURE_SHADER);
VAO vaoQuad(VAO_TYPE::VAO_TEXTURE, SHAPE::QUAD);

bool bFirstResize = true;
bool bIsFBOInitialized = false;

GLfloat gWindowWidth = 0.0f;
GLfloat gWindowHeight = 0.0f;

GLfloat gInitWindowWidth = 0.0f;
GLfloat gInitWindowHeight = 0.0f;

GLfloat gMouseX = 0.0f;
GLfloat gMouseY = 0.0f;
bool bIsMouseButtonPressed = false;

GLuint gFbo;
GLuint colorRenderBuffer;
GLuint depthRenderBuffer;

vmath::mat4 gPerspectiveProjectionMatrix;

//WinMain
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int nCmdShow)
{
	//function prototype
	void initialize(void);
	void display(void);
	void update(void);
	void uninitialize(void);

	//variables declaration
	WNDCLASSEX wndclass;
	MSG msg;
	HWND hwnd;
	TCHAR szClassName[] = TEXT("OpenGLPP");

	bool bDone = false;

	//code
	//log file
	if (fopen_s(&gpFile, "Log.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Log File can not be created"), TEXT("Error"), MB_OK);
		ExitProcess(EXIT_FAILURE);
	}
	else
		fprintf(gpFile, "Log file created successfully\n");

	//initialize wndclass
	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.hInstance = hInstance;
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.lpfnWndProc = WndProc;
	wndclass.lpszClassName = szClassName;
	wndclass.lpszMenuName = NULL;
	wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);

	//register class
	if (!RegisterClassEx(&wndclass))
	{
		fprintf(gpFile, "Error while registering the class\n");
		fclose(gpFile);
		ExitProcess(EXIT_FAILURE);
	}

	int x = GetSystemMetrics(SM_CXSCREEN);
	int y = GetSystemMetrics(SM_CYSCREEN);

	//Create Window
	hwnd = CreateWindowEx(WS_EX_APPWINDOW,
		szClassName,
		TEXT("Programmable Rotating 3D Two shapes Using FBO"),
		WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
		(x / 2) - (OGL_WINDOW_WIDTH / 2),
		(y / 2) - (OGL_WINDOW_HEIGHT / 2),
		OGL_WINDOW_WIDTH,
		OGL_WINDOW_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL);

	if (!hwnd)
	{
		fprintf(gpFile, "Error while creating window\n");
		fclose(gpFile);
		ExitProcess(EXIT_FAILURE);
	}

	ghwnd = hwnd;

	ShowWindow(hwnd, nCmdShow);
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);

	//initialize
	initialize();

	//game loop
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
			//render'
			update();
			display();

			if (gbActiveWindow == true)
			{
				if (gbEscapeKeyIsPressed == true)
					bDone = true;
			}
		}
	}

	//uninitialize
	uninitialize();

	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	//function prototype
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

		//case WM_ERASEBKGND:
		//	return(0);

	case WM_SIZE:
		Resize(LOWORD(lParam), HIWORD(lParam));
		gWindowWidth = LOWORD(lParam);
		gWindowHeight = HIWORD(lParam);
		break;

	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			if (gbEscapeKeyIsPressed == false)
				gbEscapeKeyIsPressed = true;
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

		default:
			break;
		}
		break;

	case WM_LBUTTONDOWN:
		gMouseX = LOWORD(lParam);
		gMouseY = HIWORD(lParam);
		bIsMouseButtonPressed = true;
		break;

	case WM_CLOSE:
		uninitialize();
		break;

	case WM_DESTROY:
		PostQuitMessage(0);
		break;

	default:
		break;
	}

	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void ToggleFullscreen(void)
{
	//variable declaration
	MONITORINFO mi;

	//code
	if (gbFullscreen == false)
	{
		dwStyle = GetWindowLong(ghwnd, GWL_STYLE);
		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{
			mi = { sizeof(MONITORINFO) };
			if (GetWindowPlacement(ghwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &mi))
			{
				SetWindowLong(ghwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOZORDER | SWP_FRAMECHANGED);

			}
		}
		//ShowCursor(FALSE);
	}
	else
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);

		ShowCursor(TRUE);
	}
}

//initialize
void initialize(void)
{
	///function prototype
	void uninitialize(void);
	void resize(int, int);

	//variables
	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;

	bool bRet = false;

	//code
	ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));

	//Initialize the structure pfd
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

	ghdc = GetDC(ghwnd);

	iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);
	if (iPixelFormatIndex == 0)
	{
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == false)
	{
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	ghrc = wglCreateContext(ghdc);
	if (ghrc == NULL)
	{
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}
	if (wglMakeCurrent(ghdc, ghrc) == false)
	{
		wglDeleteContext(ghrc);
		ghrc = NULL;
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	//GLEW initialization code
	GLenum glew_error = glewInit();
	if (glew_error != GLEW_OK)
	{
		fprintf(gpFile, "Error at glewInit()\n");
		wglDeleteContext(ghrc);
		ghrc = NULL;
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	bRet = PickingShaderProgram.Init();
	if (false == bRet)
	{
		fprintf(gpFile, "Error while initializing PickingShaderProgram\n");
		uninitialize();
		ExitProcess(EXIT_FAILURE);
	}

	bRet = TextureShaderProgram.Init();
	if (false == bRet)
	{
		fprintf(gpFile, "Error while initializing TextureShaderProgram\n");
		uninitialize();
		ExitProcess(EXIT_FAILURE);
	}

	bRet = vaoSun.Init();
	if (false == bRet)
	{
		fprintf(gpFile, "Error while initializing vaoSun\n");
		uninitialize();
		ExitProcess(EXIT_FAILURE);
	}

	bRet = vaoMercury.Init();
	if (false == bRet)
	{
		fprintf(gpFile, "Error while initializing vaoMercury\n");
		uninitialize();
		ExitProcess(EXIT_FAILURE);
	}

	bRet = vaoVenus.Init();
	if (false == bRet)
	{
		fprintf(gpFile, "Error while initializing vaoVenus\n");
		uninitialize();
		ExitProcess(EXIT_FAILURE);
	}

	bRet = vaoSunTexture.Init();
	if (false == bRet)
	{
		fprintf(gpFile, "Error while initializing vaoSunTexture\n");
		uninitialize();
		ExitProcess(EXIT_FAILURE);
	}

	bRet = vaoMercuryTexture.Init();
	if (false == bRet)
	{
		fprintf(gpFile, "Error while initializing vaoMercuryTexture\n");
		uninitialize();
		ExitProcess(EXIT_FAILURE);
	}

	bRet = vaoVenusTexture.Init();
	if (false == bRet)
	{
		fprintf(gpFile, "Error while initializing vaoVenusTexture\n");
		uninitialize();
		ExitProcess(EXIT_FAILURE);
	}

	bRet = TextureProgramQuad.Init();
	if (false == bRet)
	{
		fprintf(gpFile, "Error while initializing QuadTextureProgram\n");
		uninitialize();
		ExitProcess(EXIT_FAILURE);
	}

	bRet = vaoQuad.Init();
	if (false == bRet)
	{
		fprintf(gpFile, "Error while initializing vaoPyramid\n");
		uninitialize();
		ExitProcess(EXIT_FAILURE);
	}

	//FRAME BUFFER----------------------------------------------------------------------------------------------
	glGenFramebuffers(1, &gFbo);
	glBindFramebuffer(GL_FRAMEBUFFER, gFbo);

	glGenTextures(1, &colorRenderBuffer);
	glBindTexture(GL_TEXTURE_2D, colorRenderBuffer);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, OGL_WINDOW_WIDTH, OGL_WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorRenderBuffer, 0);
	glBindTexture(GL_TEXTURE_2D, 0);

	//DEPTH BUFFER
	glGenRenderbuffers(1, &depthRenderBuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, depthRenderBuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, OGL_WINDOW_WIDTH, OGL_WINDOW_HEIGHT);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderBuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	bIsFBOInitialized = true;

	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glShadeModel(GL_SMOOTH);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	gPerspectiveProjectionMatrix = vmath::mat4::identity();;

	Resize(gWindowWidth, gWindowHeight);
}

//DISPLAY
void display(void)
{
	//code
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, gFbo);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

	DrawSunPicking(PickingShaderProgram, vaoSun);
	DrawMercuryPicking(PickingShaderProgram, vaoMercury);
	DrawVenusPicking(PickingShaderProgram, vaoVenus);

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

	////Render to texture-----------------------------------------------------------------------------------------
	//glClear(GL_COLOR_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	//glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	//glUseProgram(TextureProgramQuad.GetShaderProgramObject());

	//glEnable(GL_TEXTURE_2D);

	//glActiveTexture(GL_TEXTURE0);
	//glBindTexture(GL_TEXTURE_2D, colorRenderBuffer);
	//glUniform1i(TextureProgramQuad.GetTextureSamplerUniform(), 0);

	//glBindVertexArray(vaoQuad.GetVAO());
	//glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	//glBindVertexArray(0);

	//glUseProgram(0);

	if (true == bIsMouseButtonPressed)
	{
		GLint viewport[4] = { 0 };
		unsigned char data[4] = { 0 };
		glGetIntegerv(GL_VIEWPORT, viewport);

		glBindFramebuffer(GL_READ_FRAMEBUFFER, gFbo);
		glReadBuffer(GL_COLOR_ATTACHMENT0);

		//glReadPixels(gMouseX, viewport[3] - gMouseY, 1, 1, GL_RGBA, GL_FLOAT, &data);
		//fprintf(gpFile, "x: %d, y: %d, z: %d, MouseX : %f, MouseY : %f, height: %d\n", data.fObjectIndex, data.fY, data.fZ, gMouseX, gMouseY, viewport[3]);
		glReadPixels(gMouseX, viewport[3] - gMouseY, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, &data);
		glReadBuffer(GL_NONE);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

		fprintf(gpFile, "x: %d, y: %d, z: %d, MouseX : %f, MouseY : %f\n", data[0], data[1], data[2], gMouseX, gMouseY);
		switch (data[0])
		{
		case 0:
			SetWindowText(ghwnd, TEXT("nothing picked"));
			MessageBox(ghwnd, L"nothing picked", L"message", MB_OK);
			break;
		case 1:
			SetWindowText(ghwnd, TEXT("SUN picked"));
			MessageBox(ghwnd, L"SUN picked", L"message", MB_OK);
			break;
		case 2:
			SetWindowText(ghwnd, TEXT("MERCURY picked"));
			MessageBox(ghwnd, L"MERCURY picked", L"message", MB_OK);
			break;
		case 3:
			SetWindowText(ghwnd, TEXT("VENUS picked"));
			MessageBox(ghwnd, L"VENUS picked", L"message", MB_OK);
			break;
		case 4:
			SetWindowText(ghwnd, TEXT("EARTH picked"));
			MessageBox(ghwnd, L"EARTH picked", L"message", MB_OK);
			break;
		default:
			fprintf(gpFile, "Res: %d\n", data[0]);
			break;
		}
		bIsMouseButtonPressed = false;
	}
	
	DrawSun(TextureShaderProgram, vaoSunTexture);
	DrawMercury(TextureShaderProgram, vaoMercuryTexture);
	DrawVenus(TextureShaderProgram, vaoVenusTexture);

	SwapBuffers(ghdc);
}

void update(void)
{
	return;
}

//resize
void Resize(int width, int height)
{
	if (height == 0)
		height = 1;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	if (bIsFBOInitialized)
	{
		glBindTexture(GL_TEXTURE_2D, colorRenderBuffer);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

		// create depth buffer (renderbuffer)
		glBindRenderbuffer(GL_RENDERBUFFER, depthRenderBuffer);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);

		// attach buffers
		glBindFramebuffer(GL_FRAMEBUFFER, gFbo);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorRenderBuffer, 0);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderBuffer);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	gPerspectiveProjectionMatrix = vmath::perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}

//uninitialize
void uninitialize(void)
{
	//code
	if (gbFullscreen == true)
	{
		dwStyle = GetWindowLong(ghwnd, GWL_STYLE);
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);

		ShowCursor(TRUE);
	}

	vaoSun.DeInit();
	vaoMercury.DeInit();
	vaoVenus.DeInit();

	vaoSunTexture.DeInit();
	vaoMercuryTexture.DeInit();
	vaoVenusTexture.DeInit();

	PickingShaderProgram.DeInit();
	TextureShaderProgram.DeInit();

	//unlink shader program
	glUseProgram(0);

	wglMakeCurrent(NULL, NULL);

	wglDeleteContext(ghrc);
	ghrc = NULL;

	ReleaseDC(ghwnd, ghdc);
	ghdc = NULL;

	if (gpFile)
	{
		fprintf(gpFile, "Log file closed successfully\n");
		fclose(gpFile);
		gpFile = NULL;
	}
}
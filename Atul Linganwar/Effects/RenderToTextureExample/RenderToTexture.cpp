#include<Windows.h>
#include<stdio.h>
#include<stdlib.h>

#include<gl/glew.h>
#include<gl/GL.h>

#include "../common/vmath.h"
#include "VertexData.h"
#include "VAOs.h"
#include "ShaderProgram.h"
#include "RenderToTexture.h"

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

GLuint gFbo;
GLuint colorRenderBuffer;
GLuint depthRenderBuffer;

VAO vaoPyramid(VAO_TYPE::VAO_COLOR, SHAPE::PYRAMID);
VAO vaoCube(VAO_TYPE::VAO_COLOR, SHAPE::CUBE);
VAO vaoPyramidTexture(VAO_TYPE::VAO_TEXTURE, SHAPE::PYRAMID);
VAO vaoCubeTexture(VAO_TYPE::VAO_TEXTURE, SHAPE::CUBE);
VAO vaoQuad(VAO_TYPE::VAO_TEXTURE, SHAPE::QUAD);

//ShaderProgram ColorProgram(SHADER_TYPE::COLOR_SHADER);
ShaderProgram TextureProgram(SHADER_TYPE::TEXTURE_SHADER);
ShaderProgram TextureProgramQuad(SHADER_TYPE::QUAD_TEXTURE_SHADER);

GLfloat anglePyramid = 0.0f;
GLfloat angleCube = 0.0f;

GLfloat gWindowWidth = 0.0f;
GLfloat gWindowHeight = 0.0f;

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

	//variable declaration
	static WORD xMouse = NULL;
	static WORD yMouse = NULL;

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
		resize(LOWORD(lParam), HIWORD(lParam));
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
		ShowCursor(FALSE);
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

	/*bRet = ColorProgram.Init();
	if (false == bRet)
	{
		fprintf(gpFile, "Error while initializing ColorProgram\n");
		uninitialize();
		ExitProcess(EXIT_FAILURE);
	}*/

	bRet = vaoPyramid.Init();
	if (false == bRet)
	{
		fprintf(gpFile, "Error while initializing vaoPyramid\n");
		uninitialize();
		ExitProcess(EXIT_FAILURE);
	}

	bRet = vaoCube.Init();
	if (false == bRet)
	{
		fprintf(gpFile, "Error while initializing vaoCube\n");
		uninitialize();
		ExitProcess(EXIT_FAILURE);
	}

	bRet = vaoPyramidTexture.Init();
	if (false == bRet)
	{
		fprintf(gpFile, "Error while initializing vaoPyramidTexture\n");
		uninitialize();
		ExitProcess(EXIT_FAILURE);
	}

	bRet = vaoCubeTexture.Init();
	if (false == bRet)
	{
		fprintf(gpFile, "Error while initializing vaoCubeTexture\n");
		uninitialize();
		ExitProcess(EXIT_FAILURE);
	}

	bRet = TextureProgram.Init();
	if (false == bRet)
	{
		fprintf(gpFile, "Error while initializing QuadTextureProgram\n");
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

	/*glGenBuffers(1, &gFbo);
	glBindFramebuffer(GL_FRAMEBUFFER, gFbo);

	glGenTextures(1, &colorRenderBuffer);
	glBindTexture(GL_TEXTURE_2D, colorRenderBuffer);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, OGL_WINDOW_WIDTH, OGL_WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorRenderBuffer, 0);
	glBindTexture(GL_TEXTURE_2D, 0);

	glGenTextures(1, &depthRenderBuffer);
	glBindTexture(GL_TEXTURE_2D, depthRenderBuffer);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, OGL_WINDOW_WIDTH, OGL_WINDOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthRenderBuffer, 0);

	glReadBuffer(GL_NONE);
	glDrawBuffer(GL_COLOR_ATTACHMENT0);

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);*/

	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glShadeModel(GL_SMOOTH);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	gPerspectiveProjectionMatrix = vmath::mat4::identity();;

	resize(OGL_WINDOW_WIDTH, OGL_WINDOW_HEIGHT);
}

//DISPLAY
void display(void)
{
	//code
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, gFbo);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	glUseProgram(TextureProgram.GetShaderProgramObject());

	//TRIANGLE
	vmath::mat4 modelViewMatrix = vmath::mat4(1.0f);
	vmath::mat4 modelViewProjectionMatrix = vmath::mat4(1.0f);

	modelViewMatrix = vmath::translate(-2.5f, 0.0f, -10.0f);

	vmath::mat4 rotationMatrix = vmath::mat4(1.0f);
	rotationMatrix = vmath::rotate(anglePyramid, 0.0f, 1.0f, 0.0f);
	modelViewMatrix = modelViewMatrix * rotationMatrix;

	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(TextureProgram.GetMVPUniform(), 1, GL_FALSE, modelViewProjectionMatrix);

	glViewport(0, 0, (GLsizei)OGL_WINDOW_WIDTH, (GLsizei)OGL_WINDOW_HEIGHT);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, TextureProgram.GetTextureStone());
	glUniform1i(TextureProgram.GetTextureSamplerUniform(), 0);

	glBindVertexArray(vaoPyramidTexture.GetVAO());
	glDrawArrays(GL_TRIANGLES, 0, 12);
	glBindVertexArray(0);

	//SQUARE
	modelViewMatrix = vmath::mat4(1.0f);
	modelViewProjectionMatrix = vmath::mat4(1.0f);

	modelViewMatrix = vmath::translate(2.5f, 0.0f, -10.0f);

	rotationMatrix = vmath::mat4(1.0f);
	rotationMatrix = vmath::rotate(angleCube, 1.0f, 0.0f, 0.0f);
	modelViewMatrix = modelViewMatrix * rotationMatrix;
	rotationMatrix = vmath::rotate(angleCube, 0.0f, 1.0f, 0.0f);
	modelViewMatrix = modelViewMatrix * rotationMatrix;
	rotationMatrix = vmath::rotate(angleCube, 0.0f, 0.0f, 1.0f);
	modelViewMatrix = modelViewMatrix * rotationMatrix;

	vmath::mat4 scaleMatrix = vmath::mat4(1.0f);
	scaleMatrix = vmath::scale(0.75f, 0.75f, 0.75f);
	modelViewMatrix = modelViewMatrix * scaleMatrix;

	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(TextureProgram.GetMVPUniform(), 1, GL_FALSE, modelViewProjectionMatrix);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, TextureProgram.GetTextureKundali());
	glUniform1i(TextureProgram.GetTextureSamplerUniform(), 0);

	glBindVertexArray(vaoCubeTexture.GetVAO());
	
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 16, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 20, 4);

	glBindVertexArray(0);

	glUseProgram(0);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	//Render to texture-----------------------------------------------------------------------------------------
	glClear(GL_COLOR_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	glUseProgram(TextureProgramQuad.GetShaderProgramObject());

	glViewport(0, 0, (GLsizei)gWindowWidth, (GLsizei)gWindowHeight);

	glEnable(GL_TEXTURE_2D);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, colorRenderBuffer);
	glUniform1i(TextureProgramQuad.GetTextureSamplerUniform(), 0);

	glBindVertexArray(vaoQuad.GetVAO());
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glBindVertexArray(0);

	glUseProgram(0);

	SwapBuffers(ghdc);
}

void update(void)
{
	anglePyramid = anglePyramid + 0.03f;
	if (anglePyramid >= 360.0f)
		anglePyramid = 0.0f;

	angleCube = angleCube + 0.03f;
	if (angleCube >= 360.0f)
		angleCube = 0.0f;
}

//resize
void resize(int width, int height)
{
	if (height == 0)
		height = 1;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

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

	vaoPyramid.DeInit();
	vaoCube.DeInit();
	vaoQuad.DeInit();
	vaoPyramidTexture.DeInit();
	vaoCubeTexture.DeInit();

	TextureProgram.DeInit();
	TextureProgramQuad.DeInit();

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
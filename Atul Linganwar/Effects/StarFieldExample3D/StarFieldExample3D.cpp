#include <Windows.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include "../common/vmath.h"
#include "../common/Common.h"
#include "../common/Shapes.h"
#include "Header.h"
#include "ObjectTransformations.h"
#include "Objects.h"
#include "BasicPlanetShader.h"
#include "BasicTextureShader.h"
#include "Basic3DTextureShader.h"
#include "BasicQuadRTTShader.h"
#include "StarField.h"
#include "SphereMap.h"
#include "StarFieldExample3D.h"

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "GeometricShapes.lib")

HDC ghDC = NULL;
HWND ghWnd = NULL;
HGLRC ghRC = NULL;

FILE* gpFile = NULL;

bool gbIsActivate = false;
bool gbIsFullScreen = false;
bool gbIsEscKeyPressed = false;

DWORD dwStyle = 0;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };

vmath::mat4 gPerspectiveProjectionMatrix;

BASIC_TEXTURE_SHADER BasicTextureShader;
BASIC_3DTEXTURE_SHADER Basic3DTextureShader;
BASIC_QUAD_RTT_TEXTURE_SHADER BasicQuadRTTTextureShader;

STAR_FIELD gStarField;
SPHERE_MAP gSphereMap;

GLuint guiTexture;

GLuint guiVAO;
GLuint guiVBOPosition;
GLuint guiVBOTexture;

PLANETS SphereSun;
BASIC_PLANET_SHADER BasicPlanetShader;

GLfloat gfSunAngle = 0.0f;


// WinMain
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	MSG msg = { 0 };
	HWND hWnd = NULL;
	bool bDone = false;
	WNDCLASSEX WndClass = { 0 };
	WCHAR wszClassName[] = L"3D Texture";

	fopen_s(&gpFile, "OGLLog.txt", "w");
	if (NULL == gpFile)
	{
		MessageBox(NULL, L"Error while creating log file", L"Error", MB_OK);
		return EXIT_FAILURE;
	}

	// WndClass
	WndClass.cbSize = sizeof(WNDCLASSEX);
	WndClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	WndClass.lpfnWndProc = WndProc;
	WndClass.lpszClassName = wszClassName;
	WndClass.lpszMenuName = NULL;
	WndClass.hInstance = hInstance;
	WndClass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	WndClass.hIcon = (HICON)LoadIcon(NULL, IDI_APPLICATION);
	WndClass.hIconSm = (HICON)LoadIcon(NULL, IDI_APPLICATION);
	WndClass.hCursor = (HCURSOR)LoadCursor(NULL, IDC_ARROW);
	WndClass.cbWndExtra = 0;
	WndClass.cbClsExtra = 0;


	// Register WndClass
	if (!RegisterClassEx(&WndClass))
	{
		fprintf(gpFile, "Error while registering WndClass.\n");
		fclose(gpFile);
		gpFile = NULL;
		return EXIT_FAILURE;
	}

	int iScreenWidth = GetSystemMetrics(SM_CXSCREEN);
	int iScreenHeight = GetSystemMetrics(SM_CYSCREEN);

	// Create Window
	hWnd = CreateWindowEx(
		WS_EX_APPWINDOW,
		wszClassName,
		wszClassName,
		WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
		(iScreenWidth / 2) - (OGL_WINDOW_WIDTH / 2),
		(iScreenHeight / 2) - (OGL_WINDOW_HEIGHT / 2),
		OGL_WINDOW_WIDTH,
		OGL_WINDOW_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL
	);
	if (NULL == hWnd)
	{
		fprintf(gpFile, "Error while creating window.\n");
		fclose(gpFile);
		gpFile = NULL;
		return EXIT_FAILURE;
	}

	ghWnd = hWnd;

	ShowWindow(hWnd, iCmdShow);
	SetForegroundWindow(hWnd);
	SetFocus(hWnd);

	bool bRet = Initialize();
	if (false == bRet)
	{
		fprintf(gpFile, "Error while Initialize().\n");
		fclose(gpFile);
		gpFile = NULL;
		DestroyWindow(hWnd);
		hWnd = NULL;
		return EXIT_FAILURE;
	}

	// Game loop
	while (bDone == false)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (WM_QUIT == msg.message)
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
			Display();
			Update();

			if (true == gbIsActivate)
			{
				if (true == gbIsEscKeyPressed)
				{
					bDone = true;
				}
			}
		}
	}

	// Uninitialize
	UnInitialize();

	return (int)msg.wParam;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	switch (iMsg)
	{
	case WM_ACTIVATE:
		if (HIWORD(lParam) == 0)
		{
			gbIsActivate = true;
		}
		else
		{
			gbIsActivate = false;
		}
		break;

	case WM_ERASEBKGND:
		return 0;

	case WM_SIZE:
		Resize(LOWORD(lParam), HIWORD(lParam));
		break;

	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			if (false == gbIsEscKeyPressed)
			{
				gbIsEscKeyPressed = true;
			}
			break;

		case 0x46:
			if (false == gbIsFullScreen)
			{
				ToggleFullScreen();
				gbIsFullScreen = true;
			}
			else
			{
				ToggleFullScreen();
				gbIsFullScreen = false;
			}
			break;

		default:
			break;
		}
		break;

	case WM_LBUTTONDOWN:
		break;

	case WM_CLOSE:
		UnInitialize();
		break;

	case WM_DESTROY:
		PostQuitMessage(0);
		break;

	default:
		break;
	}

	return DefWindowProc(hWnd, iMsg, wParam, lParam);
}

bool Initialize()
{
	int iPixelFormatIndex = 0;
	PIXELFORMATDESCRIPTOR pfd = { 0 };

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

	ghDC = GetDC(ghWnd);
	if (NULL == ghDC)
	{
		fprintf(gpFile, "Error while GetDC().\n");
		return false;
	}

	iPixelFormatIndex = ChoosePixelFormat(ghDC, &pfd);
	if (0 == iPixelFormatIndex)
	{
		fprintf(gpFile, "Error while ChoosePixelFormat().\n");
		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
		return false;
	}

	if (false == SetPixelFormat(ghDC, iPixelFormatIndex, &pfd))
	{
		fprintf(gpFile, "Error while SetPixelFormat().\n");
		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
		return false;
	}

	ghRC = wglCreateContext(ghDC);
	if (NULL == ghRC)
	{
		fprintf(gpFile, "Error while wglCreateContext().\n");
		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
		return false;
	}

	if (false == wglMakeCurrent(ghDC, ghRC))
	{
		fprintf(gpFile, "Error while wglMakeCurrent().\n");
		wglDeleteContext(ghRC);
		ghRC = NULL;
		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
		return false;
	}

	GLenum glError = glewInit();
	if (GLEW_OK != glError)
	{
		fprintf(gpFile, "Error while glewInit().\n");
		wglDeleteContext(ghRC);
		ghRC = NULL;
		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
		return false;
	}

	if (false == InitializeStarField(&gStarField))
	{
		fprintf(gpFile, "Error while InitializeStarField().\n");
		UnInitialize();
		return false;
	}

	if (false == InitializeBasicTextureShaderProgram(&BasicTextureShader))
	{
		fprintf(gpFile, "Error while InitializeBasicTextureShaderProgram().\n");
		UnInitialize();
		return false;
	}

	if (false == InitializeBasic3DTextureShaderProgram(&Basic3DTextureShader))
	{
		fprintf(gpFile, "Error while InitializeBasicTextureShaderProgram().\n");
		UnInitialize();
		return false;
	}

	if (false == InitializeBasicPlanetShaderProgram(&BasicPlanetShader))
	{
		fprintf(gpFile, "Error while InitializeBasicShaderProgram().\n");
		UnInitialize();
		return false;
	}

	if (false == InitializeBasicQuadRTTTextureShaderProgram(&BasicQuadRTTTextureShader))
	{
		fprintf(gpFile, "Error while InitializeBasicQuadRTTTextureShaderProgram().\n");
		UnInitialize();
		return false;
	}

	if (false == InitializeSphereMapData(100.0f, 200, 120, &gSphereMap))
	{
		fprintf(gpFile, "Error while InitializeSphereMapData().\n");
		UnInitialize();
		return false;
	}

	if (FALSE == InitializePlanetsData(10.0f, 300, 300, &SphereSun))
	{
		fprintf(gpFile, "Error while InitializePlanetsData(PlanetSun).\n");
		UnInitialize();
		return false;
	}

	BasicPlanetShader.uiModelMatrixUniform = glGetUniformLocation(BasicPlanetShader.ShaderObject.uiShaderProgramObject, "u_model_matrix");
	BasicPlanetShader.uiViewMatrixUniform = glGetUniformLocation(BasicPlanetShader.ShaderObject.uiShaderProgramObject, "u_view_matrix");
	BasicPlanetShader.uiProjectionMatrixUniform = glGetUniformLocation(BasicPlanetShader.ShaderObject.uiShaderProgramObject, "u_projection_matrix");
	BasicPlanetShader.uiTextureSamplerUniform = glGetUniformLocation(BasicPlanetShader.ShaderObject.uiShaderProgramObject, "u_texture0_sampler");

	const GLfloat quadVertices[] =
	{
		1.0f, 1.0f, 0.0f,
		-1.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f
	};

	const GLfloat quadColor[] =
	{
		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
	};

	const GLfloat quadTexCoords[] =
	{
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f
	};

	glGenVertexArrays(1, &guiVAO);
	glBindVertexArray(guiVAO);

	glGenBuffers(1, &guiVBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, guiVBOPosition);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &guiVBOTexture);
	glBindBuffer(GL_ARRAY_BUFFER, guiVBOTexture);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadTexCoords), quadTexCoords, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_TEXTURE0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);
	
	glShadeModel(GL_SMOOTH);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	glEnable(GL_CULL_FACE);
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_TEXTURE_3D);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	LoadGLTexture(&guiTexture, MAKEINTRESOURCE(ID_BITMAP_MARS));

	gPerspectiveProjectionMatrix = vmath::mat4::identity();

	Resize(OGL_WINDOW_WIDTH, OGL_WINDOW_HEIGHT);

	return true;
}

void ToggleFullScreen()
{
	MONITORINFO mi = { 0 };

	if (false == gbIsFullScreen)
	{
		dwStyle = GetWindowLong(ghWnd, GWL_STYLE);
		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{
			mi.cbSize = sizeof(MONITORINFO);
			if (
				GetWindowPlacement(ghWnd, &wpPrev) &&
				GetMonitorInfo(MonitorFromWindow(ghWnd, MONITORINFOF_PRIMARY), &mi)
				)
			{
				SetWindowLong(ghWnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(
					ghWnd,
					HWND_TOP,
					mi.rcMonitor.left,
					mi.rcMonitor.top,
					mi.rcMonitor.right - mi.rcMonitor.left,
					mi.rcMonitor.bottom - mi.rcMonitor.top,
					SWP_NOZORDER | SWP_FRAMECHANGED
				);
			}
		}
		ShowCursor(FALSE);
	}
	else
	{
		SetWindowLong(ghWnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghWnd, &wpPrev);
		SetWindowPos(
			ghWnd,
			HWND_TOP,
			0, 0, 0, 0,
			SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED
		);

		ShowCursor(TRUE);
	}
}

void Display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//DrawStarField(&gStarField);
	//DrawStarFieldToFrameBuffer(&gStarField);
	//DrawSphereMap(BasicTextureShader, &gSphereMap, gStarField.uiStarFieldFB2DTexture);

	//DrawStarFieldToFrameBuffer3D(&gStarField);
	//DrawSphereMap3D(Basic3DTextureShader, &gSphereMap, gStarField.uiStarFieldFB3DTexture, gStarField.uiStarFieldFB3DTextureDepth);

	//DrawStarFieldToFrameBuffer3D(&gStarField);
	//DrawSphereMap(BasicTextureShader, &gSphereMap, gStarField.uiStarFieldFB3DTexture);

	DrawStarFieldToFrameBuffer(&gStarField);
	DrawStarFieldToFrameBuffer3D(&gStarField);

	/*glClear(GL_COLOR_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	glUseProgram(BasicQuadRTTTextureShader.ShaderObject.uiShaderProgramObject);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, gStarField.uiStarFieldFB3DTexture);
	glUniform1i(BasicQuadRTTTextureShader.uiTextureSamplerUniform, 0);

	glBindVertexArray(guiVAO);
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glBindVertexArray(0);

	glUseProgram(0);*/

	glUseProgram(Basic3DTextureShader.ShaderObject.uiShaderProgramObject);

	vmath::mat4 viewMatrix = vmath::mat4::identity();
	vmath::mat4 modelMatrix = vmath::mat4::identity();
	vmath::mat4 translationMatrix = vmath::mat4::identity();
	vmath::mat4 rotationMatrix = vmath::mat4::identity();
	vmath::mat4 scaleMatrix = vmath::mat4::identity();

	viewMatrix = vmath::lookat(vmath::vec3(0.0f, 10.0f, 30.0f), vmath::vec3(0.0f, 0.0f, 0.0f), vmath::vec3(0.0f, 1.0f, 0.0f));

	//
	//	Sphere
	//
	rotationMatrix = vmath::rotate(-90.0f, 1.0f, 0.0f, 0.0f); // after this y becomes -z and z becomes y
	//rotationMatrix = rotationMatrix * vmath::rotate(gfSunAngle, 0.0f, 0.0f, 1.0f);

	modelMatrix = modelMatrix * rotationMatrix;

	glUniformMatrix4fv(Basic3DTextureShader.uiModelMatrixUniform, 1, GL_FALSE, modelMatrix);
	glUniformMatrix4fv(Basic3DTextureShader.uiViewMatrixUniform, 1, GL_FALSE, viewMatrix);
	glUniformMatrix4fv(Basic3DTextureShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, gStarField.uiStarFieldFB3DTexture);
	glUniform1i(Basic3DTextureShader.uiTextureSamplerUniform, 0);

	DrawPlanet(&SphereSun);

	glUseProgram(0);

	SwapBuffers(ghDC);
}

void Update(void)
{

	gfSunAngle = gfSunAngle + 0.4f;
	if (gfSunAngle > 360.0f)
	{
		gfSunAngle = 0.0f;
	}

	UpdateStarField();
}

void Resize(int iWidth, int iHeight)
{
	if (iHeight == 0)
	{
		iHeight = 1;
	}

	glViewport(0, 0, (GLsizei)iWidth, (GLsizei)iHeight);

	ResizeStarFieldFBO2D(&gStarField, iWidth, iHeight);
	ResizeStarFieldFBO3D(&gStarField, iWidth, iHeight, 128);

	gPerspectiveProjectionMatrix = vmath::perspective(45.0f, (GLfloat)iWidth / (GLfloat)iHeight, 0.1f, 10000.0f);
}

int LoadGLTexture(GLuint* texture, TCHAR imageResourceId[])
{
	//local variables
	HBITMAP hBitmap;
	BITMAP bmp;
	int iStatus = FALSE;

	//code
	glGenTextures(1, texture);
	hBitmap = (HBITMAP)LoadImage(GetModuleHandle(NULL), imageResourceId, IMAGE_BITMAP, 0, 0, LR_CREATEDIBSECTION);
	if (hBitmap)
	{
		iStatus = TRUE;
		GetObject(hBitmap, sizeof(bmp), &bmp);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glBindTexture(GL_TEXTURE_2D, *texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, bmp.bmWidth, bmp.bmHeight, 0, GL_BGR, GL_UNSIGNED_BYTE, bmp.bmBits);

		glGenerateMipmap(GL_TEXTURE_2D);
		
		glBindTexture(GL_TEXTURE_2D, 0);

		DeleteObject(hBitmap);
	}
	return(iStatus);
}

void UnInitialize()
{
	if (true == gbIsFullScreen)
	{
		dwStyle = GetWindowLong(ghWnd, GWL_STYLE);
		SetWindowLong(ghWnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghWnd, &wpPrev);
		SetWindowPos(ghWnd, HWND_TOP, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED | SWP_NOMOVE);

		ShowCursor(TRUE);
	}

	UnInitializeStarField(&gStarField);

	glUseProgram(0);

	wglMakeCurrent(NULL, NULL);

	wglDeleteContext(ghRC);
	ghRC = NULL;

	ReleaseDC(ghWnd, ghDC);
	ghDC = NULL;

	if (NULL != gpFile)
	{
		fprintf(gpFile, "Exitting successfully.\n");
		fclose(gpFile);
		gpFile = NULL;
	}

	return;
}
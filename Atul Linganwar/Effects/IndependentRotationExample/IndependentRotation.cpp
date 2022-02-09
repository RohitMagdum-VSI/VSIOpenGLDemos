#include <Windows.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include "../common/vmath.h"
#include "../common/Common.h"
#include "Header.h"
#include "../common/Shapes.h"
#include "ObjectTransformations.h"
#include "Objects.h"
#include "BasicPlanetShader.h"
#include "EllipseData.h"
#include "BasicColorShader.h"
#include "IndependentRotation.h"

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

GLuint guiVAO;
GLuint guiVBOPosition;
GLuint guiVBOColor;

PLANETS SphereSun;
PLANETS SphereMoon;
PLANETS SphereEarth;
BASIC_PLANET_SHADER BasicPlanetShader;

PELLIPSE_DATA gpEllipseData;
BASIC_COLOR_SHADER BasicColorShader;

GLfloat giIndex = 0.0f;
GLfloat gfSunAngle = 0.0f;
GLfloat gfMoonAngle = 0.0f;
GLfloat gfEarthAngle = 0.0f;
GLfloat gfMoonAngleTranslate = 0.0f;
GLfloat gfEarthAngleTranslate = 0.0f;

GLint giCirclePoints = 10000;
GLint giEllipsePoints = 10000;

GLfloat gfEarthTranslationX = 0.0f;
GLfloat gfEarthTranslationZ = 0.0f;

GLfloat gfMoonTranslationX = 0.0f;
GLfloat gfMoonTranslationZ = 0.0f;

GLfloat gfSunRadius = 3.0f;
GLfloat gfEarthRadius = 2.0f;
GLfloat gfMoonRadius = 1.0f;

// WinMain
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	MSG msg = { 0 };
	HWND hWnd = NULL;
	bool bDone = false;
	WNDCLASSEX WndClass = { 0 };
	WCHAR wszClassName[] = L"First Scene Rotating Planets";

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

	if (false == InitializeBasicColorShaderProgram(&BasicColorShader))
	{
		fprintf(gpFile, "Error while InitializeBasicShaderProgram().\n");
		UnInitialize();
		return false;
	}

	if (false == InitializeBasicPlanetShaderProgram(&BasicPlanetShader))
	{
		fprintf(gpFile, "Error while InitializeBasicShaderProgram().\n");
		UnInitialize();
		return false;
	}

	if (FALSE == InitializePlanetsData(gfSunRadius, 30, 30, &SphereSun))
	{
		fprintf(gpFile, "Error while InitializePlanetsData(PlanetSun).\n");
		UnInitialize();
		return false;
	}

	if (FALSE == InitializePlanetsData(gfMoonRadius, 30, 30, &SphereMoon))
	{
		fprintf(gpFile, "Error while InitializePlanetsData(PlanetMoon).\n");
		UnInitialize();
		return false;
	}

	if (FALSE == InitializePlanetsData(gfEarthRadius, 30, 30, &SphereEarth))
	{
		fprintf(gpFile, "Error while InitializePlanetsData(PlanetEarth).\n");
		UnInitialize();
		return false;
	}

	// get mvp uniform location
	BasicPlanetShader.uiModelMatrixUniform = glGetUniformLocation(BasicPlanetShader.ShaderObject.uiShaderProgramObject, "u_model_matrix");
	BasicPlanetShader.uiViewMatrixUniform = glGetUniformLocation(BasicPlanetShader.ShaderObject.uiShaderProgramObject, "u_view_matrix");
	BasicPlanetShader.uiProjectionMatrixUniform = glGetUniformLocation(BasicPlanetShader.ShaderObject.uiShaderProgramObject, "u_projection_matrix");
	BasicPlanetShader.uiTextureSamplerUniform = glGetUniformLocation(BasicPlanetShader.ShaderObject.uiShaderProgramObject, "u_texture0_sampler");

	gpEllipseData = GetEllipseData(100, 12, 6);
	if (NULL == gpEllipseData)
	{
		fprintf(gpFile, "Error while GetEllipseData()\n");
		UnInitialize();
		return false;
	}

	BasicColorShader.uiModelMatrixUniform = glGetUniformLocation(BasicColorShader.ShaderObject.uiShaderProgramObject, "u_model_matrix");
	BasicColorShader.uiViewMatrixUniform = glGetUniformLocation(BasicColorShader.ShaderObject.uiShaderProgramObject, "u_view_matrix");
	BasicColorShader.uiProjectionMatrixUniform = glGetUniformLocation(BasicColorShader.ShaderObject.uiShaderProgramObject, "u_projection_matrix");

	glGenVertexArrays(1, &guiVAO);
	glBindVertexArray(guiVAO);

	glGenBuffers(1, &guiVBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, guiVBOPosition);
	glBufferData(GL_ARRAY_BUFFER, gpEllipseData->iVerticesSize, (const void*)gpEllipseData->pfVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &guiVBOColor);
	glBindBuffer(GL_ARRAY_BUFFER, guiVBOColor);
	glBufferData(GL_ARRAY_BUFFER, gpEllipseData->iColorsSize, (const void*)gpEllipseData->pfColors, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, NULL, 0);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_COLOR);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	glShadeModel(GL_SMOOTH);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glEnable(GL_TEXTURE_2D);

	LoadGLTextures(&SphereSun.uiTexture, MAKEINTRESOURCE(ID_BITMAP_SUN));
	LoadGLTextures(&SphereMoon.uiTexture, MAKEINTRESOURCE(ID_BITMAP_MOON));
	LoadGLTextures(&SphereEarth.uiTexture, MAKEINTRESOURCE(ID_BITMAP_EARTH));

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	gPerspectiveProjectionMatrix = vmath::mat4::identity();

	Resize(OGL_WINDOW_WIDTH, OGL_WINDOW_HEIGHT);

	return true;
}

void Display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(BasicPlanetShader.ShaderObject.uiShaderProgramObject);

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
	rotationMatrix = rotationMatrix * vmath::rotate(gfSunAngle, 0.0f, 0.0f, 1.0f);

	modelMatrix = modelMatrix * rotationMatrix;

	glUniformMatrix4fv(BasicPlanetShader.uiModelMatrixUniform, 1, GL_FALSE, modelMatrix);
	glUniformMatrix4fv(BasicPlanetShader.uiViewMatrixUniform, 1, GL_FALSE, viewMatrix);
	glUniformMatrix4fv(BasicPlanetShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, SphereSun.uiTexture);
	glUniform1i(BasicPlanetShader.uiTextureSamplerUniform, 0);

	DrawPlanet(&SphereSun);

	//
	// Earth
	//
	modelMatrix = vmath::mat4::identity();
	translationMatrix = vmath::mat4::identity();
	rotationMatrix = vmath::mat4::identity();
	scaleMatrix = vmath::mat4::identity();

	gfEarthTranslationX = 12 * (GLfloat)cos(0.0f);
	gfEarthTranslationZ = 6 * (GLfloat)sin(0.0f);

	translationMatrix = vmath::translate(gfEarthTranslationX, 0.0f, -gfEarthTranslationZ);
	modelMatrix = modelMatrix * translationMatrix;

	rotationMatrix = vmath::rotate(-90.0f, 1.0f, 0.0f, 0.0f); // after this y becomes -z and z becomes y
	rotationMatrix = rotationMatrix * vmath::rotate(gfEarthAngle, 0.0f, 0.0f, 1.0f);
	modelMatrix = modelMatrix * rotationMatrix;

	glUniformMatrix4fv(BasicPlanetShader.uiModelMatrixUniform, 1, GL_FALSE, modelMatrix);
	glUniformMatrix4fv(BasicPlanetShader.uiViewMatrixUniform, 1, GL_FALSE, viewMatrix);
	glUniformMatrix4fv(BasicPlanetShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, SphereEarth.uiTexture);
	glUniform1i(BasicPlanetShader.uiTextureSamplerUniform, 0);

	DrawPlanet(&SphereEarth);

	//
	// Moon
	//
	modelMatrix = vmath::mat4::identity();
	translationMatrix = vmath::mat4::identity();
	rotationMatrix = vmath::mat4::identity();
	scaleMatrix = vmath::mat4::identity();

	translationMatrix = vmath::translate(gfEarthTranslationX, 0.0f, -gfEarthTranslationZ);
	modelMatrix = modelMatrix * translationMatrix;

	gfMoonTranslationX = (gfEarthRadius + gfMoonRadius + 0.5f) * (GLfloat)cos(gfMoonAngleTranslate);
	gfMoonTranslationZ = (gfEarthRadius + gfMoonRadius + 0.5f) * (GLfloat)sin(gfMoonAngleTranslate);

	translationMatrix = vmath::mat4::identity();
	translationMatrix = vmath::translate(gfMoonTranslationX, 0.0f, -gfMoonTranslationZ);
	modelMatrix = modelMatrix * translationMatrix;

	rotationMatrix = vmath::rotate(-90.0f, 1.0f, 0.0f, 0.0f);
	rotationMatrix = rotationMatrix * vmath::rotate(gfMoonAngle, 0.0f, 0.0f, 1.0f);
	modelMatrix = modelMatrix * rotationMatrix;

	glUniformMatrix4fv(BasicPlanetShader.uiModelMatrixUniform, 1, GL_FALSE, modelMatrix);
	glUniformMatrix4fv(BasicPlanetShader.uiViewMatrixUniform, 1, GL_FALSE, viewMatrix);
	glUniformMatrix4fv(BasicPlanetShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, SphereMoon.uiTexture);
	glUniform1i(BasicPlanetShader.uiTextureSamplerUniform, 0);

	DrawPlanet(&SphereMoon);

	glUseProgram(0);

	//
	// Ellipse
	//
	glUseProgram(BasicColorShader.ShaderObject.uiShaderProgramObject);

	modelMatrix = vmath::mat4::identity();
	translationMatrix = vmath::mat4::identity();
	rotationMatrix = vmath::mat4::identity();
	scaleMatrix = vmath::mat4::identity();

	glUniformMatrix4fv(BasicColorShader.uiModelMatrixUniform, 1, GL_FALSE, modelMatrix);
	glUniformMatrix4fv(BasicColorShader.uiViewMatrixUniform, 1, GL_FALSE, viewMatrix);
	glUniformMatrix4fv(BasicColorShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

	glBindVertexArray(guiVAO);

	glDrawArrays(GL_LINE_STRIP, 0, gpEllipseData->iVerticesCount);

	glBindVertexArray(0);

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

	gfMoonAngle = gfMoonAngle + 0.6f;
	if (gfMoonAngle > 360.0f)
	{
		gfMoonAngle = 0.0f;
	}

	gfEarthAngle = gfEarthAngle + 0.2f;
	if (gfEarthAngle > 360.0f)
	{
		gfEarthAngle = 0.0f;
	}

	giIndex = giIndex + 10.0f;
	if (giIndex >= 10000.0f)
	{
		giIndex = giIndex - 10000.0f;
	}
	gfMoonAngleTranslate = 2 * 3.1415 * giIndex / giCirclePoints;
	gfEarthAngleTranslate = 2 * 3.1415 * giIndex / giEllipsePoints;

	return;
}

int LoadGLTextures(GLuint* texture, TCHAR imageResourceId[])
{
	HBITMAP hBitmap;
	BITMAP bmp;
	int iStatus = FALSE;

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

		glTexImage2D(
			GL_TEXTURE_2D,
			0,
			GL_RGB,
			bmp.bmWidth,
			bmp.bmHeight,
			0,
			GL_BGR,
			GL_UNSIGNED_BYTE,
			bmp.bmBits
		);

		glGenerateMipmap(GL_TEXTURE_2D);

		DeleteObject(hBitmap);
	}

	return iStatus;
}

void Resize(int iWidth, int iHeight)
{
	if (iHeight == 0)
	{
		iHeight = 1;
	}

	glViewport(0, 0, (GLsizei)iWidth, (GLsizei)iHeight);

	gPerspectiveProjectionMatrix = vmath::perspective(45.0f, (GLfloat)iWidth / (GLfloat)iHeight, 0.1f, 100.0f);
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
	
	FreeEllipseData(&gpEllipseData);

	UnInitializeBasicColorShaderProgram(&BasicColorShader);
	UnInitializeBasicPlanetShaderProgram(&BasicPlanetShader);

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
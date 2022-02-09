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
#include "PickedPlanetViewportAnimation.h"
#include "PickingShader.h"
#include "FrameBuffers.h"
#include "PickedPlanetAnimation.h"

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

GLfloat gMouseX = 0.0f;
GLfloat gMouseY = 0.0f;
bool bIsMouseButtonPressed = false;

DWORD dwStyle = 0;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };

vmath::mat4 gPerspectiveProjectionMatrix;

PLANETS Planet;

FRAMEBUFFER_OBJECT gFBO;
PICKING_SHADER PickingShader;
BASIC_COLOR_SHADER BasicColorShader;
BASIC_PLANET_SHADER BasicPlanetShader;

GLuint guiTexturePlanets[TOTAL_PLANETS];

GLfloat giIndex = 0.0f;
GLfloat gfSunAngle = 0.0f;
GLfloat gfMoonAngle = 0.0f;
GLfloat gfEarthAngle = 0.0f;
GLfloat gfMoonAngleTranslate = 0.0f;
GLfloat gfEarthAngleTranslate = 0.0f;

GLfloat gWindowWidth = 0.0f;
GLfloat gWindowHeight = 0.0f;

GLint giCirclePoints = 10000;
GLint giEllipsePoints = 10000;

GLfloat gfTranslationX = 0.0f;
GLfloat gfTranslationZ = 0.0f;

GLfloat gfMoonTranslationX = 0.0f;
GLfloat gfMoonTranslationZ = 0.0f;

GLfloat gfSunRadius = 3.0f;
GLfloat gfEarthRadius = 2.0f;
GLfloat gfMoonRadius = 1.0f;

vmath::mat4 gmat4ViewMatrix = vmath::mat4::identity();
vmath::mat4 gmat4ModelMatrix = vmath::mat4::identity();
vmath::mat4 gmat4TranslationMatrix = vmath::mat4::identity();
vmath::mat4 gmat4RotationMatrix = vmath::mat4::identity();
vmath::mat4 gmat4ScaleMatrix = vmath::mat4::identity();

bool gbIsAnimationDone = true;
bool gbIsRightMouseButtonPressed = false;
PLANET_AND_MOONS gpPlanetPick = NONE;

void ClearMatrices();

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
		gWindowWidth = LOWORD(lParam);
		gWindowHeight = HIWORD(lParam);
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

		case VK_RETURN:
			gbIsRightMouseButtonPressed = true;
			break;

		default:
			break;
		}
		break;

	case WM_LBUTTONDOWN:
		if (true == gbIsAnimationDone)
		{
			gMouseX = LOWORD(lParam);
			gMouseY = HIWORD(lParam);
			bIsMouseButtonPressed = true;
		}
		else
		{
			bIsMouseButtonPressed = false;
		}
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

	if (false == InitializePickingShaderProgram(&PickingShader))
	{
		fprintf(gpFile, "Error while InitializePickingShaderProgram().\n");
		UnInitialize();
		return false;
	}

	if (FALSE == InitializePlanetsData(gfSunRadius, 30, 30, &Planet))
	{
		fprintf(gpFile, "Error while InitializePlanetsData().\n");
		UnInitialize();
		return false;
	}

	if (false == InitializeAllPlanetsOrbitPath())
	{
		fprintf(gpFile, "Error while InitializeAllPlanetsOrbitPath().\n");
		UnInitialize();
		return false;
	}

	if (false == InitializeFrameBuffer(&gFBO))
	{
		fprintf(gpFile, "Error while InitializeFrameBuffer()\n");
		UnInitialize();
		return false;
	}

	glShadeModel(GL_SMOOTH);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glEnable(GL_TEXTURE_2D);

	LoadGLTextures(&guiTexturePlanets[PLANET_AND_MOONS::SUN], MAKEINTRESOURCE(ID_BITMAP_SUN));
	LoadGLTextures(&guiTexturePlanets[PLANET_AND_MOONS::MERCURY], MAKEINTRESOURCE(ID_BITMAP_MERCURY));
	LoadGLTextures(&guiTexturePlanets[PLANET_AND_MOONS::VENUS], MAKEINTRESOURCE(ID_BITMAP_VENUS));
	LoadGLTextures(&guiTexturePlanets[PLANET_AND_MOONS::EARTH], MAKEINTRESOURCE(ID_BITMAP_EARTH));
	LoadGLTextures(&guiTexturePlanets[PLANET_AND_MOONS::EARTH_MOON], MAKEINTRESOURCE(ID_BITMAP_EARTH_MOON));
	LoadGLTextures(&guiTexturePlanets[PLANET_AND_MOONS::MARS], MAKEINTRESOURCE(ID_BITMAP_MARS));
	//LoadGLTextures(&guiTexturePlanets[PLANET_AND_MOONS::MARS_PHOBOS], MAKEINTRESOURCE(ID_BITMAP_MARS_PHOBOS));
	LoadGLTextures(&guiTexturePlanets[PLANET_AND_MOONS::JUPITER], MAKEINTRESOURCE(ID_BITMAP_JUPITER));
	//LoadGLTextures(&guiTexturePlanets[PLANET_AND_MOONS::JUPITER_EUROPA], MAKEINTRESOURCE(ID_BITMAP_JUPITER_EUROPA));
	LoadGLTextures(&guiTexturePlanets[PLANET_AND_MOONS::SATURN], MAKEINTRESOURCE(ID_BITMAP_SATURN));
	//LoadGLTextures(&guiTexturePlanets[PLANET_AND_MOONS::SATURN_TITAN], MAKEINTRESOURCE(ID_BITMAP_SATURN_TITAN));
	LoadGLTextures(&guiTexturePlanets[PLANET_AND_MOONS::URANUS], MAKEINTRESOURCE(ID_BITMAP_URANUS));
	//LoadGLTextures(&guiTexturePlanets[PLANET_AND_MOONS::URANUS_AERIAL], MAKEINTRESOURCE(ID_BITMAP_URANUS_AERIAL));
	LoadGLTextures(&guiTexturePlanets[PLANET_AND_MOONS::NEPTUNE], MAKEINTRESOURCE(ID_BITMAP_NEPTUNE));
	//LoadGLTextures(&guiTexturePlanets[PLANET_AND_MOONS::NEPTUNE_TRITON], MAKEINTRESOURCE(ID_BITMAP_NEPTUNE_TRITON));
	LoadGLTextures(&guiTexturePlanets[PLANET_AND_MOONS::PLUTO], MAKEINTRESOURCE(ID_BITMAP_PLUTO));

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	gPerspectiveProjectionMatrix = vmath::mat4::identity();

	Resize(gWindowWidth, gWindowHeight);

	return true;
}

void Display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	gmat4ViewMatrix = vmath::lookat(vmath::vec3(0.0f, 40.0f, 170.0f), vmath::vec3(0.0f, 0.0f, 0.0f), vmath::vec3(0.0f, 1.0f, 0.0f));
	
	if (true == gbIsAnimationDone)
	{
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, gFBO.uiFBO);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		DrawPlanetsForPicking();

		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		gpPlanetPick = ProcessForPicking();

		if (NONE != gpPlanetPick)
		{
			gbIsAnimationDone = false;
		}
	}

	if (gpPlanetPick != NONE && false == gbIsAnimationDone)
	{
		DrawPlanetPickAnimation(gpPlanetPick);
	}

	DrawPlanetsForRendering();
	
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

	giIndex = giIndex + 3.0f;
	if (giIndex >= 10000.0f)
	{
		giIndex = giIndex - 10000.0f;
	}

	gfMoonAngleTranslate = 2 * (GLfloat)3.1415 * giIndex / giCirclePoints;
	gfEarthAngleTranslate = 2 * (GLfloat)3.1415 * giIndex / giEllipsePoints;

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

	ResizeFrameBuffer(iWidth, iHeight, &gFBO);

	gPerspectiveProjectionMatrix = vmath::perspective(45.0f, (GLfloat)iWidth / (GLfloat)iHeight, 0.1f, 1000.0f);
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
		// ShowCursor(FALSE);
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
	
	UnInitializeAllPlanetsOrbitPath();

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

void DrawPlanetsForPicking()
{
	glUseProgram(PickingShader.ShaderObject.uiShaderProgramObject);

	ClearMatrices();

	//
	//	Sun
	//
	gmat4RotationMatrix = vmath::rotate(-90.0f, 1.0f, 0.0f, 0.0f); // after this y becomes -z and z becomes y
	gmat4RotationMatrix = gmat4RotationMatrix * vmath::rotate(gfSunAngle, 0.0f, 0.0f, 1.0f);

	gmat4ModelMatrix = gmat4ModelMatrix * gmat4RotationMatrix;

	gmat4ScaleMatrix = vmath::scale(SCALE_FACTOR_SUN);
	gmat4ModelMatrix = gmat4ModelMatrix * gmat4ScaleMatrix;

	glUniformMatrix4fv(PickingShader.uiModelMatrixUniform, 1, GL_FALSE, gmat4ModelMatrix);
	glUniformMatrix4fv(PickingShader.uiViewMatrixUniform, 1, GL_FALSE, gmat4ViewMatrix);
	glUniformMatrix4fv(PickingShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);
	glUniform1i(PickingShader.uiObjectIdUniform, (GLint)PLANET_AND_MOONS::SUN);

	DrawPlanetPicking(&Planet);

	//
	// Mercury
	//
	DrawPlanetAtOrbitPicking(gpEllipsePathMercury, SCALE_FACTOR_MERCURY, (GLint)PLANET_AND_MOONS::MERCURY);

	//
	// Venus
	//
	DrawPlanetAtOrbitPicking(gpEllipsePathVenus, SCALE_FACTOR_VENUS, (GLint)PLANET_AND_MOONS::VENUS);

	//
	// Earth
	//
	DrawPlanetAtOrbitPicking(gpEllipsePathEarth, SCALE_FACTOR_EARTH, (GLint)PLANET_AND_MOONS::EARTH);

	//
	// Mars
	//
	DrawPlanetAtOrbitPicking(gpEllipsePathMars, SCALE_FACTOR_MARS, (GLint)PLANET_AND_MOONS::MARS);

	// 
	// Jupiter
	//
	DrawPlanetAtOrbitPicking(gpEllipsePathJupiter, SCALE_FACTOR_JUPITER, (GLint)PLANET_AND_MOONS::JUPITER);

	//
	// Saturn
	//
	DrawPlanetAtOrbitPicking(gpEllipsePathSaturn, SCALE_FACTOR_SATURN, (GLint)PLANET_AND_MOONS::SATURN);

	//
	// Uranus
	//
	DrawPlanetAtOrbitPicking(gpEllipsePathUranus, SCALE_FACTOR_URANUS, (GLint)PLANET_AND_MOONS::URANUS);

	//
	// Neptune
	//
	DrawPlanetAtOrbitPicking(gpEllipsePathNeptune, SCALE_FACTOR_NEPTUNE, (GLint)PLANET_AND_MOONS::NEPTUNE);

	//
	// Pluto
	//
	DrawPlanetAtOrbitPicking(gpEllipsePathPluto, SCALE_FACTOR_PLUTO, (GLint)PLANET_AND_MOONS::PLUTO);

	glUseProgram(0);
}

void DrawPlanetsForRendering()
{
	glUseProgram(BasicPlanetShader.ShaderObject.uiShaderProgramObject);

	ClearMatrices();

	//
	//	Sun
	//
	gmat4RotationMatrix = vmath::rotate(-90.0f, 1.0f, 0.0f, 0.0f); // after this y becomes -z and z becomes y
	gmat4RotationMatrix = gmat4RotationMatrix * vmath::rotate(gfSunAngle, 0.0f, 0.0f, 1.0f);

	gmat4ModelMatrix = gmat4ModelMatrix * gmat4RotationMatrix;

	gmat4ScaleMatrix = vmath::scale(SCALE_FACTOR_SUN);
	gmat4ModelMatrix = gmat4ModelMatrix * gmat4ScaleMatrix;

	glUniformMatrix4fv(BasicPlanetShader.uiModelMatrixUniform, 1, GL_FALSE, gmat4ModelMatrix);
	glUniformMatrix4fv(BasicPlanetShader.uiViewMatrixUniform, 1, GL_FALSE, gmat4ViewMatrix);
	glUniformMatrix4fv(BasicPlanetShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, guiTexturePlanets[PLANET_AND_MOONS::SUN]);
	glUniform1i(BasicPlanetShader.uiTextureSamplerUniform, 0);

	DrawPlanet(&Planet);

	//
	// Mercury
	//
	DrawPlanetAtOrbitRendering(gpEllipsePathMercury, SCALE_FACTOR_MERCURY, guiTexturePlanets[PLANET_AND_MOONS::MERCURY]);

	//
	// Venus
	//
	DrawPlanetAtOrbitRendering(gpEllipsePathVenus, SCALE_FACTOR_VENUS, guiTexturePlanets[PLANET_AND_MOONS::VENUS]);

	//
	// Earth
	//
	DrawPlanetAtOrbitRendering(gpEllipsePathEarth, SCALE_FACTOR_EARTH, guiTexturePlanets[PLANET_AND_MOONS::EARTH]);

	//
	// Mars
	//
	DrawPlanetAtOrbitRendering(gpEllipsePathMars, SCALE_FACTOR_MARS, guiTexturePlanets[PLANET_AND_MOONS::MARS]);

	// 
	// Jupiter
	//
	DrawPlanetAtOrbitRendering(gpEllipsePathJupiter, SCALE_FACTOR_JUPITER, guiTexturePlanets[PLANET_AND_MOONS::JUPITER]);

	//
	// Saturn
	//
	DrawPlanetAtOrbitRendering(gpEllipsePathSaturn, SCALE_FACTOR_SATURN, guiTexturePlanets[PLANET_AND_MOONS::SATURN]);

	//
	// Uranus
	//
	DrawPlanetAtOrbitRendering(gpEllipsePathUranus, SCALE_FACTOR_URANUS, guiTexturePlanets[PLANET_AND_MOONS::URANUS]);

	//
	// Neptune
	//
	DrawPlanetAtOrbitRendering(gpEllipsePathNeptune, SCALE_FACTOR_NEPTUNE, guiTexturePlanets[PLANET_AND_MOONS::NEPTUNE]);

	//
	// Pluto
	//
	DrawPlanetAtOrbitRendering(gpEllipsePathPluto, SCALE_FACTOR_PLUTO, guiTexturePlanets[PLANET_AND_MOONS::PLUTO]);
	
	glUseProgram(0);

	//
	// Ellipse
	//
	glUseProgram(BasicColorShader.ShaderObject.uiShaderProgramObject);

	// Mercury
	ClearMatrices();

	glUniformMatrix4fv(BasicColorShader.uiModelMatrixUniform, 1, GL_FALSE, gmat4ModelMatrix);
	glUniformMatrix4fv(BasicColorShader.uiViewMatrixUniform, 1, GL_FALSE, gmat4ViewMatrix);
	glUniformMatrix4fv(BasicColorShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

	DrawEllipsePath(gpEllipsePathMercury);

	// Venus
	DrawEllipsePath(gpEllipsePathVenus);

	// Earth
	DrawEllipsePath(gpEllipsePathEarth);

	// Mars
	DrawEllipsePath(gpEllipsePathMars);

	// Jupiter
	DrawEllipsePath(gpEllipsePathJupiter);

	// Saturn
	DrawEllipsePath(gpEllipsePathSaturn);

	// Uranus
	DrawEllipsePath(gpEllipsePathUranus);

	// Neptune
	DrawEllipsePath(gpEllipsePathNeptune);

	// Pluto
	DrawEllipsePath(gpEllipsePathPluto);

	glUseProgram(0);
}

PLANET_AND_MOONS ProcessForPicking()
{
	PLANET_AND_MOONS Pick = PLANET_AND_MOONS::NONE;

	if (true == bIsMouseButtonPressed)
	{
		GLint viewport[4] = { 0 };
		unsigned char data[4] = { 0 };
		glGetIntegerv(GL_VIEWPORT, viewport);

		glBindFramebuffer(GL_READ_FRAMEBUFFER, gFBO.uiFBO);
		glReadBuffer(GL_COLOR_ATTACHMENT0);
		glReadPixels(gMouseX, viewport[3] - gMouseY, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, &data);
		glReadBuffer(GL_NONE);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

		switch (data[0])
		{
		case PLANET_AND_MOONS::SUN:
			Pick = PLANET_AND_MOONS::SUN;
			break;
		case PLANET_AND_MOONS::MERCURY:
			Pick = PLANET_AND_MOONS::MERCURY;
			break;
		case PLANET_AND_MOONS::VENUS:
			Pick = PLANET_AND_MOONS::VENUS;
			break;
		case PLANET_AND_MOONS::EARTH:
			Pick = PLANET_AND_MOONS::EARTH;
			break;
		case PLANET_AND_MOONS::MARS:
			Pick = PLANET_AND_MOONS::MARS;
			break;
		case PLANET_AND_MOONS::JUPITER:
			Pick = PLANET_AND_MOONS::JUPITER;
			break;
		case PLANET_AND_MOONS::SATURN:
			Pick = PLANET_AND_MOONS::SATURN;
			break;
		case PLANET_AND_MOONS::URANUS:
			Pick = PLANET_AND_MOONS::URANUS;
			break;
		case PLANET_AND_MOONS::NEPTUNE:
			Pick = PLANET_AND_MOONS::NEPTUNE;
			break;
		case PLANET_AND_MOONS::PLUTO:
			Pick = PLANET_AND_MOONS::PLUTO;
			break;
		default:
			break;
		}
		bIsMouseButtonPressed = false;
	}

	return Pick;
}

void DrawPlanetAtOrbitPicking(PELLIPTICAL_PATH pEllipsePath, GLfloat fScaleFactor, GLint iObjectId)
{
	ClearMatrices();

	gfTranslationX = pEllipsePath->EllipseData.fSemiMajorAxis * (GLfloat)cos(gfEarthAngleTranslate);
	gfTranslationZ = pEllipsePath->EllipseData.fSemiMinorAxis * (GLfloat)sin(gfEarthAngleTranslate);

	gmat4TranslationMatrix = vmath::translate(gfTranslationX, 0.0f, -gfTranslationZ);
	gmat4ModelMatrix = gmat4ModelMatrix * gmat4TranslationMatrix;

	gmat4RotationMatrix = vmath::rotate(-90.0f, 1.0f, 0.0f, 0.0f); // after this y becomes -z and z becomes y
	gmat4RotationMatrix = gmat4RotationMatrix * vmath::rotate(gfEarthAngle, 0.0f, 0.0f, 1.0f);
	gmat4ModelMatrix = gmat4ModelMatrix * gmat4RotationMatrix;

	gmat4ScaleMatrix = vmath::scale(fScaleFactor);
	gmat4ModelMatrix = gmat4ModelMatrix * gmat4ScaleMatrix; 

	glUniformMatrix4fv(PickingShader.uiModelMatrixUniform, 1, GL_FALSE, gmat4ModelMatrix);
	glUniformMatrix4fv(PickingShader.uiViewMatrixUniform, 1, GL_FALSE, gmat4ViewMatrix);
	glUniformMatrix4fv(PickingShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);
	glUniform1i(PickingShader.uiObjectIdUniform, iObjectId);

	DrawPlanetPicking(&Planet);
}

void DrawPlanetAtOrbitRendering(PELLIPTICAL_PATH pEllipsePath, GLfloat fScaleFactor, GLuint uiTexture)
{
	ClearMatrices();

	gfTranslationX = pEllipsePath->EllipseData.fSemiMajorAxis * (GLfloat)cos(gfEarthAngleTranslate);
	gfTranslationZ = pEllipsePath->EllipseData.fSemiMinorAxis * (GLfloat)sin(gfEarthAngleTranslate);

	gmat4TranslationMatrix = vmath::translate(gfTranslationX, 0.0f, -gfTranslationZ);
	gmat4ModelMatrix = gmat4ModelMatrix * gmat4TranslationMatrix;

	gmat4RotationMatrix = vmath::rotate(-90.0f, 1.0f, 0.0f, 0.0f); // after this y becomes -z and z becomes y
	gmat4RotationMatrix = gmat4RotationMatrix * vmath::rotate(gfEarthAngle, 0.0f, 0.0f, 1.0f);
	gmat4ModelMatrix = gmat4ModelMatrix * gmat4RotationMatrix;

	gmat4ScaleMatrix = vmath::scale(fScaleFactor);
	gmat4ModelMatrix = gmat4ModelMatrix * gmat4ScaleMatrix;

	glUniformMatrix4fv(BasicPlanetShader.uiModelMatrixUniform, 1, GL_FALSE, gmat4ModelMatrix);
	glUniformMatrix4fv(BasicPlanetShader.uiViewMatrixUniform, 1, GL_FALSE, gmat4ViewMatrix);
	glUniformMatrix4fv(BasicPlanetShader.uiProjectionMatrixUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, uiTexture);
	glUniform1i(BasicPlanetShader.uiTextureSamplerUniform, 0);

	DrawPlanet(&Planet);
}

void ClearMatrices()
{
	gmat4ModelMatrix = vmath::mat4::identity();
	gmat4TranslationMatrix = vmath::mat4::identity();
	gmat4RotationMatrix = vmath::mat4::identity();
	gmat4ScaleMatrix = vmath::mat4::identity();
}
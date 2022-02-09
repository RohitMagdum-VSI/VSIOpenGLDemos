#include <windows.h>
#include <windowsx.h>

#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "OpenGL32.lib")

#include <stdio.h>
#include <gl\glew.h>
#include <gl\wglew.h>
#include <gl\GL.h>
#include <vector>
#include <string>
#include <sstream>


//#define GLM_FORCE_CUDA
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <glm\gtc\quaternion.hpp>
#include <glm\gtc\random.hpp>
#include <glm\gtx\rotate_vector.hpp>
#include <glm\ext.hpp>
#include <glm\gtx\string_cast.hpp>
#include <glm\gtc\noise.hpp>

#include "Data.h"
#include "Camera/Camera.h"
#include "Camera/Player.h"
#include "LightsMaterial/LightsMaterial.h"
#include "Texture/Texture.h"
#include "ShaderLoad/LoadShaders.h"

#include "Skydome/Skydome.h"
#include "Terrain/Terrain.h"

#include "Texture/Texture.h"
#include "Texture/Tga.h"

#include "BasicShapes/BasicShapes.h"


bool gbEscapeKeyIsPressed = false;
bool gbIsFullscreen = false;
bool gbActiveWindow = false;

GLboolean gbIsFirstPlay = GL_FALSE;

void ToggleFullscreen(void);

//BOOL gbDone = FALSE;
HWND ghWnd = NULL;
HDC ghDC = NULL;
HGLRC ghRC = NULL;

POINT ClickPoint, MovePoint, OldMovePoint;
GLboolean gbFirstMouse = GL_TRUE;

CCommon *pCCommon = NULL; // Cleaned in Uninitialize()
CCamera *pCCamera = NULL; // no heap memory allocation still cleaned up

TGALoader *pTGA = NULL; // cleaned up
CDirectionalLight *pSunLight = NULL; // cleaned up
CPointLight *pLight = NULL;

GLuint MVPMatrixUniformLocationBasicShader;
GLuint MVPMatrixUniformLocationBasicShaderTexture;
GLuint TextureUniformLocationBasicShaderTexture;

////////////////////////////////////////////////////////
// basic shapes
CRing *pCRing[] = { NULL, NULL };
CSphere *pCSphere[] = { NULL, NULL };
CTorus *pCTorus[] = { NULL, NULL };
CCylinder *pCCylinder[] = { NULL, NULL };
CCone *pCCone[] = { NULL, NULL };

// texture IDs for basic shapes
GLuint BrickDiffuse, BrickSpecular, BrickNormal, BrickNormalInv;
GLuint BrickDiffuseGranite, BrickDiffuseGrey, BrickDiffuseOldRed, BrickSpecular2, BrickNormal2, BrickNormalInv2;
GLuint GraniteDiffuse, GraniteSpecular, GraniteNormal, GraniteNormalInv;
GLuint PavingDiffuse, PavingSpecular, PavingNormal, PavingNormalInv;
GLuint PlasterDiffuseGreen, PlasterDiffuseWhite, PlasterDiffuseYellow, PlasterSpecular, PlasterNormal, PlasterNormalInv;
GLuint WoodPlatesDiffuse, WoodPlatesSpecular, WoodNormal, WoodNormalInv;

void DrawAllBasicShapesInOneRowWithNormalMap(GLuint DiffuseTexture, GLuint SpecularTexture, GLuint NormalTexture, glm::mat4 ModelMatrix, glm::mat4 TranslationMatrix, glm::mat4 RotationMatrix, glm::mat4 ViewMatrix);
void DrawAllBasicShapesBlack(glm::mat4 ModelMatrix, glm::mat4 TranslationMatrix, glm::mat4 RotationMatrix, glm::mat4 ViewMatrix);
void SetDayLightNormalMap(GLuint ShaderObject);

/////////////////////////////////////////////////////////

glm::vec3 Positions[10] = {
	glm::vec3(-5.0f, 0.0f, -5.0f),
	glm::vec3(5.0f, 0.0f, -5.0f),

	glm::vec3(-15.0f, 0.0f, -20.0f),
	glm::vec3(15.0f, 0.0f, -20.0f),

	glm::vec3(-15.0f, 15.0f, -15.0f),
	glm::vec3(15.0f, 15.0f, -15.0f),

	glm::vec3(-15.0f, -15.0f, -15.0f),
	glm::vec3(15.0f, -15.0f, -15.0f),

	glm::vec3(-15.0f, 15.0f, -20.0f),
	glm::vec3(15.0f, 15.0f, -20.0f),
};

//GLboolean IsMyShape = GL_TRUE;
GLboolean IsMyShape = GL_FALSE;
GLuint gTextureRingID;
GLuint gShaderProgramBasicShapes;
GLuint gShaderProgramObjectBasicShapesNormalMap;

glm::vec3 CameraPosition;
glm::vec3 CameraFront;
glm::vec3 CameraUp;
glm::vec3 CameraLook;
glm::vec3 CameraRight;

DWORD dwStyle;

GLboolean gbIsPlaying = GL_FALSE;

WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };
GLuint giWinHeight = 0;
GLuint giWinWidth = 0;
TCHAR str[255];

GLfloat gfLastX = giWinWidth / 2.0f;
GLfloat gfLastY = giWinHeight / 2.0f;

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

GLfloat gfFOVY = FOV_Y;

GLuint gModelMatrixUniform, gViewMatrixUniform, gProjectionMatrixUniform, gViewPositionUniform, gMVPMatrixUniform;
glm::mat4 gPerspectiveProjectionMatrix;
glm::mat4 gPerspectiveProjectionMatrixFont;
glm::mat4 gOrthographicProjectionMatrix;

GLint Samples;

//Timing related
double gd_resolution;
unsigned __int64 gu64_base;

float gfDeltaTime = 0.0f;
float gfElapsedTime = 0.0f;
int giFrameCount = 0;
__int64 giPreviousTime = 0;
__int64 giCurrentTime = 0;
__int64 giCountsPerSecond;
float fSecondsPerCount;
int fps__;

float LastFrame = 0.0f;

RECT screen;

std::stringstream strLightProperty;

GLboolean gbIsDayNightCyclePaused = GL_FALSE;
CSkyDome *pCSkyDome = NULL; // cleaned up

GLfloat Kr = 0.0030f; // Rayleigh scattering constant
GLfloat Km = 0.0015f; // Mie scattering constant
GLfloat ESun = 16.0f; // Sun brightness constant
GLfloat g = -0.95f; // The Mie phase asymmetry factor

//GLfloat InnerRadius = 10.0f * 25.0f;
//GLfloat OuterRadius = 10.25f * 25.0f;

GLfloat InnerRadius = 10.0f;
GLfloat OuterRadius = 10.25f;

GLfloat ZDistance = 50.0f;

GLfloat Scale = 1.0f / (OuterRadius - InnerRadius);
GLfloat ScaleDepth = 0.25f;
GLfloat ScaleOverScaleDepth = Scale / ScaleDepth;

GLfloat WavelengthRed = 0.650f; // 650 nm for red
GLfloat WavelengthGreen = 0.570f; // 570 nm for green
GLfloat WavelengthBlue = 0.475f; // 475 nm for blue

GLfloat WavelengthRed4 = glm::pow(WavelengthRed, 4.0f);
GLfloat WavelengthGreen4 = glm::pow(WavelengthGreen, 4.0f);
GLfloat WavelengthBlue4 = glm::pow(WavelengthBlue, 4.0f);

glm::vec3 InvWaveLength = glm::vec3(1.0f / WavelengthRed4, 1.0f / WavelengthGreen4, 1.0f / WavelengthBlue4);


///////////////////////////////////////
// set the lighting data

void SetDayLight(GLuint ShaderObject);


///////////////////////////////////////////////////////////////////
// SunRays and Halo related
#define OFF_SCREEN_RENDER_RATIO 1

GLuint gShaderProgramObjectBasicShader; // cleaned up
GLuint gShaderProgramObjectBasicShaderTexture; // cleaned up

GLuint ShaderProgramObjectVolumeLight;

// Uniform Locations
GLuint ModelMatrixUniformLocationVolumeLight;
GLuint ViewMatrixUniformLocationVolumeLight;
GLuint ProjectionMatrixUniformLocationVolumeLight;

GLuint ExposureUniformLocationVolumeLight;
GLuint DecayUniformLocationVolumeLight;
GLuint DensityUniformLocationVolumeLight;
GLuint WeightUniformLocationVolumeLight;
GLuint TextureUniformLocationVolumeLight;

GLuint LightPositionOnScreenUniformLocationVolumeLight;

GLfloat Exposure, Decay, Density, Weight;
GLuint FramebufferObjectVolumeLight;
GLuint RenderBufferObjectVolumeLight;
GLuint FBOTexture;
GLuint ScreenCopyTexture;

CSphere *pCSphereSun = NULL;
GLfloat SunRadius = 3.75f;

GLfloat ScreenSpaceLightPositionX, ScreenSpaceLightPositionY;

GLuint VAOSquare;
GLuint VBOSquarePosition, VBOSquareTexCoord;

GLuint VAOSquareDebug;
GLuint VBOSquarePositionDebug, VBOSquareTexCoordDebug;

void CreateFramebuffer(GLuint Width, GLuint Height);
void ResizeFramebuffers(GLuint Width, GLuint Height);
glm::vec3 GetLightScreenSpaceCoordinates(glm::vec3 LightPosition, glm::mat4 ModelMatrix, glm::mat4 ProjectionMatrix);
///////////////////////////////////////////////////////////////////

//main function
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int nCmdShow)
{
	MSG msg;
	bool bDone = false;

	void Initialize(HINSTANCE hInstance, int Samples);
	void InitializeTimer(void);
	void CalculateDelta(void);
	void InitializeOpenGL(void);
	void Uninitialize(void);
	void Display(void);
	void ToggleFullscreen(void);
	void Update(void);

	void CheckAsyncKeyDown(void);
	void CalculateFPS(void);

	pCCommon = new CCommon();

	if (fopen_s(&pCCommon->pLogFile, pCCommon->logfilepath, "w") != 0)
	{
		MessageBox(NULL, TEXT("Failed to create the log file. Exiting now."), TEXT("Error"), MB_OK);
		exit(0);
	}
	else
	{
		fprintf_s(pCCommon->pLogFile, "Log file created successfully.\n");
	}

	Initialize(hInstance, MSAA_SAMPLES);

	//display created window and update window
	ShowWindow(ghWnd, nCmdShow);
	SetForegroundWindow(ghWnd);
	SetFocus(ghWnd);
	//ToggleFullscreen();

	InitializeTimer();

	//fclose(pCCommon->pLogFile);
	InitializeOpenGL();

	// Game Loop
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
			CalculateDelta();
			CheckAsyncKeyDown();
			Update();
			Display();
			CalculateFPS();

			if (gbActiveWindow == true)
			{
				if (gbEscapeKeyIsPressed == true)
					bDone = true;
			}
		}
	}

	Uninitialize();
	return((int)msg.wParam);
}

void InitializeTimer(void)
{
	// Counts per second
	QueryPerformanceFrequency((LARGE_INTEGER *)&giCountsPerSecond);
	// seconds per count
	fSecondsPerCount = 1.0f / giCountsPerSecond;

	// previous time
	QueryPerformanceCounter((LARGE_INTEGER *)&giPreviousTime);
}

void CalculateDelta(void)
{
	// get current count
	QueryPerformanceCounter((LARGE_INTEGER *)&giCurrentTime);

	// Delta time
	gfDeltaTime = (giCurrentTime - giPreviousTime) * fSecondsPerCount;
	gfElapsedTime += gfDeltaTime;
}

void CalculateFPS(void)
{
	giFrameCount++;

	TCHAR str[255];

	// frames per second
	if (gfElapsedTime >= 1.0f)
	{
		fps__ = giFrameCount;

		swprintf_s(str, 255, TEXT("A Walk in the Fields FPS : %d"), fps__);
		SetWindowText(ghWnd, str);

		giFrameCount = 0;
		gfElapsedTime = 0.0f;
	}

	giPreviousTime = giCurrentTime;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	void Resize(int Width, int Height);
	void Uninitialize(void);

	void OnLButtonDown(INT, INT, DWORD);
	void OnLButtonUp(INT, INT, DWORD);
	void OnRButtonDown(INT, INT, DWORD);
	void OnRButtonUp(INT, INT, DWORD);
	void OnMouseMove(INT, INT, DWORD);
	void OnMButtonDown(INT, INT, DWORD);
	void OnMButtonUp(INT, INT, DWORD);
	void OnMouseWheelScroll(short zDelta);

	void OnUpArrowPress(void);
	void OnLeftArrowPress(void);
	void OnRightArrowPress(void);
	void OnDownArrowPress(void);
	void OnPageUpPress(void);
	void OnPageDownPress(void);

	void CheckAsyncKeyDown(void);


	static GLfloat floatval = 0.0f;
	static glm::vec3 vector;

	switch (iMsg)
	{
	case WM_CREATE:
		break;

	case WM_ACTIVATE:
		if (HIWORD(wParam) == 0)
			gbActiveWindow = true;
		else
			gbActiveWindow = false;
		break;

		//case WM_ERASEBKGND:
		//return(0);

	case WM_SIZE:
		giWinWidth = LOWORD(lParam);
		giWinHeight = HIWORD(lParam);
		Resize(giWinWidth, giWinHeight);
		//Resize(LOWORD(lParam), HIWORD(lParam));
		break;

	case WM_LBUTTONDOWN:
		OnLButtonDown(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (DWORD)wParam);
		break;

	case WM_LBUTTONUP:
		OnLButtonUp(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (DWORD)wParam);
		break;

	case WM_RBUTTONDOWN:
		OnRButtonDown(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (DWORD)wParam);
		break;

	case WM_RBUTTONUP:
		OnRButtonUp(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (DWORD)wParam);
		break;

	case WM_MOUSEWHEEL:
		OnMouseWheelScroll(GET_WHEEL_DELTA_WPARAM(wParam));
		break;

	case WM_MOUSEMOVE:
		OnMouseMove(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (DWORD)wParam);
		break;

	case WM_MBUTTONDOWN:
		OnMButtonDown(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (DWORD)wParam);
		break;

	case WM_MBUTTONUP:
		OnMButtonUp(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (DWORD)wParam);
		break;

	case WM_KEYDOWN:
		//CheckAsyncKeyDown();
		switch (wParam)
		{
		case VK_ESCAPE:
			gbEscapeKeyIsPressed = true;
			break;

		case VK_RETURN:
			if (gbIsFullscreen == false)
			{
				ToggleFullscreen();
				gbIsFullscreen = true;
			}
			else
			{
				ToggleFullscreen();
				gbIsFullscreen = false;
			}
			break;

		case 0x46:	//F key
			if (gbIsDayNightCyclePaused == GL_FALSE)
				gbIsDayNightCyclePaused = GL_TRUE;
			else
				gbIsDayNightCyclePaused = GL_FALSE;
			break;

			//case VK_UP:
			//	OnUpArrowPress();
			//	break;

			//case VK_DOWN:
			//	OnDownArrowPress();
			//	break;

			//case VK_LEFT:
			//	OnLeftArrowPress();
			//	break;

			//case VK_RIGHT:
			//	OnRightArrowPress();
			//	break;

			//	// PAGE UP key
			//case VK_PRIOR:
			//	OnPageUpPress();
			//	break;

			//	// PAGE DOWN KEY
			//case VK_NEXT:
			//	OnPageDownPress();
			//	break;

			// W key
		case 0x57:
			break;

			// S key
		case 0x53:
			break;

			// D key
		case 0x41:
			break;

			// E key to end the scene
		case 0x45:
			break;

			// A key
		case 0x44:
			break;

		case VK_NUMPAD8:
			break;

		case VK_NUMPAD2:
			break;

		case VK_NUMPAD4:
			break;

		case VK_NUMPAD6:
			break;

		case VK_NUMPAD7:
			break;

		case VK_NUMPAD1:
			break;

			// E key
			//case 0x45:
			//cam.Roll(-1.0);
			//break;

			// Q key
		case 0x51:
			//cam.Roll(1.0);
			break;

		case 0x30:

			break;

		case 0x70: //F1

			break;

			// P key
		case 0x50:
			break;

			// L key
		case 0x4C:

			break;

		case VK_ADD:
			pCSkyDome->SetSunCPos(glm::rotate(pCSkyDome->GetSunCPos(), +0.1f, pCSkyDome->GetSunRotVec()));
			break;

		case VK_SUBTRACT:
			pCSkyDome->SetSunCPos(glm::rotate(pCSkyDome->GetSunCPos(), -0.1f, pCSkyDome->GetSunRotVec()));
			break;

		case VK_CONTROL:
			break;

		case VK_SHIFT:
			break;

		default:
			break;
		}
		break;

	case WM_CLOSE:
		//Uninitialize();
		break;

	case WM_DESTROY:
		PostQuitMessage(0);
		break;

	default:
		break;
	}

	return(DefWindowProc(hWnd, iMsg, wParam, lParam));
}

void CheckAsyncKeyDown(void)
{
	if (GetAsyncKeyState(VK_UP) & 0x8000)
	{
		if (gbIsPlaying == GL_FALSE)
			pCCamera->ProcessNavigationKeys(CameraMovement::CAMERA_FORWARD, gfDeltaTime);
	}

	if (GetAsyncKeyState(VK_DOWN) & 0x8000)
	{
		if (gbIsPlaying == GL_FALSE)
			pCCamera->ProcessNavigationKeys(CameraMovement::CAMERA_BACKWARD, gfDeltaTime);
	}

	if (GetAsyncKeyState(VK_LEFT) & 0x8000)
	{
		if (gbIsPlaying == GL_FALSE)
			pCCamera->ProcessNavigationKeys(CameraMovement::CAMERA_LEFT, gfDeltaTime);
	}

	if (GetAsyncKeyState(VK_RIGHT) & 0x8000)
	{
		if (gbIsPlaying == GL_FALSE)
			pCCamera->ProcessNavigationKeys(CameraMovement::CAMERA_RIGHT, gfDeltaTime);
	}

	if (GetAsyncKeyState(VK_PRIOR) & 0x8000)
	{
		pCCamera->ProcessNavigationKeys(CameraMovement::CAMERA_UP, gfDeltaTime);
	}

	if (GetAsyncKeyState(VK_NEXT) & 0x8000)
	{
		pCCamera->ProcessNavigationKeys(CameraMovement::CAMERA_DOWN, gfDeltaTime);
	}
}

void ToggleFullscreen(void)
{
	HMONITOR hMonitor;
	MONITORINFO mi;
	BOOL bWindowPlacement;
	BOOL bMonitorInfo;

	if (gbIsFullscreen == FALSE)
	{
		dwStyle = GetWindowLong(ghWnd, GWL_STYLE);
		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{

			mi.cbSize = { sizeof(MONITORINFO) };

			bWindowPlacement = GetWindowPlacement(ghWnd, &wpPrev);
			hMonitor = MonitorFromWindow(ghWnd, MONITOR_DEFAULTTOPRIMARY);
			bMonitorInfo = GetMonitorInfo(hMonitor, &mi);

			if (bWindowPlacement == TRUE && bMonitorInfo == TRUE)
			{
				SetWindowLong(ghWnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghWnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}

		if (WGLEW_EXT_swap_control)
		{
			wglSwapIntervalEXT(0);
		}
	}
	else
	{
		SetWindowLong(ghWnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghWnd, &wpPrev);
		SetWindowPos(ghWnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);
	}
}

void Initialize(HINSTANCE hInstance, int Samples)
{
	//variable declarations
	WNDCLASSEX wndclass;
	HWND hWnd;
	TCHAR szClassName[] = TEXT("RTROpenGLProject");
	TCHAR szAppName[] = TEXT("A Walk in the Fields");

	void Initialize(HINSTANCE hInstance, int Samples);

	//WNDCLASSEX initialization
	wndclass.cbClsExtra = 0;
	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.cbWndExtra = 0;
	wndclass.hbrBackground = (HBRUSH)GetStockObject(GRAY_BRUSH);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hInstance = hInstance;
	wndclass.lpfnWndProc = WndProc;
	wndclass.lpszClassName = szClassName;
	wndclass.lpszMenuName = NULL;
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;

	//register wndclass
	RegisterClassEx(&wndclass);

	int iWidth = GetSystemMetrics(SM_CXSCREEN);
	int iHeight = GetSystemMetrics(SM_CYSCREEN);

	int iXFinalPosition = (iWidth / 2) - (WIN_WIDTH / 2);
	int iYFinalPosition = (iHeight / 2) - (WIN_HEIGHT / 2);

	//create window
	hWnd = CreateWindowEx(WS_EX_APPWINDOW, szClassName, szAppName, WS_OVERLAPPEDWINDOW, iXFinalPosition, iYFinalPosition, WIN_WIDTH, WIN_HEIGHT, NULL, NULL, hInstance, NULL);
	ghWnd = hWnd;

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
	pfd.iLayerType = PFD_MAIN_PLANE;

	ghDC = GetDC(ghWnd);

	iPixelFormatIndex = ChoosePixelFormat(ghDC, &pfd);
	if (iPixelFormatIndex == 0)
	{
		fprintf(pCCommon->pLogFile, "ChoosePixelFormat() Error : Pixel Format Index is 0\n");
		//fclose(Essential.gpFile);
		//Essential.gpFile = NULL;

		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
	}

	static int MSAAPixelFormat = 0;

	if (SetPixelFormat(ghDC, MSAAPixelFormat == 0 ? iPixelFormatIndex : MSAAPixelFormat, &pfd) == false)
	{
		fprintf(pCCommon->pLogFile, "SetPixelFormat() Error : Failed to set the pixel format\n");
		//fclose(Essential.gpFile);
		//Essential.gpFile = NULL;

		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
	}

	ghRC = wglCreateContext(ghDC);
	if (ghRC == NULL)
	{
		fprintf(pCCommon->pLogFile, "wglCreateContext() Error : Rendering context ghRC is NULL\n");
		//fclose(Essential.gpFile);
		//Essential.gpFile = NULL;

		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
	}

	if (wglMakeCurrent(ghDC, ghRC) == false)
	{
		fprintf(pCCommon->pLogFile, "wglMakeCurrent() Error : wglMakeCurrent() Failed to set rendering context as current context\n");
		//fclose(Essential.gpFile);
		//Essential.gpFile = NULL;

		wglDeleteContext(ghRC);
		ghRC = NULL;
		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
	}

	GLenum glew_error = glewInit();
	if (glew_error != GLEW_OK)
	{
		wsprintf(str, TEXT("Error: %s\n"), glewGetErrorString(glew_error));
		MessageBox(ghWnd, str, TEXT("GLEW Error"), MB_OK);

		wglDeleteContext(ghRC);
		ghRC = NULL;
		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
	}

	//MessageBox(ghWnd, TEXT("Before if condition"), TEXT("Message"), MB_OK | MB_TOPMOST);
	if (MSAAPixelFormat == 0 && Samples > 0)
	{
		//MessageBox(ghWnd, TEXT("Inside first if condition"), TEXT("Message"), MB_OK | MB_TOPMOST);
		if (WGLEW_ARB_pixel_format && GLEW_ARB_multisample)
		{
			//MessageBox(ghWnd, TEXT("Inside second if condition"), TEXT("Message"), MB_OK | MB_TOPMOST);
			while (Samples > 0)
			{
				//MessageBox(ghWnd, TEXT("Inside while condition"), TEXT("Message"), MB_OK | MB_TOPMOST);

				UINT NumFormats = 0;

				int PFAttribs[] =
				{
					WGL_DRAW_TO_WINDOW_ARB, GL_TRUE,
					WGL_SUPPORT_OPENGL_ARB, GL_TRUE,
					WGL_DOUBLE_BUFFER_ARB, GL_TRUE,
					WGL_PIXEL_TYPE_ARB, WGL_TYPE_RGBA_ARB,
					WGL_COLOR_BITS_ARB, 32,
					WGL_DEPTH_BITS_ARB, 24,
					WGL_ACCELERATION_ARB, WGL_FULL_ACCELERATION_ARB,
					WGL_SAMPLE_BUFFERS_ARB, GL_TRUE,
					WGL_SAMPLES_ARB, Samples,
					0
				};



				if (wglChoosePixelFormatARB(ghDC, PFAttribs, NULL, 1, &MSAAPixelFormat, &NumFormats) == TRUE && NumFormats > 0)
				{
					//MessageBox(ghWnd, TEXT("Inside third if condition"), TEXT("Message"), MB_OK | MB_TOPMOST);
					break;
				}
				Samples--;
			}



			//MessageBox(ghWnd, TEXT("Out of while condition"), TEXT("Message"), MB_OK | MB_TOPMOST);

			//wglDeleteContext(ghRC);
			//DestroyWindow(hWnd);
			//UnregisterClass(wndclass.lpszClassName, hInstance);
			return(Initialize(hInstance, Samples));
		}
		else
		{
			//MessageBox(ghWnd, TEXT("Inside else condition"), TEXT("Message"), MB_OK | MB_TOPMOST);
			Samples = 0;
		}
	}

	if (WGLEW_EXT_swap_control)
	{
		wglSwapIntervalEXT(0);
	}

	if (!GLEW_ARB_texture_non_power_of_two)
	{
		fprintf(pCCommon->pLogFile, "GL_ARB_texture_non_power_of_two not supported!\n");
	}

	if (!GLEW_ARB_depth_texture)
	{
		fprintf(pCCommon->pLogFile, "GLEW_ARB_depth_texture not supported!\n");
	}

	if (!GLEW_EXT_framebuffer_object)
	{
		fprintf(pCCommon->pLogFile, "GLEW_EXT_framebuffer_object not supported!\n");
	}
}

void InitializeOpenGL(void)
{
	void Uninitialize(void);
	void Resize(int, int);
	void InitializeLightCube(void);
	void InitializeModels(void);

	pSunLight = new CDirectionalLight();
	pLight = new CPointLight();
	pCSkyDome = new CSkyDome();

	pCCamera = new CCamera();

	pTGA = new TGALoader();

	pCRing[0] = new CRing(2.5f, 5.0f, 100, GL_FALSE);
	pCRing[1] = new CRing(2.5f, 5.0f, 100, GL_TRUE);

	pCSphere[0] = new CSphere(5.0f, 100, 100, GL_FALSE);
	pCSphere[1] = new CSphere(5.0f, 100, 100, GL_TRUE);
	pCSphereSun = new CSphere(SunRadius, 16, 16, GL_TRUE);

	pCTorus[0] = new CTorus(2.5f, 5.0f, 100, 100, GL_FALSE);
	pCTorus[1] = new CTorus(2.5f, 5.0f, 100, 100, GL_TRUE);

	pCCylinder[0] = new CCylinder(5.0f, 7.0f, 100, GL_TRUE, GL_TRUE, GL_FALSE);
	pCCylinder[1] = new CCylinder(5.0f, 7.0f, 100, GL_TRUE, GL_TRUE, GL_TRUE);

	pCCone[0] = new CCone(5.0f, 7.0f, 100, GL_TRUE, GL_FALSE);
	pCCone[1] = new CCone(5.0f, 7.0f, 100, GL_TRUE, GL_TRUE);


	ShaderInfo BasicShader[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/BasicShader.vert.glsl", 0 },
	{ GL_FRAGMENT_SHADER, "Resources/Shaders/BasicShader.frag.glsl", 0 },
	{ GL_NONE, NULL, 0 }
	};

	gShaderProgramObjectBasicShader = LoadShaders(BasicShader);

	ShaderInfo BasicShaderTexture[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/BasicShaderTexture.vert.glsl", 0 },
	{ GL_FRAGMENT_SHADER, "Resources/Shaders/BasicShaderTexture.frag.glsl", 0 },
	{ GL_NONE, NULL, 0 }
	};

	gShaderProgramObjectBasicShaderTexture = LoadShaders(BasicShaderTexture);

	//ShaderInfo SampleTextureShader[] =
	//{
	//	{ GL_VERTEX_SHADER, "Resources/Shaders/SampleTexture.vert.glsl", 0 },
	//{ GL_FRAGMENT_SHADER, "Resources/Shaders/SampleTexture.frag.glsl", 0 },
	//{ GL_NONE, NULL, 0 }
	//};

	//gShaderObjectSampleTexture = LoadShaders(SampleTextureShader);

	ShaderInfo SkyShader[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/AtmosphericScatterringGPUGems2.vert.glsl", 0 },
		{ GL_FRAGMENT_SHADER, "Resources/Shaders/AtmosphericScatterringGPUGems2.frag.glsl", 0 },
		{ GL_NONE, NULL, 0 }
	};

	ShaderInfo BasicShapes[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/BasicShapesIQFog1.vert.glsl", 0 },
	{ GL_FRAGMENT_SHADER, "Resources/Shaders/BasicShapesIQFog1.frag.glsl", 0 },
	{ GL_NONE, NULL, 0 }
	};

	gShaderProgramBasicShapes = LoadShaders(BasicShapes);

	ShaderInfo BasicShapesNM[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/BasicShapesIQFog1NM.vert.glsl", 0 },
	{ GL_FRAGMENT_SHADER, "Resources/Shaders/BasicShapesIQFog1NM.frag.glsl", 0 },
	{ GL_NONE, NULL, 0 }
	};

	gShaderProgramObjectBasicShapesNormalMap = LoadShaders(BasicShapesNM);

	ShaderInfo VolumeLight[] =
	{
		{ GL_VERTEX_SHADER, "Resources/Shaders/LightScattering.vert.glsl", 0 },
		{ GL_FRAGMENT_SHADER, "Resources/Shaders/LightScattering.frag.glsl", 0 },
		{ GL_NONE, NULL, 0 }
	};

	ShaderProgramObjectVolumeLight = LoadShaders(VolumeLight);

	//pCCommon->_check_gl_error(pCCommon->pLogFile, "GeneralLog.txt", __FILE__, __LINE__);

	////////////////////////////////////////////////////////////////////////
	// Basic Shader and Basic Shader Texture
	glUseProgram(gShaderProgramObjectBasicShader);
	MVPMatrixUniformLocationBasicShader = glGetUniformLocation(gShaderProgramObjectBasicShader, "uMVPMatrix");
	glUseProgram(0);

	glUseProgram(gShaderProgramObjectBasicShaderTexture);
	MVPMatrixUniformLocationBasicShaderTexture = glGetUniformLocation(gShaderProgramObjectBasicShaderTexture, "uMVPMatrix");
	TextureUniformLocationBasicShaderTexture = glGetUniformLocation(gShaderProgramObjectBasicShaderTexture, "Texture");
	glUniform1i(TextureUniformLocationBasicShaderTexture, 0);
	glUseProgram(0);

	/////////////////////////////////////////////////////////////////////////////////////////////////

	// initialize light values
	pSunLight->SetAmbient(glm::vec3(0.2f, 0.2f, 0.2f));
	pSunLight->SetDiffuse(glm::vec3(1.0f, 1.0f, 1.0f));
	pSunLight->SetSpecular(glm::vec3(1.0f, 1.0f, 1.0f));
	pSunLight->SetDirection(glm::vec3(0.0f, 1.0f, 0.0f));

	///////////////////////////////////////////////////////////////////
	// Initializing Volume Light
	Exposure = 0.0034f;
	Decay = 1.0f;
	Density = 0.84f;
	Weight = 5.65f;

	CreateFramebuffer(giWinWidth, giWinHeight);

	glUseProgram(ShaderProgramObjectVolumeLight);

	ModelMatrixUniformLocationVolumeLight = glGetUniformLocation(ShaderProgramObjectVolumeLight, "uModelMatrix");
	ViewMatrixUniformLocationVolumeLight = glGetUniformLocation(ShaderProgramObjectVolumeLight, "uViewMatrix");
	ProjectionMatrixUniformLocationVolumeLight = glGetUniformLocation(ShaderProgramObjectVolumeLight, "uProjectionMatrix");

	LightPositionOnScreenUniformLocationVolumeLight = glGetUniformLocation(ShaderProgramObjectVolumeLight, "LightPositionOnScreen");

	ExposureUniformLocationVolumeLight = glGetUniformLocation(ShaderProgramObjectVolumeLight, "Exposure");
	DecayUniformLocationVolumeLight = glGetUniformLocation(ShaderProgramObjectVolumeLight, "Decay");
	DensityUniformLocationVolumeLight = glGetUniformLocation(ShaderProgramObjectVolumeLight, "Density");
	WeightUniformLocationVolumeLight = glGetUniformLocation(ShaderProgramObjectVolumeLight, "Weight");
	TextureUniformLocationVolumeLight = glGetUniformLocation(ShaderProgramObjectVolumeLight, "Texture");

	//glUniform2f(LightPositionOnScreenUniformLocationVolumeLight, ScreenSpaceLightPositionX, ScreenSpaceLightPositionY);
	glUniform1f(ExposureUniformLocationVolumeLight, Exposure);
	glUniform1f(DecayUniformLocationVolumeLight, Decay);
	glUniform1f(DensityUniformLocationVolumeLight, Density);
	glUniform1f(WeightUniformLocationVolumeLight, Weight);
	glUniform1i(TextureUniformLocationVolumeLight, 0);

	glUseProgram(0);


	const GLfloat SquareVertices[] =
	{
		1.0f, 1.0f, 0.0f,	// top right
		-1.0f, 1.0f, 0.0f,	// top left
		-1.0f, -1.0f, 0.0f,	// bottom left
		1.0f, -1.0f, 0.0f	// bottom right
	};

	const GLfloat SquareTexcoords[] =
	{
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f
	};

	glGenVertexArrays(1, &VAOSquare);
	glBindVertexArray(VAOSquare);


	// square position
	glGenBuffers(1, &VBOSquarePosition);
	glBindBuffer(GL_ARRAY_BUFFER, VBOSquarePosition);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(SquareVertices), SquareVertices, GL_STATIC_DRAW);
	glBufferData(GL_ARRAY_BUFFER, sizeof(SquareVertices), NULL, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_VERTEX);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	// square tex coords
	glGenBuffers(1, &VBOSquareTexCoord);
	glBindBuffer(GL_ARRAY_BUFFER, VBOSquareTexCoord);
	glBufferData(GL_ARRAY_BUFFER, sizeof(SquareTexcoords), SquareTexcoords, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_TEXTURE0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	glGenVertexArrays(1, &VAOSquareDebug);
	glBindVertexArray(VAOSquareDebug);


	// square position
	glGenBuffers(1, &VBOSquarePositionDebug);
	glBindBuffer(GL_ARRAY_BUFFER, VBOSquarePositionDebug);
	glBufferData(GL_ARRAY_BUFFER, sizeof(SquareVertices), SquareVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_VERTEX);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	// square tex coords
	glGenBuffers(1, &VBOSquareTexCoordDebug);
	glBindBuffer(GL_ARRAY_BUFFER, VBOSquareTexCoordDebug);
	glBufferData(GL_ARRAY_BUFFER, sizeof(SquareTexcoords), SquareTexcoords, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_TEXTURE0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	//////////////////////////////////////
	// Sky initialization

	pCSkyDome->InitializeSky(LoadShaders(SkyShader));


	///////////////////////////////////////////////////////////////////
	// Basic shapes initialization
	gTextureRingID = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/BricksDiffuse.tga");


	BrickDiffuse = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/bricks_02_dif.tga");
	BrickSpecular = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/bricks_02_spec.tga");
	BrickNormal = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/bricks_02_nm.tga");
	BrickNormalInv = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/bricks_02_nm_inv.tga");

	BrickDiffuseGranite = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/bricks_03_dif_granite.tga");
	BrickDiffuseGrey = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/bricks_03_dif_grey.tga");
	BrickDiffuseOldRed = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/bricks_03_dif_old_red.tga");
	BrickSpecular2 = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/Bricks_03_spec.tga");
	BrickNormal2 = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/Bricks_03_nm.tga");
	BrickNormalInv2 = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/Bricks_03_nm_inv.tga");

	GraniteDiffuse = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/granite_01_dif.tga");
	GraniteSpecular = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/granite_01_spec.tga");
	GraniteNormal = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/granite_01_nm.tga");
	GraniteNormalInv = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/granite_01_nm_inv.tga");

	PavingDiffuse = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/paving_02_dif.tga");
	PavingSpecular = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/paving_02_spec.tga");
	PavingNormal = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/paving_02_nm.tga");
	PavingNormalInv = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/paving_02_nm_inv.tga");

	PlasterDiffuseGreen = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/plaster_02_dif_green.tga");
	PlasterDiffuseWhite = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/plaster_02_dif_white.tga");
	PlasterDiffuseYellow = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/plaster_02_dif_white.tga");
	PlasterSpecular = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/plaster_02_spec.tga");
	PlasterNormal = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/plaster_02_nm.tga");
	PlasterNormalInv = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/plaster_02_nm_inv.tga");

	WoodPlatesDiffuse = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/wood_plates_05_dif.tga");
	WoodPlatesSpecular = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/wood_plates_05_spec.tga");
	WoodNormal = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/wood_plates_05_nm.tga");
	WoodNormalInv = pTGA->LoadTGATexture("Resources/ImageTextures/TGA/GamePack/wood_plates_05_nm_inv.tga");


	glUseProgram(gShaderProgramObjectBasicShapesNormalMap);

	glUniform1i(glGetUniformLocation(gShaderProgramObjectBasicShapesNormalMap, "material.DiffuseTexture"), 0);
	glUniform1i(glGetUniformLocation(gShaderProgramObjectBasicShapesNormalMap, "material.SpecularTexture"), 1);
	glUniform1i(glGetUniformLocation(gShaderProgramObjectBasicShapesNormalMap, "material.NormalTexture"), 2);

	glUseProgram(0);

	//////////////////////////////////////////////////////////////////

	///////////////////////////////////////////////////////////////////

	gPerspectiveProjectionMatrix = glm::mat4(1.0);
	gOrthographicProjectionMatrix = glm::mat4(1.0);

	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glFrontFace(GL_CCW);
	glCullFace(GL_BACK);
	glEnable(GL_CULL_FACE);

	

	Resize(giWinWidth, giWinHeight);
	//Resize(WIN_WIDTH, WIN_HEIGHT);
}

void Display(void)
{
	void RenderScene(void);

	RenderScene();

	SwapBuffers(ghDC);
}

void RenderScene(void)
{
	glm::vec3 SunPos = pCSkyDome->GetSunWPos();

	glm::mat4 ModelMatrix = glm::mat4(1.0f);
	glm::mat4 ModelViewMatrix = glm::mat4(1.0f);
	glm::mat4 TranslationMatrix = glm::mat4(1.0f);
	glm::mat4 RotationMatrix = glm::mat4(1.0f);
	glm::mat4 ScaleMatrix = glm::mat4(1.0f);
	glm::mat4 ViewMatrix = glm::mat4(1.0f);
	glm::mat4 ModelViewProjectionMatrix = glm::mat4(1.0f);

	if (gbIsPlaying == GL_FALSE)
	{
		CameraPosition = pCCamera->GetCameraPosition();
		ViewMatrix = pCCamera->GetViewMatrix();
		CameraFront = pCCamera->GetCameraFront();
		CameraUp = pCCamera->GetCameraUp();
		CameraLook = pCCamera->GetCameraFront();
		CameraRight = pCCamera->GetCameraRight();
	}

	// calculate the ambient and diffuse light color
	pSunLight->SetAmbient(glm::vec3(pCSkyDome->GetAmbientIntensity()));
	pSunLight->SetDiffuse(glm::vec3(pCSkyDome->GetLightColor() * pCSkyDome->GetDiffuseIntensity()));
	pSunLight->SetSpecular(glm::vec3(pCSkyDome->GetLightColor() * pCSkyDome->GetDiffuseIntensity()));
	pSunLight->SetDirection(glm::vec3(-pCSkyDome->GetSunWPos()));

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	//glClearColor(0.2f, 0.2f, 0.3f, 1.0f);
	glm::vec3 LightColor = pCSkyDome->GetLightColor();
	//glClearColor(LightColor.r, LightColor.g, LightColor.b, 1.0f);

	// First Pass
	// Render to offscreen buffer via FrameBuffer
	glBindFramebuffer(GL_FRAMEBUFFER, FramebufferObjectVolumeLight);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glViewport(0, 0, (GLsizei)(giWinWidth / OFF_SCREEN_RENDER_RATIO), (GLsizei)(giWinHeight / OFF_SCREEN_RENDER_RATIO));

	glUseProgram(gShaderProgramObjectBasicShader);

	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), SunPos);
	ModelMatrix = ModelMatrix * TranslationMatrix;
	ModelViewProjectionMatrix = gPerspectiveProjectionMatrix * ViewMatrix * ModelMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectBasicShader, "uMVPMatrix"), 1, GL_FALSE, glm::value_ptr(ModelViewProjectionMatrix));
	pCSphereSun->DrawSphere(glm::vec4(LightColor, 1.0f));

	glUseProgram(0);

	//glm::vec3 LightCoords = GetLightScreenSpaceCoordinates(pSunLight->GetDirection(), ModelMatrix, gPerspectiveProjectionMatrix);
	ModelMatrix = glm::mat4(1.0f);
	ModelViewMatrix = ViewMatrix * ModelMatrix;
	glm::vec3 LightCoords = GetLightScreenSpaceCoordinates(SunPos, ModelViewMatrix, gPerspectiveProjectionMatrix);

	// Draw the occluding source black with light
	glDisable(GL_DEPTH_TEST);
	DrawAllBasicShapesBlack(ModelMatrix, TranslationMatrix, RotationMatrix, ViewMatrix);
	glEnable(GL_DEPTH_TEST);

	// Switch to regular rendering
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// Second Pass
	// Render the scene with no light scattering
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glViewport(0, 0, (GLsizei)giWinWidth, (GLsizei)giWinHeight);

	////////////////////////////////////////////////////////////////////////////////////////////////
	// Draw the contents in framebuffer to the texture

	/*glUseProgram(gShaderProgramObjectBasicShaderTexture);

	ModelMatrix = glm::mat4(1.0f);
	ModelViewProjectionMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -5.0f));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	ModelViewProjectionMatrix = gPerspectiveProjectionMatrix * ViewMatrix * ModelMatrix;
	glUniformMatrix4fv(MVPMatrixUniformLocationBasicShaderTexture, 1, GL_FALSE, glm::value_ptr(ModelViewProjectionMatrix));

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, FBOTexture);
	glUniform1i(TextureUniformLocationBasicShaderTexture, 0);

	glBindVertexArray(VAOSquareDebug);
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glBindVertexArray(0);

	glUseProgram(0);*/


	///////////////////////////////////////////////////////////////////////////////////////////////
	// render basic shapes

	//DrawAllBasicShapesInOneRowWithNormalMap(BrickDiffuse, BrickSpecular, BrickNormalInv, ModelMatrix, TranslationMatrix, RotationMatrix, ViewMatrix);
	//DrawAllBasicShapesInOneRowWithNormalMap(BrickDiffuseGranite, BrickSpecular2, BrickNormalInv2, ModelMatrix, TranslationMatrix, RotationMatrix, ViewMatrix);
	//DrawAllBasicShapesInOneRowWithNormalMap(BrickDiffuseGrey, BrickSpecular2, BrickNormalInv2, ModelMatrix, TranslationMatrix, RotationMatrix, ViewMatrix);
	//DrawAllBasicShapesInOneRowWithNormalMap(BrickDiffuseOldRed, BrickSpecular2, BrickNormalInv2, ModelMatrix, TranslationMatrix, RotationMatrix, ViewMatrix);
	//DrawAllBasicShapesInOneRowWithNormalMap(GraniteDiffuse, GraniteSpecular, GraniteNormalInv, ModelMatrix, TranslationMatrix, RotationMatrix, ViewMatrix);
	//DrawAllBasicShapesInOneRowWithNormalMap(PavingDiffuse, PavingSpecular, PavingNormalInv, ModelMatrix, TranslationMatrix, RotationMatrix, ViewMatrix);
	DrawAllBasicShapesInOneRowWithNormalMap(PlasterDiffuseGreen, PlasterSpecular, PlasterNormalInv, ModelMatrix, TranslationMatrix, RotationMatrix, ViewMatrix);
	//DrawAllBasicShapesInOneRowWithNormalMap(PlasterDiffuseWhite, PlasterSpecular, PlasterNormalInv, ModelMatrix, TranslationMatrix, RotationMatrix, ViewMatrix);
	//DrawAllBasicShapesInOneRowWithNormalMap(PlasterDiffuseYellow, PlasterSpecular, PlasterNormalInv, ModelMatrix, TranslationMatrix, RotationMatrix, ViewMatrix);
	//DrawAllBasicShapesInOneRowWithNormalMap(WoodPlatesDiffuse, WoodPlatesSpecular, WoodNormalInv, ModelMatrix, TranslationMatrix, RotationMatrix, ViewMatrix);

	// Draw the light scattering effect
	ModelMatrix = glm::mat4(1.0f);
	ViewMatrix = glm::mat4(1.0f);

	const GLfloat SquareVertices[] =
	{
		(GLfloat)giWinWidth / 2.0f, (GLfloat)giWinHeight / 2.0f, 0.0f,	// top right
		-(GLfloat)giWinWidth / 2.0f, (GLfloat)giWinHeight / 2.0f, 0.0f,	// top left
		-(GLfloat)giWinWidth / 2.0f, -(GLfloat)giWinHeight / 2.0f, 0.0f,	// bottom left
		(GLfloat)giWinWidth / 2.0f, -(GLfloat)giWinHeight / 2.0f, 0.0f	// bottom right
	};

	glClear(GL_DEPTH_BUFFER_BIT);

	glUseProgram(ShaderProgramObjectVolumeLight);
	glUniformMatrix4fv(ModelMatrixUniformLocationVolumeLight, 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	glUniformMatrix4fv(ViewMatrixUniformLocationVolumeLight, 1, GL_FALSE, glm::value_ptr(ViewMatrix));
	glUniformMatrix4fv(ProjectionMatrixUniformLocationVolumeLight, 1, GL_FALSE, glm::value_ptr(gOrthographicProjectionMatrix));
	//glUniformMatrix4fv(ProjectionMatrixUniformLocationVolumeLight, 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));

	glUniform2f(LightPositionOnScreenUniformLocationVolumeLight, ScreenSpaceLightPositionX, ScreenSpaceLightPositionY);
	//glUniform2f(LightPositionOnScreenUniformLocationVolumeLight, LightCoords.x, LightCoords.y);

	glUniform1f(ExposureUniformLocationVolumeLight, Exposure);
	glUniform1f(DecayUniformLocationVolumeLight, Decay);
	glUniform1f(DensityUniformLocationVolumeLight, Density);
	glUniform1f(WeightUniformLocationVolumeLight, Weight);

	glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	glEnable(GL_BLEND);
	glDisable(GL_CULL_FACE);

	glActiveTexture(GL_TEXTURE0);

	glBindTexture(GL_TEXTURE_2D, FBOTexture);
	glUniform1i(TextureUniformLocationVolumeLight, 0);
	glBindVertexArray(VAOSquare);
	glBindBuffer(GL_ARRAY_BUFFER, VBOSquarePosition);
	glBufferData(GL_ARRAY_BUFFER, sizeof(SquareVertices), SquareVertices, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glBindVertexArray(0);

	glUseProgram(0);

	glEnable(GL_CULL_FACE);
	glDisable(GL_BLEND);
}


void DrawAllBasicShapesInOneRowWithNormalMap(GLuint DiffuseTexture, GLuint SpecularTexture, GLuint NormalTexture, glm::mat4 ModelMatrix, glm::mat4 TranslationMatrix, glm::mat4 RotationMatrix, glm::mat4 ViewMatrix)
{
	glUseProgram(gShaderProgramObjectBasicShapesNormalMap);

	glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectBasicShapesNormalMap, "u_view_matrix"), 1, GL_FALSE, glm::value_ptr(ViewMatrix));
	glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectBasicShapesNormalMap, "u_projection_matrix"), 1, GL_FALSE, glm::value_ptr(gPerspectiveProjectionMatrix));

	SetDayLightNormalMap(gShaderProgramObjectBasicShapesNormalMap);

	//bind texture
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, DiffuseTexture);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, SpecularTexture);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, NormalTexture);

	glUniform1f(glGetUniformLocation(gShaderProgramObjectBasicShapesNormalMap, "material.Shininess"), 128.0f);

	// Ring 
	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -10.0f));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	RotationMatrix = glm::rotate(glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	ModelMatrix = ModelMatrix * RotationMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectBasicShapesNormalMap, "u_model_matrix"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));

	pCRing[0]->DrawRing();

	// Sphere
	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(15.0f, 0.0f, -10.0f));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	RotationMatrix = glm::rotate(glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	ModelMatrix = ModelMatrix * RotationMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectBasicShapesNormalMap, "u_model_matrix"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));

	pCSphere[0]->DrawSphere();

	// Torus
	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(30.0f, 0.0f, -10.0f));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectBasicShapesNormalMap, "u_model_matrix"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));

	pCTorus[0]->DrawTorus();

	// Cylinder
	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(-15.0f, 0.0f, -10.0f));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	RotationMatrix = glm::rotate(glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	ModelMatrix = ModelMatrix * RotationMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectBasicShapesNormalMap, "u_model_matrix"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));

	pCCylinder[0]->DrawCylinder();

	// Cone
	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(-30.0f, 0.0f, -10.0f));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	RotationMatrix = glm::rotate(glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	ModelMatrix = ModelMatrix * RotationMatrix;
	glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectBasicShapesNormalMap, "u_model_matrix"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));

	pCCone[0]->DrawCone();

	glUseProgram(0);
}

void DrawAllBasicShapesBlack(glm::mat4 ModelMatrix, glm::mat4 TranslationMatrix, glm::mat4 RotationMatrix, glm::mat4 ViewMatrix)
{
	glm::mat4 MVPMatrix;

	glUseProgram(gShaderProgramObjectBasicShader);


	// Ring 
	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -10.0f));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	RotationMatrix = glm::rotate(glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	ModelMatrix = ModelMatrix * RotationMatrix;
	MVPMatrix = glm::mat4(1.0f);
	MVPMatrix = gPerspectiveProjectionMatrix * ViewMatrix * ModelMatrix;
	glUniformMatrix4fv(MVPMatrixUniformLocationBasicShader, 1, GL_FALSE, glm::value_ptr(MVPMatrix));

	pCRing[1]->DrawRing(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

	// Sphere
	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(15.0f, 0.0f, -10.0f));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	RotationMatrix = glm::rotate(glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	ModelMatrix = ModelMatrix * RotationMatrix;
	MVPMatrix = glm::mat4(1.0f);
	MVPMatrix = gPerspectiveProjectionMatrix * ViewMatrix * ModelMatrix;
	glUniformMatrix4fv(MVPMatrixUniformLocationBasicShader, 1, GL_FALSE, glm::value_ptr(MVPMatrix));

	pCSphere[1]->DrawSphere(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

	// Torus
	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(30.0f, 0.0f, -10.0f));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	MVPMatrix = glm::mat4(1.0f);
	MVPMatrix = gPerspectiveProjectionMatrix * ViewMatrix * ModelMatrix;
	glUniformMatrix4fv(MVPMatrixUniformLocationBasicShader, 1, GL_FALSE, glm::value_ptr(MVPMatrix));

	pCTorus[1]->DrawTorus(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

	// Cylinder
	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(-15.0f, 0.0f, -10.0f));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	RotationMatrix = glm::rotate(glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	ModelMatrix = ModelMatrix * RotationMatrix;
	MVPMatrix = glm::mat4(1.0f);
	MVPMatrix = gPerspectiveProjectionMatrix * ViewMatrix * ModelMatrix;
	glUniformMatrix4fv(MVPMatrixUniformLocationBasicShader, 1, GL_FALSE, glm::value_ptr(MVPMatrix));

	pCCylinder[1]->DrawCylinder(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

	// Cone
	ModelMatrix = glm::mat4(1.0f);
	TranslationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(-30.0f, 0.0f, -10.0f));
	ModelMatrix = ModelMatrix * TranslationMatrix;
	RotationMatrix = glm::rotate(glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	ModelMatrix = ModelMatrix * RotationMatrix;
	MVPMatrix = glm::mat4(1.0f);
	MVPMatrix = gPerspectiveProjectionMatrix * ViewMatrix * ModelMatrix;
	glUniformMatrix4fv(MVPMatrixUniformLocationBasicShader, 1, GL_FALSE, glm::value_ptr(MVPMatrix));

	pCCone[1]->DrawCone(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

	glUseProgram(0);
}

void Update(void)
{
	if (gbIsDayNightCyclePaused == GL_FALSE)
		pCSkyDome->UpdateSky(gfDeltaTime);
}

void Resize(int Width, int Height)
{
	GLfloat fAspectRatio;

	glm::mat4 SetInfiniteProjectionMatrix(GLfloat Left, GLfloat Right, GLfloat Bottom, GLfloat Top, GLfloat Near, GLfloat Far);

	GLfloat fLeft = 0.0f;
	GLfloat fRight = 0.0f;
	GLfloat fBottom = 0.0f;
	GLfloat fTop = 0.0f;
	GLfloat fNear = FRUSTUM_NEAR;
	GLfloat fFar = FRUSTUM_FAR;

	if (Height == 0)
		Height = 1;

	if (Width == 0)
		Width = 1;

	//if (Width > Height) 
	//{
	//	glViewport(0, (Height - Width) / 2, (GLsizei)Width, (GLsizei)Width);
	//}
	//else 
	//{
	//	glViewport((Width - Height) / 2, 0, (GLsizei)Height, (GLsizei)Height);
	//}

	glViewport(0, 0, (GLsizei)Width, (GLsizei)Height);

	fTop = fNear * (GLfloat)tan(gfFOVY / 360.0 * glm::pi<float>());
	fRight = fTop * ((float)Width / (float)Height);

	fBottom = -fTop;
	fLeft = -fTop * ((float)Width / (float)Height);

	gPerspectiveProjectionMatrix = SetInfiniteProjectionMatrix(fLeft, fRight, fBottom, fTop, fNear, fFar);

	if (Width <= Height)
		fAspectRatio = (GLfloat)Height / (GLfloat)Width;
	else
		fAspectRatio = (GLfloat)Width / (GLfloat)Height;

	if (Width <= Height)
	{
		//gOrthographicProjectionMatrix = glm::ortho(-1.0f, 1.0f, (-1.0f * fAspectRatio), (1.0f * fAspectRatio), -1.0f, 1.0f);
		//gOrthographicProjectionMatrix = glm::ortho(-((GLfloat)Width / 2.0f), ((GLfloat)Width / 2.0f), -((GLfloat)Height / 2.0f) * fAspectRatio, ((GLfloat)Height / 2.0f) * fAspectRatio, 0.0f, 50000.0f);
	}
	else
	{
		//gOrthographicProjectionMatrix = glm::ortho((-1.0f * fAspectRatio), (1.0f * fAspectRatio), -1.0f, 1.0f, -1.0f, 1.0f);
		//gOrthographicProjectionMatrix = glm::ortho(-((GLfloat)Width / 2.0f) * fAspectRatio, ((GLfloat)Width / 2.0f) * fAspectRatio, -((GLfloat)Height / 2.0f), ((GLfloat)Height / 2.0f), 0.0f, 50000.0f);
	}

	gOrthographicProjectionMatrix = glm::ortho(-((GLfloat)Width / 2.0f), ((GLfloat)Width / 2.0f), -((GLfloat)Height / 2.0f), ((GLfloat)Height / 2.0f), 0.0f, 50000.0f);

	//GLfloat aspectRatio = (GLfloat)Width / (GLfloat)Height;
	ResizeFramebuffers(giWinWidth, giWinHeight);
}

void CreateFramebuffer(GLuint Width, GLuint Height)
{
	GLuint OffScreenWidth = Width / OFF_SCREEN_RENDER_RATIO;
	GLuint OffScreenHeight = Height / OFF_SCREEN_RENDER_RATIO;

	// Generate the texture to attach the FBO
	glGenTextures(1, &FBOTexture);

	glBindTexture(GL_TEXTURE_2D, FBOTexture);

	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, OffScreenWidth, OffScreenHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, OffScreenWidth, OffScreenHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	//glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	//glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE);
	glBindTexture(GL_TEXTURE_2D, 0);

	// create a renderbuffer object to store depth info
	glGenRenderbuffers(1, &RenderBufferObjectVolumeLight);
	glBindRenderbuffer(GL_RENDERBUFFER, RenderBufferObjectVolumeLight);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, OffScreenWidth, OffScreenHeight);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);

	// Setup the FBO
	// Create the Framebuffer Object
	glGenFramebuffers(1, &FramebufferObjectVolumeLight);
	glBindFramebuffer(GL_FRAMEBUFFER, FramebufferObjectVolumeLight);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, FBOTexture, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, RenderBufferObjectVolumeLight);

	// check if our fbo is complete
	GLenum FBOStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);

	if (FBOStatus != GL_FRAMEBUFFER_COMPLETE)
	{
		MessageBox(ghWnd, TEXT("FBO Error"), TEXT("Incomplete framebuffer"), MB_OK | MB_ICONERROR);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void ResizeFramebuffers(GLuint Width, GLuint Height)
{
	GLuint OffScreenWidth = Width / OFF_SCREEN_RENDER_RATIO;
	GLuint OffScreenHeight = Height / OFF_SCREEN_RENDER_RATIO;

	glBindFramebuffer(GL_FRAMEBUFFER, FramebufferObjectVolumeLight);


	glBindTexture(GL_TEXTURE_2D, FBOTexture);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, OffScreenWidth, OffScreenHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, FBOTexture, 0);

	glBindTexture(GL_TEXTURE_2D, 0);


	glBindRenderbuffer(GL_RENDERBUFFER, RenderBufferObjectVolumeLight);

	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, OffScreenWidth, OffScreenHeight);

	glBindRenderbuffer(GL_RENDERBUFFER, 0);


	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

glm::vec3 GetLightScreenSpaceCoordinates(glm::vec3 LightPosition, glm::mat4 ModelViewMatrix, glm::mat4 ProjectionMatrix)
{
	glm::vec3 ScreenSpaceCoordinates = glm::vec3(0.0f);
	GLint Viewport[4];
	GLdouble DepthRange[2];

	glGetIntegerv(GL_VIEWPORT, Viewport);
	pCCommon->_check_gl_error(pCCommon->pLogFile, "GeneralLog.txt", __FILE__, __LINE__);
	glGetDoublev(GL_DEPTH_RANGE, DepthRange);
	pCCommon->_check_gl_error(pCCommon->pLogFile, "GeneralLog.txt", __FILE__, __LINE__);

	fprintf(pCCommon->pLogFile, "Viewport X: %d\tViewport Y: %d\tViewport Width: %d\tViewport Height: %d\tDepthRange Near: %lf\tDepthRange Far: %lf\n", Viewport[0], Viewport[1], Viewport[2], Viewport[3], DepthRange[0], DepthRange[1]);

	ScreenSpaceCoordinates = glm::project(LightPosition, ModelViewMatrix, ProjectionMatrix, glm::vec4(Viewport[0], Viewport[1], Viewport[2], Viewport[3]));
	//ScreenSpaceCoordinates = glm::project(LightPosition, ModelMatrix, ProjectionMatrix, glm::vec4(0, 0, 800, 600));

	ScreenSpaceLightPositionX = ScreenSpaceCoordinates.x / ((GLfloat)giWinWidth / OFF_SCREEN_RENDER_RATIO);
	ScreenSpaceLightPositionY = ScreenSpaceCoordinates.y / ((GLfloat)giWinHeight / OFF_SCREEN_RENDER_RATIO);

	fprintf(pCCommon->pLogFile, "LightCoords: %s\tX: %d\tY: %d\n", glm::to_string(ScreenSpaceCoordinates).c_str(), ScreenSpaceLightPositionX, ScreenSpaceLightPositionY);

	return ScreenSpaceCoordinates;
}

glm::mat4 SetInfiniteProjectionMatrix(GLfloat Left, GLfloat Right, GLfloat Bottom, GLfloat Top, GLfloat Near, GLfloat Far)
{
	glm::mat4 result(glm::mat4(1.0f));

	if ((Right == Left) || (Top == Bottom) || (Near == Far) || (Near < 0.0) || (Far < 0.0))
		return result;

	result[0][0] = (2.0f * Near) / (Right - Left);
	result[1][1] = (2.0f * Near) / (Top - Bottom);

	result[2][0] = (Right + Left) / (Right - Left);
	result[2][1] = (Top + Bottom) / (Top - Bottom);
	//result[2][2] = -(Far + Near) / (Far - Near);
	result[2][2] = -1.0f;
	result[2][3] = -1.0f;

	//result[3][2] = -(2.0f * Far * Near) / (Far - Near);
	result[3][2] = -(2.0f * Near);
	result[3][3] = 0.0f;

	return result;
}

void Uninitialize(void)
{


	if (gShaderProgramObjectBasicShader)
	{
		glDeleteProgram(gShaderProgramObjectBasicShader);
		gShaderProgramObjectBasicShader = 0;
	}

	if (gShaderProgramObjectBasicShaderTexture)
	{
		glDeleteProgram(gShaderProgramObjectBasicShaderTexture);
		gShaderProgramObjectBasicShaderTexture = 0;
	}

	// basic shapes cleanup
	delete pCRing[0];
	delete pCRing[1];

	delete pCSphere[0];
	delete pCSphere[1];

	delete pCTorus[0];
	delete pCTorus[1];

	delete pCCylinder[0];
	delete pCCylinder[1];

	delete pCCone[0];
	delete pCCone[1];

	glDeleteTextures(1, &BrickDiffuse);
	glDeleteTextures(1, &BrickDiffuseGranite);
	glDeleteTextures(1, &BrickDiffuseGrey);
	glDeleteTextures(1, &BrickDiffuseOldRed);
	glDeleteTextures(1, &BrickNormal);
	glDeleteTextures(1, &BrickNormalInv);
	glDeleteTextures(1, &BrickNormal2);
	glDeleteTextures(1, &BrickNormalInv2);
	glDeleteTextures(1, &BrickSpecular);
	glDeleteTextures(1, &BrickSpecular2);

	glDeleteTextures(1, &GraniteDiffuse);
	glDeleteTextures(1, &GraniteSpecular);
	glDeleteTextures(1, &GraniteNormal);
	glDeleteTextures(1, &GraniteNormalInv);

	glDeleteTextures(1, &PlasterDiffuseGreen);
	glDeleteTextures(1, &PlasterDiffuseWhite);
	glDeleteTextures(1, &PlasterDiffuseYellow);
	glDeleteTextures(1, &PlasterNormal);
	glDeleteTextures(1, &PlasterNormalInv);
	glDeleteTextures(1, &PlasterSpecular);

	glDeleteTextures(1, &PavingDiffuse);
	glDeleteTextures(1, &PavingSpecular);
	glDeleteTextures(1, &PavingNormal);
	glDeleteTextures(1, &PavingNormalInv);

	glDeleteTextures(1, &WoodPlatesDiffuse);
	glDeleteTextures(1, &WoodPlatesSpecular);
	glDeleteTextures(1, &WoodNormal);
	glDeleteTextures(1, &WoodNormalInv);

	glDeleteProgram(gShaderProgramBasicShapes);
	glDeleteProgram(gShaderProgramObjectBasicShapesNormalMap);


	glUseProgram(0);

	if (pTGA != NULL)
	{
		delete pTGA;
		pTGA = NULL;
	}


	if (pCCamera != NULL)
	{
		delete pCCamera;
		pCCamera = NULL;
	}

	if (pCCommon)
	{
		delete pCCommon;
		pCCommon = NULL;
	}


	wglMakeCurrent(NULL, NULL);

	wglDeleteContext(ghRC);
	ghRC = NULL;

	ReleaseDC(ghWnd, ghDC);
	ghDC = NULL;

	DestroyWindow(ghWnd);
}

void OnLButtonDown(INT LeftClickX, INT LeftClickY, DWORD LeftClickFlags)
{
	RECT Clip;
	//RECT PrevClip;

	SetCapture(ghWnd);
	//GetClipCursor(&PrevClip);
	GetWindowRect(ghWnd, &Clip);
	ClipCursor(&Clip);

	GetCursorPos(&ClickPoint);
	//ShowCursor(FALSE);
}

void OnLButtonUp(INT LeftUpX, INT LeftUpY, DWORD LeftUpFlags)
{
	ClipCursor(NULL);
	ReleaseCapture();
	SetCursorPos(ClickPoint.x, ClickPoint.y);
	ShowCursor(TRUE);
}

void OnRButtonDown(INT RightClickX, INT RightClickY, DWORD RightClickFlags)
{
	RECT Clip;
	//RECT PrevClip;

	SetCapture(ghWnd);
	//GetClipCursor(&PrevClip);
	GetWindowRect(ghWnd, &Clip);
	ClipCursor(&Clip);

	GetCursorPos(&ClickPoint);
	ShowCursor(FALSE);
}

void OnRButtonUp(INT RightUpX, INT RightUpY, DWORD RightUpFlags)
{
	ClipCursor(NULL);
	ReleaseCapture();
	SetCursorPos(ClickPoint.x, ClickPoint.y);
	ShowCursor(TRUE);
}

void OnMButtonDown(INT MiddleDownX, INT MiddleDownY, DWORD MiddleDownFlags)
{
	RECT Clip;
	//RECT PrevClip;

	SetCapture(ghWnd);
	//GetClipCursor(&PrevClip);
	GetWindowRect(ghWnd, &Clip);
	ClipCursor(&Clip);

	GetCursorPos(&ClickPoint);
	ShowCursor(FALSE);
}

void OnMButtonUp(INT MiddleUpX, INT MiddleUpY, DWORD MiddleUpFlags)
{
	ClipCursor(NULL);
	ReleaseCapture();
	SetCursorPos(ClickPoint.x, ClickPoint.y);
	ShowCursor(TRUE);
}

void OnMouseMove(INT MouseMoveX, INT MouseMoveY, DWORD Flags)
{
	//fprintf(stream, "+X Threshold : %f -X Threshold : %f +Y Threshold : %f -Y Threshold : %f\n", iPositiveXThreshold, iNegativeXThreshold, iPositiveYThreshold, iNegativeYThreshold);
	//fprintf(stream, "On Mouse Move X : %d Y : %d\n", MouseMoveX, MouseMoveY);
	//ShowCursor(FALSE);


	if (Flags & MK_LBUTTON)
	{
		// Code
	}

	if (gbIsPlaying == GL_FALSE)
	{
		if (gbFirstMouse)
		{
			gfLastX = (GLfloat)MouseMoveX;
			gfLastY = (GLfloat)MouseMoveY;
			gbFirstMouse = GL_FALSE;
		}

		GLfloat fXOffset = MouseMoveX - gfLastX;
		GLfloat fYOffset = gfLastY - MouseMoveY;

		gfLastX = (GLfloat)MouseMoveX;
		gfLastY = (GLfloat)MouseMoveY;

		if (Flags & MK_RBUTTON)
		{
			pCCamera->ProcessMouseMovement(fXOffset, fYOffset, GL_TRUE);
		}
	}
	else if (gbIsPlaying == GL_TRUE)
	{
		if (gbFirstMouse)
		{
			gfLastX = (GLfloat)MouseMoveX;
			gfLastY = (GLfloat)MouseMoveY;
			gbFirstMouse = GL_FALSE;
		}

		//Player->ControlMouseInput(MouseMoveX, MouseMoveY);
	}
	ShowCursor(TRUE);
}

void OnMouseWheelScroll(short zDelta)
{
	void Resize(int Width, int Height);

	if (zDelta > 0)
	{
		gfFOVY -= ZOOM_FACTOR;
		if (gfFOVY <= 0.1f)
			gfFOVY = 0.1f;
	}
	else
	{
		gfFOVY += ZOOM_FACTOR;
		if (gfFOVY >= 45.0f)
			gfFOVY = 45.0f;
	}

	Resize(giWinWidth, giWinHeight);
}


void SetDayLight(GLuint ShaderObject)
{
	glUniform3fv(glGetUniformLocation(ShaderObject, "SunLight.ambient"), 1, glm::value_ptr(pSunLight->GetAmbient()));
	glUniform3fv(glGetUniformLocation(ShaderObject, "SunLight.diffuse"), 1, glm::value_ptr(pSunLight->GetDiffuse()));
	glUniform3fv(glGetUniformLocation(ShaderObject, "SunLight.specular"), 1, glm::value_ptr(pSunLight->GetSpecular()));
	glUniform3fv(glGetUniformLocation(ShaderObject, "SunLight.direction"), 1, glm::value_ptr(pSunLight->GetDirection()));

	glUniform3fv(glGetUniformLocation(ShaderObject, "u_view_position"), 1, glm::value_ptr(CameraPosition));
}

void SetDayLightNormalMap(GLuint ShaderObject)
{
	glUniform3fv(glGetUniformLocation(ShaderObject, "SunLight.ambient"), 1, glm::value_ptr(pSunLight->GetAmbient()));
	glUniform3fv(glGetUniformLocation(ShaderObject, "SunLight.diffuse"), 1, glm::value_ptr(pSunLight->GetDiffuse()));
	glUniform3fv(glGetUniformLocation(ShaderObject, "SunLight.specular"), 1, glm::value_ptr(pSunLight->GetSpecular()));
	glUniform3fv(glGetUniformLocation(ShaderObject, "SunLight.direction"), 1, glm::value_ptr(pSunLight->GetDirection()));

	glUniform3fv(glGetUniformLocation(ShaderObject, "u_light_direction"), 1, glm::value_ptr(pSunLight->GetDirection()));
	glUniform3fv(glGetUniformLocation(ShaderObject, "u_view_position"), 1, glm::value_ptr(CameraPosition));
}
